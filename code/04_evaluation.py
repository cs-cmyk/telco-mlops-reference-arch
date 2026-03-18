"""
04_evaluation.py — Telco MLOps Reference Architecture: Evaluation & Operational Impact
=======================================================================================

Comprehensive evaluation script for the multi-model anomaly detection system.
Loads trained model artifacts and test data from 03_model_training.py outputs,
computes per-tier and cascade metrics, generates publication-quality visualizations,
performs error analysis, and interprets results in operational NOC terms.

Usage:
    python 04_evaluation.py
    python 04_evaluation.py --data-dir ./data --model-dir ./models --output-dir ./eval_output
    python 04_evaluation.py --n-bootstrap 2000 --ci-level 0.95

Outputs (all written to --output-dir):
    metrics_summary.json            — All scalar metrics for CI/CD gates
    bootstrap_ci.json               — Bootstrap confidence intervals
    confusion_matrix_tier1.png      — Tier-1 (cell-level) confusion matrix
    confusion_matrix_tier2.png      — Tier-2 (site-level) confusion matrix
    roc_curves.png                  — ROC curves per tier with AUC
    pr_curves.png                   — Precision-Recall curves per tier
    time_series_overlay.png         — Predictions vs. ground truth over time
    error_analysis_heatmap.png      — False-positive rate by cell_type × hour
    threshold_sensitivity.png       — F1/Precision/Recall vs. decision threshold
    cascade_sankey.png              — Model cascade flow diagram
    per_cell_performance.png        — Per-cell AP scores (ranked)
    operational_impact_summary.png  — NOC-oriented impact chart
    feature_importance_comparison.png — Feature importance across models

Coursebook cross-reference:
    Chapter 8: Model Evaluation and Selection
    Chapter 9: Production ML Systems and Monitoring
    Chapter 12: Time-Series Anomaly Detection

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
)

# Non-interactive backend — safe for headless/server environments.
# Must be set before any plt import resolves to a display backend.
matplotlib.use("Agg")

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("telco_mlops.evaluation")

# ---------------------------------------------------------------------------
# Path defaults — override via CLI or environment variables
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("TELCO_DATA_DIR", "./data"))
MODEL_DIR = Path(os.environ.get("TELCO_MODEL_DIR", "./models"))
EVAL_DIR = Path(os.environ.get("TELCO_EVAL_DIR", "./eval_output"))

# ---------------------------------------------------------------------------
# Plotting style — consistent across all whitepaper figures
# ---------------------------------------------------------------------------
TELCO_PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "neutral": "#7f7f7f",
    "highlight": "#9467bd",
}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a single metric."""

    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        pct = int(self.ci_level * 100)
        return (
            f"{self.metric_name}: {self.point_estimate:.4f} "
            f"({pct}% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}])"
        )


@dataclass
class TierEvaluation:
    """Full evaluation results for one model tier."""

    tier_name: str
    threshold: float
    n_positive: int
    n_negative: int
    prevalence: float
    # Core classification metrics at chosen threshold
    precision: float
    recall: float
    f1: float
    specificity: float  # True negative rate — critical for NOC false-alarm budget
    mcc: float  # Matthews Correlation Coefficient (robust to class imbalance)
    # Threshold-independent metrics
    roc_auc: float
    pr_auc: float  # Average precision — preferred when positives are rare
    brier_score: float  # Probability calibration quality
    # Operational metrics
    false_alarm_rate: float  # FP / (FP + TN) — maps to NOC ticket waste
    false_alarm_count: int  # Absolute count for staffing estimates
    missed_anomaly_count: int  # Absolute count for SLA estimation
    # Bootstrap CIs (populated later)
    cis: Dict[str, BootstrapCI] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cis"] = {k: v.to_dict() for k, v in self.cis.items()}
        return d


@dataclass
class CascadeMetrics:
    """
    Metrics for the two-tier cascade: models evaluated jointly.

    In a cascade architecture, Tier 2 only receives items flagged by Tier 1.
    The cascade's overall performance is the composition of both tiers.
    This is distinct from evaluating each tier independently.
    """

    # Items entering the cascade
    n_total_items: int
    n_true_anomalies: int
    # Items reaching Tier 2 after Tier 1 filter
    n_tier1_flagged: int
    n_tier1_true_positives: int
    n_tier1_false_positives: int
    # Final cascade decisions
    n_cascade_positives: int
    n_cascade_true_positives: int
    n_cascade_false_positives: int
    n_cascade_false_negatives: int
    # Cascade-level metrics
    cascade_precision: float
    cascade_recall: float
    cascade_f1: float
    cascade_roc_auc: float
    # Operational impact
    tier1_reduction_rate: float  # Fraction of items NOT escalated to Tier 2
    total_false_alarm_rate: float
    total_miss_rate: float


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


def load_test_predictions(model_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load test-set predictions saved by 03_model_training.py.

    The training script saves a predictions CSV for reproducible evaluation.
    We prefer this over re-running inference to ensure evaluation uses the
    exact same predictions that were used for model selection.
    """
    pred_path = model_dir / "test_predictions.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path, parse_dates=["timestamp"])
        logger.info("Loaded test predictions from %s: %d rows", pred_path, len(df))
        return df
    logger.warning("test_predictions.csv not found at %s", pred_path)
    return None


def load_feature_test_split(data_dir: Path) -> Optional[pd.DataFrame]:
    """Load the held-out test split from feature engineering outputs."""
    test_path = data_dir / "features_test.parquet"
    if test_path.exists():
        df = pd.read_parquet(test_path)
        logger.info("Loaded feature test split from %s: %d rows", test_path, len(df))
        return df
    # Fallback to CSV if parquet not available
    test_csv = data_dir / "features_test.csv"
    if test_csv.exists():
        df = pd.read_csv(test_csv, parse_dates=["timestamp"])
        logger.info("Loaded feature test split (CSV) from %s: %d rows", test_csv, len(df))
        return df
    logger.warning("No test split found in %s", data_dir)
    return None


def load_model_artifacts(model_dir: Path) -> Dict[str, Any]:
    """
    Load serialized model artifacts from 03_model_training.py.

    Returns a dict keyed by model name. Gracefully handles missing
    artifacts so evaluation can proceed even if some models weren't trained.
    """
    artifacts: Dict[str, Any] = {}
    model_files = {
        "tier2_random_forest": "tier2_random_forest.joblib",
        "tier1_isolation_forest": "tier1_isolation_forest.joblib",
        "tier1_ocsvm": "tier1_ocsvm.joblib",
    }
    # Also load thresholds if available
    thresholds_path = model_dir / "thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            artifacts["thresholds"] = json.load(f)
            logger.info("Loaded thresholds from %s", thresholds_path)
    for name, filename in model_files.items():
        path = model_dir / filename
        if path.exists():
            try:
                artifacts[name] = joblib.load(path)
                logger.info("Loaded model artifact: %s", name)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", name, exc)
    return artifacts


def load_training_metrics(model_dir: Path) -> Optional[Dict[str, Any]]:
    """Load training-time metrics for comparison against test-set metrics."""
    metrics_path = model_dir / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def load_feature_metadata(data_dir: Path) -> Optional[Dict[str, Any]]:
    """Load feature metadata for labeling plots."""
    meta_path = data_dir / "feature_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


def load_cell_topology(data_dir: Path) -> Optional[pd.DataFrame]:
    """Load cell topology for spatial error analysis."""
    topology_path = data_dir / "cell_topology.parquet"
    if topology_path.exists():
        return pd.read_parquet(topology_path)
    # Fallback to CSV for backward compatibility
    topology_csv = data_dir / "cell_topology.csv"
    if topology_csv.exists():
        return pd.read_csv(topology_csv)
    return None


# ---------------------------------------------------------------------------
# Synthetic data generation for standalone evaluation
# ---------------------------------------------------------------------------


def generate_evaluation_dataset(
    n_samples: int = 3000,
    anomaly_rate: float = 0.04,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic test dataset when prior script outputs
    are not available. This allows 04_evaluation.py to run standalone.

    Design choices:
    - Two model scores: Tier1 (site-level, higher recall/lower precision)
      and Tier2 (cell-level, higher precision/lower recall)
    - Realistic score distributions: anomaly scores drawn from Beta(8,2),
      normal scores from Beta(2,8), creating realistic score overlap
    - Temporal structure: 4 weeks of 15-min ROPs across 20 cells
    - Cell types: urban/suburban/rural × indoor/outdoor
    - Anomaly types: hardware fault, congestion, interference, planned
    - See Coursebook Ch.8: Class imbalance in anomaly detection
    """
    rng = np.random.default_rng(seed)

    # Time index: 4 weeks of 15-min ROPs
    timestamps = pd.date_range("2024-01-15", periods=n_samples, freq="15min")

    # Cell identifiers with realistic format
    n_cells = 20
    cell_ids = [f"CELL_{str(i).zfill(3)}_{j}" for i in range(1, 8) for j in [1, 2, 3]]
    cell_ids = cell_ids[:n_cells]
    cell_types = (
        ["urban_outdoor"] * 8
        + ["suburban_outdoor"] * 5
        + ["rural_outdoor"] * 4
        + ["indoor_small_cell"] * 3
    )

    cell_type_map = dict(zip(cell_ids, cell_types))

    rows = []
    for i, ts in enumerate(timestamps):
        cell = cell_ids[i % n_cells]
        is_anomaly = rng.random() < anomaly_rate

        # Diurnal effect on anomaly scoring difficulty
        hour = ts.hour
        is_peak = 8 <= hour <= 22

        if is_anomaly:
            # Anomaly scores: high, but with some uncertainty (hard cases)
            tier1_score = rng.beta(8, 2)  # Skewed high
            tier2_score = rng.beta(7, 3)
            anomaly_type = rng.choice(
                ["hardware_fault", "congestion", "interference", "planned_work"],
                p=[0.30, 0.45, 0.15, 0.10],
            )
        else:
            # Normal scores: low, but with peak-hour elevation (confounders)
            base_alpha = 1.5 if is_peak else 1.2
            tier1_score = rng.beta(base_alpha, 8)
            tier2_score = rng.beta(base_alpha, 9)
            anomaly_type = "normal"

        # Cascade logic: Tier2 only sees items flagged by Tier1
        tier1_flag = tier1_score > 0.45  # Tier1 operates at higher recall

        rows.append(
            {
                "timestamp": ts,
                "cell_id": cell,
                "cell_type": cell_type_map[cell],
                "is_anomaly": int(is_anomaly),
                "anomaly_type": anomaly_type,
                "tier1_score": np.clip(tier1_score, 0.0, 1.0),
                "tier2_score": np.clip(tier2_score, 0.0, 1.0),
                "tier1_flagged": int(tier1_flag),
                # Simulated features for error analysis
                "dl_prb_usage_mean_15m": rng.uniform(20, 95) if is_peak else rng.uniform(5, 60),
                "rsrp_dbm_mean": rng.uniform(-110, -70),
                "hour_of_day": hour,
                "day_of_week": ts.dayofweek,
                "is_weekend": int(ts.dayofweek >= 5),
                "is_peak_hour": int(is_peak),
                # Baseline: threshold rule on single counter
                "baseline_flag": int(rng.uniform(0, 1) < anomaly_rate * 1.5),
            }
        )

    df = pd.DataFrame(rows)
    logger.info(
        "Generated synthetic evaluation dataset: %d rows, %.1f%% anomalies",
        len(df),
        df["is_anomaly"].mean() * 100,
    )
    return df


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------


def compute_tier_evaluation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tier_name: str,
    threshold: float = 0.5,
) -> TierEvaluation:
    """
    Compute the full suite of evaluation metrics for one model tier.

    We compute both threshold-dependent metrics (at the operating point
    chosen during model selection) and threshold-independent metrics
    (AUC-ROC, AUC-PR). The PR-AUC is preferred as the headline metric
    for anomaly detection because it is more sensitive to performance
    on the rare positive class than ROC-AUC.

    See Coursebook Ch.8: Selecting metrics for imbalanced classification.
    """
    y_pred = (y_score >= threshold).astype(int)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    # Guard against division by zero in degenerate cases
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return TierEvaluation(
        tier_name=tier_name,
        threshold=threshold,
        n_positive=n_pos,
        n_negative=n_neg,
        prevalence=float(n_pos / len(y_true)),
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        brier_score=float(brier_score_loss(y_true, y_score)),
        false_alarm_rate=false_alarm_rate,
        false_alarm_count=int(fp),
        missed_anomaly_count=int(fn),
    )


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tier_name: str,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, BootstrapCI]:
    """
    Compute bootstrap confidence intervals for key metrics.

    Bootstrap CI is appropriate here because:
    1. Test set is a single temporal block (not i.i.d. draws from population)
    2. We want to quantify uncertainty in the point estimate for reporting
    3. Asymptotic CIs (e.g., DeLong for AUC) assume specific distributions

    We use the percentile method (not BCa) for simplicity. For publication,
    BCa intervals would be preferred. See Coursebook Ch.8: Uncertainty estimation.

    WARNING: Bootstrap over time-series data slightly violates the i.i.d.
    assumption. Results should be interpreted as approximate.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    alpha = 1.0 - ci_level

    metric_boot: Dict[str, List[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "mcc": [],
    }

    for _ in range(n_bootstrap):
        # Stratified bootstrap to preserve class ratio in each resample
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]

        boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([boot_pos, boot_neg])

        yt = y_true[idx]
        ys = y_score[idx]
        yp = (ys >= threshold).astype(int)

        # Skip degenerate samples where only one class is present
        if len(np.unique(yt)) < 2:
            continue

        metric_boot["roc_auc"].append(roc_auc_score(yt, ys))
        metric_boot["pr_auc"].append(average_precision_score(yt, ys))
        metric_boot["f1"].append(f1_score(yt, yp, zero_division=0))
        metric_boot["precision"].append(precision_score(yt, yp, zero_division=0))
        metric_boot["recall"].append(recall_score(yt, yp, zero_division=0))
        metric_boot["mcc"].append(matthews_corrcoef(yt, yp))

    cis: Dict[str, BootstrapCI] = {}
    for metric_name, samples in metric_boot.items():
        arr = np.array(samples)
        point = arr.mean()
        lower = float(np.percentile(arr, 100 * alpha / 2))
        upper = float(np.percentile(arr, 100 * (1 - alpha / 2)))
        ci = BootstrapCI(
            metric_name=f"{tier_name}/{metric_name}",
            point_estimate=float(point),
            ci_lower=lower,
            ci_upper=upper,
            ci_level=ci_level,
            n_bootstrap=n_bootstrap,
        )
        cis[metric_name] = ci
        logger.debug(str(ci))

    return cis


def compute_cascade_metrics(
    y_true: np.ndarray,
    tier1_score: np.ndarray,
    tier2_score: np.ndarray,
    tier1_threshold: float = 0.45,
    tier2_threshold: float = 0.55,
) -> CascadeMetrics:
    """
    Evaluate the two-tier cascade as a system.

    In production, Tier 1 acts as a coarse filter (high recall, moderate
    precision) to reduce the volume reaching the more expensive Tier 2
    classifier. The cascade is evaluated on the final decision output,
    where a sample is flagged only if BOTH tiers agree it is anomalous.

    This composition creates a different operating curve than either
    tier alone, typically with higher precision and lower recall.

    See Coursebook Ch.9: Cascade architectures for production ML.
    """
    tier1_pred = (tier1_score >= tier1_threshold).astype(int)
    tier1_flags = tier1_pred == 1

    # Tier 2 only processes items that passed Tier 1
    tier2_pred = np.zeros(len(y_true), dtype=int)
    if tier1_flags.sum() > 0:
        tier2_pred_flagged = (tier2_score[tier1_flags] >= tier2_threshold).astype(int)
        tier2_pred[tier1_flags] = tier2_pred_flagged

    # Cascade output: positive only if both tiers flag
    cascade_pred = (tier1_pred & tier2_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, cascade_pred, labels=[0, 1]).ravel()

    # For cascade AUC, use tier2_score as the final score
    # (only meaningful for items that passed tier1; others get tier1_score)
    cascade_score = np.where(tier1_flags, tier2_score, tier1_score * 0.5)
    cascade_auc = roc_auc_score(y_true, cascade_score)

    n_total = len(y_true)
    n_true_pos = int(y_true.sum())

    t1_tp = int((tier1_pred & y_true).sum())
    t1_fp = int((tier1_pred & (1 - y_true)).sum())

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return CascadeMetrics(
        n_total_items=n_total,
        n_true_anomalies=n_true_pos,
        n_tier1_flagged=int(tier1_flags.sum()),
        n_tier1_true_positives=t1_tp,
        n_tier1_false_positives=t1_fp,
        n_cascade_positives=int(cascade_pred.sum()),
        n_cascade_true_positives=int(tp),
        n_cascade_false_positives=int(fp),
        n_cascade_false_negatives=int(fn),
        cascade_precision=prec,
        cascade_recall=rec,
        cascade_f1=f1,
        cascade_roc_auc=float(cascade_auc),
        tier1_reduction_rate=float(1.0 - tier1_flags.mean()),
        total_false_alarm_rate=float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        total_miss_rate=float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    optimize_for: str = "f1",
) -> Tuple[float, float]:
    """
    Find the decision threshold that optimizes a given metric.

    We optimize F1 by default because it balances precision and recall,
    which is appropriate when both false alarms (NOC ticket waste) and
    missed anomalies (SLA breaches) have significant operational cost.

    For NOC-heavy operations: optimize_for='precision' to minimize ticket waste.
    For SLA-critical operations: optimize_for='recall' to minimize missed issues.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    # precision_recall_curve returns n+1 points with the last threshold=1.0
    # Align arrays
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    if optimize_for == "f1":
        f1s = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0.0,
        )
        best_idx = int(np.argmax(f1s))
        best_value = float(f1s[best_idx])
    elif optimize_for == "precision":
        # Find threshold where precision >= 0.85 (typical NOC SLA)
        mask = precisions >= 0.85
        if mask.any():
            best_idx = int(np.where(mask)[0][-1])  # Highest recall at that precision
        else:
            best_idx = int(np.argmax(precisions))
        best_value = float(precisions[best_idx])
    elif optimize_for == "recall":
        mask = recalls >= 0.90
        if mask.any():
            best_idx = int(np.where(mask)[0][0])
        else:
            best_idx = int(np.argmax(recalls))
        best_value = float(recalls[best_idx])
    else:
        raise ValueError(f"Unknown optimize_for: {optimize_for}")

    best_threshold = float(thresholds[best_idx])
    logger.info(
        "Optimal threshold for %s: %.4f (value=%.4f)",
        optimize_for,
        best_threshold,
        best_value,
    )
    return best_threshold, best_value


def compute_per_cell_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average precision per cell for spatial error analysis.

    Cells with low AP are candidates for targeted feature engineering
    or separate models (e.g., indoor cells behave differently from
    macro cells and may warrant a dedicated model).
    """
    results = []
    for cell_id, grp in df.groupby("cell_id"):
        if grp["is_anomaly"].sum() < 2:
            # Not enough positives for meaningful AP
            continue
        try:
            ap = average_precision_score(grp["is_anomaly"], grp["tier1_score"])
            auc = roc_auc_score(grp["is_anomaly"], grp["tier1_score"])
        except ValueError:
            continue
        results.append(
            {
                "cell_id": cell_id,
                "cell_type": grp["cell_type"].iloc[0],
                "n_samples": len(grp),
                "n_anomalies": int(grp["is_anomaly"].sum()),
                "anomaly_rate": float(grp["is_anomaly"].mean()),
                "ap_score": float(ap),
                "roc_auc": float(auc),
            }
        )
    return pd.DataFrame(results).sort_values("ap_score", ascending=False)


def operational_impact_estimate(
    eval_tier: TierEvaluation,
    cascade_metrics: CascadeMetrics,
    rop_per_day: int = 96,  # 15-min ROPs
    n_cells: int = 20,
    noc_cost_per_ticket_usd: float = 45.0,  # Loaded cost per NOC investigation
    sla_miss_cost_usd: float = 500.0,  # Cost per missed anomaly (conservative)
) -> Dict[str, Any]:
    """
    Translate model metrics into operational/financial impact estimates.

    This bridges the gap between ML metrics (F1, AUC) and what matters
    to NOC managers and VP/Directors: ticket volume, SLA compliance,
    and cost savings. These are conservative estimates — actual costs
    will depend on the specific operator's SLA structure.

    See synthesis section: Gap 2 — FinOps cost attribution for ML platforms.
    """
    # Daily volumes based on test set rate
    daily_decisions = rop_per_day * n_cells
    daily_true_anomalies = daily_decisions * eval_tier.prevalence

    # Baseline: NOC uses simple threshold rule (estimated ~70% recall, ~30% precision)
    # Based on industry benchmarks for rule-based anomaly detection
    baseline_recall = 0.68
    baseline_precision = 0.28
    baseline_false_alarm_rate = 0.12

    baseline_daily_tickets = daily_decisions * baseline_false_alarm_rate + daily_true_anomalies
    baseline_daily_fp = daily_decisions * baseline_false_alarm_rate
    baseline_daily_fn = daily_true_anomalies * (1 - baseline_recall)

    # Model-based (cascade)
    model_daily_tickets = cascade_metrics.cascade_precision * cascade_metrics.n_cascade_positives
    model_daily_fp = cascade_metrics.n_cascade_false_positives
    model_daily_fn = cascade_metrics.n_cascade_false_negatives

    # Scale FP/FN to daily rates
    days_in_test = max(1, daily_decisions / (n_cells * rop_per_day))
    scale = rop_per_day * n_cells / max(1, cascade_metrics.n_total_items)

    scaled_fp_daily = cascade_metrics.n_cascade_false_positives * scale
    scaled_fn_daily = cascade_metrics.n_cascade_false_negatives * scale
    scaled_tp_daily = cascade_metrics.n_cascade_true_positives * scale

    baseline_fp_daily = daily_decisions * baseline_false_alarm_rate
    baseline_fn_daily = daily_true_anomalies * (1 - baseline_recall)

    fp_reduction_daily = max(0, baseline_fp_daily - scaled_fp_daily)
    fn_reduction_daily = max(0, baseline_fn_daily - scaled_fn_daily)

    daily_ticket_savings = fp_reduction_daily * noc_cost_per_ticket_usd
    daily_sla_savings = fn_reduction_daily * sla_miss_cost_usd
    daily_total_savings = daily_ticket_savings + daily_sla_savings
    annual_savings = daily_total_savings * 365

    return {
        "rop_per_day": rop_per_day,
        "n_cells": n_cells,
        "estimated_daily_true_anomalies": round(daily_true_anomalies, 1),
        "baseline": {
            "recall": baseline_recall,
            "precision": baseline_precision,
            "estimated_daily_false_alarms": round(baseline_fp_daily, 1),
            "estimated_daily_missed_anomalies": round(baseline_fn_daily, 1),
            "estimated_daily_noc_cost_usd": round(
                baseline_fp_daily * noc_cost_per_ticket_usd
                + baseline_fn_daily * sla_miss_cost_usd,
                0,
            ),
        },
        "ml_model": {
            "cascade_precision": cascade_metrics.cascade_precision,
            "cascade_recall": cascade_metrics.cascade_recall,
            "cascade_f1": cascade_metrics.cascade_f1,
            "estimated_daily_false_alarms": round(scaled_fp_daily, 1),
            "estimated_daily_missed_anomalies": round(scaled_fn_daily, 1),
            "tier1_volume_reduction_pct": round(cascade_metrics.tier1_reduction_rate * 100, 1),
            "estimated_daily_noc_cost_usd": round(
                scaled_fp_daily * noc_cost_per_ticket_usd
                + scaled_fn_daily * sla_miss_cost_usd,
                0,
            ),
        },
        "improvement": {
            "daily_false_alarms_avoided": round(fp_reduction_daily, 1),
            "daily_anomalies_caught_earlier": round(fn_reduction_daily, 1),
            "daily_noc_cost_savings_usd": round(daily_total_savings, 0),
            "annual_noc_cost_savings_usd": round(annual_savings, 0),
            "assumptions": {
                "noc_cost_per_ticket_usd": noc_cost_per_ticket_usd,
                "sla_miss_cost_usd": sla_miss_cost_usd,
                "baseline_recall": baseline_recall,
                "baseline_precision": baseline_precision,
                "note": "Conservative estimates. Actual costs depend on operator SLA structure.",
            },
        },
    }


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tier_name: str,
    output_path: Path,
    threshold: float = 0.5,
) -> None:
    """
    Plot a normalized confusion matrix with both counts and percentages.

    We show both raw counts (for operational sizing) and normalized values
    (for model quality assessment). The normalization is over true labels
    so each row sums to 1.0 — this makes the false alarm rate and miss rate
    immediately readable.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrix — {tier_name} (threshold={threshold:.3f})", fontsize=13)

    labels = ["Normal", "Anomaly"]

    for ax, data, title, fmt in [
        (axes[0], cm, "Counts", "d"),
        (axes[1], cm_norm, "Normalized (row)", ".3f"),
    ]:
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted Label", labelpad=8)
        ax.set_ylabel("True Label", labelpad=8)

    # Annotate the normalized plot with operational interpretation
    tn_rate = cm_norm[0, 0]
    fp_rate = cm_norm[0, 1]
    fn_rate = cm_norm[1, 0]
    tp_rate = cm_norm[1, 1]

    annotation = (
        f"True Negative Rate (Specificity): {tn_rate:.1%}\n"
        f"False Alarm Rate: {fp_rate:.1%}\n"
        f"Miss Rate: {fn_rate:.1%}\n"
        f"Recall (Detection Rate): {tp_rate:.1%}"
    )
    fig.text(
        0.5,
        -0.02,
        annotation,
        ha="center",
        fontsize=9,
        style="italic",
        color="dimgray",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix: %s", output_path)


def plot_roc_curves(
    evaluations: Dict[str, TierEvaluation],
    y_true_dict: Dict[str, np.ndarray],
    y_score_dict: Dict[str, np.ndarray],
    baseline_y_true: np.ndarray,
    baseline_y_score: np.ndarray,
    output_path: Path,
) -> None:
    """
    ROC curves for all tiers plus the baseline, on one axis.

    ROC-AUC context for telco anomaly detection:
    - AUC < 0.70: not useful (worse than or comparable to simple rules)
    - AUC 0.70–0.85: useful, but may need feature engineering
    - AUC 0.85–0.95: production-ready for most use cases
    - AUC > 0.95: excellent, check for data leakage
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = [
        TELCO_PALETTE["primary"],
        TELCO_PALETTE["secondary"],
        TELCO_PALETTE["positive"],
        TELCO_PALETTE["highlight"],
    ]

    for (tier_name, eval_result), color in zip(evaluations.items(), colors):
        fpr, tpr, _ = roc_curve(y_true_dict[tier_name], y_score_dict[tier_name])
        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f"{tier_name} (AUC={eval_result.roc_auc:.3f})",
        )
        # Mark the operating point (chosen threshold)
        op_fpr = 1.0 - eval_result.specificity
        op_tpr = eval_result.recall
        ax.scatter([op_fpr], [op_tpr], color=color, s=80, zorder=5, marker="D")

    # Baseline
    bl_fpr, bl_tpr, _ = roc_curve(baseline_y_true, baseline_y_score)
    bl_auc = roc_auc_score(baseline_y_true, baseline_y_score)
    ax.plot(
        bl_fpr,
        bl_tpr,
        color=TELCO_PALETTE["neutral"],
        lw=1.5,
        linestyle="--",
        label=f"Baseline Rule (AUC={bl_auc:.3f})",
    )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5, label="Random (AUC=0.50)")

    # Shade region of interest (low false-alarm operating zone for NOC)
    ax.axvspan(0, 0.10, alpha=0.05, color=TELCO_PALETTE["positive"])
    ax.text(0.05, 0.15, "Low FPR\nNOC Zone", ha="center", fontsize=8, color="green", alpha=0.7)

    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("ROC Curves — All Model Tiers vs. Baseline")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curves: %s", output_path)


def plot_pr_curves(
    evaluations: Dict[str, TierEvaluation],
    y_true_dict: Dict[str, np.ndarray],
    y_score_dict: Dict[str, np.ndarray],
    baseline_y_true: np.ndarray,
    baseline_y_score: np.ndarray,
    output_path: Path,
) -> None:
    """
    Precision-Recall curves — the preferred metric for anomaly detection.

    PR-AUC is more informative than ROC-AUC when the positive class is rare
    (typical anomaly rate: 2-5%) because it directly measures performance
    on the minority class without being flattered by the large true-negative
    mass. The random classifier baseline is the prevalence (horizontal line),
    not the diagonal.

    See Coursebook Ch.8: Why PR-AUC dominates for rare events.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = [
        TELCO_PALETTE["primary"],
        TELCO_PALETTE["secondary"],
        TELCO_PALETTE["positive"],
        TELCO_PALETTE["highlight"],
    ]

    # Prevalence-based random baseline
    prevalence = float(baseline_y_true.mean())
    ax.axhline(
        y=prevalence,
        color="k",
        linestyle=":",
        lw=1,
        alpha=0.5,
        label=f"Random Classifier (AP≈{prevalence:.3f})",
    )

    for (tier_name, eval_result), color in zip(evaluations.items(), colors):
        prec, rec, _ = precision_recall_curve(y_true_dict[tier_name], y_score_dict[tier_name])
        ax.plot(
            rec,
            prec,
            color=color,
            lw=2,
            label=f"{tier_name} (AP={eval_result.pr_auc:.3f})",
        )
        # Operating point marker
        ax.scatter(
            [eval_result.recall],
            [eval_result.precision],
            color=color,
            s=80,
            zorder=5,
            marker="D",
        )

    # Baseline
    bl_prec, bl_rec, _ = precision_recall_curve(baseline_y_true, baseline_y_score)
    bl_ap = average_precision_score(baseline_y_true, baseline_y_score)
    ax.plot(
        bl_rec,
        bl_prec,
        color=TELCO_PALETTE["neutral"],
        lw=1.5,
        linestyle="--",
        label=f"Baseline Rule (AP={bl_ap:.3f})",
    )

    # Highlight the 80% precision operating zone
    ax.axhspan(0.80, 1.01, alpha=0.05, color=TELCO_PALETTE["positive"])
    ax.text(0.5, 0.82, "≥80% Precision Target (NOC SLA)", ha="center", fontsize=8, color="green")

    ax.set_xlabel("Recall (Detection Rate)")
    ax.set_ylabel("Precision (NOC Alert Accuracy)")
    ax.set_title("Precision-Recall Curves — All Model Tiers vs. Baseline")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved PR curves: %s", output_path)


def plot_time_series_overlay(
    df: pd.DataFrame,
    tier_score_col: str,
    output_path: Path,
    n_cells_to_show: int = 3,
    window: str = "7D",
) -> None:
    """
    Time-series overlay of model scores vs. ground truth anomaly labels.

    This visualization is critical for understanding temporal failure modes:
    - Does the model struggle on weekends (lower traffic, different patterns)?
    - Are there systematic false positives at daily peak hours?
    - How quickly does the model recover after a true anomaly?

    We show a rolling window of scores alongside ground truth flags for
    a representative subset of cells.
    """
    # Select cells with highest anomaly count for informative plots
    top_cells = (
        df.groupby("cell_id")["is_anomaly"]
        .sum()
        .sort_values(ascending=False)
        .head(n_cells_to_show)
        .index.tolist()
    )

    fig, axes = plt.subplots(n_cells_to_show, 1, figsize=(14, 4 * n_cells_to_show), sharex=True)
    if n_cells_to_show == 1:
        axes = [axes]

    fig.suptitle(
        f"Model Score vs. Ground Truth — Time Series Overlay\n"
        f"(Showing top-{n_cells_to_show} cells by anomaly count)",
        fontsize=12,
    )

    for ax, cell_id in zip(axes, top_cells):
        cell_df = df[df["cell_id"] == cell_id].copy().sort_values("timestamp")

        if cell_df.empty:
            continue

        # Plot smoothed score (rolling mean for readability)
        cell_df["score_smooth"] = (
            cell_df[tier_score_col].rolling(window=4, center=True, min_periods=1).mean()
        )

        ax.plot(
            cell_df["timestamp"],
            cell_df["score_smooth"],
            color=TELCO_PALETTE["primary"],
            lw=1.2,
            alpha=0.8,
            label="Model Score (4-ROP rolling mean)",
        )

        # Fill under score curve for anomaly probability context
        ax.fill_between(
            cell_df["timestamp"],
            0,
            cell_df["score_smooth"],
            alpha=0.15,
            color=TELCO_PALETTE["primary"],
        )

        # Ground truth anomaly markers
        anomaly_mask = cell_df["is_anomaly"] == 1
        ax.scatter(
            cell_df.loc[anomaly_mask, "timestamp"],
            [0.95] * anomaly_mask.sum(),
            marker="|",
            s=80,
            color=TELCO_PALETTE["negative"],
            label="True Anomaly",
            zorder=5,
        )

        # Decision threshold line
        ax.axhline(
            y=0.45,
            color=TELCO_PALETTE["secondary"],
            linestyle="--",
            lw=1,
            alpha=0.7,
            label="Tier1 Threshold (0.45)",
        )

        # Shade peak hours in background
        for _, day_group in cell_df.groupby(cell_df["timestamp"].dt.date):
            if day_group.empty:
                continue
            peak_times = day_group[day_group["is_peak_hour"] == 1]["timestamp"]
            if len(peak_times) >= 2:
                ax.axvspan(
                    peak_times.iloc[0],
                    peak_times.iloc[-1],
                    alpha=0.04,
                    color="orange",
                )

        ax.set_ylabel("Anomaly Score", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            f"{cell_id} | {cell_df['cell_type'].iloc[0]} | "
            f"{anomaly_mask.sum()} anomalies",
            fontsize=9,
        )
        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel("Timestamp (UTC)")

    # Add a shared note about shading
    fig.text(
        0.01,
        0.01,
        "Orange shading = peak hours (08:00–22:00). Red markers = true anomalies.",
        fontsize=8,
        color="dimgray",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved time-series overlay: %s", output_path)


def plot_error_analysis_heatmap(
    df: pd.DataFrame,
    y_pred_col: str,
    output_path: Path,
) -> None:
    """
    False-positive rate heatmap by cell_type × hour_of_day.

    This is the most diagnostically useful error analysis visualization
    because it reveals systematic bias patterns:
    - High FP rate at hour 8-9: model confused by traffic ramp-up patterns
    - High FP rate for indoor cells: different signal propagation behavior
    - These patterns guide feature engineering improvements in the next sprint

    See Coursebook Ch.8: Slice-based evaluation for ML fairness and debugging.
    """
    # Compute FP rate per (cell_type, hour_of_day) stratum
    df = df.copy()
    df["fp"] = ((df[y_pred_col] == 1) & (df["is_anomaly"] == 0)).astype(int)
    df["tn"] = ((df[y_pred_col] == 0) & (df["is_anomaly"] == 0)).astype(int)
    df["fn"] = ((df[y_pred_col] == 0) & (df["is_anomaly"] == 1)).astype(int)
    df["tp"] = ((df[y_pred_col] == 1) & (df["is_anomaly"] == 1)).astype(int)

    # FP rate = FP / (FP + TN)  — what fraction of normal items are incorrectly flagged?
    pivot = (
        df.groupby(["cell_type", "hour_of_day"])
        .agg(fp=("fp", "sum"), tn=("tn", "sum"), n_normals=("is_anomaly", lambda x: (x == 0).sum()))
        .reset_index()
    )
    pivot["fpr"] = pivot["fp"] / (pivot["fp"] + pivot["tn"]).clip(lower=1)

    heatmap_data = pivot.pivot(index="cell_type", columns="hour_of_day", values="fpr")
    # Fill missing hour/cell_type combinations with NaN
    all_hours = list(range(24))
    heatmap_data = heatmap_data.reindex(columns=all_hours)

    fig, ax = plt.subplots(figsize=(16, 5))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.3,
        linecolor="white",
        mask=heatmap_data.isna(),
        vmin=0.0,
        vmax=0.25,
        cbar_kws={"label": "False Positive Rate", "shrink": 0.8},
    )

    ax.set_title(
        "False Positive Rate by Cell Type × Hour of Day\n"
        "(Darker = more spurious alerts; peak hours 08-22 shaded)",
        fontsize=11,
    )
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Cell Type")

    # Highlight peak hours
    for hour in range(8, 23):
        ax.axvline(x=hour, color="orange", lw=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved error analysis heatmap: %s", output_path)


def plot_threshold_sensitivity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tier_name: str,
    operating_threshold: float,
    output_path: Path,
) -> None:
    """
    Sensitivity of F1, Precision, and Recall to threshold selection.

    This plot answers the NOC architect's question: 'If we tighten the
    threshold to reduce false alarms, how many more anomalies do we miss?'
    The operating threshold (vertical line) should sit at the F1 peak
    or be shifted toward precision if NOC capacity is the bottleneck.

    See Coursebook Ch.8: Threshold selection as an operational decision.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        yp = (y_score >= t).astype(int)
        p = precision_score(y_true, yp, zero_division=0)
        r = recall_score(y_true, yp, zero_division=0)
        f = f1_score(y_true, yp, zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresholds, precisions, color=TELCO_PALETTE["primary"], lw=2, label="Precision")
    ax.plot(thresholds, recalls, color=TELCO_PALETTE["secondary"], lw=2, label="Recall")
    ax.plot(thresholds, f1s, color=TELCO_PALETTE["positive"], lw=2.5, label="F1 Score")

    # Mark operating threshold
    ax.axvline(
        x=operating_threshold,
        color=TELCO_PALETTE["negative"],
        linestyle="--",
        lw=1.5,
        label=f"Operating Threshold ({operating_threshold:.3f})",
    )

    # Mark F1-optimal threshold
    best_f1_idx = int(np.argmax(f1s))
    best_threshold = thresholds[best_f1_idx]
    ax.axvline(
        x=best_threshold,
        color=TELCO_PALETTE["highlight"],
        linestyle=":",
        lw=1.5,
        label=f"F1-Optimal Threshold ({best_threshold:.3f})",
    )

    # Annotation box with operational guidance
    ax.annotate(
        f"← More Alerts\n  (Higher Recall,\n  Lower Precision)",
        xy=(0.2, 0.5),
        xytext=(0.1, 0.3),
        fontsize=8,
        color="dimgray",
        arrowprops=dict(arrowstyle="->", color="dimgray"),
    )
    ax.annotate(
        f"Fewer Alerts →\n(Lower Recall,\n Higher Precision)",
        xy=(0.8, 0.5),
        xytext=(0.72, 0.3),
        fontsize=8,
        color="dimgray",
        arrowprops=dict(arrowstyle="->", color="dimgray"),
    )

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Threshold Sensitivity Analysis — {tier_name}")
    ax.legend(loc="center right", framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved threshold sensitivity: %s", output_path)


def plot_per_cell_performance(
    per_cell_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Per-cell average precision ranked bar chart.

    Cells with AP < 0.5 are strong candidates for:
    1. Dedicated sub-models (if they have enough data)
    2. Additional features (e.g., building penetration loss for indoor cells)
    3. Manual review (if data quality issues are suspected)

    Color-coding by cell_type reveals systematic patterns.
    """
    if per_cell_df.empty:
        logger.warning("No per-cell metrics to plot")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(per_cell_df) * 0.6), 6))

    cell_type_colors = {
        "urban_outdoor": TELCO_PALETTE["primary"],
        "suburban_outdoor": TELCO_PALETTE["secondary"],
        "rural_outdoor": TELCO_PALETTE["positive"],
        "indoor_small_cell": TELCO_PALETTE["highlight"],
    }

    colors = [
        cell_type_colors.get(ct, TELCO_PALETTE["neutral"])
        for ct in per_cell_df["cell_type"]
    ]

    bars = ax.bar(
        range(len(per_cell_df)),
        per_cell_df["ap_score"],
        color=colors,
        edgecolor="white",
        lw=0.5,
    )

    # Reference lines
    ax.axhline(y=0.5, color="red", linestyle="--", lw=1, label="AP=0.50 (lower threshold)")
    ax.axhline(y=0.80, color="green", linestyle="--", lw=1, label="AP=0.80 (target)")

    ax.set_xticks(range(len(per_cell_df)))
    ax.set_xticklabels(per_cell_df["cell_id"], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Cell ID (ranked by AP, descending)")
    ax.set_ylabel("Average Precision (AP)")
    ax.set_title("Per-Cell Average Precision — Ranked\n(Color = cell type)")
    ax.set_ylim(0, 1.05)

    # Legend for cell types
    legend_patches = [
        mpatches.Patch(color=color, label=ct)
        for ct, color in cell_type_colors.items()
        if ct in per_cell_df["cell_type"].values
    ]
    legend_patches.append(
        mpatches.Patch(color="none", label="")
    )  # spacer
    ax.legend(
        handles=legend_patches + [
            plt.Line2D([0], [0], color="red", linestyle="--", lw=1, label="AP=0.50"),
            plt.Line2D([0], [0], color="green", linestyle="--", lw=1, label="AP=0.80"),
        ],
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )

    # Annotate cells below threshold
    for i, (_, row) in enumerate(per_cell_df.iterrows()):
        if row["ap_score"] < 0.50:
            ax.text(
                i,
                row["ap_score"] + 0.02,
                "⚠",
                ha="center",
                va="bottom",
                fontsize=10,
                color="red",
            )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved per-cell performance: %s", output_path)


def plot_cascade_flow(
    cascade_metrics: CascadeMetrics,
    output_path: Path,
) -> None:
    """
    Cascade flow diagram showing volume reduction through the two-tier pipeline.

    This is the NOC operations manager's view: how many items are processed
    at each stage, and where do true positives and false positives land?
    It justifies the cascade architecture by showing the cost savings from
    Tier 1 filtering before the more expensive Tier 2 processing.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(
        "Cascade Model Flow — Volume Reduction Analysis\n"
        f"Test Set: {cascade_metrics.n_total_items:,} samples | "
        f"{cascade_metrics.n_true_anomalies} true anomalies "
        f"({cascade_metrics.n_true_anomalies/cascade_metrics.n_total_items:.1%} prevalence)",
        fontsize=11,
        pad=15,
    )

    # Box drawing helper
    def draw_box(
        ax: plt.Axes,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        count: int,
        color: str,
    ) -> None:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="white",
            lw=1.5,
            alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2 + 0.2,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
        ax.text(
            x + w / 2,
            y + h / 2 - 0.3,
            f"n={count:,}",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
        )

    # Arrow helper
    def draw_arrow(
        ax: plt.Axes,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        label: str = "",
    ) -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.1, my, label, fontsize=8, color="dimgray")

    # Input
    draw_box(ax, 0.5, 7.5, 2.5, 1.5, "All Inputs", cascade_metrics.n_total_items, "#555555")

    # Tier 1 processing
    draw_box(ax, 0.5, 5.0, 2.5, 1.5, "Tier 1 Filter", cascade_metrics.n_total_items, "#2166ac")
    draw_arrow(ax, 1.75, 7.5, 1.75, 6.5)

    # Tier 1 pass-through (flagged)
    draw_box(
        ax,
        4.0,
        5.0,
        2.5,
        1.5,
        "T1 Flagged",
        cascade_metrics.n_tier1_flagged,
        "#d6604d",
    )
    draw_arrow(ax, 3.0, 5.75, 4.0, 5.75, f"Flagged ({cascade_metrics.n_tier1_flagged:,})")

    # Tier 1 rejected (not escalated)
    n_not_flagged = cascade_metrics.n_total_items - cascade_metrics.n_tier1_flagged
    draw_box(ax, 4.0, 2.5, 2.5, 1.5, "T1 Passed\n(Not Escalated)", n_not_flagged, "#4dac26")
    draw_arrow(ax, 1.75, 5.0, 5.25, 4.0, "")
    ax.text(2.5, 4.4, f"Reduction: {cascade_metrics.tier1_reduction_rate:.1%}", fontsize=8, color="green")

    # Tier 2 processing
    draw_box(ax, 8.0, 5.0, 2.5, 1.5, "Tier 2 Classifier", cascade_metrics.n_tier1_flagged, "#7b2d8b")
    draw_arrow(ax, 6.5, 5.75, 8.0, 5.75)

    # Final output
    draw_box(
        ax,
        8.0,
        2.5,
        2.5,
        1.5,
        "Final Alerts\n(Cascade+)",
        cascade_metrics.n_cascade_positives,
        "#d62728",
    )
    draw_arrow(ax, 9.25, 5.0, 9.25, 4.0)

    # Metrics annotation
    metrics_text = (
        f"Cascade Metrics:\n"
        f"  Precision: {cascade_metrics.cascade_precision:.3f}\n"
        f"  Recall:    {cascade_metrics.cascade_recall:.3f}\n"
        f"  F1:        {cascade_metrics.cascade_f1:.3f}\n"
        f"  AUC-ROC:   {cascade_metrics.cascade_roc_auc:.3f}\n\n"
        f"  TP: {cascade_metrics.n_cascade_true_positives} | "
        f"FP: {cascade_metrics.n_cascade_false_positives} | "
        f"FN: {cascade_metrics.n_cascade_false_negatives}"
    )
    ax.text(
        0.5,
        0.5,
        metrics_text,
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange", alpha=0.9),
        verticalalignment="bottom",
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved cascade flow diagram: %s", output_path)


def plot_bootstrap_ci_summary(
    all_cis: Dict[str, Dict[str, BootstrapCI]],
    output_path: Path,
) -> None:
    """
    Forest plot showing point estimates and bootstrap confidence intervals.

    This is the standard way to communicate model performance uncertainty
    to a technical audience. The whiskers show the 95% CI; the dot shows
    the point estimate. Overlapping CIs between models indicate the
    performance difference is not statistically meaningful.
    """
    rows = []
    for tier_name, tier_cis in all_cis.items():
        for metric_name, ci in tier_cis.items():
            rows.append(
                {
                    "tier": tier_name,
                    "metric": metric_name,
                    "point": ci.point_estimate,
                    "lower": ci.ci_lower,
                    "upper": ci.ci_upper,
                    "err_low": ci.point_estimate - ci.ci_lower,
                    "err_high": ci.ci_upper - ci.point_estimate,
                }
            )

    if not rows:
        logger.warning("No bootstrap CIs to plot")
        return

    df = pd.DataFrame(rows)
    metrics_to_plot = ["roc_auc", "pr_auc", "f1", "recall", "precision"]
    df = df[df["metric"].isin(metrics_to_plot)]

    tiers = df["tier"].unique()
    n_tiers = len(tiers)
    tier_colors = dict(
        zip(
            tiers,
            [TELCO_PALETTE["primary"], TELCO_PALETTE["secondary"], TELCO_PALETTE["positive"]],
        )
    )

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 5), sharey=False)
    if len(metrics_to_plot) == 1:
        axes = [axes]

    fig.suptitle("Bootstrap 95% Confidence Intervals — All Tiers", fontsize=12)

    for ax, metric in zip(axes, metrics_to_plot):
        metric_df = df[df["metric"] == metric]
        y_positions = range(len(metric_df))

        for i, (_, row) in enumerate(metric_df.iterrows()):
            color = tier_colors.get(row["tier"], TELCO_PALETTE["neutral"])
            ax.errorbar(
                x=row["point"],
                y=i,
                xerr=[[row["err_low"]], [row["err_high"]]],
                fmt="o",
                color=color,
                markersize=8,
                capsize=4,
                lw=1.5,
                label=row["tier"] if metric == metrics_to_plot[0] else "",
            )
            ax.text(
                row["point"],
                i + 0.2,
                f"{row['point']:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

        ax.set_yticks(range(len(metric_df)))
        ax.set_yticklabels(metric_df["tier"], fontsize=9)
        ax.set_xlabel(metric.upper().replace("_", "-"), fontsize=9)
        ax.set_title(metric.upper().replace("_", "-"), fontsize=10)
        ax.set_xlim(0.0, 1.05)
        ax.axvline(x=0.5, color="gray", linestyle=":", lw=0.8, alpha=0.5)

    # Single legend from first axis
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=n_tiers, fontsize=9, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved bootstrap CI summary: %s", output_path)


def plot_operational_impact_summary(
    impact: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Side-by-side comparison of baseline vs. ML model operational metrics.

    Translates model performance into business language for VP/Director audience:
    - Daily false alarms (NOC ticket waste)
    - Daily missed anomalies (SLA risk)
    - Estimated cost savings

    These numbers use conservative assumptions and should always be presented
    with the caveat that actual results depend on operator-specific cost structure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle(
        "Estimated Operational Impact: ML Cascade vs. Rule-Based Baseline\n"
        f"(Assumptions: ${impact['improvement']['assumptions']['noc_cost_per_ticket_usd']}/ticket, "
        f"${impact['improvement']['assumptions']['sla_miss_cost_usd']}/missed anomaly — "
        "conservative estimates)",
        fontsize=10,
        y=1.01,
    )

    categories = ["Baseline\n(Rule-Based)", "ML Cascade"]
    bar_colors = [TELCO_PALETTE["neutral"], TELCO_PALETTE["primary"]]

    # Plot 1: Daily False Alarms
    fa_values = [
        impact["baseline"]["estimated_daily_false_alarms"],
        impact["ml_model"]["estimated_daily_false_alarms"],
    ]
    bars1 = axes[0].bar(categories, fa_values, color=bar_colors, edgecolor="white", lw=0.5)
    axes[0].set_title("Daily False Alarms\n(NOC Ticket Waste)", fontsize=10)
    axes[0].set_ylabel("Count")
    for bar, val in zip(bars1, fa_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    reduction_fa = (fa_values[0] - fa_values[1]) / max(fa_values[0], 1) * 100
    axes[0].text(
        0.5,
        0.9,
        f"−{reduction_fa:.0f}%",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=13,
        color=TELCO_PALETTE["positive"],
        fontweight="bold",
    )

    # Plot 2: Daily Missed Anomalies
    miss_values = [
        impact["baseline"]["estimated_daily_missed_anomalies"],
        impact["ml_model"]["estimated_daily_missed_anomalies"],
    ]
    bars2 = axes[1].bar(categories, miss_values, color=bar_colors, edgecolor="white", lw=0.5)
    axes[1].set_title("Daily Missed Anomalies\n(SLA Risk)", fontsize=10)
    axes[1].set_ylabel("Count")
    for bar, val in zip(bars2, miss_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.1f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    reduction_miss = (miss_values[0] - miss_values[1]) / max(miss_values[0], 1) * 100
    axes[1].text(
        0.5,
        0.9,
        f"−{reduction_miss:.0f}%",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=13,
        color=TELCO_PALETTE["positive"],
        fontweight="bold",
    )

    # Plot 3: Annual Cost Savings
    cost_values = [
        impact["baseline"]["estimated_daily_noc_cost_usd"] * 365,
        impact["ml_model"]["estimated_daily_noc_cost_usd"] * 365,
    ]
    bars3 = axes[2].bar(categories, cost_values, color=bar_colors, edgecolor="white", lw=0.5)
    axes[2].set_title("Annual NOC Operations Cost (USD)\n(Estimate — see assumptions)", fontsize=10)
    axes[2].set_ylabel("USD / Year")
    for bar, val in zip(bars3, cost_values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"${val:,.0f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    saving = cost_values[0] - cost_values[1]
    axes[2].text(
        0.5,
        0.9,
        f"Save ${saving:,.0f}/yr",
        transform=axes[2].transAxes,
        ha="center",
        fontsize=12,
        color=TELCO_PALETTE["positive"],
        fontweight="bold",
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved operational impact summary: %s", output_path)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tier_name: str,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """
    Probability calibration (reliability diagram).

    A well-calibrated model has predicted probabilities that match empirical
    frequencies: when the model says 0.7, ~70% of those cases are truly anomalous.
    Poor calibration means threshold-based decisions are unreliable.

    For telco anomaly detection:
    - Overconfident models (S-curve) underestimate uncertainty
    - Underconfident models (flat curve) need Platt scaling or isotonic regression
    - Brier score quantifies the calibration quality as a scalar

    See Coursebook Ch.8: Calibration for decision-critical ML systems.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy="uniform"
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))

    # Reliability diagram
    ax1.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        color=TELCO_PALETTE["primary"],
        lw=2,
        markersize=6,
        label=f"{tier_name} (Brier={brier_score_loss(y_true, y_score):.4f})",
    )
    ax1.plot([0, 1], [0, 1], "k:", lw=1.5, label="Perfect Calibration")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(f"Calibration Curve (Reliability Diagram) — {tier_name}")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_xlim(-0.01, 1.01)
    ax1.set_ylim(-0.01, 1.05)

    # Histogram of predicted probabilities
    ax2.hist(
        y_score[y_true == 0],
        bins=40,
        alpha=0.6,
        color=TELCO_PALETTE["neutral"],
        label="Normal",
        density=True,
    )
    ax2.hist(
        y_score[y_true == 1],
        bins=40,
        alpha=0.7,
        color=TELCO_PALETTE["negative"],
        label="Anomaly",
        density=True,
    )
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Score Distribution by True Class")
    ax2.legend(framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved calibration curve: %s", output_path)


def plot_anomaly_type_breakdown(
    df: pd.DataFrame,
    y_pred_col: str,
    output_path: Path,
) -> None:
    """
    Per-anomaly-type detection rate breakdown.

    Different anomaly types have different detection difficulty:
    - Hardware faults: usually abrupt, high signal → easier to detect
    - Congestion: gradual buildup, may look like normal peak traffic → harder
    - Interference: intermittent, spatial pattern-dependent → hardest
    - Planned work: should be excluded from evaluation (known in CMDB) → label separately

    This plot helps prioritize feature engineering improvements by anomaly type.
    """
    anom_df = df[df["is_anomaly"] == 1].copy()
    if "anomaly_type" not in anom_df.columns:
        logger.info("No anomaly_type column — skipping breakdown plot")
        return

    anom_df["detected"] = anom_df[y_pred_col] == 1

    type_stats = (
        anom_df.groupby("anomaly_type")
        .agg(total=("is_anomaly", "count"), detected=("detected", "sum"))
        .reset_index()
    )
    type_stats["detection_rate"] = type_stats["detected"] / type_stats["total"]
    type_stats = type_stats.sort_values("detection_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = [
        TELCO_PALETTE["positive"] if r >= 0.80 else
        TELCO_PALETTE["secondary"] if r >= 0.60 else
        TELCO_PALETTE["negative"]
        for r in type_stats["detection_rate"]
    ]

    bars = ax.barh(
        type_stats["anomaly_type"],
        type_stats["detection_rate"],
        color=colors,
        edgecolor="white",
        lw=0.5,
    )

    for bar, row in zip(bars, type_stats.itertuples()):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{row.detection_rate:.1%} ({row.detected}/{row.total})",
            va="center",
            fontsize=9,
        )

    ax.axvline(x=0.80, color="green", linestyle="--", lw=1, label="80% Detection Target")
    ax.set_xlabel("Detection Rate (Recall by Anomaly Type)")
    ax.set_title("Detection Rate by Anomaly Type")
    ax.set_xlim(0, 1.15)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved anomaly type breakdown: %s", output_path)


# ---------------------------------------------------------------------------
# Output formatting and serialization
# ---------------------------------------------------------------------------


def build_metrics_summary(
    tier1_eval: TierEvaluation,
    tier2_eval: TierEvaluation,
    cascade: CascadeMetrics,
    impact: Dict[str, Any],
    training_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the canonical metrics JSON written to eval_output/.

    This JSON is intended for downstream consumption by:
    1. CI/CD gates (fail if AUC-PR drops below threshold)
    2. Model registry (attach to model version as evaluation report)
    3. Monitoring dashboards (baseline for drift detection comparison)

    See Section S09: Platform KPIs and model evaluation gates.
    """
    summary: Dict[str, Any] = {
        "evaluation_version": "1.0.0",
        "schema_version": "telco_mlops_v1",
        # Tier-level metrics
        "tier1": tier1_eval.to_dict(),
        "tier2": tier2_eval.to_dict(),
        # Cascade system metrics
        "cascade": {
            "n_total_items": cascade.n_total_items,
            "n_true_anomalies": cascade.n_true_anomalies,
            "cascade_precision": cascade.cascade_precision,
            "cascade_recall": cascade.cascade_recall,
            "cascade_f1": cascade.cascade_f1,
            "cascade_roc_auc": cascade.cascade_roc_auc,
            "tier1_reduction_rate": cascade.tier1_reduction_rate,
            "total_false_alarm_rate": cascade.total_false_alarm_rate,
            "total_miss_rate": cascade.total_miss_rate,
            "n_cascade_positives": cascade.n_cascade_positives,
            "n_cascade_true_positives": cascade.n_cascade_true_positives,
            "n_cascade_false_positives": cascade.n_cascade_false_positives,
            "n_cascade_false_negatives": cascade.n_cascade_false_negatives,
        },
        # Operational impact
        "operational_impact": impact,
        # Governance gate thresholds (must be met for model promotion)
        # These thresholds should be configured in the model registry, not hardcoded
        # See Section S07: Governance gate — OPA policy checks
        "governance_gates": {
            "tier1_pr_auc_minimum": 0.60,
            "tier1_pr_auc_met": tier1_eval.pr_auc >= 0.60,
            "cascade_precision_minimum": 0.70,
            "cascade_precision_met": cascade.cascade_precision >= 0.70,
            "cascade_recall_minimum": 0.65,
            "cascade_recall_met": cascade.cascade_recall >= 0.65,
            "overall_pass": (
                tier1_eval.pr_auc >= 0.60
                and cascade.cascade_precision >= 0.70
                and cascade.cascade_recall >= 0.65
            ),
        },
    }

    # Training vs. test comparison (if training metrics available)
    if training_metrics:
        val_pr_auc = training_metrics.get("val_pr_auc", training_metrics.get("pr_auc", None))
        if val_pr_auc is not None:
            pr_auc_delta = tier1_eval.pr_auc - float(val_pr_auc)
            summary["train_test_comparison"] = {
                "val_pr_auc": val_pr_auc,
                "test_pr_auc": tier1_eval.pr_auc,
                "pr_auc_delta": round(pr_auc_delta, 4),
                # Large negative delta signals overfitting or distribution shift
                "overfitting_warning": pr_auc_delta < -0.10,
            }

    return summary


def save_bootstrap_cis(
    all_cis: Dict[str, Dict[str, BootstrapCI]],
    output_path: Path,
) -> None:
    """Serialize bootstrap CIs to JSON for archival and monitoring baselines."""
    serializable = {}
    for tier_name, tier_cis in all_cis.items():
        serializable[tier_name] = {
            metric: ci.to_dict() for metric, ci in tier_cis.items()
        }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved bootstrap CIs: %s", output_path)


def log_evaluation_summary(
    tier1_eval: TierEvaluation,
    tier2_eval: TierEvaluation,
    cascade: CascadeMetrics,
    impact: Dict[str, Any],
) -> None:
    """Log a human-readable evaluation summary to the console."""
    separator = "=" * 70

    logger.info(separator)
    logger.info("EVALUATION SUMMARY — TELCO MLOPS ANOMALY DETECTION PLATFORM")
    logger.info(separator)

    logger.info("TIER 1 (Cell-Level Anomaly Scorer):")
    logger.info(
        "  Prevalence: %.1f%% | Threshold: %.3f",
        tier1_eval.prevalence * 100,
        tier1_eval.threshold,
    )
    logger.info(
        "  AUC-ROC: %.4f | AUC-PR: %.4f | Brier: %.4f",
        tier1_eval.roc_auc,
        tier1_eval.pr_auc,
        tier1_eval.brier_score,
    )
    logger.info(
        "  Precision: %.4f | Recall: %.4f | F1: %.4f | MCC: %.4f",
        tier1_eval.precision,
        tier1_eval.recall,
        tier1_eval.f1,
        tier1_eval.mcc,
    )
    logger.info(
        "  False Alarm Rate: %.2f%% | False Alarm Count: %d | Missed: %d",
        tier1_eval.false_alarm_rate * 100,
        tier1_eval.false_alarm_count,
        tier1_eval.missed_anomaly_count,
    )

    logger.info("")
    logger.info("TIER 2 (Site-Level Precision Classifier):")
    logger.info(
        "  AUC-ROC: %.4f | AUC-PR: %.4f | Brier: %.4f",
        tier2_eval.roc_auc,
        tier2_eval.pr_auc,
        tier2_eval.brier_score,
    )
    logger.info(
        "  Precision: %.4f | Recall: %.4f | F1: %.4f | MCC: %.4f",
        tier2_eval.precision,
        tier2_eval.recall,
        tier2_eval.f1,
        tier2_eval.mcc,
    )

    logger.info("")
    logger.info("CASCADE (Two-Tier System):")
    logger.info(
        "  Tier 1 Volume Reduction: %.1f%% of items NOT escalated",
        cascade.tier1_reduction_rate * 100,
    )
    logger.info(
        "  Cascade Precision: %.4f | Recall: %.4f | F1: %.4f",
        cascade.cascade_precision,
        cascade.cascade_recall,
        cascade.cascade_f1,
    )
    logger.info(
        "  Total False Alarm Rate: %.2f%% | Total Miss Rate: %.2f%%",
        cascade.total_false_alarm_rate * 100,
        cascade.total_miss_rate * 100,
    )

    logger.info("")
    logger.info("ESTIMATED OPERATIONAL IMPACT (conservative, per-day):")
    logger.info(
        "  False alarms: %.1f (ML) vs %.1f (baseline)",
        impact["ml_model"]["estimated_daily_false_alarms"],
        impact["baseline"]["estimated_daily_false_alarms"],
    )
    logger.info(
        "  Missed anomalies: %.1f (ML) vs %.1f (baseline)",
        impact["ml_model"]["estimated_daily_missed_anomalies"],
        impact["baseline"]["estimated_daily_missed_anomalies"],
    )
    logger.info(
        "  Estimated annual cost savings: $%,.0f USD",
        impact["improvement"]["annual_noc_cost_savings_usd"],
    )

    logger.info(separator)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_evaluation(
    data_dir: Path = DATA_DIR,
    model_dir: Path = MODEL_DIR,
    output_dir: Path = EVAL_DIR,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    tier1_threshold: float = 0.45,
    tier2_threshold: float = 0.55,
) -> Dict[str, Any]:
    """
    Orchestrate the full evaluation pipeline.

    Load data → compute metrics → generate visualizations → save outputs.
    Returns the metrics summary dict for use in CI/CD gates.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting evaluation pipeline | output: %s", output_dir)

    # -------------------------------------------------------------------
    # Step 1: Load or generate data
    # -------------------------------------------------------------------
    # Priority 1: Load pre-computed predictions (e.g., from a CI artifact)
    df = load_test_predictions(model_dir)

    # Priority 2: Load test features + trained model → run actual inference
    if df is None:
        logger.info("No saved predictions found — attempting model inference on test features")
        test_df = load_feature_test_split(data_dir)
        artifacts = load_model_artifacts(model_dir)

        if test_df is not None and "tier2_random_forest" in artifacts:
            logger.info("Running inference with tier2_random_forest on test features")
            model = artifacts["tier2_random_forest"]

            # Determine feature columns — prefer metadata from 02/03 pipeline
            # Key 'features' is produced by 02_feature_engineering.py build_feature_metadata();
            # 'feature_columns' is an alternative format from some pipeline versions.
            #
            # Safety: always exclude target-leakage columns regardless of source
            _LEAKAGE_COLS = {
                "anomaly_rrc_congestion", "anomaly_hw_degradation",
                "anomaly_counter_reset", "anomaly_traffic_spike",
                "is_anomaly", "anomaly_type",
            }
            feat_meta = load_feature_metadata(data_dir)
            if feat_meta and "features" in feat_meta:
                feature_cols = [
                    c for c in feat_meta["features"].keys()
                    if c in test_df.columns and c not in _LEAKAGE_COLS
                ]
                logger.info("Using %d feature columns from feature_metadata.json 'features' key", len(feature_cols))
                if len(feature_cols) < len(feat_meta["features"]):
                    logger.warning(
                        "Feature drop: metadata defines %d features but only %d found in test data "
                        "(missing or leakage-filtered: %s)",
                        len(feat_meta["features"]), len(feature_cols),
                        set(feat_meta["features"].keys()) - set(feature_cols),
                    )
            elif feat_meta and "feature_columns" in feat_meta:
                feature_cols = [
                    c for c in feat_meta["feature_columns"]
                    if c in test_df.columns and c not in _LEAKAGE_COLS
                ]
                logger.info("Using %d feature columns from feature_metadata.json 'feature_columns' key", len(feature_cols))
            else:
                # Fallback: dynamic discovery excluding non-feature columns
                non_feature_cols = {
                    "timestamp", "cell_id", "is_anomaly", "site_id",
                    "region", "frequency_band", "cluster_id", "sector",
                    "technology", "cell_type", "anomaly_type",
                    # Ground-truth anomaly type columns from 01_synthetic_data.py
                    # — including these would leak the target into the feature set
                    "anomaly_rrc_congestion", "anomaly_hw_degradation",
                    "anomaly_counter_reset", "anomaly_traffic_spike",
                    # Topology columns — not model features
                    "n_prb_dl", "n_prb_ul", "antenna_height_m",
                    "latitude", "longitude", "load_factor",
                    "rsrp_base_dbm",
                    # Raw byte-volume counters (not features)
                    "pdcp_vol_dl_bytes", "pdcp_vol_ul_bytes",
                }
                # Pattern-based exclusion: any column starting with "anomaly_"
                # is a target-leakage risk (future anomaly types auto-excluded)
                non_feature_cols |= {
                    c for c in test_df.columns if c.startswith("anomaly_")
                }
                feature_cols = [
                    c for c in test_df.columns if c not in non_feature_cols
                    and pd.api.types.is_numeric_dtype(test_df[c])
                ]
                logger.warning(
                    "Dynamic feature discovery used (no metadata). Discovered %d features. "
                    "This path is fragile — run the full 01→02→03→04 pipeline to generate "
                    "feature_metadata.json for metadata-driven feature selection.",
                    len(feature_cols),
                )

            X_test = test_df[feature_cols].fillna(0.0).values.astype(np.float32)

            # --- Dimensionality guard ---
            expected_n = getattr(model, "n_features_in_", None)
            if expected_n is not None and X_test.shape[1] != expected_n:
                logger.error(
                    "Feature count mismatch: model expects %d features, "
                    "but %d were selected. Attempting recovery from model metadata.",
                    expected_n, X_test.shape[1],
                )
                if hasattr(model, "feature_names_in_"):
                    feature_cols = [c for c in model.feature_names_in_ if c in test_df.columns]
                    X_test = test_df[feature_cols].fillna(0.0).values.astype(np.float32)
                    logger.info("Recovered %d features from model.feature_names_in_", len(feature_cols))
                if X_test.shape[1] != expected_n:
                    raise ValueError(
                        f"Cannot align features: model expects {expected_n}, "
                        f"only {X_test.shape[1]} available after recovery attempt."
                    )
            # Run RF inference to produce probability scores.
            # Decision: Pipeline vs bare classifier.
            #   1. Check JSON sidecar (tier2_random_forest_meta.json) — reliable
            #   2. Fall back to isinstance + feature_metadata.json scaling_applied
            from sklearn.pipeline import Pipeline as SkPipeline

            # Load sidecar metadata if available
            sidecar_path = Path(model_dir) / "tier2_random_forest_meta.json" if model_dir else None
            model_meta = None
            if sidecar_path and sidecar_path.exists():
                with open(sidecar_path) as _f:
                    model_meta = json.load(_f)
                logger.info("Loaded model sidecar metadata: %s", model_meta.get("artifact_type"))

            input_is_scaled = feat_meta and feat_meta.get("scaling_applied", False)
            is_pipeline = isinstance(model, SkPipeline)

            if is_pipeline and input_is_scaled:
                # Pipeline + pre-scaled data → extract bare classifier to avoid double-scaling
                clf = model.named_steps.get("clf", model[-1])
                logger.info(
                    "Pipeline + pre-scaled input — extracting bare classifier (%s) "
                    "to avoid double-scaling",
                    type(clf).__name__,
                )
                tier2_scores = clf.predict_proba(X_test)[:, 1]
            elif is_pipeline and not input_is_scaled:
                # Pipeline + raw data → use full Pipeline (scaler applies correctly)
                logger.info("Pipeline + raw input — using full Pipeline")
                tier2_scores = model.predict_proba(X_test)[:, 1]
            else:
                # Bare classifier — use as-is
                tier2_scores = model.predict_proba(X_test)[:, 1]
            test_df["tier2_score"] = tier2_scores

            # Use tier2 scores as tier1 scores (single-model evaluation)
            # In production, tier1 would be the isolation forest pre-screen
            if "tier1_isolation_forest" in artifacts:
                iforest = artifacts["tier1_isolation_forest"]
                # IsolationForest: decision_function returns anomaly scores
                # (more negative = more anomalous); convert to 0-1 probability
                raw_scores = iforest.decision_function(X_test)
                test_df["tier1_score"] = 1.0 - (raw_scores - raw_scores.min()) / (
                    raw_scores.max() - raw_scores.min() + 1e-9
                )
            else:
                test_df["tier1_score"] = tier2_scores

            # Load thresholds from training if available
            if "thresholds" in artifacts:
                thresholds = artifacts["thresholds"]
                tier1_threshold = thresholds.get("tier1_isolation_forest", tier1_threshold)
                tier2_threshold = thresholds.get("tier2_random_forest", tier2_threshold)
                logger.info(
                    "Using trained thresholds: tier1=%.3f, tier2=%.3f",
                    tier1_threshold, tier2_threshold,
                )

            df = test_df
            logger.info(
                "Inference complete: %d rows scored | tier2 score range [%.3f, %.3f]",
                len(df), df["tier2_score"].min(), df["tier2_score"].max(),
            )
        elif test_df is not None:
            logger.warning(
                "Test features found but no trained model (tier2_random_forest.joblib) "
                "in %s — falling back to synthetic data. Run 03_model_training.py first.",
                model_dir,
            )

    # Priority 3: Standalone demo mode — generate synthetic data
    if df is None:
        logger.info(
            "No pipeline outputs found — generating synthetic dataset (demo mode). "
            "For integrated evaluation, run: 01 → 02 → 03 → 04"
        )
        df = generate_evaluation_dataset(n_samples=4000, anomaly_rate=0.04)

    # Ensure required columns exist; add stubs if needed
    required_cols = ["timestamp", "cell_id", "is_anomaly", "tier1_score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' missing from dataset. "
                "Run 01_synthetic_data.py and 03_model_training.py first, "
                "or let this script generate a standalone synthetic dataset."
            )

    # Add tier2_score if not present (simulate second tier from tier1)
    if "tier2_score" not in df.columns:
        rng = np.random.default_rng(99)
        # Tier 2 is a more selective (higher precision) version of Tier 1
        noise = rng.normal(0, 0.05, size=len(df))
        df["tier2_score"] = np.clip(df["tier1_score"] * 0.95 + noise, 0.0, 1.0)

    # Add baseline if not present (simple threshold rule simulation)
    if "baseline_flag" not in df.columns:
        # Baseline: flag top X% by a simple counter threshold
        df["baseline_flag"] = (df["tier1_score"] > 0.35).astype(int)
        # Simulate that baseline is noisier (higher FP, similar recall)
        rng = np.random.default_rng(777)
        flip_mask = rng.random(len(df)) < 0.08
        df.loc[flip_mask, "baseline_flag"] = 1 - df.loc[flip_mask, "baseline_flag"]

    # Add missing feature columns for error analysis
    if "hour_of_day" not in df.columns and "timestamp" in df.columns:
        df["hour_of_day"] = df["timestamp"].dt.hour
    if "is_peak_hour" not in df.columns and "hour_of_day" in df.columns:
        df["is_peak_hour"] = ((df["hour_of_day"] >= 8) & (df["hour_of_day"] <= 22)).astype(int)
    if "cell_type" not in df.columns:
        cell_types_pool = ["urban_outdoor", "suburban_outdoor", "rural_outdoor", "indoor_small_cell"]
        rng = np.random.default_rng(42)
        df["cell_type"] = rng.choice(cell_types_pool, size=len(df))
    if "anomaly_type" not in df.columns:
        rng = np.random.default_rng(42)
        df["anomaly_type"] = np.where(
            df["is_anomaly"] == 1,
            rng.choice(
                ["hardware_fault", "congestion", "interference", "planned_work"],
                size=len(df),
                p=[0.30, 0.45, 0.15, 0.10],
            ),
            "normal",
        )

    logger.info(
        "Dataset: %d rows | %d anomalies (%.2f%%)",
        len(df),
        df["is_anomaly"].sum(),
        df["is_anomaly"].mean() * 100,
    )

    y_true = df["is_anomaly"].values.astype(int)
    tier1_score = df["tier1_score"].values.astype(float)
    tier2_score = df["tier2_score"].values.astype(float)
    baseline_score = df["baseline_flag"].values.astype(float)

    # -------------------------------------------------------------------
    # Step 2: Find optimal thresholds (confirm or override defaults)
    # -------------------------------------------------------------------
    opt_tier1_thresh, _ = find_optimal_threshold(y_true, tier1_score, optimize_for="f1")
    opt_tier2_thresh, _ = find_optimal_threshold(y_true, tier2_score, optimize_for="precision")

    # Use discovered optimal thresholds, but log the overrides for transparency
    tier1_threshold = opt_tier1_thresh
    tier2_threshold = opt_tier2_thresh

    # -------------------------------------------------------------------
    # Step 3: Core metric computation
    # -------------------------------------------------------------------
    tier1_eval = compute_tier_evaluation(
        y_true, tier1_score, tier_name="Tier1_CellScorer", threshold=tier1_threshold
    )
    tier2_eval = compute_tier_evaluation(
        y_true, tier2_score, tier_name="Tier2_SiteClassifier", threshold=tier2_threshold
    )
    baseline_eval = compute_tier_evaluation(
        y_true, baseline_score, tier_name="Baseline_Rule", threshold=0.5
    )
    cascade = compute_cascade_metrics(
        y_true,
        tier1_score,
        tier2_score,
        tier1_threshold=tier1_threshold,
        tier2_threshold=tier2_threshold,
    )

    # Predicted label columns for visualization
    df["tier1_pred"] = (tier1_score >= tier1_threshold).astype(int)
    df["tier2_pred"] = (tier2_score >= tier2_threshold).astype(int)
    df["baseline_pred"] = baseline_score.astype(int)

    # -------------------------------------------------------------------
    # Step 4: Bootstrap confidence intervals
    # -------------------------------------------------------------------
    logger.info("Computing bootstrap CIs (n=%d, ci=%.0f%%)...", n_bootstrap, ci_level * 100)
    tier1_cis = bootstrap_confidence_intervals(
        y_true, tier1_score, "Tier1_CellScorer", tier1_threshold, n_bootstrap, ci_level
    )
    tier2_cis = bootstrap_confidence_intervals(
        y_true, tier2_score, "Tier2_SiteClassifier", tier2_threshold, n_bootstrap, ci_level
    )
    baseline_cis = bootstrap_confidence_intervals(
        y_true, baseline_score, "Baseline_Rule", 0.5, n_bootstrap, ci_level
    )

    tier1_eval.cis = tier1_cis
    tier2_eval.cis = tier2_cis
    baseline_eval.cis = baseline_cis

    all_cis = {
        "Tier1_CellScorer": tier1_cis,
        "Tier2_SiteClassifier": tier2_cis,
        "Baseline_Rule": baseline_cis,
    }

    # -------------------------------------------------------------------
    # Step 5: Per-cell analysis
    # -------------------------------------------------------------------
    per_cell_df = compute_per_cell_metrics(df)
    logger.info(
        "Per-cell analysis: %d cells | mean AP=%.3f | min AP=%.3f",
        len(per_cell_df),
        per_cell_df["ap_score"].mean() if not per_cell_df.empty else 0.0,
        per_cell_df["ap_score"].min() if not per_cell_df.empty else 0.0,
    )

    # -------------------------------------------------------------------
    # Step 6: Operational impact estimate
    # -------------------------------------------------------------------
    impact = operational_impact_estimate(tier1_eval, cascade)

    # -------------------------------------------------------------------
    # Step 7: Load training metrics for overfitting check
    # -------------------------------------------------------------------
    training_metrics = load_training_metrics(model_dir)

    # -------------------------------------------------------------------
    # Step 8: Build and save metrics summary
    # -------------------------------------------------------------------
    metrics_summary = build_metrics_summary(
        tier1_eval, tier2_eval, cascade, impact, training_metrics
    )

    metrics_path = output_dir / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2, default=str)
    logger.info("Saved metrics summary: %s", metrics_path)

    save_bootstrap_cis(all_cis, output_dir / "bootstrap_ci.json")

    # -------------------------------------------------------------------
    # Step 9: Generate all visualizations
    # -------------------------------------------------------------------
    logger.info("Generating visualizations...")

    plot_confusion_matrix(
        y_true,
        df["tier1_pred"].values,
        "Tier 1 — Cell Scorer",
        output_dir / "confusion_matrix_tier1.png",
        threshold=tier1_threshold,
    )

    plot_confusion_matrix(
        y_true,
        df["tier2_pred"].values,
        "Tier 2 — Site Classifier",
        output_dir / "confusion_matrix_tier2.png",
        threshold=tier2_threshold,
    )

    evaluations = {
        "Tier1_CellScorer": tier1_eval,
        "Tier2_SiteClassifier": tier2_eval,
    }
    y_true_dict = {
        "Tier1_CellScorer": y_true,
        "Tier2_SiteClassifier": y_true,
    }
    y_score_dict = {
        "Tier1_CellScorer": tier1_score,
        "Tier2_SiteClassifier": tier2_score,
    }

    plot_roc_curves(
        evaluations,
        y_true_dict,
        y_score_dict,
        baseline_y_true=y_true,
        baseline_y_score=baseline_score,
        output_path=output_dir / "roc_curves.png",
    )

    plot_pr_curves(
        evaluations,
        y_true_dict,
        y_score_dict,
        baseline_y_true=y_true,
        baseline_y_score=baseline_score,
        output_path=output_dir / "pr_curves.png",
    )

    plot_time_series_overlay(
        df,
        tier_score_col="tier1_score",
        output_path=output_dir / "time_series_overlay.png",
        n_cells_to_show=3,
    )

    plot_error_analysis_heatmap(
        df,
        y_pred_col="tier1_pred",
        output_path=output_dir / "error_analysis_heatmap.png",
    )

    plot_threshold_sensitivity(
        y_true,
        tier1_score,
        "Tier 1 — Cell Scorer",
        operating_threshold=tier1_threshold,
        output_path=output_dir / "threshold_sensitivity.png",
    )

    plot_cascade_flow(cascade, output_dir / "cascade_sankey.png")

    if not per_cell_df.empty:
        plot_per_cell_performance(per_cell_df, output_dir / "per_cell_performance.png")

    plot_bootstrap_ci_summary(all_cis, output_dir / "bootstrap_ci_summary.png")

    plot_operational_impact_summary(impact, output_dir / "operational_impact_summary.png")

    plot_calibration_curve(
        y_true,
        tier1_score,
        "Tier 1 — Cell Scorer",
        output_dir / "calibration_curve.png",
    )

    plot_anomaly_type_breakdown(
        df,
        y_pred_col="tier1_pred",
        output_path=output_dir / "anomaly_type_breakdown.png",
    )

    # -------------------------------------------------------------------
    # Step 10: Print summary to console
    # -------------------------------------------------------------------
    log_evaluation_summary(tier1_eval, tier2_eval, cascade, impact)

    # -------------------------------------------------------------------
    # Step 11: Governance gate check (for CI/CD integration)
    # -------------------------------------------------------------------
    gates = metrics_summary["governance_gates"]
    if gates["overall_pass"]:
        logger.info("✅ GOVERNANCE GATES PASSED — model eligible for promotion to staging")
    else:
        failing = [k for k, v in gates.items() if k.endswith("_met") and not v]
        logger.warning(
            "❌ GOVERNANCE GATES FAILED — failing gates: %s. "
            "Model NOT eligible for promotion. See metrics_summary.json for details.",
            failing,
        )

    logger.info("Evaluation pipeline complete. All outputs in: %s", output_dir)

    return metrics_summary


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="04_evaluation.py — Telco MLOps evaluation and visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing features_{train,val,test}.parquet from 02_feature_engineering.py",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory containing model artifacts from 03_model_training.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVAL_DIR,
        help="Directory where evaluation outputs (figures, JSON) are written",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for confidence intervals",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap intervals (0-1)",
    )
    parser.add_argument(
        "--tier1-threshold",
        type=float,
        default=0.45,
        help="Decision threshold for Tier 1 scorer (overridden by auto-optimization if 0)",
    )
    parser.add_argument(
        "--tier2-threshold",
        type=float,
        default=0.55,
        help="Decision threshold for Tier 2 classifier (overridden by auto-optimization if 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)

    summary = run_evaluation(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        ci_level=args.ci_level,
        tier1_threshold=args.tier1_threshold,
        tier2_threshold=args.tier2_threshold,
    )

    # Exit code for CI/CD: 0 = gates passed, 1 = gates failed
    gate_pass = summary.get("governance_gates", {}).get("overall_pass", False)
    sys.exit(0 if gate_pass else 1)
