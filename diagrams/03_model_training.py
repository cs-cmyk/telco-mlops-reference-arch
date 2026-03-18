"""
03_model_training.py — Telco MLOps Reference Architecture: Multi-Team, Multi-Model at Scale
============================================================================================
Companion code for Section 7 (System Design) and Section 8 (Implementation Walkthrough)
of the whitepaper "Telco MLOps Reference Architecture: How Multi-Team, Multi-Model
Organizations Ship ML at Scale Without Losing Control."

PURPOSE:
    Implements a three-tier cascade anomaly detection system for RAN cell-level PM counters:
      Tier 1 — Unsupervised: Isolation Forest + One-Class SVM ensemble auto-labeling
      Tier 2 — Supervised: Random Forest with SHAP explainability on auto-labels
      Tier 3 — Sequential: LSTM Autoencoder for temporal pattern anomaly detection

    This mirrors the MLOps model lifecycle defined in 3GPP TS 28.105 (MLEntityRepository,
    MLTrainingReport) and the O-RAN WG2 AI/ML training pipeline. Each tier produces
    registered artifacts consistent with the model registry design in the reference
    architecture.

WHITEPAPER CONTEXT:
    - Addresses Section 7: System Design — Training Pipeline (Kubeflow/Argo pattern)
    - Demonstrates governance gate inputs: model cards, evaluation metrics, thresholds
    - Models are saved in ONNX-compatible format (via joblib + threshold JSON) for
      serving via KServe InferenceService (see 05_production_patterns.py)

USAGE:
    # With outputs from prior scripts:
    python 03_model_training.py

    # Override data directory:
    python 03_model_training.py --data-dir /path/to/data --output-dir /path/to/models

    # Skip LSTM (faster runs during development):
    python 03_model_training.py --skip-lstm

    # Full grid search (slow, use in CI only):
    python 03_model_training.py --full-search

OUTPUTS:
    models/
      tier1_isolation_forest.joblib       — Fitted Isolation Forest
      tier1_ocsvm.joblib                  — Fitted One-Class SVM
      tier1_autolabels.parquet            — Auto-generated labels for Tier 2 training
      tier2_random_forest.joblib          — Fitted Random Forest classifier
      tier3_lstm_autoencoder.pt           — LSTM Autoencoder state dict (if PyTorch avail.)
      thresholds.json                     — Reconstruction threshold per tier
      model_card.json                     — 3GPP TS 28.105 aligned model card
      training_metrics.json               — Full evaluation metrics for governance gate
      feature_importance.png              — SHAP-based feature importance
      training_curves.png                 — Loss curves + PR/ROC plots

REQUIREMENTS:
    pip install numpy pandas scikit-learn joblib matplotlib seaborn shap

    OPTIONAL (Tier 3 LSTM):
    pip install torch

COURSEBOOK CROSS-REFERENCE:
    - Ch. 3: Model Evaluation & Validation (temporal splits, AUC-PR)
    - Ch. 4: Feature Engineering (SHAP, feature importance)
    - Ch. 6: Anomaly Detection (Isolation Forest, Autoencoder)
    - Ch. 7: MLOps Core (experiment tracking, model registry, reproducibility)
    - Ch. 8: Production Systems (model artifacts, threshold calibration)

Author: Telco AI Engineering Team
Whitepaper: telco-mlops-reference-architecture-multi-team-multi-model
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Non-interactive backend required for headless CI/CD environments
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Logging configuration — use structured logging compatible with log aggregators
# (Loki, ELK) used in the reference architecture monitoring stack
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("telco_mlops.model_training")

# ---------------------------------------------------------------------------
# Path configuration — must align with outputs from 01 and 02 scripts
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
# Note: 02_feature_engineering.py writes features_train/val/test.parquet
# directly to DATA_DIR, not to a features/ subdirectory.
MODELS_DIR = Path("models")

# Reproducibility seed — critical for governance: same seed = same model artifacts
# In production, seed is recorded in MLflow run metadata (MLTrainingReport IOC)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Optional PyTorch import — graceful degradation for environments without GPU
# The LSTM tier is architecturally important but not required for the RF tier
# ---------------------------------------------------------------------------
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(RANDOM_SEED)
    logger.info("PyTorch available — LSTM Autoencoder tier enabled")
else:
    logger.warning(
        "PyTorch not available — Tier 3 LSTM Autoencoder will be skipped. "
        "Install with: pip install torch"
    )

# Optional SHAP — provides explainability for governance gate model cards
SHAP_AVAILABLE = importlib.util.find_spec("shap") is not None
if SHAP_AVAILABLE:
    import shap

    logger.info("SHAP available — feature importance explainability enabled")
else:
    logger.warning(
        "SHAP not available — falling back to sklearn feature_importances_. "
        "Install with: pip install shap"
    )


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class SplitMetadata:
    """
    Tracks temporal split boundaries for reproducibility and MLflow logging.
    In the reference architecture, this maps to 3GPP TS 28.105 MLTrainingReport
    IOC fields: dataStartTime, dataEndTime, trainingStartTime.
    """

    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    train_size: int
    val_size: int
    test_size: int
    train_anomaly_rate: float
    val_anomaly_rate: float
    test_anomaly_rate: float


@dataclass
class TierMetrics:
    """
    Evaluation metrics for a single model tier.
    Maps to the validation gate criteria in the governance pipeline
    (Section 7: Model Registry & Governance Gate).
    """

    tier_name: str
    auc_roc: float
    auc_pr: float
    f1_at_threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    threshold: float
    false_positive_rate: float  # Critical for telco: FP triggers unnecessary NOC tickets
    training_time_seconds: float
    n_train_samples: int
    n_features: int
    model_version: str = "1.0.0"
    # Below fields align with TM Forum AI governance model card schema
    worst_case_latency_ms: float = 0.0
    passes_governance_gate: bool = False


@dataclass
class ModelCard:
    """
    3GPP TS 28.105 aligned model card for the governance gate.
    See Section 7: Model Registry & Governance Gate.

    Fields align with:
    - 3GPP TS 28.105 cl. 7: MLEntityRepository IOC attributes
    - TM Forum IG1230 AI Management and Governance model metadata
    - EU AI Act Annex IV technical documentation requirements
    """

    # Identity — maps to MLEntityRepository.mLEntityId
    model_id: str
    model_name: str
    model_version: str
    squad_owner: str  # e.g., "ran_optimization_squad" — Kubernetes namespace
    use_case: str

    # Training provenance — maps to MLTrainingReport IOC
    training_data_start: str
    training_data_end: str
    training_script: str
    random_seed: int
    feature_count: int
    training_sample_count: int

    # Network impact metadata — telco-specific, not in generic MLOps cards
    # Critical for blast radius assessment in the governance gate
    affected_network_elements: List[str]
    rcp_write_set: List[str]  # RAN Control Parameters this model influences
    kpi_dependency_set: List[str]  # KPIs this model reads — for conflict detection
    rollback_procedure: str
    blast_radius: str  # "cell", "site", "cluster", "region"

    # SLO targets — used by the monitoring stack (Section 7: Monitoring)
    slo_inference_latency_p99_ms: float
    slo_auc_pr_minimum: float
    slo_false_positive_rate_maximum: float
    retraining_trigger: str  # "drift_wasserstein>0.30", "scheduled_weekly", "manual"

    # Evaluation results
    tier_metrics: List[Dict[str, Any]] = field(default_factory=list)
    passes_governance_gate: bool = False
    governance_gate_notes: str = ""

    # Regulatory
    # IMPORTANT: Under the EU AI Act Annex III Section 5(b), AI systems that
    # monitor or manage critical infrastructure (including telecommunications)
    # may qualify as high-risk regardless of whether they make autonomous changes.
    # Detection-only models that generate NOC alerts influencing human operator
    # decisions on network infrastructure should be conservatively classified as
    # high-risk. Operators must conduct their own risk assessment with legal counsel.
    eu_ai_act_risk_level: str = "high"  # minimal, limited, high, unacceptable
    data_sources: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# DATA LOADING
# ============================================================================


def load_feature_splits(
    data_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load pre-computed feature splits from 02_feature_engineering.py outputs.

    Returns train, val, test DataFrames plus metadata dict.

    The temporal split boundaries are preserved from 02_feature_engineering.py —
    we must NOT re-split here. Re-splitting would break the strict temporal
    ordering guarantee required for time-series model evaluation.

    See Coursebook Ch. 3: Model Evaluation — Temporal Train/Test Splits.

    Note: 02_feature_engineering.py writes files directly to DATA_DIR as
    features_train.parquet, features_val.parquet, features_test.parquet.
    """
    logger.info(f"Loading feature splits from {data_dir}")

    # Paths must match 02_feature_engineering.py save_feature_splits() output
    train_path = data_dir / "features_train.parquet"
    val_path = data_dir / "features_val.parquet"
    test_path = data_dir / "features_test.parquet"
    meta_path = data_dir / "feature_metadata.json"

    if not all(p.exists() for p in [train_path, val_path, test_path]):
        logger.warning(
            "Feature split files not found. Running synthetic fallback data generation. "
            "In production, this should never happen — CI/CD gates on 02_feature_engineering.py "
            "completing successfully before this script runs."
        )
        return _generate_fallback_data()

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    logger.info(
        f"Loaded splits — train: {len(train_df):,} rows, "
        f"val: {len(val_df):,} rows, "
        f"test: {len(test_df):,} rows"
    )

    # Log anomaly rates to verify realistic class imbalance is preserved
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "is_anomaly" in df.columns:
            rate = df["is_anomaly"].mean()
            logger.info(f"  {name} anomaly rate: {rate:.3f} ({rate*100:.1f}%)")

    return train_df, val_df, test_df, metadata


def _generate_fallback_data() -> (
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]
):
    """
    Generate minimal synthetic feature data when 02_feature_engineering.py
    outputs are not available. Used for standalone testing of this script.

    NOTE: Feature names in this fallback do NOT exactly match the
    02_feature_engineering.py output schema. Metrics from standalone mode
    are not comparable to full-pipeline metrics. Run the full 01→02→03
    pipeline for representative results.
    """
    logger.warning(
        "⚠️ STANDALONE MODE: Using fallback synthetic data. Feature names do NOT "
        "match 02_feature_engineering.py output. Metrics are NOT comparable to "
        "full-pipeline results. Run the full 01→02→03 pipeline for representative "
        "evaluation."
    )
    logger.info("Generating fallback synthetic feature data (standalone mode)")
    rng = np.random.RandomState(RANDOM_SEED)

    n_total = 3000
    n_cells = 20
    timestamps = pd.date_range("2024-01-01", periods=n_total, freq="15min")

    # Reproduce realistic feature schema from 02_feature_engineering.py
    cells = [f"CELL_{100+i//3:03d}_{i%3+1}" for i in range(n_cells)]
    cell_ids = rng.choice(cells, size=n_total)

    # Base KPIs with realistic telco distributions
    dl_prb_base = rng.beta(2, 5, n_total) * 100  # PRB utilization 0-100%
    ul_prb_base = rng.beta(1.5, 6, n_total) * 100
    rsrp_base = rng.normal(-85, 12, n_total).clip(-140, -44)
    sinr_base = rng.normal(10, 8, n_total).clip(-10, 30)
    rrc_succ = rng.beta(18, 2, n_total) * 100  # Success rates cluster near 100%
    dl_throughput = rng.lognormal(3, 1.5, n_total).clip(0, 1000)

    # Diurnal pattern — key for temporal model validity
    hour = pd.DatetimeIndex(timestamps).hour
    diurnal = 1 + 0.5 * np.sin(2 * np.pi * (hour - 8) / 24)
    dl_prb_base *= diurnal

    # Inject anomalies at ~3% rate (realistic per spec in 01_synthetic_data.py)
    anomaly_mask = rng.random(n_total) < 0.03
    dl_prb_base[anomaly_mask] *= rng.uniform(1.8, 3.0, anomaly_mask.sum())
    rsrp_base[anomaly_mask] -= rng.uniform(15, 30, anomaly_mask.sum())
    rrc_succ[anomaly_mask] -= rng.uniform(20, 60, anomaly_mask.sum())
    dl_prb_base = dl_prb_base.clip(0, 100)
    rsrp_base = rsrp_base.clip(-140, -44)
    rrc_succ = rrc_succ.clip(0, 100)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cell_id": cell_ids,
            "dl_prb_utilization": dl_prb_base,
            "ul_prb_utilization": ul_prb_base,
            "rsrp_dbm": rsrp_base,
            "sinr_db": sinr_base,
            "rrc_conn_succ_rate": rrc_succ,
            "dl_throughput_mbps": dl_throughput,
            # Rolling features from 02_feature_engineering.py
            "dl_prb_roll_mean_4h": dl_prb_base + rng.normal(0, 2, n_total),
            "dl_prb_roll_std_4h": np.abs(rng.normal(5, 2, n_total)),
            "rsrp_roll_mean_4h": rsrp_base + rng.normal(0, 1, n_total),
            "rrc_roll_mean_4h": rrc_succ + rng.normal(0, 1, n_total),
            "dl_prb_delta_1h": rng.normal(0, 5, n_total),
            "prb_to_throughput_ratio": dl_prb_base / (dl_throughput + 1),
            # Cyclical time encodings from 02_feature_engineering.py
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_of_week": pd.DatetimeIndex(timestamps).dayofweek,
            "is_weekend": (pd.DatetimeIndex(timestamps).dayofweek >= 5).astype(int),
            "is_peak_hour": ((hour >= 8) & (hour <= 20)).astype(int),
            # Spatial peer feature from 02_feature_engineering.py
            "dl_prb_zscore_peer": rng.normal(0, 1, n_total),
            # Target
            "is_anomaly": anomaly_mask.astype(int),
        }
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Strict temporal split — 70/15/15
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    metadata = {
        "feature_columns": [
            c
            for c in df.columns
            if c not in ["timestamp", "cell_id", "is_anomaly", "anomaly_type"]
        ],
        "target_column": "is_anomaly",
        "split_boundaries": {
            "train_start": str(train_df["timestamp"].min()),
            "train_end": str(train_df["timestamp"].max()),
            "val_start": str(val_df["timestamp"].min()),
            "val_end": str(val_df["timestamp"].max()),
            "test_start": str(test_df["timestamp"].min()),
            "test_end": str(test_df["timestamp"].max()),
        },
    }

    logger.info(
        f"Fallback data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    # Fit and persist a scaler on fallback data so save_all_artifacts()
    # can embed it in the Pipeline artifact even in standalone mode.
    from sklearn.preprocessing import RobustScaler as _RS
    fallback_feature_cols = [
        c for c in train_df.columns
        if c not in {"timestamp", "cell_id", "is_anomaly", "anomaly_type"}
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    fallback_scaler = _RS(quantile_range=(5.0, 95.0))
    fallback_scaler.fit(train_df[fallback_feature_cols].fillna(0.0))
    _data_dir = Path("data")
    _data_dir.mkdir(parents=True, exist_ok=True)
    import joblib as _jl
    _jl.dump(fallback_scaler, _data_dir / "feature_scaler.joblib")
    logger.info("Fallback scaler persisted to %s", _data_dir / "feature_scaler.joblib")

    return train_df, val_df, test_df, metadata


def extract_arrays(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
]:
    """
    Extract numpy arrays for model training from DataFrames.

    Returns (X_train, y_train, X_val, y_val, X_test, y_test, feature_names).
    Labels default to 0 (normal) if 'is_anomaly' column is absent —
    supporting the unsupervised pre-labeling workflow where Tier 1 generates
    labels that Tier 2 consumes.
    """
    target_col = "target_anomaly_encoded"
    if target_col not in train_df.columns:
        target_col = "is_anomaly"

    # Identify feature columns — exclude metadata, timestamps, and target
    exclude_cols = {
        "timestamp",
        "cell_id",
        "is_anomaly",
        "target_anomaly_encoded",
        "anomaly_type",
        "anomaly_subtype",
        "rop_start",
        "rop_end",
        # Ground-truth anomaly components from 01_synthetic_data.py — target leakage
        "anomaly_rrc_congestion",
        "anomaly_hw_degradation",
        "anomaly_counter_reset",
        "anomaly_traffic_spike",
    }

    # Use metadata feature list if available (produced by 02_feature_engineering.py)
    if "feature_columns" in metadata:
        feature_cols = [
            c
            for c in metadata["feature_columns"]
            if c in train_df.columns and c not in exclude_cols
        ]
    else:
        feature_cols = [
            c
            for c in train_df.columns
            if c not in exclude_cols
            and train_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

    if not feature_cols:
        raise ValueError(
            "No feature columns found. Check that 02_feature_engineering.py ran successfully."
        )

    logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")

    def _get_X_y(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df[feature_cols].values.astype(np.float32)
        # Replace NaN with 0 — NaN handling should have been done in 02, but
        # defensive coding is important when data arrives from multiple pipeline stages
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = df[target_col].values.astype(int) if target_col in df.columns else np.zeros(len(df), dtype=int)
        return X, y

    X_train, y_train = _get_X_y(train_df)
    X_val, y_val = _get_X_y(val_df)
    X_test, y_test = _get_X_y(test_df)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# ============================================================================
# TIER 1: UNSUPERVISED AUTO-LABELING
# Isolation Forest + One-Class SVM ensemble
# ============================================================================


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.03,  # ~3% anomaly rate per telco domain knowledge
) -> Tuple[IsolationForest, float]:
    """
    Train Isolation Forest for unsupervised anomaly scoring.

    Contamination is set to 0.03 based on domain knowledge from
    01_synthetic_data.py — real networks see 1-5% anomaly rates.
    Setting this too high causes FP inflation; too low causes FN inflation.

    In the reference architecture, this feeds the auto-labeling pipeline
    that generates pseudo-labels for Tier 2 Random Forest training.
    This pattern avoids the need for expensive manual labeling while
    still producing calibrated supervised models.

    See Coursebook Ch. 6: Anomaly Detection — Isolation Forest.
    """
    logger.info("Training Tier 1a: Isolation Forest")
    t0 = time.time()

    # n_estimators=200 is a conservative choice over the default 100.
    # For telco PM data with high-dimensional, correlated features (PRB util,
    # throughput, SINR are correlated), more trees reduce variance significantly.
    # max_samples='auto' uses min(256, n_samples) which is appropriate.
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_features=1.0,  # Use all features — iForest is robust to irrelevant features
        bootstrap=False,   # Non-bootstrap gives better isolation paths for telco data
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train)

    # Raw scores: negative = anomaly, positive = normal
    # Negate so higher score = more anomalous (consistent with other scorers)
    raw_scores = model.decision_function(X_train)
    anomaly_scores = -raw_scores  # Invert: higher = more anomalous

    # Threshold at contamination percentile — scores above this are labeled anomaly
    threshold = np.percentile(anomaly_scores, (1 - contamination) * 100)

    elapsed = time.time() - t0
    predicted_anomaly_rate = (anomaly_scores > threshold).mean()
    logger.info(
        f"  Isolation Forest trained in {elapsed:.1f}s | "
        f"threshold={threshold:.4f} | "
        f"predicted anomaly rate={predicted_anomaly_rate:.3f}"
    )

    return model, threshold


def train_ocsvm(
    X_train: np.ndarray,
    nu: float = 0.03,  # Upper bound on fraction of anomalies in training data
    subsample_size: int = 2000,
) -> Tuple[OneClassSVM, float, np.ndarray]:
    """
    Train One-Class SVM for unsupervised anomaly scoring.

    OC-SVM complements Isolation Forest: iForest is strong on high-dimensional
    sparse anomalies (sudden spikes), while OC-SVM with RBF kernel captures
    smooth boundary violations (gradual degradation in RSRP + SINR together).

    COMPUTATIONAL NOTE: OC-SVM is O(n²) in training. We subsample to 2000
    points for the training fit. In production, this would use a Nyström
    approximation (sklearn.kernel_approximation.Nystroem) for full-scale data.
    The subsample_size parameter allows tuning this in CI vs. production.

    nu=0.03 matches the contamination parameter in Isolation Forest for
    consistent ensemble combination.
    """
    logger.info(
        f"Training Tier 1b: One-Class SVM (subsample={min(subsample_size, len(X_train))})"
    )
    t0 = time.time()

    # Scale features — OC-SVM is sensitive to feature scale unlike iForest
    scaler = StandardScaler()

    # Subsample for training to manage quadratic complexity
    if len(X_train) > subsample_size:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X_train), subsample_size, replace=False)
        X_fit = X_train[idx]
    else:
        X_fit = X_train

    X_fit_scaled = scaler.fit_transform(X_fit)
    X_train_scaled = scaler.transform(X_train)

    # gamma='scale' = 1/(n_features * X.var()) — better than 'auto' for high-dim data
    model = OneClassSVM(
        kernel="rbf",
        nu=nu,
        gamma="scale",
    )
    model.fit(X_fit_scaled)

    # Score entire training set
    raw_scores = model.decision_function(X_train_scaled)
    anomaly_scores = -raw_scores  # Invert: higher = more anomalous

    threshold = np.percentile(anomaly_scores, (1 - nu) * 100)

    elapsed = time.time() - t0
    predicted_anomaly_rate = (anomaly_scores > threshold).mean()
    logger.info(
        f"  OC-SVM trained in {elapsed:.1f}s | "
        f"threshold={threshold:.4f} | "
        f"predicted anomaly rate={predicted_anomaly_rate:.3f}"
    )

    # Package scaler with model for consistent inference
    pipeline = Pipeline([("scaler", scaler), ("ocsvm", model)])
    return pipeline, threshold, anomaly_scores


def generate_ensemble_autolabels(
    iforest_model: IsolationForest,
    ocsvm_pipeline: Pipeline,
    X_train: np.ndarray,
    train_df: pd.DataFrame,
    iforest_threshold: float,
    ocsvm_threshold: float,
    agreement_policy: str = "union",  # "union" | "intersection" | "majority"
) -> pd.DataFrame:
    """
    Combine Isolation Forest and OC-SVM scores into ensemble auto-labels.

    The auto-labels serve as pseudo-ground-truth for Tier 2 Random Forest.
    Policy options:
    - 'union':        label as anomaly if EITHER model flags it (high recall)
    - 'intersection': label as anomaly only if BOTH flag it (high precision)
    - 'majority':     equivalent to union for 2 models

    For telco RAN anomaly detection, 'union' is preferred because:
    - False negatives (missed anomalies) are more costly than FPs
    - The Tier 2 Random Forest will refine the noisy labels
    - NOC workflow can handle some FP investigation load

    This ensemble auto-labeling pattern is described in the whitepaper
    Section 8 as a cost-effective alternative to manual labeling for
    bootstrapping supervised models.

    See Coursebook Ch. 6: Anomaly Detection — Label Propagation Strategies.
    """
    logger.info(f"Generating ensemble auto-labels (policy='{agreement_policy}')")

    # Isolation Forest scores
    if_scores = -iforest_model.decision_function(X_train)
    if_labels = (if_scores > iforest_threshold).astype(int)

    # OC-SVM scores (includes scaling)
    oc_raw = -ocsvm_pipeline.decision_function(X_train)
    oc_labels = (oc_raw > ocsvm_threshold).astype(int)

    # Ensemble decision
    if agreement_policy == "union":
        ensemble_labels = (if_labels | oc_labels).astype(int)
    elif agreement_policy == "intersection":
        ensemble_labels = (if_labels & oc_labels).astype(int)
    else:  # majority (for 2 models, equivalent to union)
        ensemble_labels = (if_labels | oc_labels).astype(int)

    # Anomaly score: average normalized scores for ranking (useful for active learning)
    if_norm = (if_scores - if_scores.min()) / (if_scores.ptp() + 1e-10)
    oc_norm = (oc_raw - oc_raw.min()) / (oc_raw.ptp() + 1e-10)
    ensemble_score = 0.5 * if_norm + 0.5 * oc_norm

    result_df = train_df[["timestamp", "cell_id"]].copy() if "timestamp" in train_df.columns else pd.DataFrame()
    if result_df.empty:
        result_df = pd.DataFrame(index=range(len(X_train)))
    result_df["iforest_label"] = if_labels
    result_df["ocsvm_label"] = oc_labels
    result_df["auto_label"] = ensemble_labels
    result_df["ensemble_score"] = ensemble_score

    # If we have ground truth, compute auto-label quality metrics
    if "is_anomaly" in train_df.columns:
        y_true = train_df["is_anomaly"].values
        from sklearn.metrics import f1_score as sk_f1

        if_f1 = sk_f1(y_true, if_labels, zero_division=0)
        oc_f1 = sk_f1(y_true, oc_labels, zero_division=0)
        ensemble_f1 = sk_f1(y_true, ensemble_labels, zero_division=0)
        logger.info(
            f"  Auto-label quality (vs ground truth): "
            f"iForest F1={if_f1:.3f}, OC-SVM F1={oc_f1:.3f}, Ensemble F1={ensemble_f1:.3f}"
        )
        result_df["true_label"] = y_true

    anomaly_rate = ensemble_labels.mean()
    logger.info(
        f"  Auto-label anomaly rate: {anomaly_rate:.3f} "
        f"({anomaly_rate*100:.1f}% of {len(ensemble_labels):,} samples)"
    )

    return result_df


# ============================================================================
# TIER 2: SUPERVISED RANDOM FOREST WITH SHAP
# Trained on auto-labels from Tier 1
# ============================================================================


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    full_search: bool = False,
) -> Tuple[RandomForestClassifier, float, TierMetrics]:
    """
    Train Random Forest classifier on auto-labels from Tier 1.

    Hyperparameter choices are justified for telco tabular PM counter data:
    - n_estimators=300: sufficient for stable probability estimates with ~20 features
    - max_depth=12: prevents overfitting on 3% anomaly rate (imbalanced classes)
    - min_samples_leaf=5: regularization — prevents fitting to label noise from auto-labeling
    - class_weight='balanced': critical for 3% anomaly rate without this,
      the model will predict all-normal and achieve 97% accuracy trivially

    full_search=True enables grid search over hyperparameters — use in CI pipelines,
    not in interactive development.

    See Coursebook Ch. 3: Model Evaluation — Class Imbalance Handling.
    See Coursebook Ch. 4: Feature Engineering — SHAP Values.
    """
    logger.info("Training Tier 2: Random Forest Classifier")
    t0 = time.time()

    logger.info(
        f"  Training data: {len(X_train):,} samples, "
        f"anomaly rate={y_train.mean():.3f}, "
        f"{X_train.shape[1]} features"
    )

    if full_search:
        # Grid search — used in automated CI pipeline, not interactive dev
        # Coursebook Ch. 7: MLOps — Hyperparameter Optimization
        model = _grid_search_rf(X_train, y_train, X_val, y_val)
    else:
        # Expert-tuned defaults for telco PM counter anomaly detection
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            min_samples_split=10,
            max_features="sqrt",       # Standard for classification (sqrt of n_features)
            class_weight="balanced",   # Critical: addresses 3% anomaly rate imbalance
            bootstrap=True,
            oob_score=True,            # Out-of-bag score provides free validation estimate
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    elapsed = time.time() - t0

    if hasattr(model, "oob_score_"):
        logger.info(f"  OOB score (approx. val accuracy): {model.oob_score_:.4f}")

    # Calibrate threshold on validation set
    # Default 0.5 threshold is suboptimal for imbalanced classes.
    # We find threshold that maximizes F1 on validation set.
    # This threshold is stored in the model registry for serving.
    val_proba = model.predict_proba(X_val)[:, 1]
    best_threshold, best_f1 = _find_optimal_threshold(val_proba, y_val)

    # Compute full metrics at optimal threshold
    metrics = _compute_tier_metrics(
        model_name="tier2_random_forest",
        y_true=y_val,
        y_score=val_proba,
        threshold=best_threshold,
        training_time=elapsed,
        n_train=len(X_train),
        n_features=X_train.shape[1],
    )

    logger.info(
        f"  RF trained in {elapsed:.1f}s | "
        f"threshold={best_threshold:.3f} | "
        f"Val AUC-ROC={metrics.auc_roc:.3f} | "
        f"Val AUC-PR={metrics.auc_pr:.3f} | "
        f"Val F1={metrics.f1_at_threshold:.3f}"
    )

    return model, best_threshold, metrics


def _grid_search_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> RandomForestClassifier:
    """
    Grid search over RF hyperparameters.
    Uses TimeSeriesSplit to respect temporal ordering — each validation fold
    uses only future data relative to its training fold, preventing temporal
    leakage. This is mandatory for time-series network data.

    Note: TimeSeriesSplit does not stratify by class. For rare anomalies,
    verify each fold contains positive samples; if not, increase n_splits
    or use a custom temporal CV with stratification.

    Note: This is intentionally lightweight — a production grid search
    would use Optuna or Ray Tune with a larger search space.
    The key insight is that for tabular telco data, the hyperparameter
    sensitivity ordering is: class_weight >> n_estimators >> max_depth.
    """
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [8, 12],
        "min_samples_leaf": [3, 5],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    # AUC-PR is the right metric for imbalanced anomaly detection
    # AUC-ROC is invariant to class imbalance but AUC-PR reflects operational impact
    # TimeSeriesSplit maintains temporal ordering: each fold's test set
    # is strictly after its training set.
    cv = TimeSeriesSplit(n_splits=3)

    # Verify each CV fold has positive samples — without them,
    # average_precision scoring returns nan and may select a degenerate model.
    cv_valid = True
    for fold_idx, (_train_idx, _val_idx) in enumerate(cv.split(X_train)):
        n_pos = int(y_train[_val_idx].sum())
        if n_pos < 2:
            logger.warning(
                "TimeSeriesSplit fold %d has only %d positive samples in "
                "validation — insufficient for average_precision scoring.",
                fold_idx, n_pos,
            )
            cv_valid = False
    if not cv_valid:
        fallback_params = dict(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced",
        )
        logger.warning(
            "TimeSeriesSplit produced folds with <2 positive samples. "
            "Falling back to expert-tuned defaults: %s. "
            "Grid search results NOT available.",
            fallback_params,
        )
        rf = RandomForestClassifier(
            **fallback_params,
            bootstrap=True, random_state=RANDOM_SEED, n_jobs=-1,
        ).fit(X_train, y_train)
        rf._telco_hyperparameter_source = "expert_default_fallback"
        return rf

    base = RandomForestClassifier(
        max_features="sqrt",
        bootstrap=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    gs = GridSearchCV(base, param_grid, scoring="average_precision", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    logger.info(f"  Grid search best params: {gs.best_params_}")
    logger.info(f"  Grid search best CV AUC-PR: {gs.best_score_:.4f}")
    gs.best_estimator_._telco_hyperparameter_source = "grid_search"
    return gs.best_estimator_


def _find_optimal_threshold(
    y_score: np.ndarray,
    y_true: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find the decision threshold that maximizes the target metric on validation data.

    Using F1 as the optimization metric because:
    - AUC-PR measures model quality; threshold determines operational behavior
    - For NOC workflows, F1 balances alert volume (precision) with coverage (recall)
    - Operators can override to optimize for precision or recall based on team capacity

    In the reference architecture, this threshold is stored in thresholds.json
    and loaded by the serving layer. Changing the threshold does not require
    model retraining — only a metadata update in the model registry.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one more value than thresholds
    # (the last point is precision=1, recall=0 at threshold=inf)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10),
        0.0,
    )

    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    return best_threshold, best_f1


def compute_shap_importance(
    model: RandomForestClassifier,
    X_val: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    max_samples: int = 500,
) -> pd.DataFrame:
    """
    Compute SHAP values for feature importance explainability.

    SHAP is mandatory for the governance gate model card in the reference
    architecture. Network operations engineers need to understand WHY a model
    flagged an anomaly — not just THAT it did. This directly addresses
    the EU AI Act requirement for explainability of high-impact automated decisions.

    In the whitepaper's architecture, SHAP values are:
    1. Stored in the model registry alongside model artifacts
    2. Surfaced in the NOC dashboard per inference (top-3 contributing features)
    3. Used for model-to-model conflict detection (feature overlap analysis)

    See Coursebook Ch. 4: Feature Engineering — SHAP Values and Model Interpretability.
    """
    logger.info("Computing SHAP feature importance")
    t0 = time.time()

    # Subsample validation set for SHAP computation — full set is slow
    if len(X_val) > max_samples:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X_val), max_samples, replace=False)
        X_explain = X_val[idx]
    else:
        X_explain = X_val

    if SHAP_AVAILABLE:
        # TreeExplainer is exact and fast for Random Forest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)

        # For binary classification, shap_values is a list [class0_vals, class1_vals]
        # We want class 1 (anomaly) SHAP values
        if isinstance(shap_values, list):
            shap_vals_anomaly = shap_values[1]
        else:
            shap_vals_anomaly = shap_values

        # Mean absolute SHAP value per feature — global importance measure
        mean_abs_shap = np.abs(shap_vals_anomaly).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": feature_names, "shap_importance": mean_abs_shap}
        ).sort_values("shap_importance", ascending=False)

        method = "SHAP TreeExplainer"
    else:
        # Fallback to sklearn built-in feature importances (Gini-based)
        # Less reliable than SHAP for correlated features — common in PM data
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_importance": model.feature_importances_,
            }
        ).sort_values("shap_importance", ascending=False)
        method = "sklearn feature_importances_ (Gini, SHAP unavailable)"

    elapsed = time.time() - t0
    logger.info(f"  Feature importance computed via {method} in {elapsed:.1f}s")
    logger.info("  Top 10 features:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"    {row['feature']:<40} {row['shap_importance']:.4f}")

    # Save importance figure
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(20, len(importance_df))
    plot_df = importance_df.head(top_n).sort_values("shap_importance")
    ax.barh(plot_df["feature"], plot_df["shap_importance"], color="#2196F3", alpha=0.8)
    ax.set_xlabel("Mean |SHAP Value| (anomaly class)", fontsize=12)
    ax.set_title(
        f"Tier 2 Random Forest — Feature Importance\n(Top {top_n}, {method})",
        fontsize=13,
    )
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved feature_importance.png")

    return importance_df


# ============================================================================
# TIER 3: LSTM AUTOENCODER
# Sequential temporal anomaly detection
# ============================================================================


def build_lstm_autoencoder(
    seq_len: int,
    n_features: int,
    hidden_dim: int = 64,
    latent_dim: int = 16,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> "nn.Module":
    """
    Build LSTM Autoencoder for temporal anomaly detection.

    Architecture rationale:
    - Encoder LSTM learns compressed latent representation of normal 15min PM windows
    - Decoder LSTM reconstructs the input sequence from the latent state
    - Reconstruction error on anomalous sequences is significantly higher
      because the model has only seen normal patterns during training

    This is the "train on normal only" paradigm:
    - Training data: ONLY samples labeled as normal by Tier 1 ensemble
    - Inference: reconstruction error > threshold → anomaly

    seq_len=8 corresponds to a 2-hour window at 15-min granularity.
    This captures diurnal patterns at a sub-period resolution while being
    short enough for real-time xApp inference latency budgets.

    See Coursebook Ch. 5: Time Series Analysis — LSTM Architectures.
    See Coursebook Ch. 6: Anomaly Detection — Autoencoder-based Methods.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for LSTM Autoencoder")

    class LSTMAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            # Encoder: compress sequence to latent representation
            self.encoder = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False,  # Causal: no future leakage for online serving
            )

            # Bottleneck projection
            self.to_latent = nn.Linear(hidden_dim, latent_dim)
            self.from_latent = nn.Linear(latent_dim, hidden_dim)

            # Decoder: reconstruct sequence from latent state
            self.decoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False,
            )

            # Output projection back to input space
            self.output_proj = nn.Linear(hidden_dim, n_features)

            # Activation for bottleneck
            self.relu = nn.ReLU()

        def encode(
            self, x: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Encode input sequence to latent vector."""
            _, (h_n, c_n) = self.encoder(x)
            # Use final layer hidden state
            z = self.relu(self.to_latent(h_n[-1]))
            return z, (h_n, c_n)

        def decode(
            self, z: "torch.Tensor", seq_len: int
        ) -> "torch.Tensor":
            """Decode latent vector to reconstructed sequence."""
            h0 = self.relu(self.from_latent(z))
            # Expand to (num_layers, batch, hidden_dim)
            h0 = h0.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
            c0 = torch.zeros_like(h0)
            # Repeat latent as input to each decoder step
            decoder_input = h0[-1].unsqueeze(1).repeat(1, seq_len, 1)
            output, _ = self.decoder(decoder_input, (h0, c0))
            reconstructed = self.output_proj(output)
            return reconstructed

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            z, _ = self.encode(x)
            return self.decode(z, x.size(1))

    return LSTMAutoencoder()


def prepare_lstm_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 8,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert tabular feature matrix to sliding window sequences for LSTM input.

    seq_len=8 = 2 hours at 15-min granularity.
    A window is labeled anomalous if ANY row in the window is anomalous.
    This "any-positive" labeling is appropriate for detection use cases.

    stride=1 is used for training diversity. In production serving,
    stride=seq_len (non-overlapping) is used for efficiency.
    """
    n_samples = len(X)
    n_windows = max(0, (n_samples - seq_len) // stride + 1)

    if n_windows == 0:
        return np.empty((0, seq_len, X.shape[1])), np.empty(0)

    X_seq = np.zeros((n_windows, seq_len, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros(n_windows, dtype=np.int64)

    for i in range(n_windows):
        start = i * stride
        end = start + seq_len
        X_seq[i] = X[start:end]
        y_seq[i] = int(y[start:end].any())

    return X_seq, y_seq


def train_lstm_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path,
    seq_len: int = 8,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> Tuple[Any, float, TierMetrics]:
    """
    Train LSTM Autoencoder on NORMAL sequences only.

    Training protocol:
    1. Filter training data to normal-only samples (using Tier 1 auto-labels)
    2. Train autoencoder to minimize reconstruction MSE on normal sequences
    3. Calibrate reconstruction error threshold on validation set
    4. Evaluate on full validation set including anomalies

    The threshold calibration on validation (not training) data is critical:
    the model has never seen anomalous sequences, so the training loss alone
    cannot inform the threshold. Validation set represents the deployment
    distribution including both normal and anomalous sequences.

    See Coursebook Ch. 6: Anomaly Detection — Reconstruction Error Thresholds.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for LSTM Autoencoder training")

    logger.info("Training Tier 3: LSTM Autoencoder")
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    # Feature normalization — LSTM is sensitive to scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    # Prepare sequences
    X_train_seq, y_train_seq = prepare_lstm_sequences(
        X_train_scaled, y_train, seq_len=seq_len, stride=1
    )
    X_val_seq, y_val_seq = prepare_lstm_sequences(
        X_val_scaled, y_val, seq_len=seq_len, stride=seq_len  # Non-overlapping for eval
    )

    if len(X_train_seq) == 0:
        raise ValueError(f"Insufficient training data for seq_len={seq_len}")

    # CRITICAL: Train only on normal sequences
    # This is the "train on normal" paradigm for autoencoder anomaly detection
    normal_mask = y_train_seq == 0
    X_normal_seq = X_train_seq[normal_mask]
    logger.info(
        f"  Using {len(X_normal_seq):,} normal sequences for training "
        f"({normal_mask.sum()/len(normal_mask)*100:.1f}% of sequences)"
    )

    if len(X_normal_seq) < batch_size:
        logger.warning(
            f"  Insufficient normal sequences ({len(X_normal_seq)} < {batch_size}). "
            "Reducing batch size."
        )
        batch_size = max(1, len(X_normal_seq) // 4)

    # PyTorch DataLoader
    train_tensor = torch.FloatTensor(X_normal_seq).to(device)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle is OK here — sequences are independent
        drop_last=True,
    )

    n_features = X_train.shape[1]
    model = build_lstm_autoencoder(seq_len, n_features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # ReduceLROnPlateau: reduce LR when val loss stagnates
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )
    criterion = nn.MSELoss()

    # Early stopping: stop if validation reconstruction loss doesn't improve
    best_val_loss = float("inf")
    patience_count = 0
    early_stop_patience = 10
    best_state_dict = None
    train_losses, val_losses = [], []

    logger.info(f"  Training {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            # Gradient clipping: important for LSTM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        # Validation reconstruction loss on ALL sequences (normal + anomaly)
        model.eval()
        with torch.no_grad():
            val_tensor = torch.FloatTensor(X_val_seq).to(device)
            val_recon = model(val_tensor)
            # Per-sample reconstruction error (mean over seq_len and features)
            val_recon_errors = ((val_tensor - val_recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
            # Report validation loss on normal sequences only (fair comparison to train loss)
            val_normal_mask = y_val_seq == 0
            if val_normal_mask.sum() > 0:
                val_loss = float(val_recon_errors[val_normal_mask].mean())
            else:
                val_loss = float(val_recon_errors.mean())
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_count = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"patience={patience_count}/{early_stop_patience}"
            )

        if patience_count >= early_stop_patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Save model artifact
    model_path = output_dir / "tier3_lstm_autoencoder.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "seq_len": seq_len,
                "n_features": n_features,
                "hidden_dim": 64,
                "latent_dim": 16,
                "num_layers": 2,
                "dropout": 0.1,
            },
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "random_seed": RANDOM_SEED,
        },
        model_path,
    )

    # Calibrate threshold on validation set
    # We use the 95th percentile of normal reconstruction errors as threshold.
    # This means ~5% of normal sequences will be flagged as anomalies (FP rate target).
    # Operators can tune this percentile based on NOC alert tolerance.
    if val_normal_mask.sum() > 0:
        normal_errors = val_recon_errors[val_normal_mask]
        threshold = float(np.percentile(normal_errors, 95))
    else:
        threshold = float(np.percentile(val_recon_errors, 95))

    elapsed = time.time() - t0
    metrics = _compute_tier_metrics(
        model_name="tier3_lstm_autoencoder",
        y_true=y_val_seq,
        y_score=val_recon_errors,
        threshold=threshold,
        training_time=elapsed,
        n_train=len(X_normal_seq),
        n_features=n_features,
    )

    logger.info(
        f"  LSTM trained in {elapsed:.1f}s | "
        f"threshold={threshold:.6f} | "
        f"Val AUC-ROC={metrics.auc_roc:.3f} | "
        f"Val AUC-PR={metrics.auc_pr:.3f} | "
        f"Val F1={metrics.f1_at_threshold:.3f}"
    )

    # Plot training curves
    _plot_training_curves(train_losses, val_losses, output_dir)

    return model, threshold, metrics, scaler


def _plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    output_dir: Path,
) -> None:
    """Plot LSTM autoencoder training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss (normal only)", color="#2196F3")
    ax.plot(epochs, val_losses, label="Val Loss (normal only)", color="#FF5722", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("LSTM Autoencoder Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# BASELINE MODELS
# Naive comparisons required by governance gate to prove ML adds value
# ============================================================================


def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate baseline models for comparison.

    Governance gate requirement: a new model must outperform all baselines
    on AUC-PR before being promoted to staging. This prevents cases where
    an overfitted model beats naive thresholds only in training distribution.

    Baselines implemented:
    1. Static threshold on single feature (dl_prb_utilization > 90%)
       — represents current rule-based NOC approach
    2. Logistic Regression — simple linear classifier
    3. Majority class predictor — absolute floor (predicts all normal)

    See Coursebook Ch. 3: Model Evaluation — Baseline Comparisons.
    """
    logger.info("Training baselines for governance gate comparison")
    baselines = {}

    # -----------------------------------------------------------------------
    # Baseline 1: Rule-based threshold (current NOC approach proxy)
    # Uses the most predictive single feature as a threshold classifier
    # -----------------------------------------------------------------------
    # Find most correlated feature with anomaly label
    if y_train.sum() > 0:
        correlations = np.abs(
            np.array([np.corrcoef(X_train[:, i], y_train)[0, 1] for i in range(X_train.shape[1])])
        )
        correlations = np.nan_to_num(correlations, nan=0.0)
        best_feature_idx = int(np.argmax(correlations))
        best_feature_name = (
            feature_names[best_feature_idx]
            if best_feature_idx < len(feature_names)
            else f"feature_{best_feature_idx}"
        )
        logger.info(
            f"  Baseline 1 uses feature: {best_feature_name} "
            f"(correlation={correlations[best_feature_idx]:.3f})"
        )

        # Threshold at 95th percentile of training distribution
        rule_threshold = np.percentile(X_train[:, best_feature_idx], 95)
        val_scores = X_val[:, best_feature_idx]
        val_preds = (val_scores > rule_threshold).astype(int)

        # Normalize to [0,1] for AUC computation
        val_scores_norm = (val_scores - val_scores.min()) / (val_scores.ptp() + 1e-10)
        if len(np.unique(y_val)) > 1:
            auc_roc = roc_auc_score(y_val, val_scores_norm)
            auc_pr = average_precision_score(y_val, val_scores_norm)
        else:
            auc_roc, auc_pr = 0.5, y_val.mean()
        f1 = f1_score(y_val, val_preds, zero_division=0)

        baselines["rule_based_threshold"] = {
            "description": f"Single-feature threshold: {best_feature_name} > {rule_threshold:.2f}",
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "f1": f1,
        }
        logger.info(
            f"  Rule-based: AUC-ROC={auc_roc:.3f}, AUC-PR={auc_pr:.3f}, F1={f1:.3f}"
        )

    # -----------------------------------------------------------------------
    # Baseline 2: Logistic Regression (linear model)
    # -----------------------------------------------------------------------
    lr_scaler = StandardScaler()
    X_train_lr = lr_scaler.fit_transform(X_train)
    X_val_lr = lr_scaler.transform(X_val)

    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
        C=1.0,
    )
    try:
        lr_model.fit(X_train_lr, y_train)
        lr_proba = lr_model.predict_proba(X_val_lr)[:, 1]
        lr_threshold, _ = _find_optimal_threshold(lr_proba, y_val)
        lr_preds = (lr_proba > lr_threshold).astype(int)

        if len(np.unique(y_val)) > 1:
            auc_roc = roc_auc_score(y_val, lr_proba)
            auc_pr = average_precision_score(y_val, lr_proba)
        else:
            auc_roc, auc_pr = 0.5, y_val.mean()
        f1 = f1_score(y_val, lr_preds, zero_division=0)

        baselines["logistic_regression"] = {
            "description": "Logistic Regression with class_weight=balanced",
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "f1": f1,
        }
        logger.info(
            f"  Logistic Regression: AUC-ROC={auc_roc:.3f}, AUC-PR={auc_pr:.3f}, F1={f1:.3f}"
        )
    except Exception as e:
        logger.warning(f"  Logistic Regression training failed: {e}")
        baselines["logistic_regression"] = {"auc_roc": 0.5, "auc_pr": float(y_val.mean()), "f1": 0.0}

    # -----------------------------------------------------------------------
    # Baseline 3: Majority class (all-normal predictor)
    # Represents the trivial "do nothing" baseline
    # Any useful model must significantly outperform this
    # -----------------------------------------------------------------------
    majority_class_auc_pr = float(y_val.mean())  # Precision = anomaly rate
    baselines["majority_class"] = {
        "description": "Predict all-normal (majority class)",
        "auc_roc": 0.5,  # Random performance on ROC
        "auc_pr": majority_class_auc_pr,
        "f1": 0.0,
    }
    logger.info(
        f"  Majority class: AUC-ROC=0.500, AUC-PR={majority_class_auc_pr:.3f}, F1=0.000"
    )

    return baselines


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================


def _compute_tier_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    training_time: float,
    n_train: int,
    n_features: int,
) -> TierMetrics:
    """
    Compute comprehensive evaluation metrics for a single model tier.

    Metrics selected for telco anomaly detection governance gate:
    - AUC-ROC: overall discrimination ability (insensitive to threshold)
    - AUC-PR: discrimination for imbalanced classes (sensitive to anomaly rate)
    - F1 at threshold: operational metric balancing alert volume and coverage
    - False positive rate: critical for NOC alert fatigue — high FPR causes
      engineers to ignore alerts, which defeats the purpose of the system

    See Coursebook Ch. 3: Model Evaluation — Choosing the Right Metrics.
    """
    y_pred = (y_score > threshold).astype(int)

    if len(np.unique(y_true)) > 1:
        auc_roc = float(roc_auc_score(y_true, y_score))
        auc_pr = float(average_precision_score(y_true, y_score))
    else:
        # Edge case: single class in evaluation set (shouldn't happen with proper splits)
        logger.warning(f"  Only one class in y_true for {model_name}. AUC metrics defaulted.")
        auc_roc = 0.5
        auc_pr = float(y_true.mean())

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_pred == 1) + 1e-10))
    recall = float(np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-10))

    # FPR = FP / (FP + TN) = proportion of normal samples incorrectly flagged
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fpr = float(fp / (fp + tn + 1e-10))

    return TierMetrics(
        tier_name=model_name,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        f1_at_threshold=f1,
        precision_at_threshold=precision,
        recall_at_threshold=recall,
        threshold=threshold,
        false_positive_rate=fpr,
        training_time_seconds=training_time,
        n_train_samples=n_train,
        n_features=n_features,
    )


def evaluate_on_test_set(
    tier2_model: RandomForestClassifier,
    tier2_threshold: float,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Final evaluation on held-out test set.

    IMPORTANT: The test set is touched EXACTLY ONCE — after all hyperparameter
    tuning and threshold calibration are complete. Using the test set for ANY
    decision (threshold selection, feature selection) is data leakage.

    Outputs:
    - JSON metrics report (fed into governance gate CI/CD check)
    - ROC and PR curve plots
    - Confusion matrix

    See Coursebook Ch. 3: Model Evaluation — The Importance of Test Set Isolation.
    """
    logger.info("Evaluating Tier 2 Random Forest on test set (final holdout)")

    test_proba = tier2_model.predict_proba(X_test)[:, 1]
    test_preds = (test_proba > tier2_threshold).astype(int)

    if len(np.unique(y_test)) > 1:
        auc_roc = float(roc_auc_score(y_test, test_proba))
        auc_pr = float(average_precision_score(y_test, test_proba))
        fpr_curve, tpr_curve, _ = roc_curve(y_test, test_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_proba)
    else:
        logger.warning("Only one class in test set — AUC metrics unreliable")
        auc_roc, auc_pr = 0.5, float(y_test.mean())
        fpr_curve = tpr_curve = np.array([0, 1])
        precision_curve = recall_curve = np.array([1, 0])

    f1 = float(f1_score(y_test, test_preds, zero_division=0))
    cm = confusion_matrix(y_test, test_preds)
    report = classification_report(y_test, test_preds, target_names=["normal", "anomaly"], output_dict=True)

    logger.info(f"  Test AUC-ROC: {auc_roc:.4f}")
    logger.info(f"  Test AUC-PR:  {auc_pr:.4f}")
    logger.info(f"  Test F1:      {f1:.4f}")
    logger.info(f"  Confusion matrix:\n{cm}")

    # Plot ROC and PR curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ROC Curve
    axes[0].plot(fpr_curve, tpr_curve, color="#2196F3", lw=2, label=f"RF (AUC={auc_roc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — Test Set")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PR Curve
    axes[1].plot(recall_curve, precision_curve, color="#FF5722", lw=2, label=f"RF (AUC-PR={auc_pr:.3f})")
    axes[1].axhline(y_test.mean(), color="k", linestyle="--", lw=1, label=f"Random (AP={y_test.mean():.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve — Test Set")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Confusion Matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Normal", "Predicted Anomaly"],
        yticklabels=["True Normal", "True Anomaly"],
        ax=axes[2],
    )
    axes[2].set_title(f"Confusion Matrix\nF1={f1:.3f} @ threshold={tier2_threshold:.3f}")

    plt.suptitle("Tier 2 Random Forest — Test Set Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "test_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved test_evaluation.png")

    # Operational interpretation log
    tn_val, fp_val, fn_val, tp_val = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    total_anomalies = int(y_test.sum())
    total_normal = int((y_test == 0).sum())
    logger.info("  Operational interpretation:")
    logger.info(f"    Of {total_anomalies} true anomalies: {tp_val} detected, {fn_val} missed")
    logger.info(
        f"    Of {total_normal} normal periods: {tn_val} correctly cleared, "
        f"{fp_val} false alerts ({fp_val/total_normal*100:.1f}% FP rate)"
    )
    logger.info(
        f"    NOC impact: ~{fp_val} false tickets per {len(y_test)} ROP intervals "
        f"({fp_val/(len(y_test)/96):.1f} false alerts/day at 15min ROP)"
    )

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1": f1,
        "threshold": tier2_threshold,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "false_positive_rate": float(fp_val / (total_normal + 1e-10)),
        "false_negative_rate": float(fn_val / (total_anomalies + 1e-10)),
    }


# ============================================================================
# GOVERNANCE GATE
# ============================================================================


def check_governance_gate(
    tier_metrics: List[TierMetrics],
    test_metrics: Dict[str, Any],
    baselines: Dict[str, Dict[str, float]],
    split_metadata: SplitMetadata,
) -> Tuple[bool, str]:
    """
    Evaluate model against governance gate criteria before registry promotion.

    The governance gate is the most critical component in the reference
    architecture (Section 7: Model Registry & Governance Gate). Models that
    fail the gate are registered as 'failed-validation' — not promoted to staging.

    Gate criteria (telco-specific, tuned for RAN anomaly detection):
    1. AUC-PR >= 0.60  — must substantially outperform random (anomaly rate ~3%)
    2. AUC-ROC >= 0.80 — strong overall discrimination
    3. FPR <= 0.05     — NOC alert fatigue threshold: no more than 5% of normal
                         periods should generate false alerts
    4. F1 >= 0.40      — operational balance between detection and precision
    5. Must outperform best baseline on AUC-PR by >= 5%

    These thresholds are defined in the model card's SLO fields and can be
    overridden per model family in the platform policy (OPA rules in prod).

    See Coursebook Ch. 7: MLOps Core — Model Governance and Validation Gates.
    See Whitepaper Section 7: Governance Gate — OPA policy enforcement.
    """
    logger.info("Running governance gate checks")
    gate_results = []
    passed = True

    # Gate 1: AUC-PR threshold
    auc_pr = test_metrics.get("auc_pr", 0.0)
    gate1 = auc_pr >= 0.60
    gate_results.append(f"AUC-PR {auc_pr:.3f} >= 0.60: {'PASS' if gate1 else 'FAIL'}")
    if not gate1:
        passed = False

    # Gate 2: AUC-ROC threshold
    auc_roc = test_metrics.get("auc_roc", 0.0)
    gate2 = auc_roc >= 0.80
    gate_results.append(f"AUC-ROC {auc_roc:.3f} >= 0.80: {'PASS' if gate2 else 'FAIL'}")
    if not gate2:
        passed = False

    # Gate 3: False positive rate
    fpr = test_metrics.get("false_positive_rate", 1.0)
    gate3 = fpr <= 0.05
    gate_results.append(f"FPR {fpr:.3f} <= 0.05: {'PASS' if gate3 else 'FAIL'}")
    if not gate3:
        passed = False

    # Gate 4: F1 score
    f1 = test_metrics.get("f1", 0.0)
    # Demo threshold: 0.40 (permissive, because synthetic data produces lower scores).
    # Production threshold for RAN anomaly detection: 0.80 (see §8 evaluation table).
    gate4 = f1 >= 0.40
    gate_results.append(f"F1 {f1:.3f} >= 0.40 (demo; production: 0.80): {'PASS' if gate4 else 'FAIL'}")
    if not gate4:
        passed = False

    # Gate 5: Outperform best baseline on AUC-PR
    best_baseline_auc_pr = max(
        (v.get("auc_pr", 0.0) for v in baselines.values()), default=0.0
    )
    improvement_margin = auc_pr - best_baseline_auc_pr
    gate5 = improvement_margin >= 0.05
    gate_results.append(
        f"AUC-PR improvement over best baseline ({best_baseline_auc_pr:.3f}) "
        f"= {improvement_margin:.3f} >= 0.05: {'PASS' if gate5 else 'FAIL'}"
    )
    if not gate5:
        passed = False

    notes = "\n".join(gate_results)
    status = "PASSED" if passed else "FAILED"
    logger.info(f"  Governance gate: {status}")
    for result in gate_results:
        logger.info(f"    {result}")

    return passed, notes


# ============================================================================
# MODEL CARD GENERATION
# ============================================================================


def generate_model_card(
    tier_metrics: List[TierMetrics],
    test_metrics: Dict[str, Any],
    baselines: Dict[str, Dict[str, float]],
    split_metadata: SplitMetadata,
    feature_importance_df: pd.DataFrame,
    governance_passed: bool,
    governance_notes: str,
    feature_names: List[str],
) -> ModelCard:
    """
    Generate 3GPP TS 28.105 aligned model card for the model registry.

    The model card is the primary artifact for the governance gate. It
    captures all information needed for:
    - Model registry registration (TS 28.105 MLEntityRepository IOC)
    - Conflict detection (RCP write-set and KPI dependency-set)
    - EU AI Act Annex IV technical documentation
    - NOC runbook integration (rollback procedure, blast radius)

    In the reference architecture, this card is auto-generated by the
    training pipeline and reviewed by the squad lead before promotion.

    Top SHAP features are included to satisfy explainability requirements.
    """
    # Extract top 5 features for model card explainability section
    top_features = (
        feature_importance_df.head(5)["feature"].tolist()
        if not feature_importance_df.empty
        else feature_names[:5]
    )

    card = ModelCard(
        # Identity
        model_id=f"ran-anomaly-detector-v1.0.0-{datetime.utcnow().strftime('%Y%m%d')}",
        model_name="ran_cell_anomaly_detector",
        model_version="1.0.0",
        squad_owner="ran_optimization_squad",  # Kubernetes namespace
        use_case="RAN Cell-Level PM Counter Anomaly Detection",
        # Training provenance — maps to TS 28.105 MLTrainingReport IOC
        training_data_start=split_metadata.train_start,
        training_data_end=split_metadata.train_end,
        training_script="03_model_training.py",
        random_seed=RANDOM_SEED,
        feature_count=len(feature_names),
        training_sample_count=split_metadata.train_size,
        # Network impact — telco-specific fields
        affected_network_elements=["eNB", "gNB", "cell-sector"],
        # RCP write-set: parameters this model influences via A1 policy or automation
        # Empty for pure detection models; non-empty for control models (O-RAN xApps)
        rcp_write_set=[],  # Detection-only model; no direct RAN parameter writes
        # KPI dependency-set: PM counters this model reads (for conflict detection)
        kpi_dependency_set=[
            "DL.PRBUsageActive",     # 3GPP TS 28.550 PM counter
            "UL.PRBUsageActive",
            "RRC.ConnEstab.Succ",
            "DL.Throughput",
            "RSRP",
            "SINR",
        ],
        rollback_procedure=(
            "1. Set InferenceService replica count to 0 via kubectl. "
            "2. Re-enable rule-based threshold alerts in NOC dashboard. "
            "3. Notify on-call ML engineer via PagerDuty (runbook: RB-RAN-AD-001). "
            "4. Root cause analysis within 24h."
        ),
        blast_radius="cell",  # Impact scoped to individual cell-sector
        # SLO targets (monitored by Evidently + Prometheus in reference architecture)
        slo_inference_latency_p99_ms=200.0,  # KServe InferenceService must meet this
        slo_auc_pr_minimum=0.60,             # Governance gate re-triggers if drift below this
        slo_false_positive_rate_maximum=0.05,
        retraining_trigger="drift_wasserstein>0.30_OR_auc_pr<0.55_OR_weekly_scheduled",
        # EU AI Act: conservatively classified as high-risk (see model card docstring)
        eu_ai_act_risk_level="high",  # Detection model influencing NOC operator decisions
        data_sources=["3GPP PM Counters (O1/TS32.435)", "FM Alarms (VES/ONAP 7.x)"],
        governance_gate_notes=governance_notes,
        passes_governance_gate=governance_passed,
        tier_metrics=[asdict(m) for m in tier_metrics],
    )

    return card


# ============================================================================
# ARTIFACT PERSISTENCE
# ============================================================================


def save_all_artifacts(
    output_dir: Path,
    iforest_model: IsolationForest,
    ocsvm_pipeline: Pipeline,
    rf_model: RandomForestClassifier,
    autolabels_df: pd.DataFrame,
    thresholds: Dict[str, float],
    tier_metrics: List[TierMetrics],
    test_metrics: Dict[str, Any],
    baselines: Dict[str, Dict[str, float]],
    split_metadata: SplitMetadata,
    model_card: ModelCard,
    feature_names: List[str],
    lstm_artifacts: Optional[Dict[str, Any]] = None,
    data_dir: Optional[Path] = None,
) -> None:
    """
    Save all model artifacts to disk with checksums for artifact integrity.

    In the reference architecture, these artifacts are:
    1. Uploaded to the object store (S3/MinIO) under a versioned path
    2. Registered in MLflow Model Registry with artifact URI references
    3. Linked to the model card for governance audit trail

    The thresholds.json is particularly important: it's the serving-time
    configuration that the KServe InferenceService loads at startup.
    Changing thresholds without retraining (A/B threshold testing) is a
    supported workflow — only thresholds.json needs updating.

    See Coursebook Ch. 7: MLOps Core — Model Registry and Artifact Management.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tier 1 models
    joblib.dump(iforest_model, output_dir / "tier1_isolation_forest.joblib", compress=3)
    joblib.dump(ocsvm_pipeline, output_dir / "tier1_ocsvm.joblib", compress=3)
    logger.info("  Saved Tier 1 models (IsolationForest, OC-SVM)")

    # Auto-labels for audit trail and Tier 2 reproducibility
    if not autolabels_df.empty:
        autolabels_df.to_parquet(output_dir / "tier1_autolabels.parquet", index=False)
        logger.info(f"  Saved auto-labels ({len(autolabels_df):,} rows)")

    # Tier 2 Random Forest — wrap in Pipeline with scaler for deployment safety.
    # The bare classifier expects pre-scaled input; wrapping it ensures the
    # KServe serving path receives raw features from the online store and
    # produces correct predictions without a separate scaler loading step.
    scaler_path = data_dir / "feature_scaler.joblib" if data_dir else None
    if scaler_path and scaler_path.exists():
        loaded_scaler = joblib.load(scaler_path)
        from sklearn.pipeline import Pipeline as SkPipeline
        # NOTE: The scaler loaded here was fitted by 02_feature_engineering.py on
        # the same feature_cols that 03_model_training.py uses. If feature selection
        # in 03 becomes more restrictive than in 02 (e.g., excluding raw counters),
        # the scaler and model will have different n_features_in_, and the guard below
        # will fall back to saving a bare classifier. To avoid this, ensure that
        # select_feature_columns() in 02 and the feature set used in 03 stay aligned.
        # Guard: verify scaler dimensionality matches model
        scaler_n = getattr(loaded_scaler, "n_features_in_", None)
        model_n = getattr(rf_model, "n_features_in_", None)
        if scaler_n is not None and model_n is not None and scaler_n != model_n:
            logger.warning(
                "Scaler n_features_in_=%d != model n_features_in_=%d — "
                "saving bare classifier (scaler incompatible, likely standalone mode)",
                scaler_n, model_n,
            )
            joblib.dump(rf_model, output_dir / "tier2_random_forest.joblib", compress=3)
        else:
            pipeline_artifact = SkPipeline([("scaler", loaded_scaler), ("clf", rf_model)])
            joblib.dump(pipeline_artifact, output_dir / "tier2_random_forest.joblib", compress=3)
            # Write JSON sidecar — reliable metadata that survives any joblib/pickle version
            sidecar = {
                "artifact_type": "pipeline_with_scaler",
                "scaler_fitted_on": "02_feature_engineering_output",
                "expects_raw_features": True,
            }
            sidecar_path = output_dir / "tier2_random_forest_meta.json"
            with open(sidecar_path, "w") as f:
                json.dump(sidecar, f, indent=2)
            logger.info("  Saved Tier 2 Random Forest (Pipeline with embedded scaler + sidecar metadata)")
    else:
        # Standalone mode — no scaler available; save bare classifier with warning
        joblib.dump(rf_model, output_dir / "tier2_random_forest.joblib", compress=3)
        logger.warning(
            "  Saved Tier 2 Random Forest (bare classifier — no scaler embedded. "
            "Deploy with a separate scaler loading step or re-run full pipeline.)"
        )

    # Thresholds — loaded by serving layer at inference time
    with open(output_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    logger.info(f"  Saved thresholds.json: {thresholds}")

    # Training metrics — consumed by governance gate CI/CD check
    training_metrics = {
        "generated_at": datetime.utcnow().isoformat(),
        "random_seed": RANDOM_SEED,
        "hyperparameter_source": getattr(
            rf_model, "_telco_hyperparameter_source", "unknown"
        ),
        "split_metadata": asdict(split_metadata),
        "tier_metrics": [asdict(m) for m in tier_metrics],
        "test_metrics": test_metrics,
        "baselines": baselines,
        "feature_names": feature_names,
        "governance_gate": {
            "passed": model_card.passes_governance_gate,
            "notes": model_card.governance_gate_notes,
        },
    }
    if lstm_artifacts:
        training_metrics["lstm_metrics"] = lstm_artifacts.get("metrics", {})

    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2, default=str)
    logger.info("  Saved training_metrics.json")

    # Model card — primary governance artifact
    model_card_dict = asdict(model_card)
    with open(output_dir / "model_card.json", "w") as f:
        json.dump(model_card_dict, f, indent=2, default=str)
    logger.info("  Saved model_card.json")

    # Human-readable model card summary
    _print_model_card_summary(model_card, test_metrics, baselines)


def _print_model_card_summary(
    card: ModelCard,
    test_metrics: Dict[str, Any],
    baselines: Dict[str, Dict[str, float]],
) -> None:
    """Print a formatted model card summary to the log."""
    logger.info("=" * 70)
    logger.info("MODEL CARD SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Model ID:        {card.model_id}")
    logger.info(f"  Use Case:        {card.use_case}")
    logger.info(f"  Squad Owner:     {card.squad_owner}")
    logger.info(f"  Version:         {card.model_version}")
    logger.info(f"  Blast Radius:    {card.blast_radius}")
    logger.info(f"  EU AI Act Risk:  {card.eu_ai_act_risk_level}")
    logger.info(f"  RCP Write-Set:   {card.rcp_write_set or '[] (detection only)'}")
    logger.info(f"  Training Data:   {card.training_data_start} → {card.training_data_end}")
    logger.info(f"  Features:        {card.feature_count}")
    logger.info(f"  Train Samples:   {card.training_sample_count:,}")
    logger.info("  --- Test Performance ---")
    logger.info(f"  AUC-ROC:         {test_metrics.get('auc_roc', 0):.4f}  (SLO: N/A)")
    logger.info(
        f"  AUC-PR:          {test_metrics.get('auc_pr', 0):.4f}  "
        f"(SLO: >= {card.slo_auc_pr_minimum:.2f})"
    )
    logger.info(
        f"  F1:              {test_metrics.get('f1', 0):.4f}"
    )
    logger.info(
        f"  FPR:             {test_metrics.get('false_positive_rate', 0):.4f}  "
        f"(SLO: <= {card.slo_false_positive_rate_maximum:.2f})"
    )
    logger.info(f"  --- Baselines ---")
    for name, vals in baselines.items():
        logger.info(
            f"  {name:<30} AUC-PR={vals.get('auc_pr', 0):.3f} "
            f"F1={vals.get('f1', 0):.3f}"
        )
    logger.info(
        f"  --- Governance Gate: {'✓ PASSED' if card.passes_governance_gate else '✗ FAILED'} ---"
    )
    logger.info(f"  Retraining Trigger: {card.retraining_trigger}")
    logger.info("=" * 70)


def build_split_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> SplitMetadata:
    """Build split metadata struct from DataFrames."""
    def _ts(df: pd.DataFrame, side: str) -> str:
        if "timestamp" in df.columns:
            return str(df["timestamp"].min() if side == "start" else df["timestamp"].max())
        return "unknown"

    return SplitMetadata(
        train_start=_ts(train_df, "start"),
        train_end=_ts(train_df, "end"),
        val_start=_ts(val_df, "start"),
        val_end=_ts(val_df, "end"),
        test_start=_ts(test_df, "start"),
        test_end=_ts(test_df, "end"),
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
        train_anomaly_rate=float(y_train.mean()),
        val_anomaly_rate=float(y_val.mean()),
        test_anomaly_rate=float(y_test.mean()),
    )


# ============================================================================
# ORCHESTRATION
# ============================================================================


def run_training_pipeline(
    data_dir: Path = DATA_DIR,
    output_dir: Path = MODELS_DIR,
    skip_lstm: bool = False,
    full_search: bool = False,
    lstm_epochs: int = 30,
    tier1_contamination: float = 0.03,
    autolabel_policy: str = "union",
) -> Dict[str, Any]:
    """
    Orchestrate the full three-tier model training pipeline.

    This function represents the Kubeflow/Argo pipeline definition in Python form.
    In production, each step would be a separate pipeline component (container)
    with artifact passing via the platform's artifact store. The Python-level
    function decomposition here mirrors the pipeline component boundaries.

    Pipeline graph:
        load_data
            ↓
        tier1_iforest + tier1_ocsvm (parallel)
            ↓
        ensemble_autolabels
            ↓
        tier2_random_forest (uses autolabels)
            ↓
        tier3_lstm (optional, uses original labels for threshold calibration)
            ↓
        baselines (parallel with tier2)
            ↓
        governance_gate
            ↓
        save_artifacts

    See Coursebook Ch. 7: MLOps Core — Pipeline Orchestration.
    See Whitepaper Section 8: CODE-03 Kubeflow pipeline definition.
    """
    logger.info("=" * 70)
    logger.info("TELCO MLOPS REFERENCE ARCHITECTURE — Model Training Pipeline")
    logger.info("Three-Tier Cascade Anomaly Detection")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)

    pipeline_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # STEP 1: Load data                                                    #
    # ------------------------------------------------------------------ #
    logger.info("[STEP 1/8] Loading feature splits")
    train_df, val_df, test_df, metadata = load_feature_splits(data_dir)

    # Sort training data by timestamp for correct TimeSeriesSplit ordering.
    # 02_feature_engineering.py sorts by [cell_id, timestamp]; TimeSeriesSplit
    # requires global temporal ordering so each CV fold is a contiguous time block.
    if "timestamp" in train_df.columns:
        # Re-sort by timestamp for TimeSeriesSplit. Safe because features are
        # pre-computed by 02_feature_engineering.py — no rolling windows applied here.
        train_df = train_df.sort_values("timestamp").reset_index(drop=True)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = extract_arrays(
        train_df, val_df, test_df, metadata
    )

    split_metadata = build_split_metadata(
        train_df, val_df, test_df, y_train, y_val, y_test
    )

    logger.info(
        f"  Feature matrix shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
    )

    # ------------------------------------------------------------------ #
    # STEP 2: Tier 1a — Isolation Forest                                  #
    # ------------------------------------------------------------------ #
    logger.info("[STEP 2/8] Tier 1a: Isolation Forest")
    iforest_model, iforest_threshold = train_isolation_forest(
        X_train, contamination=tier1_contamination
    )

    # ------------------------------------------------------------------ #
    # STEP 3: Tier 1b — One-Class SVM                                     #
    # ------------------------------------------------------------------ #
    logger.info("[STEP 3/8] Tier 1b: One-Class SVM")
    ocsvm_pipeline, ocsvm_threshold, _ = train_ocsvm(
        X_train, nu=tier1_contamination
    )

    # ------------------------------------------------------------------ #
    # STEP 4: Ensemble auto-labeling                                       #
    # ------------------------------------------------------------------ #
    logger.info("[STEP 4/8] Generating Tier 1 ensemble auto-labels")
    autolabels_df = generate_ensemble_autolabels(
        iforest_model=iforest_model,
        ocsvm_pipeline=ocsvm_pipeline,
        X_train=X_train,
        train_df=train_df,
        iforest_threshold=iforest_threshold,
        ocsvm_threshold=ocsvm_threshold,
        agreement_policy=autolabel_policy,
    )

    # Use auto-labels as training signal for Tier 2 unless ground truth is available
    # Ground truth is preferred when available (e.g., from labeled anomaly campaigns)
    if "true_label" in autolabels_df.columns:
        # We have ground truth — use it for supervised training (best case)
        # but also report auto-label quality in model card
        y_train_for_rf = y_train
        logger.info("  Using ground truth labels for Tier 2 training (ground truth available)")
    else:
        # No ground truth — use ensemble auto-labels (typical production bootstrap)
        y_train_for_rf = autolabels_df["auto_label"].values
        logger.info("  Using ensemble auto-labels for Tier 2 training (no ground truth)")

    # ------------------------------------------------------------------ #
    # STEP 5: Tier 2 — Random Forest                                       #
    # ------------------------------------------------------------------ #
    logger.info("[STEP 5/8] Tier 2: Random Forest")
    rf_model, rf_threshold, rf_metrics = train_random_forest(
        X_train=X_train,
        y_train=y_train_for_rf,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        full_search=full_search,
    )

    # SHAP feature importance for governance gate model card
    feature_importance_df = compute_shap_importance(
        model=rf_model,
        X_val=X_val,
        feature_names=feature_names,
        output_dir=output_dir,
    )

    tier_metrics = [rf_metrics]

    # ------------------------------------------------------------------ #
    # STEP 6: Tier 3 — LSTM Autoencoder (optional)                         #
    # ------------------------------------------------------------------ #
    lstm_artifacts = None
    lstm_threshold = None
    if not skip_lstm and TORCH_AVAILABLE:
        logger.info("[STEP 6/8] Tier 3: LSTM Autoencoder")
        try:
            lstm_model, lstm_threshold, lstm_metrics, lstm_scaler = train_lstm_autoencoder(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                output_dir=output_dir,
                epochs=lstm_epochs,
            )
            tier_metrics.append(lstm_metrics)
            lstm_artifacts = {
                "metrics": asdict(lstm_metrics),
                "threshold": lstm_threshold,
            }
        except Exception as e:
            logger.warning(f"  LSTM training failed (non-fatal): {e}")
    else:
        reason = "skipped by user" if skip_lstm else "PyTorch not available"
        logger.info(f"[STEP 6/8] Tier 3: LSTM Autoencoder — SKIPPED ({reason})")

    # ------------------------------------------------------------------ #
    # STEP 7: Baselines                                                    #
    # ------------------------------------------------------------------ #
    logger.info("[STEP 7/8] Training baselines for governance comparison")
    baselines = train_baselines(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
    )

    # ------------------------------------------------------------------ #
    # STEP 8: Final test evaluation + governance gate                      #
    # ------------------------------------------------------------------ #
    logger.info("[STEP 8/8] Final test evaluation and governance gate")
    test_metrics = evaluate_on_test_set(
        tier2_model=rf_model,
        tier2_threshold=rf_threshold,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        output_dir=output_dir,
    )

    governance_passed, governance_notes = check_governance_gate(
        tier_metrics=tier_metrics,
        test_metrics=test_metrics,
        baselines=baselines,
        split_metadata=split_metadata,
    )

    # Update RF metrics with governance result
    rf_metrics.passes_governance_gate = governance_passed

    # ------------------------------------------------------------------ #
    # Generate model card                                                  #
    # ------------------------------------------------------------------ #
    model_card = generate_model_card(
        tier_metrics=tier_metrics,
        test_metrics=test_metrics,
        baselines=baselines,
        split_metadata=split_metadata,
        feature_importance_df=feature_importance_df,
        governance_passed=governance_passed,
        governance_notes=governance_notes,
        feature_names=feature_names,
    )

    # ------------------------------------------------------------------ #
    # Save artifacts                                                       #
    # ------------------------------------------------------------------ #
    thresholds = {
        "tier1_isolation_forest": iforest_threshold,
        "tier1_ocsvm": ocsvm_threshold,
        "tier2_random_forest": rf_threshold,
        "tier2_version": "1.0.0",
        "calibrated_on": "val_set",
        "calibration_metric": "f1",
    }
    if lstm_threshold is not None:
        thresholds["tier3_lstm_autoencoder"] = lstm_threshold

    save_all_artifacts(
        output_dir=output_dir,
        iforest_model=iforest_model,
        ocsvm_pipeline=ocsvm_pipeline,
        rf_model=rf_model,
        autolabels_df=autolabels_df,
        thresholds=thresholds,
        tier_metrics=tier_metrics,
        test_metrics=test_metrics,
        baselines=baselines,
        split_metadata=split_metadata,
        model_card=model_card,
        feature_names=feature_names,
        lstm_artifacts=lstm_artifacts,
        data_dir=data_dir,
    )

    total_elapsed = time.time() - pipeline_start
    logger.info(f"Pipeline complete in {total_elapsed:.1f}s")
    logger.info(
        f"Governance gate: {'✓ PASSED — model eligible for staging promotion' if governance_passed else '✗ FAILED — model blocked from staging'}"
    )

    return {
        "governance_passed": governance_passed,
        "test_metrics": test_metrics,
        "thresholds": thresholds,
        "model_card_id": model_card.model_id,
        "output_dir": str(output_dir),
        "elapsed_seconds": total_elapsed,
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Telco MLOps Reference Architecture — Model Training Pipeline\n"
            "Three-tier cascade: Isolation Forest + OC-SVM → Random Forest → LSTM Autoencoder\n"
            "\n"
            "Requires outputs from:\n"
            "  01_synthetic_data.py (generates data/)\n"
            "  02_feature_engineering.py (generates data/features_train.parquet, etc.)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Root data directory (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"Model output directory (default: {MODELS_DIR})",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip LSTM Autoencoder tier (faster, for development)",
    )
    parser.add_argument(
        "--full-search",
        action="store_true",
        help="Run grid search for Random Forest hyperparameters (slow, for CI)",
    )
    parser.add_argument(
        "--lstm-epochs",
        type=int,
        default=30,
        help="LSTM training epochs (default: 30)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.03,
        help="Expected anomaly rate for Tier 1 models (default: 0.03)",
    )
    parser.add_argument(
        "--autolabel-policy",
        choices=["union", "intersection"],
        default="union",
        help="Tier 1 ensemble auto-labeling policy (default: union)",
    )
    return parser.parse_args()


# ============================================================================
# ENTRYPOINT
# ============================================================================


if __name__ == "__main__":
    args = parse_args()

    result = run_training_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        skip_lstm=args.skip_lstm,
        full_search=args.full_search,
        lstm_epochs=args.lstm_epochs,
        tier1_contamination=args.contamination,
        autolabel_policy=args.autolabel_policy,
    )

    # Exit code communicates governance gate result to CI/CD pipeline
    # KFP/Argo condition nodes read this exit code to decide whether to
    # continue to the 'promote to staging' step or halt the pipeline
    sys.exit(0 if result["governance_passed"] else 1)
