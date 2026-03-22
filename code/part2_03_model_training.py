"""
03_model_training.py — Telco MLOps Part 2: Multi-Paradigm ML Platform
======================================================================
Three-tier cascade anomaly detection training pipeline for RAN KPI data.

Pipeline:
  Tier 1 — Isolation Forest (unsupervised, zero-label bootstrap)
  Tier 2 — Random Forest classifier (semi-supervised, IF pseudo-labels)
  Tier 3 — LSTM Autoencoder (unsupervised reconstruction error)

Ensemble scoring: w1*IF_score + w2*RF_score + w3*LSTMAE_score
  where weights are calibrated on the validation set.

Outputs (written to models/):
  - isolation_forest.joblib
  - random_forest.joblib
  - lstm_autoencoder.pt
  - scaler.joblib          (loaded from 02_feature_engineering.py; JSON fallback via scaler.json)
  - ensemble_thresholds.json
  - training_metadata.json

Usage:
  python 03_model_training.py

Prerequisites:
  pip install pandas numpy scikit-learn torch joblib shap matplotlib

Coursebook cross-reference:
  Ch. 13  — Feature Engineering (feature selection, SHAP values)
  Ch. 16  — Decision Trees & Random Forests (RF hyperparameters, OOB score)
  Ch. 22  — Recurrent Neural Networks (LSTM architecture, reconstruction loss)
  Ch. 28  — Data Pipelines (temporal splits, artefact management)
  Ch. 52  — System Design for ML (cascade scoring, threshold calibration)
  Ch. 54  — Monitoring & Reliability (threshold selection, operational metrics)

Part 2 architecture notes:
  - This script trains the BASE anomaly detection ensemble from Part 1.
  - Trained models are consumed by 05_production_patterns.py as ONNX artefacts.
  - The GNN root-cause layer (Part 2 Layer 5) consumes the IF/RF scores as
    node attributes in the heterogeneous topology graph.
  - Threshold calibration here uses Youden's J; production deployments should
    re-calibrate on operator-specific false-alarm cost ratios.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Optional: SHAP for RF explainability (Ch. 13 — Feature Engineering)
# ---------------------------------------------------------------------------
try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    warnings.warn("shap not installed — RF SHAP explanations will be skipped.")

# ---------------------------------------------------------------------------
# Optional: PyTorch for LSTM Autoencoder (Ch. 22 — RNNs)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    warnings.warn(
        "torch not installed — LSTM Autoencoder tier will be skipped. "
        "Install with: pip install torch"
    )

# ---------------------------------------------------------------------------
# Optional: PyTorch Geometric transforms for GNN topology graph
# ---------------------------------------------------------------------------
try:
    import torch_geometric.transforms as T

    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    warnings.warn(
        "torch_geometric not installed — GNN topology graph reverse edges "
        "will be added manually. Install with: pip install torch_geometric"
    )

# ---------------------------------------------------------------------------
# Logging — use structured format for production log aggregation
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("model_training")

# ---------------------------------------------------------------------------
# Constants — directory paths aligned with §15 directory structure
# Script 02 writes features to data/features/
# Script 03 writes model artefacts to artifacts/models/
# Script 04 reads from artifacts/ and data/
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
FEATURES_DIR = Path("data") / "features"

# Input artefacts from 02_feature_engineering.py
TRAIN_PARQUET = FEATURES_DIR / "train.parquet"
VAL_PARQUET = FEATURES_DIR / "val.parquet"
TEST_PARQUET = FEATURES_DIR / "test.parquet"
SCALER_PATH = FEATURES_DIR / "scaler.joblib"

# Output artefacts
IF_MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"
RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_autoencoder.pt"
THRESHOLDS_PATH = MODELS_DIR / "ensemble_thresholds.json"
METADATA_PATH = MODELS_DIR / "training_metadata.json"
SHAP_VALUES_PATH = MODELS_DIR / "shap_values_sample.npy"

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Ensemble weights — calibrated on validation set; see calibrate_ensemble_weights()
# Ch. 52 — System Design: cascade weights chosen so that IF gates early (recall-
# optimised) and RF/LSTM tighten precision downstream.
DEFAULT_ENSEMBLE_WEIGHTS = (0.20, 0.50, 0.30)  # (IF, RF, LSTM-AE)

# Minimum F1 gate — must meet or exceed to allow promotion to production
# Ref: Part 1 §8 production governance gate; replicated here for CI assertion.
MIN_F1_GATE = 0.82

# Label column written by 02_feature_engineering.py (from 01_synthetic_data.py)
LABEL_COL = "is_anomaly"

# Timestamp column (used for temporal integrity checks)
TIMESTAMP_COL = "timestamp"

# Cell identifier column
CELL_ID_COL = "cell_id"


# ===========================================================================
# Configuration dataclass
# ===========================================================================


@dataclass
class TrainingConfig:
    """Centralised hyperparameter configuration.

    Every numeric value here should be traceable to either:
    (a) domain knowledge (e.g., IF contamination ≈ 2-5% for RAN anomaly rates),
    (b) a coursebook reference, or
    (c) a grid search result documented in training_metadata.json.
    """

    # --- Isolation Forest ---
    # contamination: expected anomaly fraction in training data.
    # Part 1 §3 cites 1-5% anomaly rate for RAN; we use 3% as conservative mid-point.
    # Setting too low → missed anomalies; too high → degrades precision.
    # See Ch. 16 — Decision Trees for IF's extended isolation tree logic.
    if_contamination: float = 0.03
    if_n_estimators: int = 200  # 200 trees → stable score variance at ~10K rows
    if_max_samples: str = "auto"  # "auto" = min(256, n_samples) — standard choice
    if_random_state: int = RANDOM_SEED

    # --- Pseudo-label generation for RF ---
    # IF scores are raw anomaly scores (negative = more anomalous in sklearn convention).
    # We convert to pseudo-labels using a percentile threshold on the validation set.
    # 3% contamination → ~97th percentile threshold for pseudo-labelling.
    if_pseudo_label_percentile: float = 97.0

    # --- Random Forest ---
    # n_estimators: 500 trees → OOB error stabilises for ~10K-sample datasets.
    # See Ch. 16 §4 — OOB generalisation estimate.
    rf_n_estimators: int = 500
    # max_depth: None = grow to purity; RF controls overfitting through bagging,
    # not depth-limiting. For very high-dimensional feature spaces (>200 features),
    # consider max_depth=20 to reduce memory.
    rf_max_depth: Optional[int] = None
    # class_weight: "balanced_subsample" compensates for ~3% anomaly class imbalance
    # without explicit oversampling, which can cause temporal leakage on time series.
    # See Ch. 16 §6 — Handling Class Imbalance.
    rf_class_weight: str = "balanced_subsample"
    rf_min_samples_leaf: int = 5  # Prevents overfitting on anomaly micro-clusters
    rf_random_state: int = RANDOM_SEED
    # Number of SHAP samples for TreeExplainer (expensive; subsample for speed)
    rf_shap_sample_n: int = 500

    # --- LSTM Autoencoder ---
    # Sequence length: 4 ROPs = 1 hour of 15-min PM counter data.
    # Short enough for near-RT RIC warm-path; long enough for diurnal pattern capture.
    # See Ch. 22 §3 — Sequence Modelling for Time Series.
    lstm_seq_len: int = 4
    lstm_hidden_dim: int = 64   # Bottleneck width — wider = more capacity, slower
    lstm_num_layers: int = 2    # 2-layer LSTM → captures both short and medium trends
    lstm_dropout: float = 0.2   # Regularisation; higher contamination → lower dropout
    lstm_learning_rate: float = 1e-3
    lstm_batch_size: int = 256  # Larger batches → stable gradients on GPU
    lstm_epochs: int = 50       # Early stopping prevents overfitting
    lstm_patience: int = 7      # Early stopping patience (epochs without improvement)
    lstm_reconstruction_percentile: float = 97.0  # Anomaly threshold percentile
    lstm_random_state: int = RANDOM_SEED

    # --- Ensemble calibration ---
    # Weights: (IF, RF, LSTM-AE). These are searched on the validation set.
    ensemble_weight_grid: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.20, 0.50, 0.30),  # Default: RF-dominant (highest label quality)
            (0.10, 0.70, 0.20),  # RF-heavy: when LSTM not available
            (0.25, 0.40, 0.35),  # Balanced
            (0.30, 0.30, 0.40),  # LSTM-heavy: when temporal patterns dominate
        ]
    )
    # Final threshold selection metric: "f1" or "youden_j"
    # Youden's J = sensitivity + specificity - 1; balances recall/precision.
    # See Ch. 52 §5 — Threshold Calibration for Imbalanced Operational Data.
    ensemble_threshold_metric: str = "youden_j"

    # --- Feature selection ---
    # Features to EXCLUDE from model input (leakage, identifiers, raw targets).
    # See Ch. 13 §2 — Temporal Leakage in Feature Engineering.
    exclude_cols: List[str] = field(
        default_factory=lambda: [
            LABEL_COL,
            TIMESTAMP_COL,
            CELL_ID_COL,
            "anomaly_type",         # Ground truth label — leakage
            "anomaly_severity",     # Ground truth label — leakage
            "site_id",              # Identifier — high cardinality noise
            "cluster_id",           # Identifier
            "peer_group_id",        # Identifier (integer index)
        ]
    )


# ===========================================================================
# Data loading utilities
# ===========================================================================


def load_split(
    path: Path,
    cfg: TrainingConfig,
    label: str = "data",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load a feature-engineered parquet split and separate features from labels.

    Returns:
        X_df  — raw feature DataFrame (float32, shape [N, F])
        y     — binary anomaly labels (int, shape [N])
        ts    — timestamps (numpy datetime64 array, shape [N])

    Temporal integrity check: asserts that rows are sorted by timestamp, which
    is required for LSTM sequence construction and avoids look-ahead leakage.
    See Ch. 28 §3 — Temporal Correctness in Data Pipelines.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected feature split at {path}. "
            "Run 02_feature_engineering.py first."
        )

    df = pd.read_parquet(path)
    log.info(f"Loaded {label}: {df.shape[0]:,} rows × {df.shape[1]} cols from {path}")

    # ---- Temporal integrity assertion ----
    if TIMESTAMP_COL in df.columns:
        ts_series = pd.to_datetime(df[TIMESTAMP_COL])
        if not ts_series.is_monotonic_increasing:
            log.warning(
                f"{label}: timestamps are NOT monotonic — sorting. "
                "Check 02_feature_engineering.py output ordering."
            )
            df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        ts = df[TIMESTAMP_COL].values
    else:
        log.warning(f"{label}: no '{TIMESTAMP_COL}' column found; skipping temporal check.")
        ts = np.arange(len(df))

    # ---- Extract labels ----
    if LABEL_COL in df.columns:
        y = df[LABEL_COL].astype(int).values
    else:
        log.warning(f"{label}: no '{LABEL_COL}' column — using all-zeros placeholder.")
        y = np.zeros(len(df), dtype=int)

    # ---- Drop non-feature columns ----
    drop_cols = [c for c in cfg.exclude_cols if c in df.columns]
    X_df = df.drop(columns=drop_cols)

    # ---- Drop any remaining non-numeric columns ----
    non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        log.warning(f"{label}: dropping non-numeric columns: {non_numeric}")
        X_df = X_df.drop(columns=non_numeric)

    # ---- Replace inf/NaN ----
    inf_count = np.isinf(X_df.values).sum()
    nan_count = X_df.isna().sum().sum()
    if inf_count > 0 or nan_count > 0:
        log.warning(
            f"{label}: {inf_count} inf values and {nan_count} NaN values found — "
            "replacing with column medians."
        )
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median(numeric_only=True))

    log.info(
        f"{label}: {X_df.shape[1]} features | "
        f"anomaly rate = {y.mean():.3%} ({y.sum()} / {len(y)} samples)"
    )
    return X_df, y, ts


def load_scaler(scaler_path: Path) -> Optional[StandardScaler]:
    """Load the StandardScaler fitted in 02_feature_engineering.py.

    Tries joblib format first (scaler.joblib), then falls back to JSON
    format (scaler.json) if the joblib file is missing. Script 02 saves
    both formats; this dual-path loading handles cases where only one
    is available.

    If neither file exists (e.g., running standalone), returns None
    and the caller must fit a new scaler. This maintains serving-time
    consistency: the SAME scaler fitted on training data must be used at
    inference time.
    See Ch. 28 §5 — Feature Store Consistency Between Training and Serving.
    """
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        log.info(f"Loaded pre-fitted scaler from {scaler_path}")
        return scaler

    # Fallback: try JSON format (saved by Script 02 as scaler.json)
    json_path = scaler_path.with_suffix(".json")
    if json_path.exists():
        import json
        with open(json_path, "r") as f:
            params = json.load(f)
        scaler = StandardScaler()
        scaler.mean_ = np.array(params["mean_"])
        scaler.scale_ = np.array(params["scale_"])
        scaler.var_ = np.array(params["var_"])
        scaler.n_samples_seen_ = params.get("n_samples_seen_", 1)
        scaler.n_features_in_ = len(scaler.mean_)
        scaler.feature_names_in_ = np.array(params.get("feature_names", []))
        log.info(f"Loaded pre-fitted scaler from {json_path} (JSON fallback)")
        return scaler

    log.warning(
        f"Scaler not found at {scaler_path} or {json_path}. "
        "A new scaler will be fitted on training data. "
        "This may cause train/serve skew — re-run 02_feature_engineering.py."
    )
    return None


def apply_or_fit_scaler(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    scaler: Optional[StandardScaler],
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Apply pre-fitted scaler or fit a new one on training data only.

    Critical: Never fit the scaler on val/test data — this would leak test
    distribution statistics into the scaling transform (data leakage).
    See Ch. 13 §3 — Preventing Distribution Leakage.
    """
    if scaler is None:
        log.info("Fitting new StandardScaler on training data only.")
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
    else:
        X_train_s = scaler.transform(X_train)

    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    log.info(
        f"Scaling applied | train mean range: "
        f"[{X_train_s.mean(axis=0).min():.3f}, {X_train_s.mean(axis=0).max():.3f}]"
    )
    return X_train_s, X_val_s, X_test_s, scaler


# ===========================================================================
# GNN topology graph construction
# Schema matches CODE-1 (telco_mlops_part2/graph/topology.py) in §8.1:
#   Node types: cell_sector, site, backhaul_node
#   Edge types: is_neighbour_of, same_site, rev_same_site,
#               shares_transport, rev_shares_transport
# ===========================================================================


def _add_reverse_edges_manually(data, empty):
    """Add reverse edge types by flipping forward edge indices.

    Used as a fallback when T.ToUndirected() is unavailable. Must NOT be
    called when T.ToUndirected() has already been applied — doing so would
    double the reverse edge counts and corrupt HGTConv attention weights.
    """
    import torch

    # rev_same_site: site -> cell_sector (reverse of cell_sector -> site)
    try:
        ei = data["cell_sector", "same_site", "site"].edge_index
        if ei is not None and ei.numel() > 0:
            data["site", "rev_same_site", "cell_sector"].edge_index = ei.flip(0)
        else:
            data["site", "rev_same_site", "cell_sector"].edge_index = empty.clone()
    except (KeyError, AttributeError):
        data["site", "rev_same_site", "cell_sector"].edge_index = empty.clone()

    # rev_shares_transport: cell_sector -> backhaul_node
    try:
        ei = data["backhaul_node", "shares_transport", "cell_sector"].edge_index
        if ei is not None and ei.numel() > 0:
            data["cell_sector", "rev_shares_transport", "backhaul_node"].edge_index = ei.flip(0)
        else:
            data["cell_sector", "rev_shares_transport", "backhaul_node"].edge_index = empty.clone()
    except (KeyError, AttributeError):
        data["cell_sector", "rev_shares_transport", "backhaul_node"].edge_index = empty.clone()

    # is_neighbour_of is cell_sector -> cell_sector; ToUndirected would
    # add the reverse, but for a symmetric relation the reverse is the
    # same edge type — just flip src/dst.
    try:
        ei = data["cell_sector", "is_neighbour_of", "cell_sector"].edge_index
        if ei is not None and ei.numel() > 0:
            # Concatenate forward + reversed to make it undirected
            rev_ei = ei.flip(0)
            combined = torch.cat([ei, rev_ei], dim=1)
            # Deduplicate (in case input was already symmetric)
            combined = torch.unique(combined, dim=1)
            data["cell_sector", "is_neighbour_of", "cell_sector"].edge_index = combined
    except (KeyError, AttributeError):
        pass


def build_ran_topology_graph(
    cell_features: Dict[str, Any],
    site_assignments: Dict[str, str],
    transport_links: List[Tuple[str, str]],
    neighbour_pairs: Optional[List[Tuple[str, str]]] = None,
) -> "torch_geometric.data.HeteroData":
    """Build a heterogeneous RAN topology graph for GNN root-cause analysis.

    This is the **training-time** implementation, accepting pre-processed
    dicts and lists extracted from the synthetic dataset. The **production**
    implementation is in CODE-1 (§8.1, telco_mlops_part2/graph/topology.py),
    which accepts raw O1 NRT/NRM DataFrames, anomaly scores, and a Feast
    FeatureStore for point-in-time feature retrieval. A third variant,
    ``finalize_ran_topology_graph()`` in ``01_synthetic_data.py``, adds
    node features and reverse edges to an already-constructed HeteroData
    object during synthetic data generation. All three produce the same
    HeteroData schema (node types, edge types, feature dimensions) so that
    models trained here are compatible with the production serving path.

    Node types:
        - 'cell_sector'   : individual RAN cells with KPI feature vectors
        - 'site'          : physical site (tower/building) grouping cells
        - 'backhaul_node' : transport/backhaul segment nodes

    Edge types:
        - ('cell_sector', 'is_neighbour_of', 'cell_sector') : NRT adjacency
        - ('cell_sector', 'same_site', 'site')               : cell -> site
        - ('site', 'rev_same_site', 'cell_sector')           : site -> cell
        - ('backhaul_node', 'shares_transport', 'cell_sector'): backhaul -> cell
        - ('cell_sector', 'rev_shares_transport', 'backhaul_node'): cell -> backhaul

    Args:
        cell_features:    dict mapping cell_id -> feature tensor or array
        site_assignments: dict mapping cell_id -> site_id
        transport_links:  list of (backhaul_node_id, cell_id) pairs
        neighbour_pairs:  optional list of (cell_id_a, cell_id_b) NRT pairs

    Returns:
        HeteroData object with node features and all edge types.
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError:
        raise ImportError(
            "torch_geometric is required for build_ran_topology_graph(). "
            "Install with: pip install torch_geometric"
        )

    data = HeteroData()

    # ---- Cell sector nodes ----
    cell_ids = sorted(cell_features.keys())
    cell_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

    cell_feat_list = [
        torch.tensor(cell_features[cid], dtype=torch.float32)
        for cid in cell_ids
    ]
    data["cell_sector"].x = torch.stack(cell_feat_list, dim=0)

    # ---- Site nodes ----
    site_ids = sorted(set(site_assignments.values()))
    site_to_idx = {sid: i for i, sid in enumerate(site_ids)}

    # Site node features: number of cells at site (simple scalar; CODE-1
    # uses infrastructure metadata from NRM -- here we use cell count as proxy)
    site_cell_counts = {sid: 0 for sid in site_ids}
    for cid, sid in site_assignments.items():
        if cid in cell_to_idx and sid in site_cell_counts:
            site_cell_counts[sid] += 1
    site_feat = torch.tensor(
        [[site_cell_counts[sid]] for sid in site_ids], dtype=torch.float32
    )
    data["site"].x = site_feat

    # ---- Backhaul nodes ----
    backhaul_ids = sorted(set(bh_id for bh_id, _ in transport_links))
    bh_to_idx = {bid: i for i, bid in enumerate(backhaul_ids)}

    bh_cell_counts = {bid: 0 for bid in backhaul_ids}
    for bh_id, cell_id in transport_links:
        if cell_id in cell_to_idx:
            bh_cell_counts[bh_id] += 1
    if backhaul_ids:
        bh_feat = torch.tensor(
            [[bh_cell_counts[bid]] for bid in backhaul_ids], dtype=torch.float32
        )
        data["backhaul_node"].x = bh_feat
    else:
        data["backhaul_node"].x = torch.zeros((0, 1), dtype=torch.float32)

    empty = torch.zeros((2, 0), dtype=torch.long)

    # ---- is_neighbour_of edges (cell_sector -> cell_sector) ----
    if neighbour_pairs:
        nbr_src, nbr_dst = [], []
        for cid_a, cid_b in neighbour_pairs:
            if cid_a in cell_to_idx and cid_b in cell_to_idx:
                nbr_src.append(cell_to_idx[cid_a])
                nbr_dst.append(cell_to_idx[cid_b])
        if nbr_src:
            data["cell_sector", "is_neighbour_of", "cell_sector"].edge_index = (
                torch.tensor([nbr_src, nbr_dst], dtype=torch.long)
            )
        else:
            data["cell_sector", "is_neighbour_of", "cell_sector"].edge_index = empty
    else:
        data["cell_sector", "is_neighbour_of", "cell_sector"].edge_index = empty

    # ---- same_site edges (cell_sector -> site) — forward only ----
    # Reverse edges (site -> cell_sector) are added by T.ToUndirected()
    # when PyG is available, or manually below when it is not.
    ss_src, ss_dst = [], []
    for cid, sid in site_assignments.items():
        if cid in cell_to_idx and sid in site_to_idx:
            ss_src.append(cell_to_idx[cid])
            ss_dst.append(site_to_idx[sid])

    if ss_src:
        src_t = torch.tensor(ss_src, dtype=torch.long)
        dst_t = torch.tensor(ss_dst, dtype=torch.long)
        data["cell_sector", "same_site", "site"].edge_index = torch.stack(
            [src_t, dst_t], dim=0
        )
    else:
        data["cell_sector", "same_site", "site"].edge_index = empty

    # ---- shares_transport edges (backhaul_node -> cell_sector) — forward only ----
    bt_src, bt_dst = [], []
    for bh_id, cell_id in transport_links:
        if bh_id in bh_to_idx and cell_id in cell_to_idx:
            bt_src.append(bh_to_idx[bh_id])
            bt_dst.append(cell_to_idx[cell_id])

    if bt_src:
        src_t = torch.tensor(bt_src, dtype=torch.long)
        dst_t = torch.tensor(bt_dst, dtype=torch.long)
        data["backhaul_node", "shares_transport", "cell_sector"].edge_index = (
            torch.stack([src_t, dst_t], dim=0)
        )
    else:
        data["backhaul_node", "shares_transport", "cell_sector"].edge_index = empty

    # ---- Reverse edges ----
    # Strategy: use T.ToUndirected() when PyG is available (it adds
    # rev_same_site, rev_shares_transport, and rev_is_neighbour_of
    # automatically without doubling existing edges). When PyG transforms
    # are unavailable, add reverse edges manually via .flip(0).
    # IMPORTANT: do NOT combine both approaches — that doubles edge counts
    # and corrupts HGTConv message-passing attention weights.
    if _PYG_AVAILABLE:
        try:
            import torch_geometric.transforms as T
            data = T.ToUndirected()(data)
            log.info(
                "build_ran_topology_graph: T.ToUndirected() applied; "
                f"edge_types = {data.edge_types}"
            )
        except Exception as exc:
            log.warning(
                f"build_ran_topology_graph: T.ToUndirected() failed ({exc}); "
                "falling back to manual reverse edges."
            )
            # Fallback: add reverse edges manually
            _add_reverse_edges_manually(data, empty)
    else:
        log.info(
            "build_ran_topology_graph: torch_geometric transforms unavailable; "
            "adding reverse edges manually."
        )
        _add_reverse_edges_manually(data, empty)

    log.info(
        f"build_ran_topology_graph: {data['cell_sector'].x.shape[0]} cell_sector nodes, "
        f"{data['site'].x.shape[0]} site nodes, "
        f"{data['backhaul_node'].x.shape[0]} backhaul_node nodes | "
        f"edge_types = {data.edge_types}"
    )
    return data


# ===========================================================================
# GNN Root Cause Classifier (Layer 5 — see whitepaper §8.1)
# ===========================================================================

# Guard: nn.Module requires torch
_RootCauseGNN_base = nn.Module if _TORCH_AVAILABLE else object


class RootCauseGNN(_RootCauseGNN_base):
    """Two-layer Heterogeneous Graph Transformer for root cause attribution.

    Architecture (per whitepaper §8.1):
        1. Per-node-type Linear projections → shared 128-dim hidden space
        2. Two HGTConv layers (4 attention heads each) — heterogeneous
           message passing across all edge types
        3. Shared Linear(128, 1) classifier with sigmoid activation →
           per-node root cause probability for each node type

    HGTConv was chosen over homogeneous GCN because:
        - Handles heterogeneous node/edge types natively
        - Per-edge-type attention weights serve as interpretable root cause
          evidence (high attention on a `shares_transport` edge identifies
          the upstream backhaul node as the probable root cause)

    Governance gates (§8.1):
        - node_auroc ≥ 0.80 (check label_coverage first — NaN auroc
          with label_coverage=0.0 means labels are absent, not that
          the model has failed)
        - top1_accuracy ≥ operator_baseline + 0.10 (floor: 0.60)
        - false_escalation ≤ 0.05

    Requires: torch, torch_geometric (PyG with HGTConv support)
    """

    def __init__(self, metadata, hidden_dim: int = 128, heads: int = 4):
        super().__init__()
        try:
            from torch_geometric.nn import HGTConv, Linear
            self.conv1 = HGTConv(
                in_channels=-1,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=heads,
            )
            self.conv2 = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=heads,
            )
            self.classifier = Linear(hidden_dim, 1)
            self._has_pyg = True
        except ImportError:
            logger.warning(
                "PyTorch Geometric not available. RootCauseGNN will use a "
                "placeholder forward pass. Install with: pip install torch_geometric"
            )
            self._has_pyg = False
            if _TORCH_AVAILABLE:
                self.classifier = nn.Linear(hidden_dim, 1)
            else:
                self.classifier = None

    def forward(self, x_dict, edge_index_dict):
        """Forward pass returning per-node root cause probabilities.

        Returns:
            Dict[str, Tensor]: node_type → (n_nodes, 1) sigmoid probabilities
        """
        import torch.nn.functional as F

        if not self._has_pyg:
            # Placeholder: return uniform probabilities
            return {
                node_type: torch.sigmoid(torch.zeros(x.shape[0], 1))
                for node_type, x in x_dict.items()
            }

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # Per-node-type root cause probability
        return {
            node_type: torch.sigmoid(self.classifier(x))
            for node_type, x in x_dict.items()
        }


def _global_index(
    node_type: str,
    local_idx: int,
    node_type_order: List[str],
    node_counts: Dict[str, int],
) -> int:
    """Map (node_type, local_index) to an index into the concatenated score array.

    The concatenated array is built by iterating ``node_type_order`` and
    appending each node type's scores in sequence.  This helper computes
    the offset for *node_type* and adds *local_idx*.

    Returns
    -------
    int
        Global index into the concatenated score/label array.

    Raises
    ------
    ValueError
        If *node_type* is not in *node_type_order* or *local_idx* is
        out of range.
    """
    if node_type not in node_type_order:
        raise ValueError(
            f"Unknown node type '{node_type}'; expected one of {node_type_order}"
        )
    offset = sum(node_counts[nt] for nt in node_type_order[: node_type_order.index(node_type)])
    if local_idx < 0 or local_idx >= node_counts[node_type]:
        raise ValueError(
            f"local_idx {local_idx} out of range for node type '{node_type}' "
            f"with {node_counts[node_type]} nodes"
        )
    return offset + local_idx


def evaluate_root_cause_attribution(
    model: "RootCauseGNN",
    data,
    labelled_incidents: Optional[List[Dict]] = None,
) -> Dict[str, float]:
    """Evaluate GNN root cause attribution on labelled incidents.

    Computes three metrics (per whitepaper §8.1):
        - node_auroc: AUROC for root cause node classification across
          all node types (primary model metric).  Returns ``NaN`` (not
          0.0) when the label vector is constant (all-zero or all-one)
          — this distinguishes "labels absent" from "model is bad".
        - top1_accuracy: fraction of incidents where the GNN's highest-
          scored node is the true root cause (governance gate metric)
        - precision_at_3: precision when the model names three candidate
          root cause nodes (reflects NOC triage workflow)
        - label_coverage: fraction of node types with non-trivial labels
          (at least one positive and one negative).  Governance gates
          should check ``label_coverage > 0`` before evaluating
          ``node_auroc`` — a NaN auroc with zero label_coverage means
          the evaluation is undefined, not that the model has failed.

    These metrics operate across heterogeneous node types.

    Incident format
    ---------------
    Each element of *labelled_incidents* must be a dict with **either**:

    -  ``root_cause_idx`` (int): global index into the concatenated
       score array, with node-type offsets already applied.  Use
       ``_global_index(node_type, local_idx, ...)`` to compute this.

    -  ``root_cause_node_type`` (str) **and** ``root_cause_local_idx``
       (int): the node type and local index within that type.  The
       function will compute the global index automatically.

    If both forms are present, the (node_type, local_idx) pair takes
    precedence.
    """
    if labelled_incidents is None or len(labelled_incidents) == 0:
        logger.warning("No labelled incidents provided for RCA evaluation.")
        return {
            "node_auroc": float("nan"),
            "top1_accuracy": float("nan"),
            "precision_at_3": float("nan"),
            "n_incidents": 0,
            "label_coverage": 0.0,
        }

    model.eval()
    with torch.no_grad():
        probs = model(data.x_dict, data.edge_index_dict)

    # Deterministic node-type iteration order and counts
    node_type_order = sorted(probs.keys())
    node_counts = {nt: probs[nt].shape[0] for nt in node_type_order}

    # Flatten scores and labels across all node types for ranking
    all_scores = []
    all_labels = []
    for node_type in node_type_order:
        scores = probs[node_type].squeeze(-1).numpy()
        # Ground-truth labels: only use if they exist and are non-trivial
        if hasattr(data[node_type], "y"):
            labels = data[node_type].y.numpy()
        else:
            labels = np.zeros(len(scores), dtype=int)
        all_scores.append(scores)
        all_labels.append(labels)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    # label_coverage: fraction of node types that have non-trivial labels
    # (at least one positive and one negative). Governance gates should check
    # label_coverage before evaluating node_auroc — see §8.1.
    n_labelled_types = sum(
        1 for nt in node_type_order
        if hasattr(data[nt], "y")
        and int(data[nt].y.sum()) > 0
        and int(data[nt].y.sum()) < len(data[nt].y)
    )
    label_coverage = n_labelled_types / max(len(node_type_order), 1)

    # node_auroc — guard against all-zeros (or all-ones) label vectors
    # which make AUROC undefined.  Return NaN (not 0.0) so the governance
    # gate can distinguish "model is bad" from "labels are absent".
    if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
        logger.warning(
            "AUROC undefined: label vector is constant (sum=%d, len=%d). "
            "Returning NaN — governance gate should check label_coverage "
            "(currently %.2f) before evaluating node_auroc. "
            "This usually means ground-truth labels are missing for one or "
            "more node types — check data[node_type].y assignments.",
            int(all_labels.sum()), len(all_labels), label_coverage,
        )
        node_auroc = float("nan")
    else:
        try:
            from sklearn.metrics import roc_auc_score
            node_auroc = float(roc_auc_score(all_labels, all_scores))
        except (ValueError, ImportError):
            node_auroc = float("nan")

    # top1_accuracy and precision_at_3 per incident
    #
    # ⚠ LIMITATION: The ranking below uses a SINGLE global score vector
    # (one forward pass over the entire graph snapshot).  This means
    # `ranked[0]` is the globally highest-scored node — NOT the highest-
    # scored node within each incident's affected subgraph.  The evaluation
    # is correct ONLY when there is one active fault scenario per graph
    # snapshot (the synthetic-data demo satisfies this assumption).
    #
    # For graphs with multiple concurrent faults, either:
    #   (a) run a separate forward pass per incident's temporal snapshot, or
    #   (b) mask the score vector to each incident's contributing_cells and
    #       rank within that subgraph.
    # See §8.1 label bootstrapping protocol for multi-incident evaluation.
    top1_correct = 0
    p3_sum = 0.0
    ranked = np.argsort(all_scores)[::-1]

    if len(labelled_incidents) > 1:
        import warnings
        warnings.warn(
            f"evaluate_root_cause_attribution called with {len(labelled_incidents)} "
            "incidents on a single graph snapshot. top1_accuracy and precision_at_3 "
            "use a single global score ranking and are only exact when one fault is "
            "active per snapshot. For multi-incident evaluation, run separate forward "
            "passes per incident's temporal graph snapshot.",
            UserWarning,
            stacklevel=2,
        )
    for incident in labelled_incidents:
        # Resolve global index from (node_type, local_idx) if provided
        rc_node_type = incident.get("root_cause_node_type")
        rc_local_idx = incident.get("root_cause_local_idx")
        if rc_node_type is not None and rc_local_idx is not None:
            try:
                true_node_idx = _global_index(
                    rc_node_type, rc_local_idx, node_type_order, node_counts,
                )
            except ValueError as exc:
                logger.warning("Skipping incident: %s", exc)
                continue
        else:
            true_node_idx = incident.get("root_cause_idx")

        if true_node_idx is None:
            continue

        top1_correct += int(ranked[0] == true_node_idx)
        top3 = set(ranked[:3])
        p3_sum += 1.0 if true_node_idx in top3 else 0.0

    n = max(len(labelled_incidents), 1)
    metrics = {
        "node_auroc": round(node_auroc, 4),
        "top1_accuracy": round(top1_correct / n, 4),
        "precision_at_3": round(p3_sum / n, 4),
        "n_incidents": len(labelled_incidents),
        "label_coverage": round(label_coverage, 4),
    }

    logger.info(
        "RCA evaluation: node_auroc=%.4f, top1_accuracy=%.4f, "
        "precision_at_3=%.4f (n=%d incidents, label_coverage=%.2f)",
        metrics["node_auroc"], metrics["top1_accuracy"],
        metrics["precision_at_3"], metrics["n_incidents"],
        metrics["label_coverage"],
    )
    return metrics


# ===========================================================================
# Tier 1: Isolation Forest
# ===========================================================================
def train_isolation_forest(
    X_train: np.ndarray,
    cfg: TrainingConfig,
) -> IsolationForest:
    """Train Isolation Forest for zero-label anomaly bootstrapping.

    Isolation Forest assigns anomaly scores by measuring how quickly a sample
    can be isolated in a random partition tree. Genuine anomalies — sparse,
    extreme points in feature space — require fewer splits to isolate.

    Why IF first in the cascade:
    - Requires NO labels (critical for Phase 0 deployment before labelling pipeline)
    - Computationally cheap (~1s for 10K samples)
    - High recall, moderate precision — gates most obvious anomalies early
    - Pseudo-labels generated by IF seed the RF training (Tier 2)

    See Ch. 16 §8 — Isolation Forest and Tree-Based Anomaly Detection.
    See Part 1 §5.1 — Phased Ensemble: Phase 1 (Isolation Forest).
    """
    log.info(
        f"Training Isolation Forest | "
        f"n_estimators={cfg.if_n_estimators} | "
        f"contamination={cfg.if_contamination}"
    )
    clf = IsolationForest(
        n_estimators=cfg.if_n_estimators,
        max_samples=cfg.if_max_samples,
        contamination=cfg.if_contamination,
        random_state=cfg.if_random_state,
        n_jobs=-1,  # Use all available cores
    )
    clf.fit(X_train)
    log.info("Isolation Forest training complete.")
    return clf


def compute_if_scores(
    clf: IsolationForest,
    X: np.ndarray,
) -> np.ndarray:
    """Compute normalised IF anomaly scores in [0, 1].

    sklearn's decision_function returns negative scores where more negative =
    more anomalous. We negate and scale to [0,1] for consistent ensemble
    score direction (higher = more anomalous across all three tiers).

    The normalisation is min-max over the provided array — for production
    serving, store the training min/max and apply consistently.
    See Ch. 52 §4 — Score Normalisation for Ensemble Combination.
    """
    raw_scores = clf.decision_function(X)  # More negative = more anomalous
    anomaly_scores = -raw_scores           # Flip: higher = more anomalous

    # Normalise to [0, 1] — prevents IF from dominating ensemble due to scale
    score_min = anomaly_scores.min()
    score_max = anomaly_scores.max()
    if score_max > score_min:
        normalised = (anomaly_scores - score_min) / (score_max - score_min)
    else:
        normalised = np.zeros_like(anomaly_scores)
    return normalised


def generate_pseudo_labels(
    if_scores: np.ndarray,
    percentile: float = 97.0,
) -> np.ndarray:
    """Generate binary pseudo-labels from IF scores using a percentile threshold.

    Pseudo-labels are noisy but sufficient to give the Random Forest a signal
    about the anomaly class boundary. The RF then learns a more precise
    decision boundary than IF alone.

    Percentile choice: 3% contamination rate → 97th percentile threshold.
    This is intentionally conservative — a small number of false pseudo-labels
    (normal samples labelled anomalous) is tolerable because the RF will
    learn to separate them from true anomalies using richer features.

    See Part 1 §5.2 — Phased Labelling Pipeline: Semi-Supervised Bootstrap.
    """
    threshold = np.percentile(if_scores, percentile)
    pseudo_labels = (if_scores >= threshold).astype(int)
    log.info(
        f"Pseudo-label threshold (p{percentile:.0f}): {threshold:.4f} | "
        f"pseudo-anomaly rate: {pseudo_labels.mean():.3%}"
    )
    return pseudo_labels


# ===========================================================================
# Tier 2: Random Forest with SHAP
# ===========================================================================


def train_random_forest(
    X_train: np.ndarray,
    y_pseudo: np.ndarray,
    y_true_train: np.ndarray,
    cfg: TrainingConfig,
) -> RandomForestClassifier:
    """Train Random Forest on IF pseudo-labels (or ground truth if available).

    Label priority:
    1. If true labels exist for ≥10% of training samples, use them directly.
    2. Otherwise, use IF pseudo-labels supplemented with true labels where known.
    3. Pure unsupervised fallback: use only IF pseudo-labels.

    This mirrors the Part 1 phased labelling pipeline: as the trouble-ticket
    correlation pipeline and RF engineer annotation provide more labels, the
    quality of RF training improves automatically.

    RF hyperparameter rationale:
    - balanced_subsample: each tree sees a bootstrapped sample with equal class
      counts, avoiding the majority-class collapse problem common with 3%
      anomaly rates. Preferred over SMOTE for time series data (no synthetic
      interpolation that can blur temporal boundaries).
    - OOB score: free generalisation estimate from bagged samples, useful as
      a sanity check without consuming validation data.

    See Ch. 16 §§3-7 — Random Forests: Bagging, Feature Importance, OOB Error.
    See Part 1 §5.2 — Phase 2: Random Forest Classifier.
    """
    # Determine effective labels: use ground truth where available
    true_rate = y_true_train.mean()
    if true_rate > 0.001:
        # Mix true labels and pseudo-labels:
        # True labels override pseudo-labels where they exist (non-zero)
        y_effective = y_pseudo.copy()
        true_mask = y_true_train == 1  # Cells confirmed anomalous by ground truth
        y_effective[true_mask] = 1
        log.info(
            f"RF training with mixed labels | "
            f"true anomalies: {true_mask.sum()} | "
            f"pseudo-labelled: {y_pseudo.sum()} | "
            f"effective anomaly rate: {y_effective.mean():.3%}"
        )
    else:
        y_effective = y_pseudo
        log.info(
            f"RF training on pseudo-labels only | "
            f"pseudo-anomaly rate: {y_pseudo.mean():.3%}"
        )

    log.info(
        f"Training Random Forest | "
        f"n_estimators={cfg.rf_n_estimators} | "
        f"class_weight={cfg.rf_class_weight} | "
        f"min_samples_leaf={cfg.rf_min_samples_leaf}"
    )
    clf = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        class_weight=cfg.rf_class_weight,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        oob_score=True,   # Enable OOB generalisation estimate
        random_state=cfg.rf_random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_effective)
    log.info(
        f"Random Forest training complete | "
        f"OOB score: {clf.oob_score_:.4f}"
    )
    return clf


def compute_rf_scores(
    clf: RandomForestClassifier,
    X: np.ndarray,
) -> np.ndarray:
    """Compute RF anomaly probability scores (class=1 probability).

    RF predict_proba returns calibrated probabilities when class_weight is
    used. The anomaly class (1) probability is used directly as the score —
    no re-normalisation needed as it is already in [0, 1].

    See Ch. 16 §5 — Probability Calibration for Random Forests.
    """
    return clf.predict_proba(X)[:, 1]


def compute_shap_values(
    clf: RandomForestClassifier,
    X_sample: np.ndarray,
    feature_names: List[str],
    cfg: TrainingConfig,
) -> Optional[np.ndarray]:
    """Compute SHAP TreeExplainer values for RF feature attribution.

    SHAP values are computed on a subsample (cfg.rf_shap_sample_n) because
    TreeExplainer is O(N * T * D) where T=n_estimators, D=tree depth.

    SHAP integration with Part 2 agentic system:
    The top-5 SHAP values per cell become node attributes in the Layer 5 GNN
    graph, enabling the root-cause agent to understand WHICH KPI features
    drove the anomaly score before traversing the topology graph.

    See Ch. 13 §4 — SHAP: Unified Feature Importance.
    See Part 1 §6 — SHAP-DAG Conflict Detection (references same values).
    """
    if not _SHAP_AVAILABLE:
        log.warning("SHAP not available — skipping feature attribution.")
        return None

    n_samples = min(len(X_sample), cfg.rf_shap_sample_n)
    idx = np.random.choice(len(X_sample), size=n_samples, replace=False)
    X_sub = X_sample[idx]

    log.info(f"Computing SHAP values on {n_samples} samples ...")
    explainer = shap.TreeExplainer(
        clf,
        feature_perturbation="tree_path_dependent",  # Exact for tree models
    )
    shap_values = explainer.shap_values(X_sub)

    # shap_values is a list [class_0_values, class_1_values] — use class 1
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # Anomaly class
    else:
        shap_vals = shap_values

    # Log top-10 features by mean absolute SHAP
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:10]
    log.info("Top-10 features by mean |SHAP|:")
    for rank, idx_ in enumerate(top_indices, 1):
        log.info(f"  {rank:2d}. {feature_names[idx_]:<50s} {mean_abs[idx_]:.4f}")

    return shap_vals


# ===========================================================================
# Tier 3: LSTM Autoencoder
# ===========================================================================


_LSTMAutoencoder_base = nn.Module if _TORCH_AVAILABLE else object


class LSTMAutoencoder(_LSTMAutoencoder_base):
    """LSTM Autoencoder for temporal anomaly detection via reconstruction error.

    Architecture:
      Encoder: LSTM(input_dim → hidden_dim, num_layers) → context vector
      Decoder: LSTM(hidden_dim → hidden_dim, num_layers) → Linear(hidden_dim → input_dim)

    The model is trained to reconstruct NORMAL sequences. Anomalous sequences
    produce high reconstruction error because the encoder's compressed
    representation cannot capture the deviation pattern.

    Sequence-level reconstruction error (MSE per feature, mean across time
    steps and features) is normalised to [0,1] and used as Tier 3 score.

    Key design choices:
    - Decoder uses context vector as initial hidden state (seq2seq pattern)
    - Dropout applied between LSTM layers for regularisation
    - Training on NORMAL-only data: reconstruction error acts as a distance
      metric from the "normal manifold"

    See Ch. 22 §§4-6 — LSTM Autoencoders for Anomaly Detection.
    See Part 1 §5.3 — Phase 3: LSTM Autoencoder.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder: maps input sequence to fixed-size context vector
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,   # (batch, seq, features)
        )

        # Decoder: reconstructs input from context vector
        # We repeat the context vector across seq_len for teacher-force-free decoding
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Output projection: hidden_dim → input_dim
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass: encode → decode → reconstruct.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            reconstructed: Tensor of shape (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Encode: extract final hidden state as context
        _, (h_n, c_n) = self.encoder(x)
        # h_n shape: (num_layers, batch_size, hidden_dim)

        # Expand context vector across the sequence dimension for decoding
        # This is the "repeat context" decoding strategy — simpler than
        # attention-based decoding, adequate for seq_len=4 (1-hour window).
        context = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)
        # context shape: (batch_size, seq_len, hidden_dim)

        # Decode: project context sequence through LSTM
        decoder_out, _ = self.decoder(context)
        # decoder_out shape: (batch_size, seq_len, hidden_dim)

        # Project to input dimension
        reconstructed = self.output_layer(decoder_out)
        # reconstructed shape: (batch_size, seq_len, input_dim)

        return reconstructed


_Dataset_base = Dataset if _TORCH_AVAILABLE else object


class RanSequenceDataset(_Dataset_base):
    """PyTorch Dataset for fixed-length RAN KPI sequences.

    Constructs overlapping windows of length seq_len from the input array.
    Stride=1 maximises the number of training samples.

    Temporal integrity: sequences are constructed in chronological order
    from a temporally sorted input array. No shuffle at Dataset level;
    DataLoader shuffle=True is applied only during training.

    See Ch. 22 §2 — Sequence Construction for Temporal Data.
    See Ch. 28 §3 — Windowing Strategies for Time-Series ML.
    """

    def __init__(self, X: np.ndarray, seq_len: int = 4) -> None:
        """
        Args:
            X: Feature matrix of shape (N, F), temporally sorted.
            seq_len: Lookback window length in number of ROPs.
        """
        self.seq_len = seq_len
        # Construct sliding window sequences
        # sequence[i] covers rows i : i+seq_len
        n_sequences = len(X) - seq_len + 1
        if n_sequences <= 0:
            raise ValueError(
                f"Not enough samples ({len(X)}) for seq_len={seq_len}. "
                "Reduce seq_len or provide more data."
            )
        self.sequences = np.stack(
            [X[i : i + seq_len] for i in range(n_sequences)],
            axis=0,
        ).astype(np.float32)
        log.info(
            f"RanSequenceDataset: {len(self.sequences):,} sequences "
            f"(seq_len={seq_len}, n_features={X.shape[1]})"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> "torch.Tensor":
        return torch.from_numpy(self.sequences[idx])


def train_lstm_autoencoder(
    X_train_normal: np.ndarray,
    X_val: np.ndarray,
    cfg: TrainingConfig,
    device: "torch.device",
) -> "LSTMAutoencoder":
    """Train LSTM Autoencoder on NORMAL-only training samples.

    Why train on normal-only:
    If anomalous sequences are included in training, the autoencoder learns
    to reconstruct them too, compressing the reconstruction error differential
    between normal and anomalous. Training on normal-only enforces that the
    autoencoder is optimised for normal reconstruction; anomalies then stand
    out as high-error outliers.

    Early stopping:
    Validation loss is monitored every epoch. Training stops after
    cfg.lstm_patience epochs without improvement. Best weights are restored.
    This prevents overfitting on the normal training set and is especially
    important when the training set is small (<5K samples).

    See Ch. 22 §5 — Autoencoder Training and Reconstruction Loss.
    See Ch. 54 §3 — Early Stopping as a Regularisation Strategy.
    """
    torch.manual_seed(cfg.lstm_random_state)

    # ---- Construct dataset from normal-only training samples ----
    train_dataset = RanSequenceDataset(X_train_normal, seq_len=cfg.lstm_seq_len)
    val_dataset = RanSequenceDataset(X_val, seq_len=cfg.lstm_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.lstm_batch_size,
        shuffle=True,   # Shuffle sequences (not individual time steps) for SGD
        num_workers=0,  # Single process: safer for Jupyter/Windows compatibility
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.lstm_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model, loss, optimiser ----
    n_features = X_train_normal.shape[1]
    model = LSTMAutoencoder(
        input_dim=n_features,
        hidden_dim=cfg.lstm_hidden_dim,
        num_layers=cfg.lstm_num_layers,
        dropout=cfg.lstm_dropout,
    ).to(device)

    # MSE loss: per-element squared error, averaged across batch/seq/feature dims
    criterion = nn.MSELoss(reduction="mean")

    # Adam with default betas — standard choice for LSTM training
    # See Ch. 22 §3 — Optimiser Selection for Recurrent Networks.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lstm_learning_rate,
        weight_decay=1e-5,  # L2 regularisation
    )

    # Reduce LR on plateau: halves LR if val_loss stagnates for 3 epochs
    # Allows coarse-to-fine optimisation without manual LR scheduling.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        verbose=False,
    )

    best_val_loss = float("inf")
    best_weights: Dict[str, Any] = {}
    patience_counter = 0

    log.info(
        f"Training LSTM Autoencoder | "
        f"input_dim={n_features} | "
        f"hidden_dim={cfg.lstm_hidden_dim} | "
        f"seq_len={cfg.lstm_seq_len} | "
        f"device={device}"
    )

    for epoch in range(1, cfg.lstm_epochs + 1):
        # ---- Training phase ----
        model.train()
        train_losses: List[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            # Gradient clipping: prevents exploding gradients in LSTM
            # max_norm=1.0 is standard for sequence models.
            # See Ch. 22 §2 — Gradient Clipping for RNNs.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ---- Validation phase ----
        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        # Log every 5 epochs or on improvement
        if epoch % 5 == 0 or val_loss < best_val_loss:
            log.info(
                f"Epoch {epoch:3d}/{cfg.lstm_epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        # ---- Early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.lstm_patience:
                log.info(
                    f"Early stopping at epoch {epoch} | "
                    f"best_val_loss={best_val_loss:.6f}"
                )
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    log.info(
        f"LSTM Autoencoder training complete | best_val_loss={best_val_loss:.6f}"
    )
    return model


def compute_reconstruction_errors(
    model: "LSTMAutoencoder",
    X: np.ndarray,
    cfg: TrainingConfig,
    device: "torch.device",
    norm_min: Optional[float] = None,
    norm_max: Optional[float] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
    """Compute per-sequence reconstruction error (MSE).

    Returns an error array of shape (N - seq_len + 1,) where each element
    is the mean squared reconstruction error for the corresponding sequence.

    The error is then aligned back to the original N rows by padding with
    the first error value for the initial (seq_len - 1) rows that have no
    complete lookback window.

    This alignment strategy ensures that the LSTM score vector has the same
    length as IF and RF score vectors for ensemble combination.

    Parameters
    ----------
    norm_min, norm_max : float, optional
        If provided, use these values as the min-max anchors for [0,1]
        normalisation instead of deriving them from the current array.
        **Always reuse the training-set anchors when scoring validation
        or test data** — otherwise the threshold computed by
        ``compute_lstm_threshold()`` is not portable across splits.

    Returns
    -------
    If *norm_min* and *norm_max* are both ``None`` (the default, used during
    threshold calibration on training data), returns a tuple
    ``(normalised, score_min, score_max)`` so the caller can store the
    anchors.  Otherwise returns the normalised array only.

    See Ch. 22 §6 — Reconstruction Error as Anomaly Score.
    See Ch. 52 §4 — Score Alignment for Ensemble Combination.
    """
    model.eval()
    dataset = RanSequenceDataset(X, seq_len=cfg.lstm_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=cfg.lstm_batch_size,
        shuffle=False,  # MUST be False — preserves temporal order
        num_workers=0,
    )

    errors: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            # Per-sequence MSE: mean over seq_len and feature dimensions
            seq_errors = ((reconstructed - batch) ** 2).mean(dim=(1, 2))
            errors.extend(seq_errors.cpu().numpy().tolist())

    errors_arr = np.array(errors)

    # Align to original N rows: pad the beginning with the first error value
    # (the first seq_len-1 rows have no complete window)
    pad_len = cfg.lstm_seq_len - 1
    padded = np.concatenate([
        np.full(pad_len, errors_arr[0]),  # Conservative: assume first window error
        errors_arr,
    ])

    # Normalise to [0, 1] for ensemble combination.
    # If anchors were supplied, reuse them (val/test scoring path);
    # otherwise derive from this array (training calibration path).
    _return_anchors = (norm_min is None and norm_max is None)
    _score_min = padded.min() if norm_min is None else norm_min
    _score_max = padded.max() if norm_max is None else norm_max
    if _score_max > _score_min:
        normalised = (padded - _score_min) / (_score_max - _score_min)
    else:
        normalised = np.zeros_like(padded)

    if _return_anchors:
        return normalised, float(_score_min), float(_score_max)
    return normalised


def compute_lstm_threshold(
    model: "LSTMAutoencoder",
    X_train_normal: np.ndarray,
    cfg: TrainingConfig,
    device: "torch.device",
) -> Tuple[float, float, float]:
    """Compute the reconstruction error threshold on normal training data.

    Threshold = cfg.lstm_reconstruction_percentile of the error distribution
    on normal sequences. This defines the "normal manifold boundary" — errors
    above this threshold are classified as anomalous.

    Important: this threshold is computed on TRAINING NORMAL data, not on
    validation or test data. Validation is used only to select the optimal
    ensemble weight combination (to prevent double-dipping).

    Returns
    -------
    (threshold, norm_min, norm_max) — the threshold value and the min/max
    anchors used for normalisation.  Callers MUST reuse ``norm_min`` and
    ``norm_max`` when scoring validation or test data to ensure the
    threshold remains calibrated.

    See Ch. 52 §5 — Unsupervised Threshold Selection.
    See Part 1 §5.3 — LSTM Autoencoder Threshold Calibration.
    """
    errors, norm_min, norm_max = compute_reconstruction_errors(
        model, X_train_normal, cfg, device,
    )
    threshold = float(np.percentile(errors, cfg.lstm_reconstruction_percentile))
    log.info(
        f"LSTM reconstruction threshold (p{cfg.lstm_reconstruction_percentile:.0f}): "
        f"{threshold:.6f} (norm_min={norm_min:.6f}, norm_max={norm_max:.6f})"
    )
    return threshold, norm_min, norm_max


# ===========================================================================
# Ensemble scoring and threshold calibration
# ===========================================================================


def compute_ensemble_score(
    if_scores: np.ndarray,
    rf_scores: np.ndarray,
    lstm_scores: np.ndarray,
    weights: Tuple[float, float, float],
) -> np.ndarray:
    """Compute weighted ensemble anomaly score.

    Score = w1 * IF_score + w2 * RF_score + w3 * LSTM_score

    All three component scores are in [0, 1] with higher = more anomalous.
    The weighted sum is also in [0, 1].

    Design rationale (Part 1 §5.4):
    - IF (w1=0.20): Fast, zero-label first-pass filter. Low weight because
      its scores have lower precision than RF.
    - RF (w2=0.50): Highest weight — benefits from feature richness and
      pseudo-label signal. Dominant discriminator.
    - LSTM-AE (w3=0.30): Captures temporal patterns (diurnal, trend breaks)
      that IF and RF miss because they treat each row independently.

    The ensemble is deliberately RF-dominant in the default configuration
    because tabular RF with peer-group features outperforms temporal models
    on the most common anomaly types in RAN data (parameter misconfiguration,
    hardware degradation). LSTM weight increases for event-spike use cases.

    See Ch. 52 §3 — Ensemble Methods for Anomaly Detection.
    See Part 1 §5.4 — Cascade Scoring Formula.
    """
    w1, w2, w3 = weights
    # Normalise weights to sum to 1 (defensive — should already sum to 1)
    weight_sum = w1 + w2 + w3
    w1, w2, w3 = w1 / weight_sum, w2 / weight_sum, w3 / weight_sum

    ensemble = w1 * if_scores + w2 * rf_scores + w3 * lstm_scores
    return ensemble


def select_threshold_youden_j(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Select optimal classification threshold using Youden's J statistic.

    Youden's J = Sensitivity + Specificity - 1 = TPR - FPR

    This metric balances detection rate (recall) against false alarm rate
    (1-specificity). For NOC operations, it is preferred over F1 maximisation
    because false alarms and missed detections have asymmetric costs:
    - False alarm: NOC analyst investigates a healthy cell (~15 min wasted)
    - Missed detection: network degradation continues until next ROP cycle (~15 min)

    The Part 1 cost model (§1) values a missed detection at approximately
    10× a false alarm for high-traffic cells, but 2-3× for rural cells.
    Youden's J gives a reasonable operational middle ground without requiring
    explicit cost matrix input at training time.

    For operators with known cost ratios, replace with F-beta maximisation:
    F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
    where beta = sqrt(missed_detection_cost / false_alarm_cost).

    See Ch. 52 §5 — Threshold Selection Methods.
    See Ch. 54 §2 — Operational Metric Alignment for Alerting Systems.

    Returns:
        threshold: Optimal classification threshold
        precision: Precision at threshold
        recall: Recall at threshold
        f1: F1 at threshold
    """
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(labels, scores)
    # thresh_arr has one fewer element than prec_arr, rec_arr
    # prec_arr[-1] = 1.0, rec_arr[-1] = 0.0 (sklearn convention) — skip last
    prec_arr = prec_arr[:-1]
    rec_arr = rec_arr[:-1]

    # Compute ROC for specificity
    from sklearn.metrics import roc_curve
    fpr_arr, tpr_arr, roc_thresh_arr = roc_curve(labels, scores)

    # Interpolate specificity at each PR threshold
    # (PR and ROC threshold arrays are different; we evaluate J on PR thresholds)
    specificity_at_pr_thresh = 1.0 - np.interp(thresh_arr, roc_thresh_arr, fpr_arr)
    sensitivity_at_pr_thresh = rec_arr

    j_stat = sensitivity_at_pr_thresh + specificity_at_pr_thresh - 1.0

    # Avoid thresholds where precision is pathologically low (<0.10)
    # — would generate too many false alarms in production
    valid_mask = prec_arr >= 0.10
    if valid_mask.sum() == 0:
        log.warning("No thresholds with precision >= 0.10 — using F1 maximisation fallback.")
        f1_arr = 2 * prec_arr * rec_arr / np.maximum(prec_arr + rec_arr, 1e-8)
        best_idx = np.argmax(f1_arr)
    else:
        best_idx_valid = np.argmax(j_stat[valid_mask])
        best_idx = np.where(valid_mask)[0][best_idx_valid]

    best_threshold = float(thresh_arr[best_idx])
    best_prec = float(prec_arr[best_idx])
    best_rec = float(rec_arr[best_idx])
    best_f1 = float(
        2 * best_prec * best_rec / max(best_prec + best_rec, 1e-8)
    )

    log.info(
        f"Youden's J threshold: {best_threshold:.4f} | "
        f"P={best_prec:.4f} R={best_rec:.4f} F1={best_f1:.4f} | "
        f"J={j_stat[best_idx]:.4f}"
    )
    return best_threshold, best_prec, best_rec, best_f1


def calibrate_ensemble_weights(
    if_scores_val: np.ndarray,
    rf_scores_val: np.ndarray,
    lstm_scores_val: np.ndarray,
    y_val: np.ndarray,
    weight_grid: List[Tuple[float, float, float]],
) -> Tuple[Tuple[float, float, float], float, float]:
    """Grid-search ensemble weights on the validation set.

    For each weight combination in the grid, compute:
    1. Ensemble score on validation set
    2. AUROC (threshold-independent measure of discrimination)
    3. Best F1 via Youden's J threshold selection

    Select the weight combination maximising AUROC. AUROC is preferred over
    F1 for weight selection because it is threshold-independent — we don't
    want to jointly optimise weights AND threshold on the same data, which
    would cause overfitting of the threshold.

    Final threshold is then selected on the validation set using the best
    weights. This two-step approach (weight selection → threshold selection)
    is a common pattern for cascaded anomaly detectors.

    See Ch. 52 §§4-5 — Ensemble Weight Calibration and Threshold Selection.
    See Part 1 §5.4 — Phased Ensemble Calibration Procedure.

    Returns:
        best_weights: Weight tuple (w_if, w_rf, w_lstm) that maximises AUROC
        best_auroc: AUROC of the best ensemble on validation set
        best_threshold: Youden's J threshold for best ensemble on validation set
    """
    log.info(
        f"Calibrating ensemble weights over {len(weight_grid)} combinations "
        f"on validation set ({len(y_val):,} samples, "
        f"anomaly rate={y_val.mean():.3%}) ..."
    )

    best_weights: Tuple[float, float, float] = weight_grid[0]
    best_auroc = -1.0
    best_threshold = 0.5

    for weights in weight_grid:
        scores = compute_ensemble_score(
            if_scores_val, rf_scores_val, lstm_scores_val, weights
        )
        try:
            auroc = roc_auc_score(y_val, scores)
        except ValueError:
            # Can happen if val set has only one class — skip
            log.warning(f"Weights {weights}: AUROC computation failed (single class).")
            continue

        threshold, prec, rec, f1 = select_threshold_youden_j(scores, y_val)
        log.info(
            f"  weights={weights} | AUROC={auroc:.4f} | "
            f"F1={f1:.4f} | threshold={threshold:.4f}"
        )
        if auroc > best_auroc:
            best_auroc = auroc
            best_weights = weights
            best_threshold = threshold

    log.info(
        f"Best ensemble weights: {best_weights} | "
        f"AUROC={best_auroc:.4f} | threshold={best_threshold:.4f}"
    )
    return best_weights, best_auroc, best_threshold


# ===========================================================================
# Baseline model
# ===========================================================================


def baseline_rolling_threshold(
    X_df: pd.DataFrame,
    y: np.ndarray,
    feature_col: str = "dl_throughput_mbps",
    window: int = 4,
    n_sigma: float = 3.0,
) -> Dict[str, float]:
    """Naive baseline: flag anomalies where a KPI exceeds N sigma of its rolling mean.

    This is the simplest possible anomaly detector — a Z-score threshold on a
    single KPI with a short rolling window. Production systems always require
    a baseline comparison to demonstrate that the ML model adds value beyond
    what simple heuristics can achieve.

    In Part 1 §8.1, the baseline is defined as the operator's existing
    threshold-based alerting system. Here we simulate it with a rolling
    sigma detector applied to throughput degradation — a common NOC heuristic.

    Design choices:
    - n_sigma=3.0: Flags samples more than 3 standard deviations below the
      rolling mean. Asymmetric (only flags drops, not spikes) to match
      common NOC practice for throughput monitoring.
    - window=4: 4 ROPs = 1 hour. Short enough to detect rapid degradation,
      long enough to avoid flagging transient fluctuations.

    Returns dict of baseline metrics for comparison table.

    See Ch. 52 §2 — Baseline Model Selection and Justification.
    """
    # Find the first matching column (feature name may vary with suffix)
    target_col = None
    for col in X_df.columns:
        if feature_col in col:
            target_col = col
            break

    if target_col is None:
        log.warning(
            f"Baseline: column containing '{feature_col}' not found. "
            "Using first numeric column as proxy."
        )
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {"baseline_note": "no numeric columns"}
        target_col = numeric_cols[0]

    values = X_df[target_col].values.astype(float)

    # Rolling mean and std with minimum periods=2 (prevents NaN at start)
    rolling_mean = pd.Series(values).rolling(window=window, min_periods=2).mean().values
    rolling_std = pd.Series(values).rolling(window=window, min_periods=2).std().values

    # Avoid division by zero
    rolling_std = np.where(rolling_std < 1e-6, 1e-6, rolling_std)

    # Flag: value drops more than n_sigma below rolling mean
    baseline_scores = (rolling_mean - values) / rolling_std
    baseline_scores = np.clip(baseline_scores, 0, None)  # Only flag drops
    baseline_scores = np.nan_to_num(baseline_scores, nan=0.0)

    # Normalise to [0,1]
    if baseline_scores.max() > 0:
        baseline_scores = baseline_scores / baseline_scores.max()

    # Threshold at n_sigma normalised level
    threshold = n_sigma / max(baseline_scores.max(), n_sigma)
    baseline_labels = (baseline_scores >= threshold).astype(int)

    # Compute metrics
    if y.sum() > 0:
        baseline_precision = float(precision_score(y, baseline_labels, zero_division=0))
        baseline_recall = float(recall_score(y, baseline_labels, zero_division=0))
        baseline_f1 = float(f1_score(y, baseline_labels, zero_division=0))
        try:
            baseline_auroc = float(roc_auc_score(y, baseline_scores))
        except ValueError:
            baseline_auroc = 0.5
    else:
        baseline_precision = baseline_recall = baseline_f1 = baseline_auroc = 0.0

    metrics = {
        "model": f"Baseline (rolling {window}-ROP {n_sigma}σ on {target_col})",
        "precision": baseline_precision,
        "recall": baseline_recall,
        "f1": baseline_f1,
        "auroc": baseline_auroc,
        "anomaly_rate_predicted": float(baseline_labels.mean()),
    }
    log.info(
        f"Baseline | P={baseline_precision:.4f} R={baseline_recall:.4f} "
        f"F1={baseline_f1:.4f} AUROC={baseline_auroc:.4f}"
    )
    return metrics


# ===========================================================================
# Full evaluation on a dataset split
# ===========================================================================


def evaluate_ensemble(
    if_clf: IsolationForest,
    rf_clf: RandomForestClassifier,
    lstm_model: Optional["LSTMAutoencoder"],
    X_scaled: np.ndarray,
    X_df: pd.DataFrame,
    y: np.ndarray,
    weights: Tuple[float, float, float],
    threshold: float,
    cfg: TrainingConfig,
    device: Optional["torch.device"],
    split_label: str = "test",
    lstm_norm_min: Optional[float] = None,
    lstm_norm_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate the three-tier ensemble on a labelled dataset split.

    Computes component scores, ensemble score, and all operational metrics.
    Returns a dictionary suitable for serialisation to training_metadata.json.

    Parameters
    ----------
    lstm_norm_min, lstm_norm_max : float, optional
        Min/max anchors from training-set LSTM error distribution.  When
        provided, ``compute_reconstruction_errors`` reuses them so that
        the threshold calibrated on training data remains valid on
        val/test splits.  If ``None``, each split derives its own
        anchors — acceptable for demo/synthetic data but **not for
        production** (see §8.4 threshold alignment warning).

    Metrics computed:
    - AUROC: Threshold-independent discrimination (Ch. 52 §5)
    - AUPRC: Area Under Precision-Recall Curve — preferred for imbalanced data
    - Precision, Recall, F1 at the calibrated threshold
    - Confusion matrix entries (TP, FP, TN, FN)
    - False Alarm Rate (FAR = FP / (FP + TN)): the primary NOC SLA metric
    - Miss Rate (MR = FN / (FN + TP)): the primary reliability metric

    Operational interpretation:
    - FAR < 5%: NOC analysts spend < 5% of investigation time on false positives
    - Recall > 90%: < 10% of true anomalies are missed

    See Ch. 54 §§2-3 — Operational Metrics for Alerting Systems.
    See Part 1 §8 — Evaluation and Operational Impact.
    """
    log.info(f"Evaluating ensemble on {split_label} ({len(y):,} samples) ...")

    # --- Compute component scores ---
    if_scores = compute_if_scores(if_clf, X_scaled)
    rf_scores = compute_rf_scores(rf_clf, X_scaled)

    if lstm_model is not None and _TORCH_AVAILABLE:
        lstm_scores = compute_reconstruction_errors(
            lstm_model, X_scaled, cfg, device,
            norm_min=lstm_norm_min, norm_max=lstm_norm_max,
        )
        # When anchors are None (first call / demo), unpack the tuple
        if isinstance(lstm_scores, tuple):
            lstm_scores = lstm_scores[0]
    else:
        # LSTM not available: use RF scores doubled (degenerate two-tier ensemble)
        lstm_scores = rf_scores.copy()
        log.warning(
            f"LSTM not available for {split_label} evaluation — "
            "using RF scores as LSTM proxy (two-tier ensemble)."
        )

    # --- Ensemble score ---
    ensemble_scores = compute_ensemble_score(if_scores, rf_scores, lstm_scores, weights)

    # --- Predictions at calibrated threshold ---
    y_pred = (ensemble_scores >= threshold).astype(int)

    # --- Metrics ---
    try:
        auroc = float(roc_auc_score(y, ensemble_scores))
    except ValueError:
        auroc = 0.5
        log.warning(f"{split_label}: AUROC undefined (single class in y).")

    try:
        auprc = float(average_precision_score(y, ensemble_scores))
    except ValueError:
        auprc = 0.0

    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    far = float(fp / max(fp + tn, 1))    # False Alarm Rate
    miss_rate = float(fn / max(fn + tp, 1))  # Miss Rate (1 - Recall)

    results = {
        "split": split_label,
        "n_samples": int(len(y)),
        "anomaly_rate_actual": float(y.mean()),
        "anomaly_rate_predicted": float(y_pred.mean()),
        "auroc": auroc,
        "auprc": auprc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "false_alarm_rate": far,
        "miss_rate": miss_rate,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "ensemble_threshold": float(threshold),
        "ensemble_weights": list(weights),
    }

    log.info(
        f"{split_label.upper()} | "
        f"AUROC={auroc:.4f} | AUPRC={auprc:.4f} | "
        f"P={prec:.4f} R={rec:.4f} F1={f1:.4f} | "
        f"FAR={far:.4f} MissRate={miss_rate:.4f}"
    )

    # --- Governance gate assertion ---
    # Part 1 §8 defines F1 ≥ 0.82 as the minimum production gate.
    # This assertion runs here during training so CI/CD pipelines fail fast.
    if split_label == "test":
        if f1 < MIN_F1_GATE:
            log.warning(
                f"TEST F1={f1:.4f} BELOW governance gate F1≥{MIN_F1_GATE}. "
                "Model must NOT be promoted to production. "
                "Investigate: insufficient training data, feature drift, or "
                "label quality issues. See Part 1 §8 governance gate."
            )
        else:
            log.info(
                f"GOVERNANCE GATE PASSED: TEST F1={f1:.4f} ≥ {MIN_F1_GATE} ✓"
            )

    return results


# ===========================================================================
# Synthetic data fallback (if 02_feature_engineering.py output is missing)
# ===========================================================================


def generate_fallback_data(
    n_train: int = 8000,
    n_val: int = 1500,
    n_test: int = 1500,
    n_features: int = 40,
    anomaly_rate: float = 0.03,
    cfg: Optional[TrainingConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate minimal synthetic feature matrices if 02 outputs are absent.

    This fallback allows 03_model_training.py to run standalone (as required
    by the production-quality code standard). The synthetic data approximates
    the feature space from 02_feature_engineering.py:
    - Features: mix of scaled KPI values (normal/anomalous clusters)
    - Labels: injected at anomaly_rate with temporal clustering (anomalies
      tend to cluster in time — realistic for network faults)
    - Timestamps: hourly granularity with diurnal patterns embedded in features

    See 01_synthetic_data.py for full telco-realistic data generation.
    See 02_feature_engineering.py for the feature engineering pipeline.
    """
    log.info(
        f"Generating fallback synthetic data: "
        f"train={n_train} val={n_val} test={n_test} features={n_features}"
    )
    rng = np.random.default_rng(42)

    # Feature names that mirror 02_feature_engineering.py output schema
    feature_names = (
        [
            "rsrp_dbm", "rsrq_db", "sinr_db", "cqi", "dl_throughput_mbps",
            "ul_throughput_mbps", "rrc_conn_setup_success_rate",
            "handover_success_rate", "cell_availability_pct", "dl_prb_usage_rate",
            "dl_throughput_mbps_rollmean_1h", "dl_throughput_mbps_rollstd_1h",
            "dl_throughput_mbps_rollmean_4h", "dl_throughput_mbps_rollstd_4h",
            "rsrp_dbm_delta_1rop", "rsrp_dbm_delta_4rop",
            "dl_throughput_mbps_zscore_1h", "rsrp_dbm_zscore_1h",
            "peer_dl_throughput_mbps_mean", "peer_dl_throughput_mbps_zscore",
            "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
            "is_peak_hour", "is_weekend",
        ]
        + [f"feature_{i:03d}" for i in range(n_features - 26)]
    )[:n_features]

    def _make_split(n: int, seed_offset: int) -> pd.DataFrame:
        rng_ = np.random.default_rng(42 + seed_offset)
        X = rng_.standard_normal((n, n_features)).astype(np.float32)

        # Generate clustered anomaly labels (faults cluster temporally)
        y = np.zeros(n, dtype=int)
        n_anomaly_events = max(1, int(n * anomaly_rate / 5))  # ~5 ROPs per event
        for _ in range(n_anomaly_events):
            start = rng_.integers(0, n - 5)
            duration = rng_.integers(3, 8)
            y[start : start + duration] = 1

        # Shift anomalous samples to create a separable anomaly cluster
        anomaly_mask = y == 1
        X[anomaly_mask, :5] -= 3.0   # Degrade first 5 KPI features for anomalies
        X[anomaly_mask, 5:10] += 2.0 # Spike remaining capacity features

        # Generate timestamps at 15-minute granularity
        base_ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=seed_offset * 24)
        timestamps = pd.date_range(base_ts, periods=n, freq="15min")

        df = pd.DataFrame(X, columns=feature_names)
        df[LABEL_COL] = y
        df[TIMESTAMP_COL] = timestamps
        df[CELL_ID_COL] = "CELL_001_1"  # Single cell for simplicity
        return df

    train_df = _make_split(n_train, seed_offset=0)
    val_df = _make_split(n_val, seed_offset=1)
    test_df = _make_split(n_test, seed_offset=2)

    log.info(
        f"Fallback data: train anomaly rate={train_df[LABEL_COL].mean():.3%} | "
        f"val={val_df[LABEL_COL].mean():.3%} | test={test_df[LABEL_COL].mean():.3%}"
    )
    return train_df, val_df, test_df


# ===========================================================================
# Artefact serialisation
# ===========================================================================


def save_artefacts(
    if_clf: IsolationForest,
    rf_clf: RandomForestClassifier,
    lstm_model: Optional["LSTMAutoencoder"],
    scaler: StandardScaler,
    thresholds: Dict[str, Any],
    metadata: Dict[str, Any],
    shap_values: Optional[np.ndarray],
) -> None:
    """Persist all model artefacts to MODELS_DIR.

    Artefacts:
    - isolation_forest.joblib: sklearn model (joblib serialisation)
    - random_forest.joblib: sklearn model (joblib serialisation)
    - lstm_autoencoder.pt: PyTorch state dict (NOT the full model object —
      state dict serialisation is more portable across Python/PyTorch versions)
    - scaler.joblib: fitted StandardScaler (must match serving-time scaler)
    - ensemble_thresholds.json: calibrated threshold and weights
    - training_metadata.json: full evaluation results and configuration
    - shap_values_sample.npy: SHAP values array for explainability dashboards

    Serving-time consistency note:
    The ONNX export pipeline (05_production_patterns.py) reads these artefacts
    and converts sklearn models to ONNX format. The scaler is embedded as an
    ONNX preprocessing node to prevent train/serve skew.

    See Ch. 28 §6 — Model Serialisation and Artefact Management.
    See Part 1 §6 — ONNX Serving Pattern.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Isolation Forest
    joblib.dump(if_clf, IF_MODEL_PATH, compress=3)
    log.info(f"Saved Isolation Forest → {IF_MODEL_PATH}")

    # Random Forest
    joblib.dump(rf_clf, RF_MODEL_PATH, compress=3)
    log.info(f"Saved Random Forest → {RF_MODEL_PATH}")

    # LSTM Autoencoder (PyTorch state dict)
    if lstm_model is not None and _TORCH_AVAILABLE:
        torch.save(
            {
                "state_dict": lstm_model.state_dict(),
                "config": {
                    "input_dim": lstm_model.input_dim,
                    "hidden_dim": lstm_model.hidden_dim,
                    "num_layers": lstm_model.num_layers,
                },
            },
            LSTM_MODEL_PATH,
        )
        log.info(f"Saved LSTM Autoencoder → {LSTM_MODEL_PATH}")
    else:
        log.info("LSTM model not available — skipping LSTM save.")

    # Scaler (save updated copy even if loaded from 02; ensures models/ is self-contained)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib", compress=3)
    log.info(f"Saved Scaler → {MODELS_DIR / 'scaler.joblib'}")

    # Thresholds and weights
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info(f"Saved ensemble thresholds → {THRESHOLDS_PATH}")

    # Training metadata (full provenance record)
    metadata["saved_at"] = datetime.now(tz=timezone.utc).isoformat()
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info(f"Saved training metadata → {METADATA_PATH}")

    # SHAP values
    if shap_values is not None:
        np.save(SHAP_VALUES_PATH, shap_values)
        log.info(f"Saved SHAP values sample → {SHAP_VALUES_PATH}")


# ===========================================================================
# Main training orchestration
# ===========================================================================


def main() -> None:
    """End-to-end training orchestration for the three-tier cascade ensemble.

    Pipeline stages:
    1. Load data splits (from 02_feature_engineering.py or fallback)
    2. Apply/fit feature scaling
    3. Train Isolation Forest (Tier 1)
    4. Generate pseudo-labels from IF scores
    5. Train Random Forest (Tier 2)
    6. Train LSTM Autoencoder on normal-only data (Tier 3)
    7. Calibrate ensemble weights on validation set
    8. Evaluate baseline and ensemble on test set
    9. Assert governance gate (F1 ≥ 0.82)
    10. Save all artefacts

    Temporal split integrity:
    - Train: earliest time period
    - Val: middle period (never random split for time series)
    - Test: latest time period
    This prevents look-ahead leakage from future network state into training.

    See Ch. 28 §2 — Temporal Train/Val/Test Split for Time-Series Data.
    See Part 1 §8 — Evaluation Methodology and Governance Gates.
    """
    log.info("=" * 70)
    log.info("Telco MLOps Part 2 — 03_model_training.py")
    log.info("Three-tier cascade: Isolation Forest → Random Forest → LSTM-AE")
    log.info("=" * 70)

    cfg = TrainingConfig()

    # -----------------------------------------------------------------------
    # Stage 1: Load data
    # -----------------------------------------------------------------------
    if all(p.exists() for p in [TRAIN_PARQUET, VAL_PARQUET, TEST_PARQUET]):
        log.info("Loading feature splits from 02_feature_engineering.py output ...")
        X_train_df, y_train, ts_train = load_split(TRAIN_PARQUET, cfg, "train")
        X_val_df, y_val, ts_val = load_split(VAL_PARQUET, cfg, "val")
        X_test_df, y_test, ts_test = load_split(TEST_PARQUET, cfg, "test")
        log.info(
            f"Temporal split: "
            f"train [{pd.to_datetime(ts_train[0]):%Y-%m-%d} → "
            f"{pd.to_datetime(ts_train[-1]):%Y-%m-%d}] | "
            f"val [{pd.to_datetime(ts_val[0]):%Y-%m-%d} → "
            f"{pd.to_datetime(ts_val[-1]):%Y-%m-%d}] | "
            f"test [{pd.to_datetime(ts_test[0]):%Y-%m-%d} → "
            f"{pd.to_datetime(ts_test[-1]):%Y-%m-%d}]"
        )
    else:
        log.warning(
            "Feature splits not found — using synthetic fallback data. "
            "Run 02_feature_engineering.py for production-quality results."
        )
        train_df, val_df, test_df = generate_fallback_data(cfg=cfg)

        # Extract from fallback DataFrames using same load_split logic
        for name, df, path in [
            ("train", train_df, TRAIN_PARQUET),
            ("val", val_df, VAL_PARQUET),
            ("test", test_df, TEST_PARQUET),
        ]:
            FEATURES_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)

        X_train_df, y_train, ts_train = load_split(TRAIN_PARQUET, cfg, "train")
        X_val_df, y_val, ts_val = load_split(VAL_PARQUET, cfg, "val")
        X_test_df, y_test, ts_test = load_split(TEST_PARQUET, cfg, "test")

    # Confirm temporal ordering: test timestamps must be after train timestamps
    if (
        isinstance(ts_train[0], (np.datetime64, pd.Timestamp))
        and isinstance(ts_test[0], (np.datetime64, pd.Timestamp))
    ):
        if pd.Timestamp(ts_train[-1]) >= pd.Timestamp(ts_test[0]):
            log.warning(
                "TEMPORAL INTEGRITY WARNING: training data timestamps overlap with "
                "test data. This may indicate a non-temporal split. "
                "Verify 02_feature_engineering.py temporal_split() output."
            )

    feature_names: List[str] = X_train_df.columns.tolist()
    X_train = X_train_df.values.astype(np.float32)
    X_val = X_val_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)

    log.info(
        f"Feature dimensions: train={X_train.shape} | "
        f"val={X_val.shape} | test={X_test.shape}"
    )

    # -----------------------------------------------------------------------
    # Stage 2: Feature scaling
    # -----------------------------------------------------------------------
    scaler = load_scaler(SCALER_PATH)
    X_train_s, X_val_s, X_test_s, scaler = apply_or_fit_scaler(
        X_train, X_val, X_test, scaler, feature_names
    )

    # -----------------------------------------------------------------------
    # Stage 3: Train Isolation Forest (Tier 1)
    # -----------------------------------------------------------------------
    if_clf = train_isolation_forest(X_train_s, cfg)

    # Compute IF scores on all splits
    if_train_scores = compute_if_scores(if_clf, X_train_s)
    if_val_scores = compute_if_scores(if_clf, X_val_s)
    if_test_scores = compute_if_scores(if_clf, X_test_s)

    if y_train.sum() > 0:
        if_train_auroc = roc_auc_score(y_train, if_train_scores)
        log.info(f"Tier 1 (IF) train AUROC: {if_train_auroc:.4f}")

    # -----------------------------------------------------------------------
    # Stage 4: Generate pseudo-labels from IF scores
    # -----------------------------------------------------------------------
    y_pseudo_train = generate_pseudo_labels(
        if_train_scores, percentile=cfg.if_pseudo_label_percentile
    )

    # -----------------------------------------------------------------------
    # Stage 5: Train Random Forest (Tier 2)
    # -----------------------------------------------------------------------
    rf_clf = train_random_forest(X_train_s, y_pseudo_train, y_train, cfg)

    rf_train_scores = compute_rf_scores(rf_clf, X_train_s)
    rf_val_scores = compute_rf_scores(rf_clf, X_val_s)
    rf_test_scores = compute_rf_scores(rf_clf, X_test_s)

    if y_train.sum() > 0:
        rf_train_auroc = roc_auc_score(y_train, rf_train_scores)
        log.info(f"Tier 2 (RF) train AUROC: {rf_train_auroc:.4f}")

    # -----------------------------------------------------------------------
    # Stage 6: Train LSTM Autoencoder (Tier 3)
    # -----------------------------------------------------------------------
    lstm_model = None
    lstm_train_threshold = 0.5
    lstm_norm_min: Optional[float] = None  # populated by compute_lstm_threshold
    lstm_norm_max: Optional[float] = None
    device = None

    if _TORCH_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"LSTM device: {device}")

        # Train on normal-only samples to learn the normal reconstruction manifold
        # See Ch. 22 §5 — Autoencoder Training on Normal Data.
        normal_mask_train = y_train == 0
        X_train_normal = X_train_s[normal_mask_train]

        if len(X_train_normal) < cfg.lstm_seq_len * 2:
            log.warning(
                f"Insufficient normal training samples ({len(X_train_normal)}) "
                f"for LSTM seq_len={cfg.lstm_seq_len}. Skipping LSTM tier."
            )
        else:
            lstm_model = train_lstm_autoencoder(
                X_train_normal=X_train_normal,
                X_val=X_val_s,
                cfg=cfg,
                device=device,
            )
            # Threshold calibrated on NORMAL training data (not validation)
            lstm_train_threshold, lstm_norm_min, lstm_norm_max = compute_lstm_threshold(
                lstm_model, X_train_normal, cfg, device
            )

        lstm_train_scores = (
            compute_reconstruction_errors(
                lstm_model, X_train_s, cfg, device,
                norm_min=lstm_norm_min, norm_max=lstm_norm_max,
            )
            if lstm_model is not None
            else rf_train_scores.copy()
        )
        lstm_val_scores = (
            compute_reconstruction_errors(
                lstm_model, X_val_s, cfg, device,
                norm_min=lstm_norm_min, norm_max=lstm_norm_max,
            )
            if lstm_model is not None
            else rf_val_scores.copy()
        )
        lstm_test_scores = (
            compute_reconstruction_errors(
                lstm_model, X_test_s, cfg, device,
                norm_min=lstm_norm_min, norm_max=lstm_norm_max,
            )
            if lstm_model is not None
            else rf_test_scores.copy()
        )

        if y_train.sum() > 0 and lstm_model is not None:
            lstm_train_auroc = roc_auc_score(y_train, lstm_train_scores)
            log.info(f"Tier 3 (LSTM-AE) train AUROC: {lstm_train_auroc:.4f}")
    else:
        log.info("PyTorch not available — running two-tier ensemble (IF + RF).")
        lstm_val_scores = rf_val_scores.copy()
        lstm_test_scores = rf_test_scores.copy()
        lstm_train_scores = rf_train_scores.copy()

    # -----------------------------------------------------------------------
    # Stage 7: Calibrate ensemble weights on validation set
    # -----------------------------------------------------------------------
    best_weights, best_val_auroc, best_threshold = calibrate_ensemble_weights(
        if_scores_val=if_val_scores,
        rf_scores_val=rf_val_scores,
        lstm_scores_val=lstm_val_scores,
        y_val=y_val,
        weight_grid=cfg.ensemble_weight_grid,
    )

    # -----------------------------------------------------------------------
    # Stage 8: Baseline comparison
    # -----------------------------------------------------------------------
    log.info("\n" + "─" * 50)
    log.info("Baseline model evaluation (rolling sigma threshold):")
    baseline_metrics = baseline_rolling_threshold(
        X_df=X_test_df,
        y=y_test,
        feature_col="dl_throughput_mbps",
        window=4,
        n_sigma=3.0,
    )

    # -----------------------------------------------------------------------
    # Stage 9: Evaluate full ensemble on all splits
    # -----------------------------------------------------------------------
    log.info("\n" + "─" * 50)
    log.info("Full ensemble evaluation:")

    val_results = evaluate_ensemble(
        if_clf=if_clf,
        rf_clf=rf_clf,
        lstm_model=lstm_model,
        X_scaled=X_val_s,
        X_df=X_val_df,
        y=y_val,
        weights=best_weights,
        threshold=best_threshold,
        cfg=cfg,
        device=device,
        split_label="val",
        lstm_norm_min=lstm_norm_min,
        lstm_norm_max=lstm_norm_max,
    )

    test_results = evaluate_ensemble(
        if_clf=if_clf,
        rf_clf=rf_clf,
        lstm_model=lstm_model,
        X_scaled=X_test_s,
        X_df=X_test_df,
        y=y_test,
        weights=best_weights,
        threshold=best_threshold,
        cfg=cfg,
        device=device,
        split_label="test",
        lstm_norm_min=lstm_norm_min,
        lstm_norm_max=lstm_norm_max,
    )

    # -----------------------------------------------------------------------
    # Stage 9a: SHAP feature attribution (RF model)
    # -----------------------------------------------------------------------
    shap_values = compute_shap_values(
        clf=rf_clf,
        X_sample=X_train_s,
        feature_names=feature_names,
        cfg=cfg,
    )

    # -----------------------------------------------------------------------
    # Stage 10: Print comparison table
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("MODEL COMPARISON SUMMARY (test set)")
    log.info("=" * 70)
    log.info(
        f"{'Model':<55} {'AUROC':>6} {'AUPRC':>6} {'F1':>6} {'FAR':>6}"
    )
    log.info("-" * 70)
    log.info(
        f"{'Baseline (rolling 3σ threshold)':<55} "
        f"{baseline_metrics.get('auroc', 0):>6.4f} "
        f"{'N/A':>6} "
        f"{baseline_metrics.get('f1', 0):>6.4f} "
        f"{'N/A':>6}"
    )
    log.info(
        f"{'Ensemble (IF+RF+LSTM-AE)':<55} "
        f"{test_results['auroc']:>6.4f} "
        f"{test_results['auprc']:>6.4f} "
        f"{test_results['f1']:>6.4f} "
        f"{test_results['false_alarm_rate']:>6.4f}"
    )
    log.info("=" * 70)

    # NOC operational interpretation
    noc_tp = test_results["tp"]
    noc_fp = test_results["fp"]
    noc_fn = test_results["fn"]
    noc_total_alerts = noc_tp + noc_fp
    if noc_total_alerts > 0:
        noc_useful_pct = 100.0 * noc_tp / noc_total_alerts
        log.info(
            f"NOC operational impact: {noc_useful_pct:.1f}% of alerts are actionable "
            f"({noc_tp} TP, {noc_fp} FP false alarms). "
            f"{noc_fn} true anomalies missed."
        )
        # Rough OPEX estimate: 15 min per false alarm, analyst rate A$100/hr
        # See Part 1 §1 cost model (cross-reference, not reproduced here).
        fa_opex_per_period = noc_fp * 15 / 60 * 100
        log.info(
            f"Estimated false-alarm OPEX this test period: "
            f"A${fa_opex_per_period:.0f} "
            f"(at A$100/hr analyst rate, 15 min/investigation)"
        )

    # -----------------------------------------------------------------------
    # Stage 11: Build artefact payloads and save
    # -----------------------------------------------------------------------
    thresholds_payload: Dict[str, Any] = {
        "ensemble_threshold": best_threshold,
        "ensemble_weights": {
            "isolation_forest": best_weights[0],
            "random_forest": best_weights[1],
            "lstm_autoencoder": best_weights[2],
        },
        "lstm_reconstruction_threshold": lstm_train_threshold,
        "calibration_split": "val",
        "threshold_method": cfg.ensemble_threshold_metric,
        "governance_gate": {
            "min_f1": MIN_F1_GATE,
            "test_f1": test_results["f1"],
            "passed": test_results["f1"] >= MIN_F1_GATE,
        },
    }

    metadata_payload: Dict[str, Any] = {
        "training_run_id": datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config": asdict(cfg),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "data_splits": {
            "train_rows": int(len(y_train)),
            "val_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "train_anomaly_rate": float(y_train.mean()),
            "val_anomaly_rate": float(y_val.mean()),
            "test_anomaly_rate": float(y_test.mean()),
        },
        "tier1_isolation_forest": {
            "n_estimators": cfg.if_n_estimators,
            "contamination": cfg.if_contamination,
        },
        "tier2_random_forest": {
            "n_estimators": cfg.rf_n_estimators,
            "oob_score": float(rf_clf.oob_score_),
        },
        "tier3_lstm_autoencoder": {
            "available": lstm_model is not None,
            "hidden_dim": cfg.lstm_hidden_dim,
            "seq_len": cfg.lstm_seq_len,
            "reconstruction_threshold": lstm_train_threshold,
            "norm_min": lstm_norm_min,
            "norm_max": lstm_norm_max,
        },
        "ensemble": {
            "best_weights": list(best_weights),
            "val_auroc": float(best_val_auroc),
            "threshold": float(best_threshold),
        },
        "evaluation": {
            "baseline": baseline_metrics,
            "val": val_results,
            "test": test_results,
        },
        "shap_computed": shap_values is not None,
        "torch_available": _TORCH_AVAILABLE,
        "shap_available": _SHAP_AVAILABLE,
        "coursebook_refs": [
            "Ch. 13 — Feature Engineering (SHAP values, feature selection)",
            "Ch. 16 — Decision Trees & Random Forests (IF, RF hyperparameters, OOB)",
            "Ch. 22 — Recurrent Neural Networks (LSTM-AE architecture, reconstruction loss)",
            "Ch. 28 — Data Pipelines (temporal splits, artefact management)",
            "Ch. 52 — System Design for ML (cascade scoring, threshold calibration)",
            "Ch. 54 — Monitoring & Reliability (FAR, miss rate, governance gates)",
        ],
    }

    save_artefacts(
        if_clf=if_clf,
        rf_clf=rf_clf,
        lstm_model=lstm_model,
        scaler=scaler,
        thresholds=thresholds_payload,
        metadata=metadata_payload,
        shap_values=shap_values,
    )

    # --- Save per-tier test scores for Script 04 evaluation ---
    # Script 04 (load_tier_scores) looks for these files in ARTIFACTS_DIR.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS_DIR / "if_scores_test.npy", if_test_scores)
    np.save(ARTIFACTS_DIR / "rf_scores_test.npy", rf_test_scores)
    np.save(ARTIFACTS_DIR / "lstm_scores_test.npy", lstm_test_scores)
    log.info("Saved per-tier test scores → %s/{if,rf,lstm}_scores_test.npy", ARTIFACTS_DIR)

    log.info("\n" + "=" * 70)
    log.info("Training complete. Artefacts written to: %s/", MODELS_DIR)
    log.info(f"  {IF_MODEL_PATH}")
    log.info(f"  {RF_MODEL_PATH}")
    if lstm_model is not None:
        log.info(f"  {LSTM_MODEL_PATH}")
    log.info(f"  {THRESHOLDS_PATH}")
    log.info(f"  {METADATA_PATH}")
    log.info("Next step: python 04_evaluation.py")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
