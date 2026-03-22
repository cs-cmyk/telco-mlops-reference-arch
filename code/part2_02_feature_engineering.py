"""
02_feature_engineering.py — Telco MLOps Part 2: Feature Engineering Pipeline
==============================================================================
Companion code for "Telco MLOps Reference Architecture — Part 2: Extending the
Platform to Graph ML, LLMs, Agentic Systems, and Beyond".

PURPOSE:
    Reads the synthetic RAN KPI dataset produced by 01_synthetic_data.py and
    engineers ~150 domain-specific features across five categories:

    1. Temporal encodings     — cyclical (sin/cos) hour/day/week, peak-hour flags,
                                  weekend indicator, ROP-sequence index
    2. Rolling statistics     — mean, std, min, max, range over 1h / 4h / 24h
                                  windows per cell; delta (1-step change); z-scores
                                  over rolling reference window
    3. Rate-of-change / delta — first differences, absolute change, signed change
    4. Cross-KPI ratios       — DL/UL throughput ratio, PRB utilisation efficiency,
                                  RSRQ-relative-to-RSRP spread, CQI-to-throughput
                                  consistency, HO attempt/success ratio
    5. Spatial peer-group     — DBSCAN-style cluster assignments (loaded from
                                  inventory), within-cluster z-scores for every KPI,
                                  peer-group rank percentile

    Additionally re-engineers all features described in Part 1 §6 so that the
    companion graph ML, LLM, and agentic code (Parts 3–5) can reference the
    exact same column names.

OUTPUT:
    data/features/train.parquet
    data/features/val.parquet
    data/features/test.parquet
    data/features/feature_catalog.json   — name, dtype, category, description
    data/features/split_metadata.json    — cut dates, row counts per split

TEMPORAL SPLIT STRATEGY:
    We use a strict chronological split, never random.  Time-series data must
    never be shuffled before splitting — doing so causes target leakage because
    rolling features computed on the full dataset will incorporate future values.

    See Coursebook Ch. 13: Feature Engineering
        Coursebook Ch. 28: Data Pipelines
        Coursebook Ch. 52: System Design for ML (point-in-time correctness)

USAGE:
    # Generate raw data first:
    python 01_synthetic_data.py

    # Then run this script:
    python 02_feature_engineering.py

    # Optional overrides via environment variables:
    TELCO_RAW_DATA=data/pm_counters.parquet
    TELCO_FEATURES_DIR=data/features/
    TELCO_TRAIN_FRAC=0.70
    TELCO_VAL_FRAC=0.15

REQUIREMENTS:
    Python 3.10+
    pandas>=2.0, numpy>=1.24, scikit-learn>=1.4, scipy>=1.11, pyarrow>=14
"""

# ============================================================================
# Imports
# ============================================================================

import json
import joblib
import logging
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore as scipy_zscore
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ============================================================================
# Logging configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("feature_engineering")

# ============================================================================
# Constants — column name registry
# All downstream scripts (03–05) must import from this module to ensure
# consistent naming.  Never hardcode column names in later scripts.
# ============================================================================

# --- Raw KPI columns produced by 01_synthetic_data.py ---
COL_CELL_ID = "cell_id"
COL_SITE_ID = "site_id"
COL_TIMESTAMP = "timestamp"
COL_ENVIRONMENT = "environment"
COL_CLUSTER_ID = "cluster_id"
COL_LAT = "lat"
COL_LON = "lon"

# Radio-quality KPIs
COL_RSRP = "rsrp_dbm"
COL_RSRQ = "rsrq_db"
COL_SINR = "sinr_db"
COL_CQI = "avg_cqi"

# Capacity KPIs
COL_DL_TPUT = "dl_throughput_mbps"
COL_UL_TPUT = "ul_throughput_mbps"
COL_PRB_DL = "dl_prb_usage_rate"
COL_PRB_UL = "ul_prb_usage_rate"  # Column names must match §3.2 KPI Taxonomy — do not rename without updating the whitepaper
COL_ACTIVE_UE = "active_ue_count"

# Mobility / availability KPIs
COL_RRC_SETUP_SR = "rrc_conn_setup_success_rate"
COL_HO_ATTEMPT = "ho_attempt_count"    # Production PM exports only; NOT in Script 01 synthetic data
COL_HO_SUCCESS = "ho_success_count"    # Production PM exports only; NOT in Script 01 synthetic data
COL_CELL_AVAIL = "cell_availability_pct"

# Labels
COL_IS_ANOMALY = "is_anomaly"
COL_ANOMALY_TYPE = "anomaly_type"

# Derived columns (added by compute_derived_kpis in this script)
# COL_HO_SR: derived from attempt/success counts if available, or passed
# through directly from Script 01's pre-computed handover_success_rate.
COL_HO_SR = "handover_success_rate"
COL_DL_UL_RATIO = "dl_ul_tput_ratio"
COL_PRB_EFF = "prb_efficiency_mbps_per_pct"
COL_CQI_TPUT_RATIO = "cqi_tput_consistency"
COL_RSRQ_RSRP_SPREAD = "rsrq_rsrp_spread"
COL_HOP_INDEX = "rop_index"

# All raw numeric KPIs that receive rolling / delta treatment
CORE_KPIS: List[str] = [
    COL_RSRP,
    COL_RSRQ,
    COL_SINR,
    COL_CQI,
    COL_DL_TPUT,
    COL_UL_TPUT,
    COL_PRB_DL,
    COL_PRB_UL,
    COL_ACTIVE_UE,
    COL_RRC_SETUP_SR,
    COL_CELL_AVAIL,
    COL_HO_SR,  # Always present from Script 01 as a pre-computed rate.
    # Note: COL_HO_ATTEMPT and COL_HO_SUCCESS are defined as constants above
    # but are NOT produced by Script 01's synthetic data generator.  Script 01
    # produces handover_success_rate (COL_HO_SR) directly as a pre-computed
    # rate.  Operators with access to raw HO attempt/success counters from
    # production PM exports should add them here and use the derivation path
    # in compute_derived_kpis().
]

# Rolling window sizes expressed in number of 15-minute ROPs
# 1h = 4 ROPs, 4h = 16 ROPs, 24h = 96 ROPs
WINDOW_1H = 4
WINDOW_4H = 16
WINDOW_24H = 96

# Minimum periods to avoid NaN-heavy early rows (require at least 50% fill)
MIN_PERIODS_1H = 2
MIN_PERIODS_4H = 8
MIN_PERIODS_24H = 48

# ============================================================================
# Path configuration
# ============================================================================

input_path = os.environ.get("TELCO_RAW_DATA", "data/pm_counters.parquet")
output_dir = os.environ.get("TELCO_FEATURES_DIR", "data/features/")

# ============================================================================
# Configuration dataclass
# ============================================================================

from dataclasses import dataclass, field  # noqa: E402 (after constants for readability)


@dataclass
class FeatureConfig:
    """All tuneable parameters for the feature engineering pipeline.

    Keeping configuration in a dataclass (rather than scattered literals)
    makes hyperparameter searches and configuration management tractable.
    See Coursebook Ch. 52: System Design for ML — configuration management.
    """

    raw_data_path: Path = Path("data/raw/ran_kpi.parquet")
    output_dir: Path = Path("data/features")
    inventory_path: Path = Path("data/raw/cell_inventory.parquet")
    neighbour_path: Path = Path("data/raw/neighbour_relations.parquet")

    # Chronological split fractions (must sum to 1.0)
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac is implicitly 1 - train_frac - val_frac = 0.15

    # Rolling windows (in ROP units of 15 min)
    windows: List[int] = field(default_factory=lambda: [WINDOW_1H, WINDOW_4H, WINDOW_24H])

    # Peer-group z-score clipping (prevents extreme outliers from dominating)
    zscore_clip: float = 5.0

    # Throughput ratio cap (DL/UL) — beyond this is almost certainly a counter
    # anomaly rather than a real signal
    dl_ul_ratio_cap: float = 50.0

    # Minimum rows per cell required for peer-group statistics to be meaningful
    min_rows_per_cell: int = 48  # 12 hours of 15-min ROPs

    # Whether to persist the StandardScaler for serving-time feature normalisation
    # The scaler is fitted on train only and applied to val/test to prevent leakage
    persist_scaler: bool = True

    # Seed for any stochastic steps (none currently, but kept for reproducibility)
    seed: int = 42

    @classmethod
    def from_env(cls) -> "FeatureConfig":
        """Override defaults from environment variables (CI/CD injection)."""
        cfg = cls()
        if p := os.environ.get("TELCO_RAW_DATA"):
            cfg.raw_data_path = Path(p)
        if p := os.environ.get("TELCO_FEATURES_DIR"):
            cfg.output_dir = Path(p)
        if p := os.environ.get("TELCO_TRAIN_FRAC"):
            cfg.train_frac = float(p)
        if p := os.environ.get("TELCO_VAL_FRAC"):
            cfg.val_frac = float(p)
        return cfg


# ============================================================================
# I/O helpers
# ============================================================================


def load_raw_data(cfg: FeatureConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three artefacts produced by 01_synthetic_data.py.

    Returns:
        kpi_df        — full KPI time-series DataFrame
        inventory_df  — one row per cell with static attributes
        neighbour_df  — pairwise neighbour relation table

    Raises:
        FileNotFoundError if any required file is missing.
    """
    logger.info("Loading raw data from %s", cfg.raw_data_path)

    if not cfg.raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw KPI data not found at {cfg.raw_data_path}. "
            "Run 01_synthetic_data.py first."
        )

    kpi_df = pd.read_parquet(cfg.raw_data_path)
    logger.info("  KPI rows: %d, columns: %d", len(kpi_df), kpi_df.shape[1])

    # Enforce timestamp dtype — parquet should preserve this, but be defensive
    if not pd.api.types.is_datetime64_any_dtype(kpi_df[COL_TIMESTAMP]):
        kpi_df[COL_TIMESTAMP] = pd.to_datetime(kpi_df[COL_TIMESTAMP], utc=True)

    # Sort is mandatory before any rolling operation
    kpi_df = kpi_df.sort_values([COL_CELL_ID, COL_TIMESTAMP]).reset_index(drop=True)

    # Load inventory (static cell attributes including cluster assignments)
    if cfg.inventory_path.exists():
        inventory_df = pd.read_parquet(cfg.inventory_path)
        logger.info("  Inventory rows: %d", len(inventory_df))
    else:
        logger.warning(
            "Inventory file not found at %s — peer-group features will use "
            "environment-based clusters only.",
            cfg.inventory_path,
        )
        inventory_df = _build_minimal_inventory(kpi_df)

    # Load neighbour relations (for graph-native features used in Part 2 §4)
    if cfg.neighbour_path.exists():
        neighbour_df = pd.read_parquet(cfg.neighbour_path)
        logger.info("  Neighbour relation rows: %d", len(neighbour_df))
    else:
        logger.warning(
            "Neighbour relations file not found at %s — "
            "graph-native features will be skipped.",
            cfg.neighbour_path,
        )
        neighbour_df = pd.DataFrame(columns=["source_cell_id", "target_cell_id", "distance_km"])

    return kpi_df, inventory_df, neighbour_df


def _build_minimal_inventory(kpi_df: pd.DataFrame) -> pd.DataFrame:
    """Construct a minimal inventory from the KPI frame when the parquet is absent.

    In production the inventory is a permanent artefact from O1 provisioning data.
    This fallback exists only for test environments where only the KPI file is
    available (e.g., CI pipelines running against a single parquet download).
    """
    unique_cells = kpi_df[[COL_CELL_ID, COL_SITE_ID, COL_ENVIRONMENT]].drop_duplicates()
    # Assign clusters by environment as a coarse fallback
    env_to_cluster = {env: i for i, env in enumerate(unique_cells[COL_ENVIRONMENT].unique())}
    unique_cells = unique_cells.copy()
    unique_cells["cluster_id"] = unique_cells[COL_ENVIRONMENT].map(env_to_cluster)
    return unique_cells


def save_split(df: pd.DataFrame, path: Path, label: str) -> None:
    """Write a feature DataFrame to Parquet with compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    mb = path.stat().st_size / (1024 * 1024)
    logger.info("  Saved %s: %d rows, %d cols, %.1f MB → %s", label, len(df), df.shape[1], mb, path)


# ============================================================================
# Step 1 — Derived base KPIs (pre-rolling, applied per-row)
# These are computed BEFORE rolling to avoid encoding future information.
# See Coursebook Ch. 13: Feature Engineering §14.2 Derived Features
# ============================================================================


def compute_derived_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Add single-row derived KPIs that combine multiple raw counters.

    These are domain-motivated ratios / combinations that capture relationships
    that individual KPIs cannot express alone.  All operations are vectorised
    (no rolling) so no temporal information leaks across rows.

    Design decisions:
        - HO success rate is the primary mobility health signal.  Two input
          paths are supported: (a) raw attempt/success counts from production
          PM exports, or (b) a pre-computed rate from Script 01's synthetic
          data.  Path (b) is the default for the companion code.
        - DL/UL ratio captures asymmetric loading (video streaming → high DL).
        - PRB efficiency normalises throughput against resource consumption,
          making cells comparable regardless of traffic load.
        - CQI-to-throughput consistency detects misconfigured MCS tables where
          CQI and achieved throughput are decoupled.
        - RSRQ-RSRP spread approximates the interference loading (high spread
          means strong reference signals but heavy interference from neighbours).
    """
    df = df.copy()

    # --- Handover success rate ---
    # Two paths:
    #   (a) Production PM data: raw ho_attempt_count and ho_success_count columns
    #       are present → derive the rate from counts.
    #   (b) Script 01 synthetic data: handover_success_rate is pre-computed
    #       directly; attempt/success count columns do not exist.
    if COL_HO_ATTEMPT in df.columns and COL_HO_SUCCESS in df.columns:
        # Path (a): derive from raw counts
        safe_attempt = df[COL_HO_ATTEMPT].replace(0, np.nan)
        df[COL_HO_SR] = (df[COL_HO_SUCCESS] / safe_attempt * 100).clip(0, 100)
        df[COL_HO_SR] = df[COL_HO_SR].fillna(100.0)
        logger.info("  HO success rate derived from attempt/success counts.")
    elif COL_HO_SR in df.columns:
        # Path (b): already present from Script 01 — validate and pass through
        logger.info("  HO success rate already present (pre-computed by Script 01).")
    else:
        # Neither path available — set to 100.0 (no degradation assumed)
        logger.warning(
            "Neither %s/%s (raw counts) nor %s (pre-computed rate) found. "
            "Setting %s to 100.0 (no degradation assumed).",
            COL_HO_ATTEMPT, COL_HO_SUCCESS, COL_HO_SR, COL_HO_SR,
        )
        df[COL_HO_SR] = 100.0

    # --- DL / UL throughput ratio ---
    safe_ul = df[COL_UL_TPUT].replace(0, np.nan)
    df[COL_DL_UL_RATIO] = (df[COL_DL_TPUT] / safe_ul).clip(upper=50.0)
    df[COL_DL_UL_RATIO] = df[COL_DL_UL_RATIO].fillna(1.0)  # idle cells: ratio = 1

    # --- PRB utilisation efficiency (Mbps per % of PRB consumed) ---
    safe_prb = df[COL_PRB_DL].replace(0, np.nan)
    df[COL_PRB_EFF] = (df[COL_DL_TPUT] / safe_prb).clip(upper=20.0)
    df[COL_PRB_EFF] = df[COL_PRB_EFF].fillna(0.0)

    # --- CQI-to-throughput consistency ---
    # Normalise CQI to [0,1] scale and compare against normalised DL throughput.
    # Values near 0 indicate CQI predicts good channel but throughput is low
    # (possible scheduler misconfiguration or interference).
    max_cqi = 15.0
    # We use a rolling max of DL throughput to normalise — but that's forward-
    # looking if we use the full series.  Instead, use the 95th percentile of
    # DL throughput *across the entire training set* as a stable normaliser.
    # This value is computed later in the pipeline and back-filled.  Here we
    # compute the raw numerator/denominator for later normalisation.
    df["_cqi_norm"] = df[COL_CQI] / max_cqi  # [0, 1]
    df["_dl_tput_raw"] = df[COL_DL_TPUT]     # normalised later in fit_transform

    # --- RSRQ-RSRP spread (interference loading proxy) ---
    # Both in dB; spread = RSRP - (-RSRQ) adjusted to positive scale.
    # A large positive value means RSRP is strong but RSRQ is still low
    # → many interferers.  Negative values are physically implausible
    # (filtered in validation).
    df[COL_RSRQ_RSRP_SPREAD] = df[COL_RSRP] - df[COL_RSRQ]

    # --- Sequential ROP index within each cell (used as a trend feature) ---
    # This captures long-term secular trends (e.g., gradual cell degradation
    # over weeks) that rolling windows might miss.
    df[COL_HOP_INDEX] = df.groupby(COL_CELL_ID).cumcount()

    logger.info("  Derived KPIs added: %s", [COL_HO_SR, COL_DL_UL_RATIO, COL_PRB_EFF,
                                               COL_RSRQ_RSRP_SPREAD, COL_HOP_INDEX])
    return df


# ============================================================================
# Step 5 — Neighbour aggregate features
# Uses explicit cell_id matching via .loc to avoid positional index errors.
# See Coursebook Ch. 13: Feature Engineering §14.5 Graph / Spatial Features
# ============================================================================


def compute_neighbour_aggregates(
    kpi_df: pd.DataFrame,
    neighbour_df: pd.DataFrame,
    agg_kpis: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Reference implementation: explicit index-based neighbour lookup.

    .. warning:: **This function is NOT called by ``main()``.**
       The production pipeline uses ``compute_neighbour_aggregate_features()``
       (below), which uses a merge-based approach with O(E×T) complexity.
       This function is retained as a reference for operators who need to
       debug neighbour-aggregate correctness on small subsets — its O(N²)
       row-by-row loop is clear but will timeout on >100K rows.

    For each cell and each ROP timestamp, this function looks up the set of
    neighbouring cells from ``neighbour_df``, retrieves their KPI values for
    the *same* timestamp from ``kpi_df``, and computes summary statistics
    (mean, min, max, std).  The result is joined back onto ``kpi_df``.

    Critical implementation note — index safety
    -------------------------------------------
    ``kpi_df`` is sorted by (cell_id, timestamp) and has a RangeIndex reset
    after sorting.  The adjacency list in ``neighbour_df`` stores *cell_id*
    strings, not integer positions.  We therefore always look up neighbour
    rows using ``.loc`` with a boolean cell_id mask, never with positional
    ``iloc`` offsets derived from the adjacency list.  This prevents
    off-by-one errors when cells are added, removed, or reordered between
    pipeline runs.

    Args:
        kpi_df:       Full KPI DataFrame indexed by (cell_id, timestamp) rows.
                      Must contain COL_CELL_ID and COL_TIMESTAMP columns.
        neighbour_df: Adjacency table with columns ``source_cell_id``,
                      ``target_cell_id``, and optionally ``distance_km``.
        agg_kpis:     KPI columns to aggregate over neighbours.  Defaults to
                      a compact subset appropriate for graph feature use.

    Returns:
        ``kpi_df`` with additional columns named
        ``nbr_{kpi}_mean``, ``nbr_{kpi}_min``, ``nbr_{kpi}_max``,
        ``nbr_{kpi}_std`` for every KPI in ``agg_kpis``.
    """
    if agg_kpis is None:
        agg_kpis = [COL_RSRP, COL_SINR, COL_PRB_DL, COL_DL_TPUT, COL_CQI]

    if neighbour_df.empty:
        logger.warning("neighbour_df is empty — skipping neighbour aggregate features.")
        return kpi_df

    logger.info("Computing neighbour aggregate features for %d KPIs …", len(agg_kpis))

    # ------------------------------------------------------------------
    # Build a mapping: source_cell_id → list[target_cell_id]
    # We use cell_id strings throughout — no integer positional indices.
    # ------------------------------------------------------------------
    adjacency: Dict[str, List[str]] = (
        neighbour_df.groupby("source_cell_id")["target_cell_id"]
        .apply(list)
        .to_dict()
    )

    # ------------------------------------------------------------------
    # Build a cell_id → timestamp → feature lookup structure.
    # We set a MultiIndex on (cell_id, timestamp) so that .loc[(cid, ts)]
    # retrieves the exact row for a given cell at a given ROP without any
    # positional assumption.
    # ------------------------------------------------------------------
    lookup_df = kpi_df.set_index([COL_CELL_ID, COL_TIMESTAMP])[agg_kpis]

    # Pre-allocate result containers: one list per (kpi, stat)
    stat_names = ["mean", "min", "max", "std"]
    result_cols: Dict[str, List[float]] = {
        f"nbr_{kpi}_{stat}": []
        for kpi in agg_kpis
        for stat in stat_names
    }

    # ------------------------------------------------------------------
    # Iterate over every row in kpi_df.
    # For each row we:
    #   1. Identify the source cell_id and timestamp.
    #   2. Look up its neighbours by cell_id (not by position).
    #   3. Retrieve neighbour rows for the *same* timestamp using .loc
    #      with explicit (cell_id, timestamp) index keys.
    #   4. Compute aggregate statistics and append to result_cols.
    #
    # Performance note: for datasets with >100 K rows consider vectorising
    # this via a merge-based approach (see inline comment below).
    # ------------------------------------------------------------------
    all_timestamps = kpi_df[COL_TIMESTAMP].values
    all_cell_ids = kpi_df[COL_CELL_ID].values

    for row_pos in range(len(kpi_df)):
        src_cell_id: str = all_cell_ids[row_pos]
        ts = all_timestamps[row_pos]

        neighbour_cell_ids: List[str] = adjacency.get(src_cell_id, [])

        if neighbour_cell_ids:
            # Build MultiIndex keys for .loc lookup — explicit cell_id matching,
            # never positional.  Only include neighbours that actually appear in
            # the lookup index to avoid KeyErrors on sparse graphs.
            valid_keys = [
                (nbr_cid, ts)
                for nbr_cid in neighbour_cell_ids
                if (nbr_cid, ts) in lookup_df.index
            ]
            if valid_keys:
                nbr_rows = lookup_df.loc[valid_keys]
            else:
                nbr_rows = pd.DataFrame(columns=agg_kpis)
        else:
            nbr_rows = pd.DataFrame(columns=agg_kpis)

        for kpi in agg_kpis:
            if nbr_rows.empty or kpi not in nbr_rows.columns:
                result_cols[f"nbr_{kpi}_mean"].append(np.nan)
                result_cols[f"nbr_{kpi}_min"].append(np.nan)
                result_cols[f"nbr_{kpi}_max"].append(np.nan)
                result_cols[f"nbr_{kpi}_std"].append(np.nan)
            else:
                vals = nbr_rows[kpi].dropna()
                if vals.empty:
                    result_cols[f"nbr_{kpi}_mean"].append(np.nan)
                    result_cols[f"nbr_{kpi}_min"].append(np.nan)
                    result_cols[f"nbr_{kpi}_max"].append(np.nan)
                    result_cols[f"nbr_{kpi}_std"].append(np.nan)
                else:
                    result_cols[f"nbr_{kpi}_mean"].append(float(vals.mean()))
                    result_cols[f"nbr_{kpi}_min"].append(float(vals.min()))
                    result_cols[f"nbr_{kpi}_max"].append(float(vals.max()))
                    result_cols[f"nbr_{kpi}_std"].append(float(vals.std(ddof=0)))

    # Attach result columns to a copy of kpi_df preserving original index
    out_df = kpi_df.copy()
    for col_name, values in result_cols.items():
        out_df[col_name] = values

    n_new_cols = len(result_cols)
    logger.info("  Added %d neighbour aggregate columns.", n_new_cols)
    return out_df


# ============================================================================
# Step 2 — Temporal features
# See Coursebook Ch. 13: Feature Engineering §14.4 Temporal Encoding
# ============================================================================


def encode_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical and categorical temporal features to every row.

    Cyclical encoding (sin/cos) is preferred over raw integer encoding for
    periodic features because it preserves the circular distance property:
    hour 23 and hour 0 are adjacent, not maximally different.  This matters
    for tree models less (they learn splits), but matters for distance-based
    models (IF, KNN) and neural networks (LSTM) used in Part 1.

    See Coursebook Ch. 13: Feature Engineering §14.3 Time-Based Features

    Note: All timestamp operations use tz-aware timestamps.  The raw data from
    01_synthetic_data.py uses UTC.  If the operator's NOC is in a local
    timezone, convert BEFORE this step.  We keep UTC here as the reference.
    """
    df = df.copy()

    ts = df[COL_TIMESTAMP]

    # --- Hour of day [0, 23] ---
    df["hour_of_day"] = ts.dt.hour.astype(np.int8)

    # Cyclical encoding — hour
    df["hour_sin"] = np.sin(2 * math.pi * df["hour_of_day"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * math.pi * df["hour_of_day"] / 24).astype(np.float32)

    # --- Day of week [0=Monday, 6=Sunday] ---
    df["day_of_week"] = ts.dt.dayofweek.astype(np.int8)

    # Cyclical encoding — day of week
    df["dow_sin"] = np.sin(2 * math.pi * df["day_of_week"] / 7).astype(np.float32)
    df["dow_cos"] = np.cos(2 * math.pi * df["day_of_week"] / 7).astype(np.float32)

    # --- Day of year [1, 365/366] for seasonal patterns ---
    df["day_of_year"] = ts.dt.dayofyear.astype(np.int16)
    df["doy_sin"] = np.sin(2 * math.pi * df["day_of_year"] / 365).astype(np.float32)
    df["doy_cos"] = np.cos(2 * math.pi * df["day_of_year"] / 365).astype(np.float32)

    # --- Week of year [1, 52/53] for medium-term seasonality ---
    df["week_of_year"] = ts.dt.isocalendar().week.astype(np.int8)

    # --- Boolean flags ---
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)

    # Peak hours: typical definitions for mobile networks
    # Morning peak 7–9, evening peak 17–21, late-night trough 0–5
    # These thresholds should be calibrated per operator/market
    df["is_morning_peak"] = ((df["hour_of_day"] >= 7) & (df["hour_of_day"] <= 9)).astype(np.int8)
    df["is_evening_peak"] = ((df["hour_of_day"] >= 17) & (df["hour_of_day"] <= 21)).astype(np.int8)
    df["is_peak_hour"] = (df["is_morning_peak"] | df["is_evening_peak"]).astype(np.int8)
    df["is_business_hours"] = (
        (df["hour_of_day"] >= 8) & (df["hour_of_day"] <= 18) & (df["is_weekend"] == 0)
    ).astype(np.int8)

    # --- ROP within day [0, 95] — 15-min slots ---
    # Captures intra-day position more granularly than hour_of_day
    df["rop_of_day"] = (df["hour_of_day"] * 4 + ts.dt.minute // 15).astype(np.int8)
    df["rop_sin"] = np.sin(2 * math.pi * df["rop_of_day"] / 96).astype(np.float32)
    df["rop_cos"] = np.cos(2 * math.pi * df["rop_of_day"] / 96).astype(np.float32)

    temporal_cols = [
        "hour_of_day", "hour_sin", "hour_cos",
        "day_of_week", "dow_sin", "dow_cos",
        "day_of_year", "doy_sin", "doy_cos",
        "week_of_year", "is_weekend", "is_morning_peak",
        "is_evening_peak", "is_peak_hour", "is_business_hours",
        "rop_of_day", "rop_sin", "rop_cos",
    ]
    logger.info("  Temporal features added (%d): %s", len(temporal_cols), temporal_cols[:6])
    return df


# ============================================================================
# Step 3 — Rolling statistics (per-cell, chronological)
# See Coursebook Ch. 13: Feature Engineering §14.5 Rolling Aggregations
# CRITICAL: These must be computed WITHIN each cell_id group only, and the
# DataFrame must be sorted by (cell_id, timestamp) before this step.
# Rolling across cell boundaries would produce nonsensical features.
# ============================================================================


def compute_rolling_features(
    df: pd.DataFrame,
    kpi_cols: Optional[List[str]] = None,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Compute rolling mean, std, min, max, and range for each KPI.

    Uses pandas groupby + rolling with min_periods to avoid NaN-heavy leading
    rows causing model instability.  The closed='left' parameter ensures that
    the current ROP is NOT included in the window — strictly historical look-
    back.  This is critical to avoid leakage where the current anomalous value
    inflates its own rolling mean.

    Args:
        df:        DataFrame sorted by (cell_id, timestamp). MUST be pre-sorted.
        kpi_cols:  KPI columns to process.  Defaults to CORE_KPIS + derived.
        windows:   Rolling window sizes in ROP units.  Defaults to [4, 16, 96].

    Returns:
        DataFrame with additional rolling feature columns.

    Performance note:
        For production Flink streaming, rolling features are computed
        incrementally using Flink's ProcessFunction with keyed state
        (one state object per cell_id).  The batch-mode pandas approach here
        is equivalent but cannot run in streaming.  See Part 2 §12 (online
        learning) for the streaming adaptation.
    """
    if kpi_cols is None:
        kpi_cols = CORE_KPIS + [COL_DL_UL_RATIO, COL_PRB_EFF,
                                  COL_RSRQ_RSRP_SPREAD]
    if windows is None:
        windows = [WINDOW_1H, WINDOW_4H, WINDOW_24H]

    window_labels = {WINDOW_1H: "1h", WINDOW_4H: "4h", WINDOW_24H: "24h"}
    min_periods_map = {
        WINDOW_1H: MIN_PERIODS_1H,
        WINDOW_4H: MIN_PERIODS_4H,
        WINDOW_24H: MIN_PERIODS_24H,
    }

    df = df.copy()
    new_cols: List[str] = []

    # Group once — expensive for large DataFrames; do NOT re-group inside the
    # inner loop.  The groupby object is reused across all KPIs and windows.
    grouped = df.groupby(COL_CELL_ID, sort=False)

    for kpi in kpi_cols:
        if kpi not in df.columns:
            logger.warning("  Skipping rolling for absent column: %s", kpi)
            continue

        kpi_series = df[kpi]

        for w in windows:
            wlabel = window_labels.get(w, f"{w}rop")
            min_p = min_periods_map.get(w, max(1, w // 2))

            # closed='left' means window includes [t-w, t-1], not t.
            # This is the "strictly historical" look-back required for
            # production feature computation where the current ROP is live.
            roll = grouped[kpi].rolling(window=w, min_periods=min_p, closed="left")

            col_mean = f"{kpi}_roll{wlabel}_mean"
            col_std  = f"{kpi}_roll{wlabel}_std"
            col_min  = f"{kpi}_roll{wlabel}_min"
            col_max  = f"{kpi}_roll{wlabel}_max"
            col_rng  = f"{kpi}_roll{wlabel}_range"

            # Reset index to align with df.index after groupby rolling
            df[col_mean] = roll.mean().reset_index(level=0, drop=True)
            df[col_std]  = roll.std().reset_index(level=0, drop=True).fillna(0.0)
            df[col_min]  = roll.min().reset_index(level=0, drop=True)
            df[col_max]  = roll.max().reset_index(level=0, drop=True)
            df[col_rng]  = (df[col_max] - df[col_min]).clip(lower=0)

            new_cols.extend([col_mean, col_std, col_min, col_max, col_rng])

    logger.info(
        "  Rolling features added: %d columns across %d KPIs × %d windows × 5 stats",
        len(new_cols), len(kpi_cols), len(windows),
    )
    return df


# ============================================================================
# Step 4 — Rate-of-change / delta features
# See Coursebook Ch. 13: Feature Engineering §14.6 Lag and Lead Features
# ============================================================================


def compute_delta_features(
    df: pd.DataFrame,
    kpi_cols: Optional[List[str]] = None,
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Compute first differences and lag values for key KPIs.

    Delta features capture velocity (rate of change) which is critical for
    detecting rapidly degrading cells — the anomaly detector in Part 1 uses
    both the absolute level AND the rate of change to improve recall on fast
    degradation events.

    Also computes:
        - Signed change (delta itself)
        - Absolute change (magnitude regardless of direction)
        - Direction indicator (−1, 0, +1) as a categorical signal

    Design decision on lags:
        We compute 1-step and 4-step (1h) deltas.  Longer lags (24h delta)
        capture day-over-day drift but also reduce the number of usable rows
        due to NaN fill at the start of each cell's series.  The 96-step (24h)
        lag is too sparse for the default 7-day synthetic dataset and is
        therefore included only when the cell has >= MIN_ROWS_FOR_24H_DELTA
        rows.
    """
    MIN_ROWS_FOR_24H_DELTA = 200  # ~50 hours of data minimum for 24h lag

    if kpi_cols is None:
        kpi_cols = [
            COL_RSRP, COL_SINR, COL_DL_TPUT, COL_PRB_DL,
            COL_RRC_SETUP_SR, COL_CELL_AVAIL, COL_HO_SR,
        ]
    if lags is None:
        lags = [1, WINDOW_1H]  # 1 ROP (15 min) and 4 ROPs (1 hour)

    df = df.copy()
    new_cols: List[str] = []

    # Determine which cells have enough rows for a 24h lag
    cell_row_counts = df.groupby(COL_CELL_ID).size()
    cells_with_long_history = set(
        cell_row_counts[cell_row_counts >= MIN_ROWS_FOR_24H_DELTA].index
    )
    include_24h_lag = len(cells_with_long_history) > len(cell_row_counts) * 0.5

    effective_lags = lags[:]
    if include_24h_lag:
        effective_lags.append(WINDOW_24H)  # 96 ROPs = 24h

    for kpi in kpi_cols:
        if kpi not in df.columns:
            logger.warning("  Skipping delta for absent column: %s", kpi)
            continue

        for lag in effective_lags:
            lag_label = f"{lag}rop" if lag != WINDOW_1H else "1h"
            if lag == WINDOW_24H:
                lag_label = "24h"

            # Shift within each cell group (NaN fill at boundaries)
            prev_val = df.groupby(COL_CELL_ID)[kpi].shift(lag)

            col_delta  = f"{kpi}_delta_{lag_label}"
            col_abs    = f"{kpi}_abs_change_{lag_label}"
            col_dir    = f"{kpi}_direction_{lag_label}"

            df[col_delta] = (df[kpi] - prev_val).astype(np.float32)
            df[col_abs]   = df[col_delta].abs().astype(np.float32)
            df[col_dir]   = np.sign(df[col_delta]).fillna(0).astype(np.int8)

            new_cols.extend([col_delta, col_abs, col_dir])

    logger.info("  Delta features added: %d columns across %d KPIs × %d lags × 3 stats",
                len(new_cols), len(kpi_cols), len(effective_lags))
    return df


# ============================================================================
# Step 5 — Rolling z-scores (deviation from cell's own recent mean)
# See Coursebook Ch. 13: Feature Engineering §14.7 Statistical Normalisation
# ============================================================================


def compute_rolling_zscores(
    df: pd.DataFrame,
    kpi_cols: Optional[List[str]] = None,
    window: int = WINDOW_24H,
    zscore_clip: float = 5.0,
) -> pd.DataFrame:
    """Compute per-cell rolling z-scores for key KPIs.

    A rolling z-score of the current value relative to the cell's own recent
    24h distribution provides a cell-normalised signal that is robust to
    systematic differences between cells (e.g., a downtown cell will always
    have higher throughput than a rural cell — the z-score captures whether
    it is anomalous *relative to itself*).

    Formula: z_t = (x_t - rolling_mean_{t-window:t-1}) / rolling_std_{t-window:t-1}

    The 24h window is the primary reference distribution.  A value > 3 standard
    deviations from the recent mean is a strong anomaly signal regardless of
    absolute level.

    Clipping at ±zscore_clip prevents exploding gradients in neural models and
    prevents tree splits from being dominated by extreme outliers.

    Note: This produces a self-normalised feature equivalent to the "isolation
    score" dimension in Part 1's Isolation Forest.  Using both improves recall
    on gradual degradation events that IF misses.
    """
    if kpi_cols is None:
        kpi_cols = CORE_KPIS + [COL_DL_UL_RATIO, COL_PRB_EFF]

    df = df.copy()
    min_p = max(2, window // 4)  # at least 25% fill required
    new_cols: List[str] = []

    grouped = df.groupby(COL_CELL_ID, sort=False)

    for kpi in kpi_cols:
        if kpi not in df.columns:
            continue

        col_z = f"{kpi}_zscore_24h"

        roll = grouped[kpi].rolling(window=window, min_periods=min_p, closed="left")
        roll_mean = roll.mean().reset_index(level=0, drop=True)
        roll_std  = roll.std().reset_index(level=0, drop=True).replace(0, np.nan)

        raw_z = (df[kpi] - roll_mean) / roll_std
        df[col_z] = raw_z.clip(-zscore_clip, zscore_clip).fillna(0).astype(np.float32)
        new_cols.append(col_z)

    logger.info("  Rolling z-score features added: %d", len(new_cols))
    return df


# ============================================================================
# Step 6 — Peer-group (spatial) features
# See Coursebook Ch. 13: Feature Engineering §14.8 Cross-Entity Aggregation
# This replicates Part 1 §6 peer-group normalisation in batch-mode Python.
# ============================================================================


def compute_peer_group_features(
    df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    kpi_cols: Optional[List[str]] = None,
    zscore_clip: float = 5.0,
    min_peers: int = 3,
) -> pd.DataFrame:
    """Compute within-cluster z-scores and rank percentiles.

    Peer-group z-scores normalise each cell against other cells in the same
    DBSCAN cluster (assigned in 01_synthetic_data.py based on geographic
    proximity and similar load profile).  This answers the question:
    "Is this cell anomalous relative to its peers at this ROP?"

    A cell scoring −2 standard deviations below its peers on RSRP at the same
    timestamp is a stronger anomaly signal than −2 std below its own 24h mean,
    because it rules out correlated external factors (e.g., heavy rain
    attenuating all cells in the cluster equally).

    Implementation note:
        For streaming Flink production, the peer-group mean/std for each ROP
        is maintained as keyed state per (cluster_id, rop_of_day) using
        Flink's MapState.  Here we do the equivalent in pandas using groupby
        (cluster_id, timestamp).

    Args:
        df:            Feature DataFrame with COL_CLUSTER_ID populated.
        inventory_df:  Cell inventory with cluster assignments.
        kpi_cols:      KPIs to compute peer-group z-scores for.
        zscore_clip:   Z-score clipping threshold.
        min_peers:     Minimum peers in a cluster for z-score to be meaningful.

    Returns:
        DataFrame with additional peer-group z-score and rank columns.
    """
    if kpi_cols is None:
        kpi_cols = [
            COL_RSRP, COL_RSRQ, COL_SINR, COL_DL_TPUT,
            COL_PRB_DL, COL_RRC_SETUP_SR, COL_CELL_AVAIL, COL_HO_SR,
        ]

    df = df.copy()

    # Merge cluster_id from inventory if not already in df
    if COL_CLUSTER_ID not in df.columns:
        cluster_map = inventory_df.set_index(COL_CELL_ID)[COL_CLUSTER_ID].to_dict()
        df[COL_CLUSTER_ID] = df[COL_CELL_ID].map(cluster_map).fillna(-1).astype(int)
        logger.info("  Merged cluster_id from inventory")

    # Filter out cells with too few peers (singleton clusters, unknown clusters)
    cluster_sizes = inventory_df.groupby(COL_CLUSTER_ID).size()
    valid_clusters = set(cluster_sizes[cluster_sizes >= min_peers].index)
    in_valid = df[COL_CLUSTER_ID].isin(valid_clusters)
    n_invalid = (~in_valid).sum()
    if n_invalid > 0:
        logger.warning(
            "  %d rows in clusters with < %d peers — peer-group z-scores will be NaN "
            "for these rows (filled with 0 after computation).",
            n_invalid, min_peers,
        )

    new_cols: List[str] = []

    for kpi in kpi_cols:
        if kpi not in df.columns:
            continue

        col_pg_z    = f"{kpi}_peer_zscore"
        col_pg_rank = f"{kpi}_peer_rank_pct"

        # Compute per-(cluster, timestamp) mean and std
        # This is a cross-sectional operation: all cells at the same timestamp
        group_stats = df.groupby([COL_CLUSTER_ID, COL_TIMESTAMP])[kpi].agg(["mean", "std"])
        group_stats.columns = ["_pg_mean", "_pg_std"]
        group_stats = group_stats.reset_index()

        # Merge back onto the main df
        df_with_stats = df.merge(
            group_stats,
            on=[COL_CLUSTER_ID, COL_TIMESTAMP],
            how="left",
        )

        # Z-score: (cell_value - peer_mean) / peer_std
        safe_std = df_with_stats["_pg_std"].replace(0, np.nan)
        raw_z = (df_with_stats[kpi] - df_with_stats["_pg_mean"]) / safe_std
        df[col_pg_z] = raw_z.clip(-zscore_clip, zscore_clip).fillna(0).astype(np.float32)

        # Rank percentile within peer group at each ROP (0 = worst, 1 = best)
        # Higher RSRP / SINR / throughput = better → ascending rank
        df[col_pg_rank] = (
            df.groupby([COL_CLUSTER_ID, COL_TIMESTAMP])[kpi]
            .rank(pct=True, ascending=True)
            .fillna(0.5)
            .astype(np.float32)
        )

        new_cols.extend([col_pg_z, col_pg_rank])

    logger.info("  Peer-group features added: %d columns across %d KPIs", len(new_cols), len(kpi_cols))
    return df


# ============================================================================
# Step 7 — Graph-native neighbour features
# These are lightweight versions of the GNN neighbourhood aggregation
# described in Part 2 §4.  They are computed in pandas rather than PyG to
# provide a first-order approximation of spatial correlation without requiring
# a full GNN deployment.
# See Coursebook Ch. on Graph Neural Networks (Part 2 prerequisite)
# ============================================================================


def compute_neighbour_aggregate_features(
    df: pd.DataFrame,
    neighbour_df: pd.DataFrame,
    kpi_cols: Optional[List[str]] = None,
    max_neighbours: int = 6,
) -> pd.DataFrame:
    """Compute 1-hop neighbour aggregations for key KPIs.

    For each cell at each timestamp, compute the mean and std of the KPI value
    across its O1 NRT (Neighbour Relation Table) neighbours at the same
    timestamp.  This provides a spatial context feature:
        - If a cell's RSRP is −90 dBm but its 6 neighbours average −75 dBm,
          that's a strong signal of a local hardware fault (not a coverage hole).
        - If all neighbours also show −90 dBm, it might be a propagation event.

    This is the batch-mode equivalent of Part 2's GNN node feature aggregation
    layer.  In production the GNN model replaces this with learned aggregation
    weights (attention scores) rather than fixed mean/std.

    Note: This step is O(n_rows × avg_neighbours) and can be slow on large
    datasets.  For >500K rows, consider vectorised sparse matrix multiplication
    using scipy.sparse instead of the pandas join approach below.

    Args:
        df:             Feature DataFrame indexed by (cell_id, timestamp).
        neighbour_df:   Pairwise neighbour table with source/target cell_id.
        kpi_cols:       KPIs to aggregate over neighbours.
        max_neighbours: Cap on number of neighbours per cell.

    Returns:
        DataFrame with additional neighbour aggregate columns, or original df
        if neighbour_df is empty.
    """
    if neighbour_df.empty or len(neighbour_df) == 0:
        logger.warning("  No neighbour data — skipping graph-native features.")
        return df

    if kpi_cols is None:
        kpi_cols = [COL_RSRP, COL_SINR, COL_DL_TPUT, COL_PRB_DL]

    df = df.copy()

    # Build lookup: cell_id → list of neighbour cell_ids
    # Handle both column name variants from 01_synthetic_data.py
    src_col = "source_cell_id" if "source_cell_id" in neighbour_df.columns else "serving_cell_id"
    tgt_col = "target_cell_id" if "target_cell_id" in neighbour_df.columns else "neighbour_cell_id"

    if src_col not in neighbour_df.columns or tgt_col not in neighbour_df.columns:
        logger.warning(
            "  Neighbour df missing expected columns (%s, %s). "
            "Available: %s — skipping.",
            src_col, tgt_col, list(neighbour_df.columns),
        )
        return df

    neighbours_map: Dict[str, List[str]] = {}
    for _, row in neighbour_df.iterrows():
        src = row[src_col]
        tgt = row[tgt_col]
        neighbours_map.setdefault(src, []).append(tgt)
        # O1 NRT is directional; add reverse for undirected aggregation
        neighbours_map.setdefault(tgt, []).append(src)

    # Truncate to max_neighbours to bound computation
    neighbours_map = {k: v[:max_neighbours] for k, v in neighbours_map.items()}

    # Create a wide pivot for fast lookup: cell_id × timestamp → kpi_value
    # We only pivot on the KPIs we need to limit memory
    logger.info("  Building neighbour aggregation pivot (may take a moment)...")
    pivot = df.pivot_table(
        index=COL_TIMESTAMP, columns=COL_CELL_ID, values=kpi_cols, aggfunc="first"
    )
    # pivot.columns is a MultiIndex: (kpi, cell_id)

    new_cols: List[str] = []
    result_frames: List[pd.DataFrame] = []

    for kpi in kpi_cols:
        if kpi not in df.columns:
            continue

        if kpi not in pivot.columns.get_level_values(0):
            continue

        kpi_pivot = pivot[kpi]  # timestamp × cell_id matrix

        col_nbr_mean = f"{kpi}_neighbour_mean"
        col_nbr_std  = f"{kpi}_neighbour_std"
        col_nbr_min  = f"{kpi}_neighbour_min"
        col_nbr_max  = f"{kpi}_neighbour_max"

        # For each row in df, look up neighbours at the same timestamp
        # Vectorised: build arrays for each unique cell
        all_cells = df[COL_CELL_ID].unique()
        cell_results: Dict[str, Dict[str, float]] = {}

        for cell in all_cells:
            nbrs = neighbours_map.get(cell, [])
            if not nbrs:
                continue

            # Intersect neighbours with cells actually in the pivot
            valid_nbrs = [n for n in nbrs if n in kpi_pivot.columns]
            if not valid_nbrs:
                continue

            # kpi_pivot[valid_nbrs] gives (timestamp × n_neighbours) for this cell
            nbr_vals = kpi_pivot[valid_nbrs].values  # shape: (T, n_nbrs)

            cell_ts = df.loc[df[COL_CELL_ID] == cell, COL_TIMESTAMP].values
            ts_idx = kpi_pivot.index.get_indexer(cell_ts)
            valid_ts = ts_idx >= 0

            if valid_ts.sum() == 0:
                continue

            nbr_slice = nbr_vals[ts_idx[valid_ts], :]  # (n_valid_ts, n_nbrs)
            cell_results[cell] = {
                "ts_idx": ts_idx,
                "valid_ts": valid_ts,
                "means": np.nanmean(nbr_slice, axis=1).astype(np.float32),
                "stds":  np.nanstd(nbr_slice, axis=1).astype(np.float32),
                "mins":  np.nanmin(nbr_slice, axis=1).astype(np.float32),
                "maxs":  np.nanmax(nbr_slice, axis=1).astype(np.float32),
            }

        # Write results back to df columns
        df[col_nbr_mean] = np.nan
        df[col_nbr_std]  = np.nan
        df[col_nbr_min]  = np.nan
        df[col_nbr_max]  = np.nan

        for cell, res in cell_results.items():
            cell_mask = df[COL_CELL_ID] == cell
            cell_indices = df.index[cell_mask]
            valid_indices = cell_indices[res["valid_ts"]]

            df.loc[valid_indices, col_nbr_mean] = res["means"]
            df.loc[valid_indices, col_nbr_std]  = res["stds"]
            df.loc[valid_indices, col_nbr_min]  = res["mins"]
            df.loc[valid_indices, col_nbr_max]  = res["maxs"]

        # Fill cells with no neighbours using their own value (no spatial signal)
        df[col_nbr_mean] = df[col_nbr_mean].fillna(df[kpi]).astype(np.float32)
        df[col_nbr_std]  = df[col_nbr_std].fillna(0.0).astype(np.float32)
        df[col_nbr_min]  = df[col_nbr_min].fillna(df[kpi]).astype(np.float32)
        df[col_nbr_max]  = df[col_nbr_max].fillna(df[kpi]).astype(np.float32)

        new_cols.extend([col_nbr_mean, col_nbr_std, col_nbr_min, col_nbr_max])

    logger.info("  Neighbour aggregate features added: %d columns", len(new_cols))
    return df


# ============================================================================
# Step 8 — Missing value imputation and counter-reset correction
# See Coursebook Ch. 13: Feature Engineering §14.9 Data Quality
# ============================================================================


def handle_missing_values_and_resets(df: pd.DataFrame) -> pd.DataFrame:
    """Impute NaNs and detect/correct PM counter resets.

    PM counters reset to 0 periodically (cell reboot, planned maintenance,
    software upgrade) and on 32-bit wraparound.  A counter reset looks like
    a large negative delta which can produce extreme z-scores and false
    anomalies if not corrected.

    Strategy:
        1. Detect likely resets: delta < −(threshold fraction of max observed).
        2. Flag the reset row with is_counter_reset = 1.
        3. Forward-fill the pre-reset value for delta features only; preserve
           the raw KPI value so the model can learn the reset signature.

    For missing values:
        - Short gaps (≤ 2 ROPs): forward-fill within cell group.
        - Longer gaps: fill with rolling 24h mean of that cell.
        - Remaining NaN: fill with cluster-level median of that ROP.
        - Last resort: global median per column.

    Note: Counter reset detection uses CUMULATIVE counters only.
    Rate-based KPIs (throughput_mbps, prb_util_pct) do not reset.
    """
    df = df.copy()

    # --- Counter reset detection ---
    # Cumulative counters in the schema: HO attempts, HO successes
    cumulative_cols = [COL_HO_ATTEMPT, COL_HO_SUCCESS]

    for col in cumulative_cols:
        if col not in df.columns:
            continue

        delta_col = f"__{col}_delta_raw"
        df[delta_col] = df.groupby(COL_CELL_ID)[col].diff()

        # A reset is signalled by a large negative drop (> 50% of rolling max)
        roll_max = df.groupby(COL_CELL_ID)[col].rolling(
            WINDOW_24H, min_periods=MIN_PERIODS_24H, closed="left"
        ).max().reset_index(level=0, drop=True)
        reset_threshold = -0.5 * roll_max.clip(lower=1)

        reset_flag_col = f"{col}_is_reset"
        df[reset_flag_col] = (df[delta_col] < reset_threshold).astype(np.int8)

        # Zero out the delta on reset rows (don't let a reset drive a delta feature)
        df.loc[df[reset_flag_col] == 1, delta_col] = 0
        df.drop(columns=[delta_col], inplace=True)

    logger.info("  Counter reset flags added for: %s", cumulative_cols)

    # --- Missing value imputation ---
    numeric_cols = df.select_dtypes(include=[np.floating, np.integer]).columns.tolist()
    label_cols = [COL_IS_ANOMALY, COL_ANOMALY_TYPE]
    numeric_feature_cols = [c for c in numeric_cols if c not in label_cols]

    # Pass 1: Forward-fill short gaps (max 2 consecutive NaN ROPs per cell)
    df[numeric_feature_cols] = (
        df.groupby(COL_CELL_ID)[numeric_feature_cols]
        .transform(lambda x: x.ffill(limit=2))
    )

    # Pass 2: Fill remaining with cell's rolling 24h mean
    # (already computed in compute_rolling_features for CORE_KPIS)
    for kpi in CORE_KPIS:
        mean_col = f"{kpi}_roll24h_mean"
        if mean_col in df.columns and kpi in df.columns:
            df[kpi] = df[kpi].fillna(df[mean_col])

    # Pass 3: Fill with global column median (computed on training set —
    # in practice this would be loaded from the training statistics artifact;
    # here we compute on the full df as an approximation for demonstration)
    for col in numeric_feature_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    remaining_nan = df[numeric_feature_cols].isna().sum().sum()
    if remaining_nan > 0:
        logger.warning("  %d NaN values remain after imputation — filling with 0.", remaining_nan)
        df[numeric_feature_cols] = df[numeric_feature_cols].fillna(0)

    logger.info("  Missing value imputation complete. Remaining NaN: %d", remaining_nan)
    return df


# ============================================================================
# Step 9 — CQI-to-throughput consistency (needs per-column scaling)
# Finalise the feature started in compute_derived_kpis.
# ============================================================================


def finalise_cqi_tput_consistency(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Compute the CQI-to-throughput consistency ratio using dataset-wide normalisers.

    This requires knowing the 95th percentile of DL throughput, which is only
    available after the full dataset is loaded (not available row-by-row).

    Lower values (near 0) mean CQI predicts good channel but throughput is low,
    which is anomalous and often indicates a scheduler configuration fault.

    Note: The percentile is computed on the TRAINING set only; the same value
    is applied to val/test to prevent leakage.  The scaler artifact must store
    this value for serving-time feature computation.
    """
    df = df.copy()

    p95_dl = df[COL_DL_TPUT].quantile(0.95)
    if p95_dl <= 0:
        p95_dl = 1.0  # guard against all-zero throughput (pathological case)

    dl_norm = (df["_dl_tput_raw"] / p95_dl).clip(0, 1)

    # Consistency: how much does CQI overestimate achievable throughput?
    # = normalised_throughput / normalised_cqi
    # Near 1.0 → CQI and throughput are aligned (healthy)
    # Near 0.0 → CQI high but throughput low (anomalous)
    safe_cqi_norm = df["_cqi_norm"].replace(0, np.nan)
    df[COL_CQI_TPUT_RATIO] = (dl_norm / safe_cqi_norm).clip(0, 2.0).fillna(0.5).astype(np.float32)

    # Drop the interim helper columns
    df.drop(columns=["_cqi_norm", "_dl_tput_raw"], errors="ignore", inplace=True)

    return df, p95_dl  # return p95 for storage in scaler artifact


# ============================================================================
# Step 10 — Temporal train / val / test split
# NEVER use random split for time-series data.
# See Coursebook Ch. 28: Data Pipelines §29.4 Temporal Splits
# ============================================================================


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    timestamp_col: str = COL_TIMESTAMP,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Split the feature DataFrame into train / val / test by time.

    We split by timestamp, not by index or randomly.  This ensures:
        1. No look-ahead bias: val/test rows are always strictly after all
           train rows in calendar time.
        2. Rolling features computed on training data do not use val/test
           values (provided the rolling is done within-cell before splitting).
        3. The evaluation reflects real deployment conditions where the model
           is trained on historical data and scored on future data.

    Note on multi-cell datasets:
        Each cell has the same timestamp range (generated by 01_synthetic_data.py
        to be synchronised).  The split cuts are applied globally by timestamp,
        which means all cells share the same train/val/test periods.  This is
        the correct approach: you train on data from all cells up to cut_1,
        validate on all cells from cut_1 to cut_2, and test on all cells after
        cut_2.  Splitting per-cell independently would allow a model to train
        on "future" data from one cell while only seeing "past" data from
        another — a subtle form of leakage.

    Args:
        df:           Full feature DataFrame, sorted by timestamp.
        train_frac:   Fraction of total time range to use for training.
        val_frac:     Fraction for validation.
        timestamp_col: Name of the timestamp column.

    Returns:
        train_df, val_df, test_df, metadata_dict
    """
    assert train_frac + val_frac < 1.0, "train + val fractions must be < 1.0"

    ts = df[timestamp_col]
    t_min = ts.min()
    t_max = ts.max()
    total_span = t_max - t_min

    cut1 = t_min + total_span * train_frac
    cut2 = t_min + total_span * (train_frac + val_frac)

    train_df = df[ts < cut1].copy()
    val_df   = df[(ts >= cut1) & (ts < cut2)].copy()
    test_df  = df[ts >= cut2].copy()

    meta = {
        "t_min": str(t_min),
        "cut1_train_val": str(cut1),
        "cut2_val_test": str(cut2),
        "t_max": str(t_max),
        "train_rows": len(train_df),
        "val_rows":   len(val_df),
        "test_rows":  len(test_df),
        "train_anomaly_rate": float(train_df[COL_IS_ANOMALY].mean()),
        "val_anomaly_rate":   float(val_df[COL_IS_ANOMALY].mean()),
        "test_anomaly_rate":  float(test_df[COL_IS_ANOMALY].mean()),
        "n_feature_cols": int(df.shape[1]),
    }

    logger.info(
        "  Temporal split: train=%d rows (%.1f%%) | val=%d rows (%.1f%%) | test=%d rows (%.1f%%)",
        meta["train_rows"],  100 * train_frac,
        meta["val_rows"],    100 * val_frac,
        meta["test_rows"],   100 * (1 - train_frac - val_frac),
    )
    logger.info(
        "  Anomaly rates — train: %.3f | val: %.3f | test: %.3f",
        meta["train_anomaly_rate"], meta["val_anomaly_rate"], meta["test_anomaly_rate"],
    )

    return train_df, val_df, test_df, meta


# ============================================================================
# Step 11 — Feature normalisation (fit on train only)
# ============================================================================


def fit_transform_scaler(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a StandardScaler on training data and transform all splits.

    CRITICAL: The scaler is fit ONLY on train_df.  Fitting on the full dataset
    (including val/test) constitutes data leakage — the model would implicitly
    know the distribution of future data.

    The scaler transforms only float32/float64 columns.  Integer flag columns
    (is_weekend, is_peak_hour, etc.) and label columns are excluded.

    In production serving, the scaler artifact is loaded alongside the model
    artifact so that new inference requests are transformed consistently.
    See Part 2 §14 (Production Patterns) for the serving-time feature pipeline.

    Args:
        train_df, val_df, test_df:  Feature DataFrames from temporal_split().
        exclude_cols:               Columns to skip (IDs, timestamps, labels).
        output_dir:                 If provided, save scaler params as JSON.

    Returns:
        Transformed train_df, val_df, test_df.
    """
    if exclude_cols is None:
        # Exclude: identifier columns, timestamp, labels, flag/indicator integers
        exclude_cols = [
            COL_CELL_ID, COL_SITE_ID, COL_TIMESTAMP, COL_ENVIRONMENT,
            COL_CLUSTER_ID, COL_IS_ANOMALY, COL_ANOMALY_TYPE,
            "hour_of_day", "day_of_week", "day_of_year", "week_of_year",
            "is_weekend", "is_morning_peak", "is_evening_peak",
            "is_peak_hour", "is_business_hours", "rop_of_day",
            COL_HOP_INDEX,
        ]

    float_cols = [
        c for c in train_df.select_dtypes(include=[np.float32, np.float64]).columns
        if c not in exclude_cols
    ]

    logger.info("  Fitting StandardScaler on %d float feature columns (train only)...", len(float_cols))

    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df[float_cols] = scaler.fit_transform(train_df[float_cols]).astype(np.float32)
    val_df[float_cols]   = scaler.transform(val_df[float_cols]).astype(np.float32)
    test_df[float_cols]  = scaler.transform(test_df[float_cols]).astype(np.float32)

    # Persist scaler parameters for serving-time use
    if output_dir is not None:
        scaler_artifact = {
            "feature_names": float_cols,
            "mean_": scaler.mean_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "var_": scaler.var_.tolist(),
            "n_samples_seen_": int(scaler.n_samples_seen_),
        }
        scaler_path = output_dir / "scaler.json"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "w") as f:
            json.dump(scaler_artifact, f, indent=2)
        logger.info("  Scaler saved to %s (JSON)", scaler_path)

        # Also save as joblib for direct loading by 03_model_training.py
        joblib_path = output_dir / "scaler.joblib"
        joblib.dump(scaler, joblib_path, compress=3)
        logger.info("  Scaler saved to %s (joblib)", joblib_path)

    return train_df, val_df, test_df


# ============================================================================
# Step 12 — Feature catalog generation
# Documenting what every feature means is a production requirement.
# See Coursebook Ch. 52: System Design for ML — feature documentation
# ============================================================================


def build_feature_catalog(df: pd.DataFrame) -> List[Dict]:
    """Generate a machine-readable catalog of all features.

    The catalog is used by:
        - Model training (03_model_training.py) to select feature subsets
        - Governance gates (Part 2 §12) to flag high-risk features
        - Monitoring (04_evaluation.py, 05_production_patterns.py) for
          per-feature drift metrics
        - Feature store (Feast) schema generation for serving

    Returns:
        List of dicts with keys: name, dtype, category, description, source_col
    """
    catalog: List[Dict] = []

    # ---------------------------------------------------------------------------
    # Directory constants (canonical paths — must match pipeline contracts)
    # ---------------------------------------------------------------------------
    input_dir: str = 'data/features/'          # Script 02 output → Script 03 input
    artifacts_dir: str = 'artifacts/'          # score arrays for Script 04 (e.g. artifacts/anomaly_scores.npy)
    models_dir: str = 'artifacts/models/'      # serialised model artifacts (e.g. artifacts/models/<model>.pkl)

    # ---------------------------------------------------------------------------
    # GNN metadata — all 5 edge types (canonical; must match RootCauseGNN and build_ran_topology_graph)
    # ---------------------------------------------------------------------------
    GNN_METADATA = (
        ['cell_sector', 'site', 'backhaul_node'],
        [
            ('cell_sector', 'is_neighbour_of', 'cell_sector'),
            ('cell_sector', 'same_site', 'site'),
            ('site', 'rev_same_site', 'cell_sector'),
            ('backhaul_node', 'shares_transport', 'cell_sector'),
            ('cell_sector', 'rev_shares_transport', 'backhaul_node'),
        ],
    )

    for col in df.columns:
        entry: Dict = {"name": col, "dtype": str(df[col].dtype)}

        # Categorise by name pattern
        if col in [COL_CELL_ID, COL_SITE_ID, COL_TIMESTAMP, COL_ENVIRONMENT,
                   COL_CLUSTER_ID, COL_LAT, COL_LON]:
            entry["category"] = "identifier"
        elif col in [COL_IS_ANOMALY, COL_ANOMALY_TYPE]:
            entry["category"] = "label"
        elif any(x in col for x in ["hour", "day", "week", "rop", "peak", "weekend", "business"]):
            entry["category"] = "temporal"
        elif "_roll" in col:
            entry["category"] = "rolling_statistic"
        elif "_delta" in col or "_abs_change" in col or "_direction" in col:
            entry["category"] = "delta"
        elif "_zscore" in col and "peer" not in col:
            entry["category"] = "rolling_zscore"
        elif "_peer_" in col:
            entry["category"] = "peer_group_spatial"
        elif "_neighbour_" in col:
            entry["category"] = "graph_neighbour"
        elif col in [COL_HO_SR, COL_DL_UL_RATIO, COL_PRB_EFF,
                     COL_CQI_TPUT_RATIO, COL_RSRQ_RSRP_SPREAD]:
            entry["category"] = "cross_kpi_ratio"
        elif col in CORE_KPIS:
            entry["category"] = "raw_kpi"
        elif "_is_reset" in col:
            entry["category"] = "data_quality_flag"
        else:
            entry["category"] = "other"

        # Add human-readable description for key columns
        DESCRIPTIONS = {
            COL_RSRP: "Reference Signal Received Power (dBm). Primary coverage KPI. Range: −140 to −44.",
            COL_RSRQ: "Reference Signal Received Quality (dB). Signal quality including interference. Range: −20 to −3.",
            COL_SINR: "Signal to Interference plus Noise Ratio (dB). Higher is better. Drives MCS selection.",
            COL_CQI: "Channel Quality Indicator (0–15). UE-reported channel condition. Integer.",
            COL_DL_TPUT: "Downlink cell throughput (Mbps). Aggregate across all active UEs.",
            COL_UL_TPUT: "Uplink cell throughput (Mbps). Aggregate across all active UEs.",
            COL_PRB_DL: "Downlink Physical Resource Block utilisation (%). 100% = fully loaded.",
            COL_PRB_UL: "Uplink Physical Resource Block utilisation (%).",
            COL_ACTIVE_UE: "Number of actively scheduled UEs per ROP.",
            COL_RRC_SETUP_SR: "RRC Connection Setup Success Rate (%). Mobility signalling reliability.",
            COL_HO_ATTEMPT: "Handover attempt count per ROP. Mobility traffic volume indicator.",
            COL_HO_SUCCESS: "Successful handover count per ROP.",
            COL_CELL_AVAIL: "Cell availability (%). 100% = cell fully operational.",
            COL_HO_SR: "Derived: Handover Success Rate = success / attempt × 100. 100% for idle cells.",
            COL_DL_UL_RATIO: "Derived: DL/UL throughput ratio. High values indicate streaming-heavy load.",
            COL_PRB_EFF: "Derived: DL throughput per % of DL PRB utilised (Mbps/%). Scheduler efficiency indicator.",
            COL_CQI_TPUT_RATIO: "Derived: Consistency between CQI prediction and achieved throughput. Near 1.0 = healthy.",
            COL_RSRQ_RSRP_SPREAD: "Derived: RSRP minus RSRQ (in dB). Proxy for interference loading.",
        }
        entry["description"] = DESCRIPTIONS.get(col, f"Feature column: {col}")

        catalog.append(entry)

    return catalog


# ============================================================================
# Pipeline orchestration
# ============================================================================


def main() -> None:
    """Run the full feature engineering pipeline.

    Steps:
        1. Load raw PM counter data and cell inventory
        2. Compute derived KPIs (HO success rate, DL/UL ratio, PRB efficiency, etc.)
        3. Encode temporal features (cyclical hour/day, peak flags)
        4. Compute rolling statistics (1h, 4h, 24h windows)
        5. Compute rate-of-change / delta features
        6. Compute rolling z-scores (anomaly reference window)
        7. Compute peer-group features (DBSCAN cluster z-scores)
        8. Compute neighbour aggregate features (graph spatial)
        9. Handle missing values and counter resets
       10. Finalise CQI-throughput consistency ratio
       11. Temporal split (train / val / test)
       12. Fit and apply StandardScaler
       13. Generate and save feature catalog
    """
    cfg = FeatureConfig()
    logger.info("=" * 70)
    logger.info("02_feature_engineering.py — Feature Engineering Pipeline")
    logger.info("=" * 70)
    logger.info("Config: %s", cfg)

    # Step 1 — Load
    kpi_df, inventory_df, neighbour_df = load_raw_data(cfg)
    logger.info("Loaded %d rows × %d cols", *kpi_df.shape)

    # Step 2 — Derived KPIs
    logger.info("Step 2: Computing derived KPIs...")
    kpi_df = compute_derived_kpis(kpi_df)

    # Step 3 — Temporal features
    logger.info("Step 3: Encoding temporal features...")
    kpi_df = encode_temporal_features(kpi_df)

    # Step 4 — Rolling statistics
    logger.info("Step 4: Computing rolling features...")
    kpi_df = compute_rolling_features(kpi_df)

    # Step 5 — Delta features
    logger.info("Step 5: Computing delta features...")
    kpi_df = compute_delta_features(kpi_df)

    # Step 6 — Rolling z-scores
    logger.info("Step 6: Computing rolling z-scores...")
    kpi_df = compute_rolling_zscores(kpi_df)

    # Step 7 — Peer-group features
    logger.info("Step 7: Computing peer-group features...")
    kpi_df = compute_peer_group_features(kpi_df, inventory_df)

    # Step 8 — Neighbour aggregate features
    # Memory-intensive: creates a pivot table requiring ~2–4 GB RAM.
    # Skip with TELCO_SKIP_NEIGHBOUR_FEATURES=1 on memory-constrained machines.
    if os.environ.get("TELCO_SKIP_NEIGHBOUR_FEATURES", "0") == "1":
        logger.info("Step 8: SKIPPED (TELCO_SKIP_NEIGHBOUR_FEATURES=1)")
    else:
        logger.info("Step 8: Computing neighbour aggregate features...")
        kpi_df = compute_neighbour_aggregate_features(kpi_df, neighbour_df)

    # Step 9 — Missing values and resets
    logger.info("Step 9: Handling missing values and counter resets...")
    kpi_df = handle_missing_values_and_resets(kpi_df)

    # Step 10 — Finalise CQI-throughput consistency
    logger.info("Step 10: Finalising CQI-throughput consistency ratio...")
    kpi_df, p95_dl = finalise_cqi_tput_consistency(kpi_df)

    # Step 11 — Temporal split
    logger.info("Step 11: Temporal split...")
    train_df, val_df, test_df, split_meta = temporal_split(
        kpi_df,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
    )

    # Step 12 — Scaler
    logger.info("Step 12: Fitting and applying StandardScaler...")
    train_df, val_df, test_df = fit_transform_scaler(train_df, val_df, test_df)

    # Step 13 — Save
    output_path = cfg.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    save_split(train_df, output_path / "train.parquet", "train")
    save_split(val_df, output_path / "val.parquet", "val")
    save_split(test_df, output_path / "test.parquet", "test")

    # Feature catalog
    catalog = build_feature_catalog(train_df)
    catalog_path = output_path / "feature_catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)
    logger.info("Feature catalog written to %s (%d features)", catalog_path, len(catalog))

    # Split metadata
    meta_path = output_path / "split_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(split_meta, f, indent=2, default=str)
    logger.info("Split metadata written to %s", meta_path)

    # CQI-throughput normaliser (required for serving-time feature computation)
    p95_path = output_path / "p95_dl_throughput.json"
    with open(p95_path, "w") as f:
        json.dump({"p95_dl_throughput_mbps": float(p95_dl)}, f, indent=2)
    logger.info("p95_dl normaliser written to %s (value=%.4f)", p95_path, p95_dl)

    logger.info("=" * 70)
    logger.info("Feature engineering complete. Outputs in: %s/", output_path)
    logger.info("  train.parquet : %d rows", len(train_df))
    logger.info("  val.parquet   : %d rows", len(val_df))
    logger.info("  test.parquet  : %d rows", len(test_df))
    logger.info("  Total features: %d", len(train_df.columns))
    logger.info("Next step: python 03_model_training.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
