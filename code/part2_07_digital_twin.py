"""
07_digital_twin.py — Digital Twin for RAN What-If Simulation
=============================================================
Telco MLOps Reference Architecture — Part 2

Builds a per-cell digital twin from synthetic PM counter data and validates
what-if predictions against held-out observations.

Architecture (see §8.5):
  1. Behavioural layer — per-cell diurnal KPI profiles (24 hourly medians)
  2. Structural layer — topology-aware neighbour impact propagation
  3. What-if engine — predict KPI deltas from parameter changes
  4. Validation — compare twin predictions against actual observations

Key sensitivity constants (coarse linear approximations for 65° HPBW
  macro antennas in sub-6 GHz urban environments — NOT from any single
  3GPP specification; typical ranges: urban macro -0.6 to -1.2,
  suburban -0.3 to -0.8, rural high-gain -1.0 to -2.5 CQI/degree):
  TILT_CQI_SENSITIVITY  = -0.8 CQI units per degree of electrical tilt
  TILT_HO_SENSITIVITY   = -0.005 HO success rate change per degree of tilt

  These are COARSE APPROXIMATIONS for urban macro deployments. Operators
  MUST recalibrate against their own antenna hardware, propagation
  environment, and drive test data before production use. See §8.5.

Inputs:
  - data/pm_counters.parquet       (from Script 01)
  - data/cell_inventory.parquet    (from Script 01)
  - data/neighbour_relations.parquet (from Script 01)

Outputs:
  - artifacts/digital_twin/cell_profiles.json
  - artifacts/digital_twin/validation_report.json
  - artifacts/digital_twin/what_if_examples.json

Usage:
  python 07_digital_twin.py

Prerequisites:
  pip install pandas numpy scipy

Coursebook cross-reference:
  Ch. 44  — Causal Inference (counterfactual estimation, treatment effects)
  Ch. 47  — Knowledge Graphs (structural modelling, entity relationships)
  Ch. 52  — System Design for ML (simulation in ML pipelines)
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("digital_twin")

# ── Sensitivity constants (see §8.5) ──────────────────────────────────────
# COARSE APPROXIMATIONS for 65° HPBW macro antennas in sub-6 GHz urban
# environments. NOT applicable to indoor/pico cells, narrow-beam high-gain
# antennas, or mmWave deployments without recalibration. Published field
# studies show -0.3 to -2.5 CQI/degree depending on antenna pattern,
# propagation environment, and carrier frequency. Recalibrate against
# your vendor's antenna patterns and drive test data before production use.

TILT_CQI_SENSITIVITY = -0.8   # CQI units per degree of electrical tilt increase
                               # Urban macro approx; range: -0.6 to -1.2 for urban,
                               # -0.3 to -0.8 suburban, -1.0 to -2.5 rural high-gain
TILT_HO_SENSITIVITY = -0.005  # HO success rate change per degree of tilt (urban macro approx)
POWER_TPUT_SENSITIVITY = 0.8  # Mbps per dBm of transmit power change
PRB_LOAD_SENSITIVITY = 0.15   # PRB usage change per 10% traffic redistribution

# ── Paths ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
TWIN_DIR = ARTIFACTS_DIR / "digital_twin"

RANDOM_SEED = 42


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class CellProfile:
    """Per-cell diurnal KPI profile — 24 hourly medians per KPI."""
    cell_id: str
    site_id: str
    hourly_profiles: Dict[str, List[float]]  # kpi_name → [24 hourly medians]
    hourly_std: Dict[str, List[float]]       # kpi_name → [24 hourly stds]
    n_observations: int = 0


@dataclass
class WhatIfResult:
    """Result of a what-if parameter change simulation."""
    cell_id: str
    parameter_name: str
    parameter_delta: float
    predicted_kpi_deltas: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    blast_radius_cells: List[str]
    recommendation: str  # "SAFE" | "CAUTION" | "BLOCK"


@dataclass
class ValidationResult:
    """Validation of twin predictions against actual observations."""
    cell_id: str
    hour: int
    kpi_name: str
    predicted: float
    actual: float
    error: float
    within_ci: bool


# ── Step 1: Build behavioural layer ───────────────────────────────────────

def build_cell_profiles(pm_df: pd.DataFrame, inventory_df: pd.DataFrame) -> Dict[str, CellProfile]:
    """
    Build per-cell diurnal KPI profiles from PM counter history.

    For each cell, compute the median and standard deviation of each KPI
    at each hour of the day. This captures diurnal traffic patterns.
    """
    logger.info("Building per-cell diurnal profiles...")

    kpi_cols = [
        c for c in pm_df.columns
        if c not in {"cell_id", "timestamp", "site_id", "is_anomaly", "anomaly_type"}
        and pd.api.types.is_numeric_dtype(pm_df[c])
    ]

    if "timestamp" not in pm_df.columns:
        logger.warning("No timestamp column found. Generating synthetic hours.")
        pm_df = pm_df.copy()
        rng = np.random.RandomState(RANDOM_SEED)
        pm_df["timestamp"] = pd.date_range(
            "2024-01-01", periods=len(pm_df), freq="15min"
        )

    pm_df = pm_df.copy()
    pm_df["hour"] = pd.to_datetime(pm_df["timestamp"]).dt.hour

    # Build cell_id → site_id mapping
    cell_site_map = {}
    if inventory_df is not None and "cell_id" in inventory_df.columns:
        site_col = "site_id" if "site_id" in inventory_df.columns else None
        if site_col:
            cell_site_map = dict(zip(inventory_df["cell_id"], inventory_df[site_col]))

    profiles = {}
    for cell_id, cell_data in pm_df.groupby("cell_id"):
        hourly_medians = {}
        hourly_stds = {}

        for kpi in kpi_cols:
            if kpi not in cell_data.columns:
                continue
            medians = cell_data.groupby("hour")[kpi].median()
            stds = cell_data.groupby("hour")[kpi].std().fillna(0)

            # Fill all 24 hours (some may be missing)
            full_medians = [float(medians.get(h, medians.median())) for h in range(24)]
            full_stds = [float(stds.get(h, stds.median())) for h in range(24)]

            hourly_medians[kpi] = full_medians
            hourly_stds[kpi] = full_stds

        profiles[str(cell_id)] = CellProfile(
            cell_id=str(cell_id),
            site_id=str(cell_site_map.get(cell_id, "unknown")),
            hourly_profiles=hourly_medians,
            hourly_std=hourly_stds,
            n_observations=len(cell_data),
        )

    logger.info("Built profiles for %d cells, %d KPIs", len(profiles), len(kpi_cols))
    return profiles


# ── Step 2: Structural layer — neighbour impact ──────────────────────────

def build_neighbour_map(
    neighbour_df: Optional[pd.DataFrame],
) -> Dict[str, List[str]]:
    """Build cell → neighbour list mapping from NRT data."""
    if neighbour_df is None or neighbour_df.empty:
        return {}

    src_col = "source_cell_id" if "source_cell_id" in neighbour_df.columns else None
    tgt_col = "target_cell_id" if "target_cell_id" in neighbour_df.columns else None

    if not src_col or not tgt_col:
        # Try alternative column names
        cols = neighbour_df.columns.tolist()
        if len(cols) >= 2:
            src_col, tgt_col = cols[0], cols[1]
        else:
            return {}

    neighbour_map: Dict[str, List[str]] = {}
    for _, row in neighbour_df.iterrows():
        src = str(row[src_col])
        tgt = str(row[tgt_col])
        neighbour_map.setdefault(src, []).append(tgt)

    logger.info(
        "Built neighbour map: %d cells with neighbours (avg %.1f neighbours/cell)",
        len(neighbour_map),
        np.mean([len(v) for v in neighbour_map.values()]) if neighbour_map else 0,
    )
    return neighbour_map


def compute_blast_radius(
    cell_id: str,
    neighbour_map: Dict[str, List[str]],
    profiles: Dict[str, CellProfile],
    max_hops: int = 1,
) -> List[str]:
    """
    Identify cells in the blast radius of a parameter change.

    Currently implements 1-hop neighbours. Multi-hop propagation
    (for backhaul failures affecting entire site clusters) requires
    the transport topology graph from §4.
    """
    direct_neighbours = neighbour_map.get(cell_id, [])

    # Also include co-sited cells (same site_id)
    if cell_id in profiles:
        target_site = profiles[cell_id].site_id
        co_sited = [
            cid for cid, p in profiles.items()
            if p.site_id == target_site and cid != cell_id
        ]
    else:
        co_sited = []

    blast_radius = list(set(direct_neighbours + co_sited))
    return blast_radius


# ── Step 3: What-if engine ─────────────────────────────────────────────────

def predict_what_if(
    cell_id: str,
    parameter_name: str,
    parameter_delta: float,
    hour: int,
    profiles: Dict[str, CellProfile],
    neighbour_map: Dict[str, List[str]],
) -> WhatIfResult:
    """
    Predict KPI deltas from a proposed parameter change.

    See §8.5 — the twin's value at initial deployment is in identifying
    which cells are in the blast radius and in which direction KPIs move,
    not in the precise magnitude of the predicted delta.
    """
    if cell_id not in profiles:
        return WhatIfResult(
            cell_id=cell_id,
            parameter_name=parameter_name,
            parameter_delta=parameter_delta,
            predicted_kpi_deltas={},
            confidence_intervals={},
            blast_radius_cells=[],
            recommendation="BLOCK",
        )

    profile = profiles[cell_id]
    kpi_deltas = {}
    confidence_intervals = {}

    # Apply sensitivity model based on parameter type
    if "tilt" in parameter_name.lower():
        # Antenna electrical tilt adjustment
        if "avg_cqi" in profile.hourly_profiles:
            cqi_delta = TILT_CQI_SENSITIVITY * parameter_delta
            cqi_std = profile.hourly_std.get("avg_cqi", [1.0] * 24)[hour]
            kpi_deltas["avg_cqi"] = round(cqi_delta, 3)
            confidence_intervals["avg_cqi"] = (
                round(cqi_delta - 1.96 * cqi_std * abs(parameter_delta) * 0.3, 3),
                round(cqi_delta + 1.96 * cqi_std * abs(parameter_delta) * 0.3, 3),
            )

        if "handover_success_rate" in profile.hourly_profiles:
            ho_delta = TILT_HO_SENSITIVITY * parameter_delta * 100  # percentage points
            ho_std = profile.hourly_std.get("handover_success_rate", [2.0] * 24)[hour]
            kpi_deltas["handover_success_rate"] = round(ho_delta, 3)
            confidence_intervals["handover_success_rate"] = (
                round(ho_delta - 1.96 * ho_std * abs(parameter_delta) * 0.2, 3),
                round(ho_delta + 1.96 * ho_std * abs(parameter_delta) * 0.2, 3),
            )

    elif "power" in parameter_name.lower():
        # Transmit power adjustment
        if "dl_throughput_mbps" in profile.hourly_profiles:
            tput_delta = POWER_TPUT_SENSITIVITY * parameter_delta
            tput_std = profile.hourly_std.get("dl_throughput_mbps", [5.0] * 24)[hour]
            kpi_deltas["dl_throughput_mbps"] = round(tput_delta, 3)
            confidence_intervals["dl_throughput_mbps"] = (
                round(tput_delta - 1.96 * tput_std * abs(parameter_delta) * 0.1, 3),
                round(tput_delta + 1.96 * tput_std * abs(parameter_delta) * 0.1, 3),
            )

    # Compute blast radius
    blast_cells = compute_blast_radius(cell_id, neighbour_map, profiles)

    # Determine recommendation
    critical_degradation = any(
        delta < -2.0 for delta in kpi_deltas.values()
    )
    large_blast = len(blast_cells) > 10

    if critical_degradation:
        recommendation = "BLOCK"
    elif large_blast or any(delta < -0.5 for delta in kpi_deltas.values()):
        recommendation = "CAUTION"
    else:
        recommendation = "SAFE"

    return WhatIfResult(
        cell_id=cell_id,
        parameter_name=parameter_name,
        parameter_delta=parameter_delta,
        predicted_kpi_deltas=kpi_deltas,
        confidence_intervals={
            k: list(v) for k, v in confidence_intervals.items()
        },
        blast_radius_cells=blast_cells[:20],  # Cap for readability
        recommendation=recommendation,
    )


# ── Step 4: Validation ────────────────────────────────────────────────────

def validate_profiles(
    profiles: Dict[str, CellProfile],
    pm_df: pd.DataFrame,
    holdout_fraction: float = 0.2,
) -> List[ValidationResult]:
    """
    Validate twin predictions against held-out observations.

    Splits the data temporally: last 20% of observations are held out,
    and the twin predicts the hourly median for each cell-KPI pair.
    """
    logger.info("Validating profiles against held-out observations...")

    if "timestamp" not in pm_df.columns:
        logger.warning("No timestamp column. Skipping temporal validation.")
        return []

    pm_df = pm_df.copy()
    pm_df["timestamp"] = pd.to_datetime(pm_df["timestamp"])
    pm_df = pm_df.sort_values("timestamp")

    cutoff_idx = int(len(pm_df) * (1 - holdout_fraction))
    holdout = pm_df.iloc[cutoff_idx:]
    holdout = holdout.copy()
    holdout["hour"] = holdout["timestamp"].dt.hour

    kpi_cols = [
        c for c in holdout.columns
        if c not in {"cell_id", "timestamp", "site_id", "hour", "is_anomaly", "anomaly_type"}
        and pd.api.types.is_numeric_dtype(holdout[c])
    ]

    results = []
    sample_cells = list(profiles.keys())[:50]  # Validate a sample

    for cell_id in sample_cells:
        if cell_id not in profiles:
            continue

        cell_holdout = holdout[holdout["cell_id"] == cell_id]
        if cell_holdout.empty:
            continue

        profile = profiles[cell_id]

        for kpi in kpi_cols[:5]:  # Top 5 KPIs for speed
            if kpi not in profile.hourly_profiles:
                continue

            for hour in range(24):
                hour_data = cell_holdout[cell_holdout["hour"] == hour]
                if hour_data.empty or kpi not in hour_data.columns:
                    continue

                actual = float(hour_data[kpi].median())
                predicted = profile.hourly_profiles[kpi][hour]
                std = profile.hourly_std[kpi][hour]
                error = actual - predicted

                ci_low = predicted - 1.96 * std
                ci_high = predicted + 1.96 * std

                results.append(ValidationResult(
                    cell_id=cell_id,
                    hour=hour,
                    kpi_name=kpi,
                    predicted=round(predicted, 4),
                    actual=round(actual, 4),
                    error=round(error, 4),
                    within_ci=bool(ci_low <= actual <= ci_high) if std > 0 else True,
                ))

    if results:
        coverage = sum(1 for r in results if r.within_ci) / len(results)
        mae = np.mean([abs(r.error) for r in results])
        logger.info(
            "Validation: %d cell-hour-KPI pairs, 95%% CI coverage=%.1f%%, MAE=%.4f",
            len(results), coverage * 100, mae,
        )
    else:
        logger.warning("No validation results produced.")

    return results


# ── Main pipeline ──────────────────────────────────────────────────────────

def main() -> Dict:
    logger.info("=" * 70)
    logger.info("Digital Twin — RAN What-If Simulation")
    logger.info("=" * 70)

    # Load data
    pm_path = DATA_DIR / "pm_counters.parquet"
    inv_path = DATA_DIR / "cell_inventory.parquet"
    nr_path = DATA_DIR / "neighbour_relations.parquet"

    if not pm_path.exists():
        logger.error("PM counters not found at %s. Run Script 01 first.", pm_path)
        sys.exit(1)

    pm_df = pd.read_parquet(pm_path)
    inventory_df = pd.read_parquet(inv_path) if inv_path.exists() else None
    neighbour_df = pd.read_parquet(nr_path) if nr_path.exists() else None

    logger.info("Loaded %d PM counter records", len(pm_df))

    # Step 1: Build behavioural layer
    profiles = build_cell_profiles(pm_df, inventory_df)

    # Step 2: Build structural layer
    neighbour_map = build_neighbour_map(neighbour_df)

    # Step 3: Run what-if examples
    logger.info("Running what-if simulation examples...")
    example_cells = list(profiles.keys())[:5]
    what_if_examples = []

    for cell_id in example_cells:
        # Example 1: tilt adjustment
        result = predict_what_if(
            cell_id=cell_id,
            parameter_name="electrical_tilt",
            parameter_delta=2.0,  # +2 degrees
            hour=14,  # Peak hour
            profiles=profiles,
            neighbour_map=neighbour_map,
        )
        what_if_examples.append(asdict(result))

        logger.info(
            "  Cell %s, tilt +2°: CQI delta=%.2f, recommendation=%s, blast_radius=%d cells",
            cell_id,
            result.predicted_kpi_deltas.get("avg_cqi", 0),
            result.recommendation,
            len(result.blast_radius_cells),
        )

        # Example 2: power adjustment
        result_power = predict_what_if(
            cell_id=cell_id,
            parameter_name="tx_power",
            parameter_delta=3.0,  # +3 dBm
            hour=14,
            profiles=profiles,
            neighbour_map=neighbour_map,
        )
        what_if_examples.append(asdict(result_power))

    # Step 4: Validate
    validation_results = validate_profiles(profiles, pm_df)

    # Write outputs
    TWIN_DIR.mkdir(parents=True, exist_ok=True)

    # Cell profiles (sample — full profiles can be large)
    sample_profiles = {
        k: {
            "cell_id": v.cell_id,
            "site_id": v.site_id,
            "n_observations": v.n_observations,
            "kpis_tracked": list(v.hourly_profiles.keys()),
            "sample_profile": {
                kpi: vals[:6]  # First 6 hours for readability
                for kpi, vals in list(v.hourly_profiles.items())[:3]
            },
        }
        for k, v in list(profiles.items())[:10]
    }

    with open(TWIN_DIR / "cell_profiles.json", "w") as f:
        json.dump(sample_profiles, f, indent=2)

    # What-if examples
    with open(TWIN_DIR / "what_if_examples.json", "w") as f:
        json.dump(what_if_examples, f, indent=2, default=str)

    # Validation report
    validation_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_cells_profiled": len(profiles),
        "n_kpis": len(next(iter(profiles.values())).hourly_profiles) if profiles else 0,
        "n_validation_pairs": len(validation_results),
        "ci_coverage": (
            round(sum(1 for r in validation_results if r.within_ci) / len(validation_results), 4)
            if validation_results else None
        ),
        "mae": (
            round(float(np.mean([abs(r.error) for r in validation_results])), 4)
            if validation_results else None
        ),
        "sensitivity_constants": {
            "TILT_CQI_SENSITIVITY": TILT_CQI_SENSITIVITY,
            "TILT_HO_SENSITIVITY": TILT_HO_SENSITIVITY,
            "POWER_TPUT_SENSITIVITY": POWER_TPUT_SENSITIVITY,
            "PRB_LOAD_SENSITIVITY": PRB_LOAD_SENSITIVITY,
            "calibration_status": "DEFAULT — recalibrate before production use",
        },
        "what_if_examples_count": len(what_if_examples),
    }

    with open(TWIN_DIR / "validation_report.json", "w") as f:
        json.dump(validation_summary, f, indent=2)

    logger.info("Outputs written to %s", TWIN_DIR)
    logger.info("=" * 70)
    logger.info("Digital twin build complete.")
    logger.info("=" * 70)

    return validation_summary


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
