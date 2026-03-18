#!/usr/bin/env python3
"""
01_synthetic_data.py — Telco MLOps Reference Architecture: Multi-Team, Multi-Model at Scale
============================================================================================
Generates a realistic synthetic telco dataset for demonstrating the MLOps reference
architecture described in the companion whitepaper. The data models a multi-site,
multi-cell LTE/NR network with realistic PM counter correlations, temporal patterns,
counter resets, and injected anomalies with ground truth labels.

Data sources modelled (see whitepaper Section 4 — Data Requirements):
  - Performance Management (PM) counters (TS 32.435 XML schema; TS 28.550 governs PM service architecture)
  - Fault Management (FM) alarm events (VES event format conceptually)
  - Cell configuration metadata (Cell Management, O1 interface)
  - Derived KPIs (RRC success rate, PDCP throughput, PRB utilization)

Network topology:
  - 3 geographic regions (urban, suburban, rural)
  - 5 sites per region, 3 sectors per site → 45 cells total
  - Each cell has 15-minute ROP (Result Output Period) counters per 3GPP TS 28.552

Temporal coverage:
  - 30 days of 15-minute ROPs = 2,880 intervals per cell
  - 45 cells × 2,880 intervals = 129,600 rows (PM counter table)
  - Daily diurnal cycle, weekly seasonality, regional event spikes

Anomaly injection (ground truth for model evaluation):
  - ~2% RRC congestion events (elevated RRC.ConnEstab.Fail)
  - ~1% hardware degradation (gradual RSRP/RSRQ degradation + throughput drop)
  - ~0.5% counter resets (raw counters drop to 0, must be handled downstream)
  - ~1% traffic spikes (stadium events, CBD rush hours)

Usage:
    python 01_synthetic_data.py

Outputs:
    data/pm_counters.parquet        — Main PM counter dataset (129,600 rows)
    data/cell_topology.parquet      — Cell metadata (45 rows)
    data/fm_alarms.parquet          — Alarm events (~350 rows)
    data/anomaly_labels.parquet     — Ground truth anomaly labels

Requirements:
    pip install pandas numpy scipy pyarrow

Coursebook cross-reference:
    - Feature Engineering chapter: temporal patterns, rolling aggregations
    - MLOps Core chapter: data contracts, schema definitions
    - Time Series Analysis chapter: diurnal/weekly seasonality modelling
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

# ---------------------------------------------------------------------------
# Logging configuration — use structured format for production alignment
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("telco_mlops.synthetic_data")

# ---------------------------------------------------------------------------
# Reproducibility seed — must be set globally before any numpy calls
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)  # Use the new Generator API throughout

# ---------------------------------------------------------------------------
# Constants: Network topology and simulation parameters
# ---------------------------------------------------------------------------
N_REGIONS = 3
N_SITES_PER_REGION = 5
N_SECTORS_PER_SITE = 3
N_CELLS = N_REGIONS * N_SITES_PER_REGION * N_SECTORS_PER_SITE  # = 45

# 30 days of 15-minute ROPs; 4 ROPs/hour × 24 hours × 30 days = 2,880
SIM_DAYS = 30
ROP_MINUTES = 15
ROPS_PER_DAY = (24 * 60) // ROP_MINUTES  # = 96
N_ROPS = SIM_DAYS * ROPS_PER_DAY  # = 2,880

SIM_START = datetime(2024, 1, 1, 0, 0, 0)  # UTC

# Anomaly injection rates (fraction of ROPs per cell)
ANOMALY_RRC_CONGESTION_RATE = 0.020   # ~2% of ROPs per cell
ANOMALY_HW_DEGRADATION_RATE = 0.010   # ~1%
ANOMALY_COUNTER_RESET_RATE  = 0.005   # ~0.5%
ANOMALY_TRAFFIC_SPIKE_RATE  = 0.010   # ~1%

# ---------------------------------------------------------------------------
# Regional profile definitions
# Drives capacity, load, and propagation characteristics
# ---------------------------------------------------------------------------
REGION_PROFILES: Dict[str, Dict] = {
    "urban": {
        "description": "Dense urban CBD — high load, small inter-site distance (~300m)",
        "base_prb_utilization_dl": 0.72,   # fraction of available PRBs used (DL)
        "base_prb_utilization_ul": 0.55,
        "base_rsrp_dbm": -85.0,            # typical serving cell RSRP
        "rsrp_std_db": 6.0,                # spatial variation
        "base_dl_throughput_mbps": 250.0,  # median DL cell throughput
        "base_ul_throughput_mbps": 80.0,
        "rrc_attempt_rate": 450,           # RRC attempts per 15-min ROP (per cell)
        "peak_hour_multiplier": 1.8,       # morning/evening rush amplification
        "weekend_multiplier": 0.65,        # weekends lighter in CBD
        "event_spike_probability": 0.05,   # probability of a stochastic event spike per day
    },
    "suburban": {
        "description": "Suburban residential — medium load, inter-site distance ~600m",
        "base_prb_utilization_dl": 0.48,
        "base_prb_utilization_ul": 0.32,
        "base_rsrp_dbm": -90.0,
        "rsrp_std_db": 8.0,
        "base_dl_throughput_mbps": 180.0,
        "base_ul_throughput_mbps": 55.0,
        "rrc_attempt_rate": 250,
        "peak_hour_multiplier": 1.5,
        "weekend_multiplier": 1.10,        # weekends slightly busier (home users)
        "event_spike_probability": 0.02,
    },
    "rural": {
        "description": "Rural coverage — low load, large inter-site distance ~2km",
        "base_prb_utilization_dl": 0.18,
        "base_prb_utilization_ul": 0.12,
        "base_rsrp_dbm": -100.0,
        "rsrp_std_db": 10.0,
        "base_dl_throughput_mbps": 60.0,
        "base_ul_throughput_mbps": 18.0,
        "rrc_attempt_rate": 80,
        "peak_hour_multiplier": 1.3,
        "weekend_multiplier": 0.90,
        "event_spike_probability": 0.01,
    },
}


@dataclass
class CellConfig:
    """
    Per-cell static configuration, analogous to a Cell Management (CM) record
    exposed over O1. Mirrors a 3GPP NRM LTE/NR cell object.
    """
    cell_id: str           # e.g. "CELL_001_A" — site_sector format
    site_id: str           # e.g. "SITE_001"
    region: str            # urban / suburban / rural
    sector: int            # 1, 2, 3 (alpha, beta, gamma)
    latitude: float
    longitude: float
    antenna_height_m: float
    frequency_band: int    # 3GPP band number, e.g. 3 (1800 MHz), 78 (3500 MHz)
    technology: str        # "LTE" or "NR"
    n_prb_dl: int          # Number of PRBs available in DL
    n_prb_ul: int
    # Per-cell noise floor — pulled from regional RSRP base + random offset
    rsrp_base_dbm: float
    # Cell-level load multiplier (heterogeneity within region)
    load_factor: float = 1.0


@dataclass
class AnomalyRecord:
    """Tracks injected anomaly windows for ground truth label generation."""
    cell_id: str
    anomaly_type: str   # "rrc_congestion" | "hw_degradation" | "counter_reset" | "traffic_spike"
    start_rop_idx: int  # Index into the ROP time axis
    duration_rops: int  # How many consecutive ROPs are affected
    severity: float     # 0.0–1.0 for parameterizing the effect magnitude


def _truncated_normal(
    mean: float,
    std: float,
    low: float,
    high: float,
    size: int,
    rng_instance: np.random.Generator = rng,
) -> np.ndarray:
    """
    Draw samples from a truncated normal distribution.

    Using truncated normal (rather than clipping) preserves the distributional
    shape within bounds, which matters for realistic RSRP/RSRQ distributions.

    See Coursebook chapter on Feature Engineering: handling bounded measurements.
    """
    a, b = (low - mean) / std, (high - mean) / std
    # scipy truncnorm is parameterized by standard normal bounds
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size,
                         random_state=rng_instance.integers(0, 2**31))


def generate_cell_topology() -> Tuple[pd.DataFrame, List[CellConfig]]:
    """
    Generate cell site topology with realistic geographic distribution
    and radio configuration parameters.

    Returns:
        topology_df: DataFrame suitable for storage as a CM record table
        cell_configs: List of CellConfig dataclasses for downstream generation

    Topology layout:
        urban sites: clustered around city centre (lat ~51.5, lon ~-0.1)
        suburban sites: ring ~10km radius
        rural sites: sparse, ~40km radius
    """
    logger.info("Generating cell topology for %d cells across %d regions",
                N_CELLS, N_REGIONS)

    cell_configs: List[CellConfig] = []
    region_names = list(REGION_PROFILES.keys())

    # Site centre coordinates (conceptually London UK area — arbitrary but realistic)
    region_centre: Dict[str, Tuple[float, float]] = {
        "urban": (51.507, -0.128),
        "suburban": (51.520, -0.050),
        "rural": (51.450, 0.100),
    }

    site_counter = 0
    cell_counter = 0

    for region in region_names:
        profile = REGION_PROFILES[region]
        centre_lat, centre_lon = region_centre[region]

        # Spread radius for site placement varies by region type
        site_spread_deg = {"urban": 0.03, "suburban": 0.08, "rural": 0.25}[region]

        for site_num in range(N_SITES_PER_REGION):
            site_counter += 1
            site_id = f"SITE_{site_counter:03d}"

            # Jitter site location around regional centre
            site_lat = centre_lat + rng.uniform(-site_spread_deg, site_spread_deg)
            site_lon = centre_lon + rng.uniform(-site_spread_deg, site_spread_deg)

            # Antenna height varies by environment — urban has shorter masts
            antenna_height = {
                "urban": rng.uniform(15, 30),
                "suburban": rng.uniform(25, 45),
                "rural": rng.uniform(35, 60),
            }[region]

            # Band assignment: urban → NR band 78 (3500 MHz), others → LTE band 3
            freq_band = 78 if region == "urban" else 3
            # Simplified: assigns NR to urban and LTE to others for demonstration clarity.
            # Real deployments have mixed NR/LTE across all regions in NSA configurations.
            # technology="NR" → gNB (5G NR base station); technology="LTE" → eNB (LTE base station).
            # The platform feature store handles both technology generations identically
            # via the vendor normalisation layer (§3).
            tech = "NR" if region == "urban" else "LTE"
            # NR 3500 MHz: 106 PRBs (40 MHz NR) vs LTE 1800 MHz: 100 PRBs (20 MHz)
            n_prb = 106 if tech == "NR" else 100

            for sector in range(1, N_SECTORS_PER_SITE + 1):
                cell_counter += 1

                # cell_id format: CELL_{site_number}_{sector_letter}
                # Matches the whitepaper's CELL_XXX_YYY convention
                sector_letter = ["A", "B", "C"][sector - 1]
                cell_id = f"CELL_{site_counter:03d}_{sector_letter}"

                # Per-cell RSRP base: pull from regional profile + cell-specific offset
                rsrp_base = profile["base_rsrp_dbm"] + rng.normal(0, 2.0)
                # Constrain to physically plausible range
                rsrp_base = float(np.clip(rsrp_base, -120.0, -60.0))

                # Cell-level load factor: log-normal to create realistic heterogeneity
                # Some cells carry more traffic than their neighbours (e.g., near a station)
                load_factor = float(np.clip(rng.lognormal(0.0, 0.25), 0.3, 2.5))

                cell = CellConfig(
                    cell_id=cell_id,
                    site_id=site_id,
                    region=region,
                    sector=sector,
                    latitude=round(site_lat + rng.uniform(-0.002, 0.002), 6),
                    longitude=round(site_lon + rng.uniform(-0.002, 0.002), 6),
                    antenna_height_m=round(antenna_height, 1),
                    frequency_band=freq_band,
                    technology=tech,
                    n_prb_dl=n_prb,
                    n_prb_ul=n_prb,
                    rsrp_base_dbm=rsrp_base,
                    load_factor=load_factor,
                )
                cell_configs.append(cell)

    # Build DataFrame
    topology_records = []
    for c in cell_configs:
        topology_records.append({
            "cell_id": c.cell_id,
            "site_id": c.site_id,
            "region": c.region,
            "sector": c.sector,
            "latitude": c.latitude,
            "longitude": c.longitude,
            "antenna_height_m": c.antenna_height_m,
            "frequency_band": c.frequency_band,
            "technology": c.technology,
            "n_prb_dl": c.n_prb_dl,
            "n_prb_ul": c.n_prb_ul,
            "rsrp_base_dbm": c.rsrp_base_dbm,
            "load_factor": c.load_factor,
        })

    topology_df = pd.DataFrame(topology_records)
    logger.info("Topology generated: %d cells, %d sites",
                len(topology_df), topology_df["site_id"].nunique())
    return topology_df, cell_configs


def _build_rop_timestamps(start: datetime, n_rops: int, rop_minutes: int) -> pd.DatetimeIndex:
    """
    Build the ROP timestamp axis.

    In real networks, PM file collection timestamps are UTC-aligned to ROP boundaries.
    The timestamp represents the END of the measurement period (consistent with
    3GPP TS 32.435 file naming conventions).
    """
    freq = f"{rop_minutes}min"
    # Start offset by one ROP so that index[0] = first ROP end time
    ts_start = start + timedelta(minutes=rop_minutes)
    return pd.date_range(start=ts_start, periods=n_rops, freq=freq, tz="UTC")


def _diurnal_load_factor(timestamps: pd.DatetimeIndex, region: str) -> np.ndarray:
    """
    Compute a diurnal (daily cycle) load multiplier for each ROP timestamp.

    Models the typical telco traffic pattern:
      - Night trough: 02:00-05:00 UTC (~0.15× peak)
      - Morning ramp: 06:00-09:00 UTC
      - Business peak: 09:00-12:00 UTC (weekday CBD)
      - Lunchtime secondary peak: 12:00-14:00
      - Evening peak: 18:00-21:00 (highest for residential)
      - Late night decline: 21:00-02:00

    Uses a sum of Gaussians which produces a smooth, realistic shape.
    Alternative approaches (Fourier series, sinusoidal decomposition) are
    discussed in the coursebook Time Series chapter.
    """
    hours = timestamps.hour + timestamps.minute / 60.0  # fractional hour UTC

    # Gaussian component: (centre_hour, amplitude, width_hours)
    # Tuned per region to reflect different traffic archetypes
    if region == "urban":
        components = [
            (8.5,  0.55, 1.5),   # morning commute peak
            (12.5, 0.40, 1.0),   # lunchtime
            (17.5, 0.85, 2.0),   # evening commute peak (highest)
            (21.0, 0.30, 1.5),   # late evening
        ]
        night_floor = 0.10
    elif region == "suburban":
        components = [
            (8.0,  0.35, 1.5),
            (13.0, 0.40, 1.5),
            (19.0, 0.90, 2.5),   # residential evening peak highest
            (22.0, 0.35, 1.5),
        ]
        night_floor = 0.08
    else:  # rural
        components = [
            (9.0,  0.30, 2.0),
            (14.0, 0.45, 2.0),
            (19.5, 0.70, 2.0),
        ]
        night_floor = 0.05

    load = np.full(len(hours), night_floor)
    for centre, amplitude, width in components:
        load += amplitude * np.exp(-0.5 * ((hours - centre) / width) ** 2)

    # Normalise so peak = 1.0
    load = load / load.max()
    return load


def _weekly_load_factor(timestamps: pd.DatetimeIndex, region: str) -> np.ndarray:
    """
    Compute day-of-week multiplier.
    dayofweek: 0=Monday ... 6=Sunday
    """
    profile = REGION_PROFILES[region]
    dow = timestamps.dayofweek
    # Weekdays = 0-4, weekends = 5-6
    multiplier = np.where(dow < 5, 1.0, profile["weekend_multiplier"])
    return multiplier.astype(float)


def generate_pm_counters(
    cell_configs: List[CellConfig],
) -> Tuple[pd.DataFrame, List[AnomalyRecord]]:
    """
    Generate the main PM counter table (N_CELLS × N_ROPS rows).

    Counter naming follows 3GPP TS 28.552 conventions where applicable:
      - RRC.ConnEstab.Att        (RRC connection establishment attempts)
      - RRC.ConnEstab.Succ       (RRC connection establishment successes)
      - DL.PRBUsage.Active       (DL PRB utilization in active subframes)
      - UL.PRBUsage.Active       (UL PRB utilization)
      - PDCP.VolDl               (PDCP SDU volume downlink, bytes)
      - PDCP.VolUl               (PDCP SDU volume uplink, bytes)
      - RSRP.Mean                (Mean RSRP of UEs in cell, dBm — proprietary KPI)
      - RSRQ.Mean                (Mean RSRQ, dB)
      - CQI.Mean                 (Mean CQI reported by UEs, 0-15)
      - PDSCH.BLER               (PDSCH block error rate, fraction)
      - HandoverExec.Att         (Handover execution attempts)
      - HandoverExec.Succ        (Handover execution successes)
      - RRCConn.ActiveUE         (Average connected UE count per ROP)

    All raw counters are 32-bit unsigned integers (realistic for 15-min ROP).
    The RSRP/RSRQ/CQI fields are mean-per-UE floating point values (derived KPIs).

    Returns:
        pm_df: DataFrame with schema described above
        anomaly_records: List of injected anomalies for ground truth
    """
    logger.info("Generating PM counters: %d cells × %d ROPs = %d rows",
                N_CELLS, N_ROPS, N_CELLS * N_ROPS)

    timestamps = _build_rop_timestamps(SIM_START, N_ROPS, ROP_MINUTES)
    all_anomaly_records: List[AnomalyRecord] = []
    cell_dfs: List[pd.DataFrame] = []

    for cell_idx, cell in enumerate(cell_configs):
        if cell_idx % 9 == 0:  # Log progress every 9 cells (every site)
            logger.debug("  Processing cell %d/%d: %s", cell_idx + 1, N_CELLS, cell.cell_id)

        profile = REGION_PROFILES[cell.region]
        n = N_ROPS

        # ------------------------------------------------------------------
        # Step 1: Build the time-varying base load trajectory
        # ------------------------------------------------------------------
        diurnal = _diurnal_load_factor(timestamps, cell.region)
        weekly  = _weekly_load_factor(timestamps, cell.region)

        # Combined temporal load (diurnal × weekly) × cell's static load factor
        base_load = diurnal * weekly * cell.load_factor

        # Add long-term trend: slight linear growth (1.5% over 30 days) — realistic
        # operator networks grow slowly during normal operations
        trend = 1.0 + 0.015 * np.linspace(0, 1, n)
        base_load = base_load * trend

        # Add low-frequency noise (slower than diurnal — represents weather,
        # local events, gradual subscriber behaviour shifts)
        low_freq_noise = _generate_ar_noise(n, ar_coef=0.95, noise_std=0.04, rng=rng)
        base_load = np.clip(base_load + low_freq_noise, 0.01, 1.0)

        # ------------------------------------------------------------------
        # Step 2: Inject anomalies into this cell's time axis
        # ------------------------------------------------------------------
        cell_anomalies = _inject_anomalies(cell.cell_id, n, rng)
        all_anomaly_records.extend(cell_anomalies)

        # Build anomaly masks (boolean arrays over the ROP axis)
        rrc_congestion_mask   = np.zeros(n, dtype=bool)
        hw_degradation_mask   = np.zeros(n, dtype=bool)
        counter_reset_mask    = np.zeros(n, dtype=bool)
        traffic_spike_mask    = np.zeros(n, dtype=bool)

        hw_degradation_factor = np.ones(n, dtype=float)  # multiplicative on RSRP/throughput

        for anomaly in cell_anomalies:
            s = anomaly.start_rop_idx
            e = min(s + anomaly.duration_rops, n)
            if anomaly.anomaly_type == "rrc_congestion":
                rrc_congestion_mask[s:e] = True
            elif anomaly.anomaly_type == "hw_degradation":
                hw_degradation_mask[s:e] = True
                # Ramp degradation factor from 1.0 down to (1 - severity)
                ramp = np.linspace(1.0, 1.0 - anomaly.severity * 0.5, e - s)
                hw_degradation_factor[s:e] = ramp
            elif anomaly.anomaly_type == "counter_reset":
                counter_reset_mask[s:e] = True
            elif anomaly.anomaly_type == "traffic_spike":
                traffic_spike_mask[s:e] = True

        # ------------------------------------------------------------------
        # Step 3: Generate correlated PM counters
        # ------------------------------------------------------------------

        # --- RRC connections ---
        # Base RRC attempts: Poisson-distributed around rate × base_load
        rrc_attempt_rate = profile["rrc_attempt_rate"] * base_load
        # Traffic spikes increase RRC attempts substantially
        rrc_attempt_rate = np.where(traffic_spike_mask,
                                    rrc_attempt_rate * rng.uniform(2.0, 4.0, n),
                                    rrc_attempt_rate)
        rrc_att = rng.poisson(np.maximum(rrc_attempt_rate, 1.0)).astype(np.int32)

        # Normal RRC success rate: 98–99.5% (very high in healthy network)
        base_rrc_sr = rng.uniform(0.980, 0.995, n)
        # Congestion events drop success rate significantly (rejects due to overload)
        rrc_sr = np.where(rrc_congestion_mask,
                          rng.uniform(0.70, 0.88, n),
                          base_rrc_sr)
        rrc_succ = np.floor(rrc_att * rrc_sr).astype(np.int32)

        # --- PRB utilization ---
        # DL: strongly correlated with base_load
        dl_prb_mean = profile["base_prb_utilization_dl"] * base_load
        dl_prb_mean = np.clip(dl_prb_mean, 0.01, 0.99)
        # Traffic spikes push PRB utilization toward saturation
        dl_prb_mean = np.where(traffic_spike_mask,
                               np.minimum(dl_prb_mean * 1.4, 0.97), dl_prb_mean)
        # Small measurement noise on PRB (quantized to 1/n_prb granularity)
        dl_prb_raw = dl_prb_mean + rng.normal(0, 0.015, n)
        dl_prb_active = np.clip(dl_prb_raw, 0.0, 1.0)

        ul_prb_mean = profile["base_prb_utilization_ul"] * base_load
        ul_prb_mean = np.clip(ul_prb_mean, 0.01, 0.99)
        ul_prb_active = np.clip(ul_prb_mean + rng.normal(0, 0.012, n), 0.0, 1.0)

        # Convert to integer PRB count (the actual counter value)
        # 3GPP TS 28.552 reports mean active PRB count over ROP
        dl_prb_count = np.round(dl_prb_active * cell.n_prb_dl).astype(np.int32)
        ul_prb_count = np.round(ul_prb_active * cell.n_prb_ul).astype(np.int32)

        # --- PDCP volume (bytes) ---
        # DL throughput in bps: PRB utilisation × spectral efficiency × bandwidth
        # Spectral efficiency varies with CQI; use simplified linear model
        # NR 3500 MHz: ~12.5 bit/s/Hz, LTE 1800 MHz: ~6 bit/s/Hz (peak)
        # SE degrades with hardware degradation
        se_factor = 12.5 if cell.technology == "NR" else 6.0
        # Effective SE scales with PRB utilization and degrades with hw_degradation_factor
        dl_bw_hz = cell.n_prb_dl * 180e3  # 180 kHz per PRB
        # NOTE: dl_throughput_mbps is a simplified peak-rate estimate at
        # current PRB utilisation, not a true ROP-averaged measurement.
        # Real DRB.UEThpDl values will differ due to sub-ROP traffic
        # variability. See §3 counter mapping table for the conversion.
        dl_throughput_bps_est = (dl_prb_active * dl_bw_hz * se_factor
                             * hw_degradation_factor)
        dl_throughput_bps_est = np.maximum(dl_throughput_bps_est, 0.0)

        # PDCP volume = throughput_bps × ROP_duration_sec / 8 (bytes)
        rop_sec = ROP_MINUTES * 60
        pdcp_vol_dl = (dl_throughput_bps_est * rop_sec / 8).astype(np.int64)

        ul_bw_hz = cell.n_prb_ul * 180e3
        ul_se_factor = se_factor * 0.45  # UL SE lower than DL
        ul_throughput_bps = (ul_prb_active * ul_bw_hz * ul_se_factor
                             * hw_degradation_factor)
        pdcp_vol_ul = (np.maximum(ul_throughput_bps, 0) * rop_sec / 8).astype(np.int64)

        # --- Signal quality: RSRP, RSRQ, CQI ---
        # RSRP base from cell config, with per-ROP variation
        rsrp_ts = cell.rsrp_base_dbm + rng.normal(0, 2.5, n)
        # Hardware degradation causes RSRP to drop (e.g., antenna connector issue)
        rsrp_ts = rsrp_ts + (hw_degradation_factor - 1.0) * 8.0  # up to -4 dBm additional loss
        rsrp_ts = np.clip(rsrp_ts, -140.0, -44.0)  # Physical bounds (3GPP TS 38.133)

        # RSRQ correlates with RSRP but also with interference (PRB load):
        # high PRB load → more interference → lower RSRQ
        #
        # ⚠️ SYNTHETIC RSRQ IS NOT PHYSICALLY VALID. The formula
        # (rsrp_ts / 6.0) - interference_penalty produces values in the correct
        # range [-19.5, -3] dB but has no relationship to the 3GPP RSRQ definition
        # (N × RSRP / RSSI, TS 38.215). The linear dBm scaling creates a
        # correlation structure with RSRP that is fundamentally different from
        # field measurements. Models trained on synthetic RSRQ features may fail
        # to generalise to real network data.
        #
        # For any model that uses RSRQ as a feature, validate exclusively against
        # real operator data before deployment.
        #
        # PRODUCTION: Use vendor-reported RSRQ directly from PM counters
        # (e.g., Nokia RRH.RSRQ.UE.Avg, Ericsson pmRadioRsrqAvg).
        # Do not recompute from RSRP.
        #
        # NOTE: Heavy clipping occurs for cells with RSRP < -117 dBm, producing
        # artificial RSRQ = -19.5 dB floor. This is a synthetic data artefact;
        # production RSRQ distributions should not show this pattern.
        rsrq_interference_penalty = dl_prb_active * 5.0  # 0–5 dB penalty at full load
        rsrq_base = (rsrp_ts / 6.0) - rsrq_interference_penalty
        rsrq_ts = np.clip(rsrq_base + rng.normal(0, 1.0, n), -19.5, -3.0)  # TS 38.133 valid range
        # MULTICOLLINEARITY NOTE: In this synthetic data, RSRQ = f(RSRP) + noise,
        # creating near-perfect linear correlation. Real RSRQ depends on RSSI
        # (total interference + noise), not just RSRP. Models using both RSRP
        # and RSRQ features will exhibit inflated importance for one of the pair
        # in synthetic data. When evaluating feature importance on synthetic data,
        # treat RSRP and RSRQ importance as interchangeable.

        n_clipped = int((rsrq_base < -19.5).sum())
        if n_clipped > 0.05 * n:
            logger.warning(
                "Cell %s: %.1f%% of RSRQ values clipped at -19.5 dB floor "
                "(synthetic artifact — see inline WARNING)",
                cell.cell_id, 100.0 * n_clipped / n,
            )

        # CQI: integer 0–15; correlates with RSRQ
        # Map RSRQ range [-20, -3] to CQI range [0, 15]
        cqi_float = (rsrq_ts - (-20.0)) / ((-3.0) - (-20.0)) * 15.0
        cqi_noise = rng.normal(0, 0.8, n)
        cqi_ts = np.clip(np.round(cqi_float + cqi_noise), 0, 15).astype(np.int8)

        # --- Block Error Rate (BLER) ---
        # BLER is inversely correlated with CQI; high CQI → low BLER
        # Using exponential relationship: BLER ≈ exp(-0.5 * CQI)
        bler_base = np.exp(-0.4 * cqi_ts.astype(float)) * 0.15
        bler_ts = np.clip(bler_base + rng.normal(0, 0.005, n), 0.0, 0.5)

        # --- Handovers ---
        # Handover attempts scale with connected UE count and mobility
        # Urban → more mobility → more HOs per UE
        ho_mobility_factor = {"urban": 0.12, "suburban": 0.07, "rural": 0.03}[cell.region]
        connected_ue_base = profile["rrc_attempt_rate"] * base_load * 0.8 / 4
        connected_ue = np.maximum(
            rng.poisson(np.maximum(connected_ue_base, 0.1)).astype(np.int32), 0
        )
        ho_att = rng.poisson(
            np.maximum(connected_ue * ho_mobility_factor, 0.01)
        ).astype(np.int32)
        # HO success rate: 97–99% normally, degrades with hardware issues
        ho_sr_base = rng.uniform(0.970, 0.995, n)
        ho_sr = np.where(hw_degradation_mask, rng.uniform(0.88, 0.95, n), ho_sr_base)
        ho_succ = np.floor(ho_att * ho_sr).astype(np.int32)

        # --- Counter reset injection ---
        # When a counter reset occurs, raw cumulative counters drop to near-zero
        # This is a realistic artifact from ROP measurement collection gaps or
        # equipment restarts (relevant to feature engineering — see 02 script)
        # We model this as zeroing out the affected ROPs
        if counter_reset_mask.any():
            rrc_att[counter_reset_mask]     = 0
            rrc_succ[counter_reset_mask]    = 0
            dl_prb_count[counter_reset_mask] = 0
            ul_prb_count[counter_reset_mask] = 0
            pdcp_vol_dl[counter_reset_mask]  = 0
            pdcp_vol_ul[counter_reset_mask]  = 0
            ho_att[counter_reset_mask]       = 0
            ho_succ[counter_reset_mask]      = 0
            connected_ue[counter_reset_mask] = 0

        # ------------------------------------------------------------------
        # Step 4: Build per-cell DataFrame
        # ------------------------------------------------------------------
        cell_df = pd.DataFrame({
            "timestamp":            timestamps,
            "cell_id":              cell.cell_id,
            "site_id":              cell.site_id,
            "region":               cell.region,
            "technology":           cell.technology,
            # 3GPP TS 28.552 PM counters (integer counts per ROP)
            "rrc_conn_estab_att":   rrc_att,
            "rrc_conn_estab_succ":  rrc_succ,
            "dl_prb_usage_active":  dl_prb_count,
            "ul_prb_usage_active":  ul_prb_count,
            "pdcp_vol_dl_bytes":    pdcp_vol_dl,
            "pdcp_vol_ul_bytes":    pdcp_vol_ul,
            "ho_exec_att":          ho_att,
            "ho_exec_succ":         ho_succ,
            "rrc_conn_active_ue":   connected_ue,
            # Derived/aggregated KPIs (floating point — mean per UE per ROP)
            "rsrp_mean_dbm":        np.round(rsrp_ts, 2),
            "rsrq_mean_db":         np.round(rsrq_ts, 2),
            "cqi_mean":             cqi_ts,
            "pdsch_bler":           np.round(bler_ts, 4),
            # Convenience float KPIs derived from raw counters
            "dl_prb_utilization":   np.round(dl_prb_active, 4),
            "ul_prb_utilization":   np.round(ul_prb_active, 4),
            "dl_throughput_mbps":   np.round(dl_throughput_bps_est / 1e6, 3),
            "ul_throughput_mbps":   np.round(ul_throughput_bps / 1e6, 3),
            # Ground truth anomaly flags (used in evaluation)
            "anomaly_rrc_congestion":  rrc_congestion_mask.astype(np.int8),
            "anomaly_hw_degradation":  hw_degradation_mask.astype(np.int8),
            "anomaly_counter_reset":   counter_reset_mask.astype(np.int8),
            "anomaly_traffic_spike":   traffic_spike_mask.astype(np.int8),
        })

        cell_dfs.append(cell_df)

    pm_df = pd.concat(cell_dfs, ignore_index=True)

    # Add a composite anomaly label (any anomaly present)
    pm_df["is_anomaly"] = (
        pm_df[["anomaly_rrc_congestion",
               "anomaly_hw_degradation",
               "anomaly_counter_reset",
               "anomaly_traffic_spike"]].any(axis=1).astype(np.int8)
    )

    logger.info("PM counter table: %d rows, %d columns", len(pm_df), len(pm_df.columns))
    _log_anomaly_stats(pm_df)
    return pm_df, all_anomaly_records


def _generate_ar_noise(
    n: int,
    ar_coef: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate AR(1) noise: x[t] = ar_coef × x[t-1] + ε[t], ε ~ N(0, noise_std).

    Using AR(1) rather than i.i.d. noise produces autocorrelated fluctuations
    that better represent real network load variations (consecutive ROPs are
    correlated — a busy ROP tends to be followed by another busy ROP).

    See Coursebook chapter on Time Series Analysis: AR processes.
    """
    noise = np.zeros(n)
    noise[0] = rng.normal(0, noise_std)
    for t in range(1, n):
        noise[t] = ar_coef * noise[t - 1] + rng.normal(0, noise_std)
    return noise


def _inject_anomalies(
    cell_id: str,
    n_rops: int,
    rng: np.random.Generator,
) -> List[AnomalyRecord]:
    """
    Determine which ROPs in a cell's timeline should have injected anomalies.

    Anomaly types and their realistic characteristics:
    - rrc_congestion: Short bursts (2–8 ROPs = 30min–2h), cluster during peak hours
    - hw_degradation: Longer episodes (12–48 ROPs = 3–12h), any time
    - counter_reset:  Very short (1–2 ROPs), random
    - traffic_spike:  Medium duration (4–16 ROPs = 1–4h), peak hours preferred

    Returns list of AnomalyRecord instances for this cell.
    """
    records: List[AnomalyRecord] = []

    anomaly_specs = [
        ("rrc_congestion",  ANOMALY_RRC_CONGESTION_RATE, 2,  8,  0.3, 0.9),
        ("hw_degradation",  ANOMALY_HW_DEGRADATION_RATE, 12, 48, 0.2, 0.8),
        ("counter_reset",   ANOMALY_COUNTER_RESET_RATE,  1,  2,  1.0, 1.0),
        ("traffic_spike",   ANOMALY_TRAFFIC_SPIKE_RATE,  4,  16, 0.5, 1.0),
    ]

    for atype, rate, dur_min, dur_max, sev_low, sev_high in anomaly_specs:
        # Expected number of anomaly events for this cell
        expected_count = rate * n_rops / ((dur_min + dur_max) / 2)
        n_events = rng.poisson(max(expected_count, 0))

        for _ in range(n_events):
            start_idx = int(rng.integers(0, n_rops - dur_max))
            duration = int(rng.integers(dur_min, dur_max + 1))
            severity = float(rng.uniform(sev_low, sev_high))
            records.append(AnomalyRecord(
                cell_id=cell_id,
                anomaly_type=atype,
                start_rop_idx=start_idx,
                duration_rops=duration,
                severity=severity,
            ))

    return records


def _log_anomaly_stats(pm_df: pd.DataFrame) -> None:
    """Log anomaly injection statistics for quality assurance."""
    total_rows = len(pm_df)
    for col in ["anomaly_rrc_congestion", "anomaly_hw_degradation",
                "anomaly_counter_reset", "anomaly_traffic_spike", "is_anomaly"]:
        count = pm_df[col].sum()
        pct = 100.0 * count / total_rows
        logger.info("  %-30s: %6d rows (%5.2f%%)", col, count, pct)


def generate_fm_alarms(
    cell_configs: List[CellConfig],
    anomaly_records: List[AnomalyRecord],
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Generate a Fault Management (FM) alarm table corresponding to injected anomalies.

    In real networks, FM alarms are raised by the Network Management System (NMS)
    or by the network elements themselves and forwarded via VES events (ONAP VES 7.x)
    or Netconf/YANG notifications over O1. Alarms correlate with — but do not
    perfectly map to — PM anomaly events (alarm detection lags PM degradation by
    1–3 ROPs typically).

    Alarm schema fields align with ETSI ES 203 228 alarm model and TM Forum TMF642.
    """
    logger.info("Generating FM alarms from %d injected anomaly events",
                len(anomaly_records))

    # Map anomaly type → alarm definitions
    alarm_definitions = {
        "rrc_congestion": {
            "alarm_type":     "CONGESTION",
            "probable_cause": "congestion",
            "specific_problem": "RRC Connection Setup Failure Rate Exceeded Threshold",
            "perceived_severity": "MAJOR",
            "managed_object_class": "ENodeBFunction",
            "detection_lag_min": 1,  # ROPs of lag before alarm fires
            "detection_lag_max": 3,
            "false_negative_rate": 0.25,  # 25% of events don't trigger an alarm
        },
        "hw_degradation": {
            "alarm_type":     "EQUIPMENT_ALARM",
            "probable_cause": "hardware-failure",
            "specific_problem": "Antenna VSWR Degradation Detected",
            "perceived_severity": "CRITICAL",
            "managed_object_class": "AntennaUnitGroup",
            "detection_lag_min": 4,
            "detection_lag_max": 12,
            "false_negative_rate": 0.10,
        },
        "traffic_spike": {
            "alarm_type":     "CONGESTION",
            "probable_cause": "bandwidth-reduced",
            "specific_problem": "DL PRB Utilization Sustained Above 90%",
            "perceived_severity": "WARNING",
            "managed_object_class": "EUtranCellFDD",
            "detection_lag_min": 2,
            "detection_lag_max": 5,
            "false_negative_rate": 0.40,  # Many spikes are short and auto-clear
        },
    }

    # counter_reset events don't generate alarms (silent data quality issue)
    alarm_records = []
    alarm_id_counter = 1

    # Build a cell_id → CellConfig lookup for managed object distinguished name
    cell_map = {c.cell_id: c for c in cell_configs}

    for anomaly in anomaly_records:
        if anomaly.anomaly_type not in alarm_definitions:
            continue

        defn = alarm_definitions[anomaly.anomaly_type]

        # Apply false negative rate — not every anomaly generates an alarm
        if rng.random() < defn["false_negative_rate"]:
            continue

        # Alarm raise time = anomaly start + detection lag
        lag_rops = int(rng.integers(defn["detection_lag_min"],
                                    defn["detection_lag_max"] + 1))
        raise_rop_idx = min(anomaly.start_rop_idx + lag_rops, N_ROPS - 1)
        raise_time = timestamps[raise_rop_idx]

        # Alarm clear time = anomaly end + 1–4 additional ROPs (operator clears it)
        clear_rop_idx = min(
            anomaly.start_rop_idx + anomaly.duration_rops
            + int(rng.integers(1, 5)),
            N_ROPS - 1,
        )
        clear_time = timestamps[clear_rop_idx]

        cell = cell_map.get(anomaly.cell_id)
        if cell is None:
            continue

        # Build a pseudo-3GPP Distinguished Name for the managed object
        dn = (f"SubNetwork=MNO,MeContext={cell.site_id},"
              f"ManagedElement={cell.site_id},{defn['managed_object_class']}={cell.cell_id}")

        alarm_records.append({
            "alarm_id":              f"ALARM_{alarm_id_counter:06d}",
            "cell_id":               anomaly.cell_id,
            "site_id":               cell.site_id,
            "region":                cell.region,
            "raise_time":            raise_time,
            "clear_time":            clear_time,
            "duration_minutes":      int((clear_time - raise_time).total_seconds() / 60),
            "alarm_type":            defn["alarm_type"],
            "probable_cause":        defn["probable_cause"],
            "specific_problem":      defn["specific_problem"],
            "perceived_severity":    defn["perceived_severity"],
            "managed_object_class":  defn["managed_object_class"],
            "managed_object_dn":     dn,
            "source_anomaly_type":   anomaly.anomaly_type,
            "anomaly_severity":      round(anomaly.severity, 3),
            # VES-style metadata (ONAP VES 7.x)
            "ves_event_type":        "faultNotification",
            "ves_event_id":          f"VES_{alarm_id_counter:08d}",
            "ves_sequence_number":   alarm_id_counter,
        })
        alarm_id_counter += 1

    alarms_df = pd.DataFrame(alarm_records)

    if len(alarms_df) > 0:
        alarms_df = alarms_df.sort_values("raise_time").reset_index(drop=True)
        logger.info("FM alarms generated: %d alarms from %d anomaly events",
                    len(alarms_df), len(anomaly_records))
        logger.info("  Severity distribution:\n%s",
                    alarms_df["perceived_severity"].value_counts().to_string())
    else:
        logger.warning("No FM alarms generated — check anomaly injection rates")

    return alarms_df


def generate_anomaly_labels(
    pm_df: pd.DataFrame,
    anomaly_records: List[AnomalyRecord],
) -> pd.DataFrame:
    """
    Generate a clean, separate anomaly label table for model evaluation.

    This is the ground truth used in evaluation (04_evaluation.py).
    Kept separate from PM counters to simulate the real-world scenario where
    labels are curated offline by SMEs reviewing PM + FM data.

    Schema includes anomaly type, severity, and affected cell/time window
    for use in multi-class anomaly detection evaluation.
    """
    label_records = []
    for anomaly in anomaly_records:
        label_records.append({
            "cell_id":        anomaly.cell_id,
            "anomaly_type":   anomaly.anomaly_type,
            "start_rop_idx":  anomaly.start_rop_idx,
            "duration_rops":  anomaly.duration_rops,
            "severity":       round(anomaly.severity, 3),
        })

    labels_df = pd.DataFrame(label_records)

    if len(labels_df) > 0:
        logger.info("Anomaly label table: %d records", len(labels_df))
        logger.info("  Type distribution:\n%s",
                    labels_df["anomaly_type"].value_counts().to_string())

    return labels_df


def _validate_pm_counters(pm_df: pd.DataFrame) -> None:
    """
    Basic schema and range validation — analogous to a Great Expectations suite.

    Production use would use Great Expectations or Pandera with full expectation
    suites. Here we use explicit assertions to keep the script dependency-light
    while demonstrating the validation intent.

    See Coursebook MLOps Core chapter: data contracts and validation gates.
    """
    logger.info("Running data validation checks...")

    # Schema checks
    required_cols = [
        "timestamp", "cell_id", "site_id", "region",
        "rrc_conn_estab_att", "rrc_conn_estab_succ",
        "dl_prb_utilization", "ul_prb_utilization",
        "rsrp_mean_dbm", "rsrq_mean_db", "cqi_mean",
        "dl_throughput_mbps", "ul_throughput_mbps",
        "is_anomaly",
    ]
    missing = set(required_cols) - set(pm_df.columns)
    assert not missing, f"Missing columns: {missing}"

    # Range checks for telco KPIs
    assert pm_df["rsrp_mean_dbm"].between(-140, -44).all(), \
        "RSRP values out of valid range [-140, -44] dBm"
    assert pm_df["rsrq_mean_db"].between(-19.5, -3).all(), \
        "RSRQ values out of valid range [-20, -3] dB"
    assert pm_df["cqi_mean"].between(0, 15).all(), \
        "CQI values out of valid range [0, 15]"
    assert (pm_df["dl_prb_utilization"] >= 0).all() and \
           (pm_df["dl_prb_utilization"] <= 1.0).all(), \
        "DL PRB utilization out of [0, 1] range"
    assert (pm_df["rrc_conn_estab_succ"] <= pm_df["rrc_conn_estab_att"]).all(), \
        "RRC successes exceed attempts — data generation error"
    assert (pm_df["ho_exec_succ"] <= pm_df["ho_exec_att"]).all(), \
        "HO successes exceed attempts — data generation error"

    # Temporal ordering
    cell_sample = pm_df[pm_df["cell_id"] == pm_df["cell_id"].iloc[0]]
    assert cell_sample["timestamp"].is_monotonic_increasing, \
        "Timestamps not monotonically increasing for sample cell"

    # Anomaly rate sanity (must be within reasonable bounds)
    anomaly_rate = pm_df["is_anomaly"].mean()
    assert 0.005 <= anomaly_rate <= 0.15, \
        f"Anomaly rate {anomaly_rate:.3f} outside expected [0.5%, 15%] range"

    logger.info("  All %d validation checks passed. Anomaly rate: %.2f%%",
                len(required_cols) + 5, 100.0 * anomaly_rate)


def save_datasets(
    topology_df: pd.DataFrame,
    pm_df: pd.DataFrame,
    alarms_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Persist all generated datasets to Parquet files.

    Parquet is used because:
    1. Column-oriented storage is efficient for PM counter queries (select few columns)
    2. Preserves dtypes including timestamps with timezone
    3. Directly readable by Spark, Pandas, DuckDB — matches production data lake patterns
    4. Snappy compression reduces storage by ~3-5× vs CSV

    See Coursebook MLOps Core chapter: data versioning and lineage.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: Dict[str, Path] = {}

    datasets = {
        "cell_topology": (topology_df, "Cell topology: site and cell configuration metadata"),
        "pm_counters": (pm_df, "PM counter table: main KPI dataset"),
        "fm_alarms": (alarms_df, "FM alarm events table"),
        "anomaly_labels": (labels_df, "Anomaly ground truth labels"),
    }

    for name, (df, description) in datasets.items():
        fpath = output_dir / f"{name}.parquet"
        df.to_parquet(fpath, index=False, compression="snappy")
        size_mb = fpath.stat().st_size / (1024 ** 2)
        logger.info("Saved %-20s → %s (%.1f MB, %d rows)",
                    f"{name}.parquet", fpath, size_mb, len(df))
        output_paths[name] = fpath

    return output_paths


def print_dataset_summary(
    topology_df: pd.DataFrame,
    pm_df: pd.DataFrame,
    alarms_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> None:
    """
    Print a structured summary of the generated datasets to stdout.
    Useful for verifying data generation before running downstream scripts.
    """
    sep = "=" * 70
    logger.info(sep)
    logger.info("DATASET SUMMARY")
    logger.info(sep)

    logger.info("TOPOLOGY:")
    logger.info("  Cells: %d across %d sites, %d regions",
                len(topology_df),
                topology_df["site_id"].nunique(),
                topology_df["region"].nunique())
    logger.info("  Technologies: %s",
                topology_df["technology"].value_counts().to_dict())
    logger.info("  Regions: %s",
                topology_df["region"].value_counts().to_dict())

    logger.info("PM COUNTERS:")
    logger.info("  Rows: %d (cells=%d × ROPs=%d)",
                len(pm_df), N_CELLS, N_ROPS)
    logger.info("  Time range: %s → %s",
                pm_df["timestamp"].min(), pm_df["timestamp"].max())
    logger.info("  DL throughput [Mbps]: mean=%.1f, p50=%.1f, p95=%.1f, max=%.1f",
                pm_df["dl_throughput_mbps"].mean(),
                pm_df["dl_throughput_mbps"].quantile(0.50),
                pm_df["dl_throughput_mbps"].quantile(0.95),
                pm_df["dl_throughput_mbps"].max())
    logger.info("  RSRP [dBm]: mean=%.1f, std=%.1f, min=%.1f, max=%.1f",
                pm_df["rsrp_mean_dbm"].mean(),
                pm_df["rsrp_mean_dbm"].std(),
                pm_df["rsrp_mean_dbm"].min(),
                pm_df["rsrp_mean_dbm"].max())
    logger.info("  PRB utilization DL: mean=%.3f, p95=%.3f",
                pm_df["dl_prb_utilization"].mean(),
                pm_df["dl_prb_utilization"].quantile(0.95))
    logger.info("  Anomaly rows: %d (%.2f%%)",
                pm_df["is_anomaly"].sum(),
                100.0 * pm_df["is_anomaly"].mean())

    logger.info("FM ALARMS:")
    logger.info("  Total alarms: %d", len(alarms_df))
    if len(alarms_df) > 0:
        logger.info("  Severity breakdown: %s",
                    alarms_df["perceived_severity"].value_counts().to_dict())
        logger.info("  Median alarm duration: %.0f min",
                    alarms_df["duration_minutes"].median())

    logger.info("ANOMALY LABELS:")
    logger.info("  Total anomaly events: %d", len(labels_df))
    if len(labels_df) > 0:
        logger.info("  Type breakdown: %s",
                    labels_df["anomaly_type"].value_counts().to_dict())
        logger.info("  Severity: mean=%.2f, min=%.2f, max=%.2f",
                    labels_df["severity"].mean(),
                    labels_df["severity"].min(),
                    labels_df["severity"].max())
    logger.info(sep)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Telco MLOps Reference Architecture — Synthetic Data Generator")
    logger.info("=" * 70)
    logger.info("Simulation parameters:")
    logger.info("  Cells:       %d (%d regions × %d sites × %d sectors)",
                N_CELLS, N_REGIONS, N_SITES_PER_REGION, N_SECTORS_PER_SITE)
    logger.info("  ROPs:        %d (%d days × %d ROPs/day @ %d min)",
                N_ROPS, SIM_DAYS, ROPS_PER_DAY, ROP_MINUTES)
    logger.info("  Total rows:  %d", N_CELLS * N_ROPS)
    logger.info("  Random seed: %d", RANDOM_SEED)
    logger.info("  Start time:  %s UTC", SIM_START.isoformat())

    # Step 1: Generate network topology
    topology_df, cell_configs = generate_cell_topology()

    # Step 2: Generate PM counter time series (main dataset)
    pm_df, anomaly_records = generate_pm_counters(cell_configs)

    # Step 3: Generate FM alarms correlated with anomalies
    timestamps = _build_rop_timestamps(SIM_START, N_ROPS, ROP_MINUTES)
    alarms_df = generate_fm_alarms(cell_configs, anomaly_records, timestamps)

    # Step 4: Generate ground truth anomaly labels
    labels_df = generate_anomaly_labels(pm_df, anomaly_records)

    # Step 5: Validate the PM counter dataset
    _validate_pm_counters(pm_df)

    # Step 6: Print summary
    print_dataset_summary(topology_df, pm_df, alarms_df, labels_df)

    # Step 7: Save to Parquet
    output_dir = Path("data")
    output_paths = save_datasets(topology_df, pm_df, alarms_df, labels_df, output_dir)

    logger.info("Data generation complete. Files written to: %s", output_dir.resolve())
    logger.info("Next step: python 02_feature_engineering.py")
