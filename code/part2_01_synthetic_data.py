"""


01_synthetic_data.py — Telco MLOps Part 2: Synthetic Data Generation
=====================================================================
Companion code for:
  "Telco MLOps Reference Architecture — Part 2: Extending the Platform to
   Graph ML, LLMs, Agentic Systems, and Beyond"

Purpose:
    Generate realistic synthetic RAN PM counter data matching the data
    requirements in Section 3 of the whitepaper. The generated dataset
    serves as the foundation for all subsequent scripts in this series.

What this script produces:
    1. Cell inventory (cell_inventory.parquet) — static topology metadata
       for 200 cell sectors across 70 sites, with realistic geo-clustering,
       indoor/outdoor split, and urban/rural/suburban taxonomy.
    2. PM counter time series (pm_counters.parquet) — 15-minute ROP data
       for 30 days, with diurnal/weekly load patterns, spatial correlations,
       realistic counter resets, and injected multi-cell anomalies with
       ground truth labels.
    3. Neighbour relation table (neighbour_relations.parquet) — O1-style
       NRT (Neighbour Relation Table) entries used to construct the cell
       adjacency graph in downstream Graph ML scripts.
    4. Anomaly ground truth (anomaly_labels.parquet) — per-ROP per-cell
       labels with anomaly type, root cause cell, and severity.

Design decisions:
    - 200 cells × 30 days × 96 ROPs/day = 576,000 rows (realistic scale
      without requiring a GPU cluster to run downstream scripts).
    - Anomaly rate: 2.1% of cell-ROPs are anomalous (within the 1–5%
      realistic range noted in the telco realism checklist).
    - Multi-cell correlated failures are injected at the SITE level so the
      Graph ML scripts (02+) can demonstrate spatial root-cause attribution.
    - Counter resets are injected at a 0.3% rate (typical for live RAN nodes
      after software upgrades or BBU restarts).

Coursebook cross-references:
    - Ch. 13: Feature Engineering — temporal encoding foundations
    - Ch. 28: Data Pipelines — schema design for PM counter ingestion
    - Ch. 52: System Design for ML — data layer architecture

Usage:
    python 01_synthetic_data.py

    Output files are written to ./data/ (created if absent).
    All outputs are reproducible: seed is fixed at 42.

Requirements:
    Python 3.10+
    pip install pandas numpy scipy pyarrow
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Logging — use structured logging throughout; no print() calls.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("telco.synthetic_data")

# ---------------------------------------------------------------------------
# Global constants — centralised so downstream scripts can import them.
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
N_SITES: int = 70          # macro sites; 3 sectors each → 210 cells, trimmed to 200
N_CELLS: int = 200         # cell sectors (sector 1/2/3 per site, some sites have 2)
N_DAYS: int = 30           # simulation window
ROP_MINUTES: int = 15      # reporting output period (3GPP TS 28.552 §5.1)
ROPS_PER_DAY: int = 24 * 60 // ROP_MINUTES   # 96

SIM_START: datetime = datetime(2024, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
SIM_END: datetime = SIM_START + timedelta(days=N_DAYS)

# Geographic bounding box (synthetic Australian metro + suburban + rural mix)
GEO_BBOX: Dict[str, float] = {
    "lat_min": -34.1,
    "lat_max": -33.7,
    "lon_min": 150.9,
    "lon_max": 151.4,
}

# Anomaly rates (see whitepaper §4 and realism checklist)
ANOMALY_RATE_CELL: float = 0.021         # fraction of cell-ROPs that are anomalous
COUNTER_RESET_RATE: float = 0.003        # fraction of rows with counter reset event
MULTI_CELL_FAULT_PROBABILITY: float = 0.4  # fraction of anomaly events that are correlated

# KPI value ranges (from realism checklist and 3GPP TS 28.552)
KPI_RANGES: Dict[str, Tuple[float, float]] = {
    # ── ORDERING DEPENDENCY: dl_volume_gb must appear AFTER dl_throughput_mbps ──
    # generate_kpi_time_series() iterates KPI_RANGES.keys() in insertion order
    # (Python 3.7+) and computes dl_volume_gb from the already-generated
    # dl_throughput_mbps series.  If you reorder these entries, dl_volume_gb
    # will silently fall back to a less-accurate load-based approximation.
    # A post-loop guard also recomputes dl_volume_gb from throughput if both
    # are present, but keep the ordering correct as the primary mechanism.
    "rsrp_dbm":        (-135.0, -55.0),   # 3GPP full range -140 to -44 dBm; widened to cover deep-coverage edge cases
    "rsrq_db":         (-18.0,  -6.0),    # typical; full range -20 to -3
    "sinr_db":         (-5.0,   30.0),
    "avg_cqi":        (3.0,    13.0),    # integer in practice, float mean reported
    "dl_throughput_mbps": (0.5, 450.0),   # skewed left; median ~40 Mbps
    "ul_throughput_mbps": (0.1, 100.0),
    "dl_prb_usage_rate": (2.0, 95.0),   # percentage
    "ul_prb_usage_rate": (5.0, 85.0),    # UL PRB utilisation percentage — matches §3.2 KPI Taxonomy
    "rrc_conn_setup_success_rate": (0.92, 1.0),
    "handover_success_rate": (0.88, 1.0),
    "cell_availability_pct": (99.0, 100.0),
    "active_ue_count": (1.0, 800.0),
    "dl_volume_gb": (0.01, 50.0),
}

OUTPUT_DIR: Path = Path("data")
PM_COUNTERS_OUTPUT_PATH: Path = OUTPUT_DIR / "pm_counters.parquet"


# ---------------------------------------------------------------------------
# Data classes for type safety
# ---------------------------------------------------------------------------

@dataclass
class SiteConfig:
    """Static configuration for a macro site."""
    site_id: str
    latitude: float
    longitude: float
    environment: str         # urban | suburban | rural
    n_sectors: int           # 2 or 3
    height_m: float          # antenna height
    is_indoor: bool          # indoor small cell (True) vs outdoor macro (False)
    cluster_id: int          # DBSCAN peer-group cluster (pre-assigned here)


@dataclass
class CellConfig:
    """Static configuration for a cell sector."""
    cell_id: str             # "CELL_XXX_YYY" format
    site_id: str
    sector: int              # 1, 2, or 3
    band_mhz: int            # 700, 1800, 2100, 2600, 3500
    bandwidth_mhz: int       # 10, 15, 20, 100 (NR)
    technology: str          # LTE | NR
    max_throughput_mbps: float
    latitude: float
    longitude: float
    environment: str
    is_indoor: bool
    cluster_id: int


@dataclass
class AnomalyEvent:
    """A single injected anomaly event."""
    event_id: str
    anomaly_type: str        # interference | hardware | overload | config_error | backhaul
    root_cause_cell_id: str
    affected_cell_ids: List[str]
    start_rop: int           # index into global ROP sequence
    duration_rops: int
    severity: str            # low | medium | high | critical


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def rop_timestamps(start: datetime, n_days: int, rop_minutes: int = 15) -> pd.DatetimeIndex:
    """Generate a DatetimeIndex covering n_days at rop_minutes granularity.

    Uses UTC throughout — PM counters from O1/E2SM-KPM are always UTC.
    The closed='left' interval means the timestamp represents the START of
    the measurement period, matching 3GPP TS 28.552 §5.1 convention.
    """
    freq = f"{rop_minutes}min"
    return pd.date_range(start=start, periods=n_days * (24 * 60 // rop_minutes), freq=freq, tz="UTC")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points in kilometres."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def diurnal_load_factor(hour: int, is_weekend: bool, environment: str) -> float:
    """Return a load multiplier [0.05, 1.0] for the given hour and day type.

    Models:
      - Business-district urban: sharp morning + evening rush, low overnight
      - Suburban: broader residential evening peak
      - Rural: flat with mild midday dip

    See Coursebook Ch. 13: Feature Engineering — temporal encoding.
    """
    if environment == "urban":
        if is_weekend:
            # Later wake, leisure peak afternoon/evening
            peaks = [(10, 0.55), (14, 0.75), (20, 0.85), (23, 0.65)]
        else:
            # Commuter double-hump
            peaks = [(8, 0.85), (12, 0.70), (18, 1.00), (21, 0.80)]
    elif environment == "suburban":
        if is_weekend:
            peaks = [(10, 0.50), (15, 0.70), (20, 0.90)]
        else:
            peaks = [(8, 0.60), (12, 0.55), (17, 0.75), (21, 0.95)]
    else:  # rural
        peaks = [(9, 0.55), (14, 0.65), (19, 0.80)]

    # Gaussian mixture: sum contributions from each peak, normalise
    load = 0.05  # overnight floor
    for peak_hour, peak_weight in peaks:
        sigma = 1.5  # hours
        load += peak_weight * math.exp(-0.5 * ((hour - peak_hour) / sigma) ** 2)
    return min(load, 1.0)


def inject_counter_reset(
    series: pd.Series,
    reset_rate: float,
    rng: np.random.Generator,
) -> Tuple[pd.Series, pd.Series]:
    """Simulate PM counter resets (BBU restart / software upgrade).

    When a counter resets, its value drops to a small non-negative number
    (not zero — most counters accumulate within the ROP so a restart
    mid-ROP yields a partial count).

    Returns (modified_series, reset_flag_series).
    """
    n = len(series)
    reset_mask = rng.random(n) < reset_rate
    modified = series.copy()

    # At reset points, value drops to a fraction of what it would be
    if reset_mask.any():
        partial_fraction = rng.uniform(0.01, 0.3, size=reset_mask.sum())
        modified[reset_mask] = modified[reset_mask] * partial_fraction

    return modified, pd.Series(reset_mask.astype(int), index=series.index, name="counter_reset_flag")


# ---------------------------------------------------------------------------
# 1. Generate cell inventory
# ---------------------------------------------------------------------------

def generate_site_configs(
    n_sites: int,
    bbox: Dict[str, float],
    rng: np.random.Generator,
) -> List[SiteConfig]:
    """Generate realistic macro site configurations.

    Sites are placed using a mixture of:
      - Dense urban core cluster (centre of bbox)
      - Suburban ring
      - Sparse rural scatter

    This mirrors how RAN deployments are actually built — high-density urban
    grids with thinning coverage as you move to suburbs/rural.
    """
    sites: List[SiteConfig] = []

    # Split sites by environment
    n_urban = int(n_sites * 0.35)      # 35% urban
    n_suburban = int(n_sites * 0.45)   # 45% suburban
    n_rural = n_sites - n_urban - n_suburban

    lat_centre = (bbox["lat_min"] + bbox["lat_max"]) / 2
    lon_centre = (bbox["lon_min"] + bbox["lon_max"]) / 2
    lat_span = bbox["lat_max"] - bbox["lat_min"]
    lon_span = bbox["lon_max"] - bbox["lon_min"]

    def _make_site(idx: int, env: str, lat: float, lon: float, cluster: int) -> SiteConfig:
        n_sectors = rng.choice([2, 3], p=[0.25, 0.75])  # 75% tri-sector
        height = (
            rng.uniform(25, 55) if env == "urban"
            else rng.uniform(30, 45) if env == "suburban"
            else rng.uniform(35, 60)  # rural needs taller masts
        )
        is_indoor = (env == "urban") and (rng.random() < 0.08)  # 8% indoor small cells in urban
        return SiteConfig(
            site_id=f"SITE_{idx:03d}",
            latitude=round(lat, 6),
            longitude=round(lon, 6),
            environment=env,
            n_sectors=int(n_sectors),
            height_m=round(float(height), 1),
            is_indoor=is_indoor,
            cluster_id=cluster,
        )

    site_idx = 0
    cluster_id = 0

    # Urban core — clustered near centre with small offsets
    for _ in range(n_urban):
        lat = lat_centre + rng.normal(0, lat_span * 0.07)
        lon = lon_centre + rng.normal(0, lon_span * 0.07)
        lat = float(np.clip(lat, bbox["lat_min"], bbox["lat_max"]))
        lon = float(np.clip(lon, bbox["lon_min"], bbox["lon_max"]))
        sites.append(_make_site(site_idx, "urban", lat, lon, cluster_id))
        site_idx += 1
        if site_idx % 8 == 0:
            cluster_id += 1  # roughly 8 urban sites per cluster

    cluster_id += 1

    # Suburban ring — uniform in outer 60% of bbox
    for _ in range(n_suburban):
        lat = rng.uniform(bbox["lat_min"] + lat_span * 0.15, bbox["lat_max"] - lat_span * 0.15)
        lon = rng.uniform(bbox["lon_min"] + lon_span * 0.15, bbox["lon_max"] - lon_span * 0.15)
        # Reject if too close to centre (already covered by urban)
        while abs(lat - lat_centre) < lat_span * 0.08 and abs(lon - lon_centre) < lon_span * 0.08:
            lat = rng.uniform(bbox["lat_min"] + lat_span * 0.15, bbox["lat_max"] - lat_span * 0.15)
            lon = rng.uniform(bbox["lon_min"] + lon_span * 0.15, bbox["lon_max"] - lon_span * 0.15)
        sites.append(_make_site(site_idx, "suburban", float(lat), float(lon), cluster_id))
        site_idx += 1
        if site_idx % 12 == 0:
            cluster_id += 1

    cluster_id += 1

    # Rural scatter — edges of bbox
    for _ in range(n_rural):
        lat = rng.uniform(bbox["lat_min"], bbox["lat_max"])
        lon = rng.uniform(bbox["lon_min"], bbox["lon_max"])
        sites.append(_make_site(site_idx, "rural", float(lat), float(lon), cluster_id))
        site_idx += 1
        cluster_id += 1  # each rural site is its own cluster (sparse)

    logger.info(
        "Generated %d sites: %d urban, %d suburban, %d rural",

        len(sites), n_urban, n_suburban, n_rural,
    )
    return sites


def generate_cell_configs(sites: List[SiteConfig], target_n_cells: int) -> List[CellConfig]:
    """Expand sites into individual cell sector configurations.

    Band assignment is environment-dependent:
      - Urban: 700/1800/2100/3500 MHz mix (capacity + coverage)
      - Suburban: 700/1800/2100 MHz
      - Rural: 700 MHz primary (coverage priority)

    Technology assignment:
      - 80% LTE, 20% NR (NSA NR on 3500 MHz band in urban/suburban)
    """
    cells: List[CellConfig] = []
    cell_count = 0

    URBAN_BANDS = [700, 1800, 2100, 3500]
    SUBURBAN_BANDS = [700, 1800, 2100]
    RURAL_BANDS = [700, 1800]

    BAND_BW_MAP: Dict[int, int] = {700: 20, 1800: 20, 2100: 20, 2600: 20, 3500: 100}
    BAND_MAX_TPUT: Dict[int, float] = {700: 150.0, 1800: 300.0, 2100: 300.0, 2600: 300.0, 3500: 1200.0}

    for site in sites:
        bands = (
            URBAN_BANDS if site.environment == "urban"
            else SUBURBAN_BANDS if site.environment == "suburban"
            else RURAL_BANDS
        )

        for sector in range(1, site.n_sectors + 1):
            if cell_count >= target_n_cells:
                break

            # Assign a band; sectors on different bands for co-coverage
            band = bands[(sector - 1) % len(bands)]
            tech = "NR" if band == 3500 else "LTE"
            bw = BAND_BW_MAP[band]
            max_tput = BAND_MAX_TPUT[band]

            # Small azimuth offset per sector so GPS coords differ slightly
            az_offset = (sector - 1) * 120  # degrees
            lat_offset = math.cos(math.radians(az_offset)) * 0.0002
            lon_offset = math.sin(math.radians(az_offset)) * 0.0002

            cell_id = f"CELL_{int(site.site_id.split('_')[1]):03d}_{sector}"

            cells.append(CellConfig(
                cell_id=cell_id,
                site_id=site.site_id,
                sector=sector,
                band_mhz=band,
                bandwidth_mhz=bw,
                technology=tech,
                max_throughput_mbps=max_tput,
                latitude=round(site.latitude + lat_offset, 6),
                longitude=round(site.longitude + lon_offset, 6),
                environment=site.environment,
                is_indoor=site.is_indoor,
                cluster_id=site.cluster_id,
            ))
            cell_count += 1

    logger.info("Generated %d cell configs from %d sites", len(cells), len(sites))
    return cells
# NOTE: The canonical PM counter generation implementation is
# generate_kpi_time_series() below, which uses load-factor-driven
# generation with AR(1) autocorrelation. An earlier implementation
# (generate_pm_dataset) was removed to avoid dead-code confusion.



def build_cell_inventory_df(cells: List[CellConfig]) -> pd.DataFrame:
    """Convert list of CellConfig into a tidy DataFrame for persistence."""
    records = [
        {
            "cell_id": c.cell_id,
            "site_id": c.site_id,
            "sector": c.sector,
            "band_mhz": c.band_mhz,
            "bandwidth_mhz": c.bandwidth_mhz,
            "technology": c.technology,
            "max_throughput_mbps": c.max_throughput_mbps,
            "latitude": c.latitude,
            "longitude": c.longitude,
            "environment": c.environment,
            "is_indoor": c.is_indoor,
            "cluster_id": c.cluster_id,
        }
        for c in cells
    ]
    df = pd.DataFrame(records)
    df["cell_id"] = df["cell_id"].astype("string")
    df["site_id"] = df["site_id"].astype("string")
    df["technology"] = df["technology"].astype("category")
    df["environment"] = df["environment"].astype("category")
    return df


# ---------------------------------------------------------------------------
# 2. Generate neighbour relation table (NRT)
# ---------------------------------------------------------------------------

def generate_neighbour_relations(
    cells: List[CellConfig],
    max_neighbours: int = 8,
    max_distance_km: float = 5.0,
) -> pd.DataFrame:
    """Build an O1-style Neighbour Relation Table (NRT).

    In a real network, NRT entries come from:
      - ANR (Automatic Neighbour Relation) — UE measurement reports
      - Manual planning tool exports

    We approximate this by selecting the N closest cells (within distance
    threshold) as neighbours, with a preference for same-band neighbours
    (they cause inter-cell interference). Co-site sectors are always
    neighbours regardless of distance.

    This table is used in Script 03+ to build the cell adjacency graph
    for Graph ML, following the architecture in whitepaper §4.

    Returns a DataFrame with one row per directed edge (source → target).
    The undirected graph is reconstructed by taking both directions.
    """
    coords = np.array([[c.latitude, c.longitude] for c in cells])
    n = len(cells)

    # Pairwise Haversine distances (km) — N×N matrix
    # For 200 cells this is trivial; for 10K cells use approximate NN.
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])

    # Efficient vectorised Haversine
    lat1 = lat_rad[:, np.newaxis]
    lat2 = lat_rad[np.newaxis, :]
    lon1 = lon_rad[:, np.newaxis]
    lon2 = lon_rad[np.newaxis, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist_km = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    np.fill_diagonal(dist_km, np.inf)  # exclude self

    records: List[Dict] = []

    for i, cell in enumerate(cells):
        distances = dist_km[i]

        # Sort by distance ascending
        sorted_idx = np.argsort(distances)

        neighbour_count = 0
        for j in sorted_idx:
            if neighbour_count >= max_neighbours:
                break

            target = cells[j]
            distance = distances[j]

            # Always include co-site sectors (distance ~ 0)
            is_cosite = cell.site_id == target.site_id

            # Include within distance threshold
            is_within_range = distance <= max_distance_km

            if not (is_cosite or is_within_range):
                continue

            # Edge weight: higher weight for same-band (interference coupling)
            same_band = cell.band_mhz == target.band_mhz
            edge_weight = 1.0 / (1.0 + distance) * (1.5 if same_band else 1.0)

            records.append({
                "source_cell_id": cell.cell_id,
                "target_cell_id": target.cell_id,
                "distance_km": round(float(distance), 3),
                "same_site": is_cosite,
                "same_band": same_band,
                "edge_weight": round(float(edge_weight), 4),
                "relation_type": "co_site" if is_cosite else "neighbour",
            })
            neighbour_count += 1

    nrt_df = pd.DataFrame(records)
    logger.info(
        "Generated NRT with %d directed edges (avg %.1f neighbours/cell)",
        len(nrt_df),
        len(nrt_df) / len(cells),
    )
    return nrt_df


# ---------------------------------------------------------------------------
# 3. Generate PM counter time series
# ---------------------------------------------------------------------------

def _baseline_kpi(
    cell: CellConfig,
    kpi: str,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Return (mean, std) for a cell's baseline KPI value.

    Baseline depends on environment, technology, and band — reflecting the
    realistic correlation structure of RAN KPIs.

    This per-cell heterogeneity is critical: it prevents the synthetic data
    from being trivially separable by environment label alone, matching
    the peer-group normalisation challenge described in whitepaper §4.
    """
    env = cell.environment
    is_nr = cell.technology == "NR"

    if kpi == "rsrp_dbm":
        base = {"urban": -82.0, "suburban": -88.0, "rural": -97.0}[env]
        std = 6.0 + rng.uniform(0, 3)
        return base + rng.uniform(-5, 5), std

    elif kpi == "rsrq_db":
        # RSRQ is more correlated with load than coverage
        base = {"urban": -11.0, "suburban": -10.0, "rural": -9.0}[env]
        return base + rng.uniform(-2, 2), 1.5

    elif kpi == "sinr_db":
        base = {"urban": 8.0, "suburban": 12.0, "rural": 16.0}[env]
        return base + rng.uniform(-3, 3), 3.0

    elif kpi == "avg_cqi":
        # CQI correlates with SINR
        base = {"urban": 8.5, "suburban": 10.0, "rural": 11.5}[env]
        return base + rng.uniform(-1, 1), 1.2

    elif kpi == "dl_throughput_mbps":
        # NR cells have higher peak throughput
        base = cell.max_throughput_mbps * 0.12  # typical utilisation baseline
        return base, base * 0.35

    elif kpi == "ul_throughput_mbps":
        dl_base, dl_std = _baseline_kpi(cell, "dl_throughput_mbps", rng)
        return dl_base * 0.18, dl_std * 0.18  # UL typically ~18% of DL

    elif kpi == "dl_prb_usage_rate":
        base = {"urban": 45.0, "suburban": 30.0, "rural": 15.0}[env]
        return base + rng.uniform(-10, 10), 10.0

    elif kpi == "ul_prb_usage_rate":
        # UL PRB utilisation is typically lower than DL
        base = {"urban": 35.0, "suburban": 22.0, "rural": 12.0}[env]
        return base + rng.uniform(-8, 8), 8.0

    elif kpi == "rrc_conn_setup_success_rate":
        return 0.978 + rng.uniform(-0.01, 0.01), 0.008

    elif kpi == "handover_success_rate":
        # Urban has more HO due to dense deployment
        base = {"urban": 0.960, "suburban": 0.975, "rural": 0.985}[env]
        return base + rng.uniform(-0.005, 0.005), 0.010

    elif kpi == "cell_availability_pct":
        return 99.85 + rng.uniform(-0.1, 0.1), 0.15

    elif kpi == "active_ue_count":
        base = {"urban": 180.0, "suburban": 80.0, "rural": 20.0}[env]
        return base + rng.uniform(-20, 20), base * 0.40

    elif kpi == "dl_volume_gb":
        dl_tput_mean, _ = _baseline_kpi(cell, "dl_throughput_mbps", rng)
        # volume = avg_throughput * period_in_hours * 3600 / 8 / 1024
        rops_per_hour = 60 // ROP_MINUTES
        volume_mean = dl_tput_mean * (1.0 / rops_per_hour) * 3600 / 8 / 1024
        return volume_mean, volume_mean * 0.45

    else:
        raise ValueError(f"Unknown KPI: {kpi}")


def generate_kpi_time_series(
    cells: List[CellConfig],
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate a PM counter time series for all cells.

    Architecture:
        For each KPI and each cell, we generate:
          1. Per-cell baseline (mean, std) from _baseline_kpi()
          2. Global temporal signal: diurnal + weekly patterns
          3. Per-cell autocorrelated noise: AR(1) process with phi=0.7
             (PM counters have strong autocorrelation between consecutive ROPs)
          4. Cross-KPI correlations: throughput follows PRB utilisation,
             SINR correlates inversely with PRB utilisation, etc.

    The AR(1) autocorrelation structure is important: naive i.i.d. noise
    would make anomaly detection trivially easy (every anomaly would be
    instantly visible). Real PM counters have persistence that means a
    degradation unfolds over 3–15 ROPs.
    """
    kpis = list(KPI_RANGES.keys())
    n_ts = len(timestamps)
    n_cells = len(cells)

    logger.info(
        "Generating %d KPI series × %d cells × %d ROPs = %s rows...",
        len(kpis), n_cells, n_ts,
        f"{len(kpis) * n_cells * n_ts:,}",
    )

    # Pre-compute temporal load factors for all timestamps
    # Shape: (n_ts, 3) for urban/suburban/rural
    env_types = ["urban", "suburban", "rural"]
    load_factors: Dict[str, np.ndarray] = {}

    for env in env_types:
        factors = np.zeros(n_ts)
        for t_idx, ts in enumerate(timestamps):
            hour = ts.hour
            is_weekend = ts.dayofweek >= 5
            factors[t_idx] = diurnal_load_factor(hour, is_weekend, env)
        load_factors[env] = factors

    # Build records list — more memory-efficient than building a 3D array
    # then melting, especially at 10K+ cell scale.
    all_rows: List[Dict] = []

    for cell_idx, cell in enumerate(cells):
        if cell_idx % 50 == 0:
            logger.debug("Processing cell %d/%d: %s", cell_idx + 1, n_cells, cell.cell_id)

        # Cell-specific load factor (reuse env-level factors)
        load = load_factors[cell.environment]

        # Per-cell noise: AR(1) innovations
        phi = 0.70  # AR coefficient — strong persistence in PM counters
        innovations = rng.standard_normal(n_ts)

        # Generate AR(1) process: x_t = phi * x_{t-1} + sqrt(1-phi^2) * eps_t
        ar_noise = np.zeros(n_ts)
        ar_noise[0] = innovations[0]
        for t in range(1, n_ts):
            ar_noise[t] = phi * ar_noise[t - 1] + math.sqrt(1 - phi ** 2) * innovations[t]

        # Per-KPI series
        cell_kpis: Dict[str, np.ndarray] = {}

        for kpi in kpis:
            mu, sigma = _baseline_kpi(cell, kpi, rng)

            if kpi in ("rsrp_dbm", "rsrq_db", "sinr_db"):
                # RF quality KPIs: moderate load effect, AR noise
                signal = mu + sigma * ar_noise * 0.5
                # Slight improvement at low load (less interference)
                signal = signal + (1.0 - load) * abs(sigma) * 0.3

            elif kpi == "avg_cqi":
                # CQI follows SINR which follows load inversely
                signal = mu + sigma * ar_noise * 0.5 - load * 1.5

            elif kpi in ("dl_throughput_mbps", "ul_throughput_mbps"):
                # Throughput scales with load × capacity, with noise
                ratio = 1.0 if kpi == "dl_throughput_mbps" else 0.18
                signal = (
                    cell.max_throughput_mbps * ratio * load * 0.85
                    + sigma * ar_noise * 0.6
                )

            elif kpi == "dl_prb_usage_rate":
                # PRB utilisation tracks load closely
                signal = mu * load / 0.50 + sigma * ar_noise * 0.5

            elif kpi == "active_ue_count":
                signal = mu * load / 0.50 + sigma * ar_noise * 0.5

            elif kpi == "dl_volume_gb":
                # Volume = throughput × time, so follows throughput pattern
                dl_tput = cell_kpis.get("dl_throughput_mbps")
                if dl_tput is not None:
                    rops_per_hour = 60 // ROP_MINUTES
                    signal = dl_tput * (1.0 / rops_per_hour) * 3600 / 8 / 1024
                else:
                    signal = mu * load / 0.5 + sigma * ar_noise * 0.3

            elif kpi in ("rrc_conn_setup_success_rate", "handover_success_rate", "cell_availability_pct"):
                # Success rates: high baseline, small load-correlated degradation
                load_impact = load * 0.005  # degrade slightly at peak
                signal = mu - load_impact + sigma * ar_noise * 0.3

            else:
                signal = mu + sigma * ar_noise * 0.5

            # Clip to realistic range
            lo, hi = KPI_RANGES[kpi]
            signal = np.clip(signal, lo, hi)
            cell_kpis[kpi] = signal

        # ── Post-loop guard: ensure dl_volume_gb is derived from throughput ──
        # If KPI_RANGES insertion order is accidentally changed and dl_volume_gb
        # was computed before dl_throughput_mbps, recompute it now from the
        # correct throughput series.  This is a safety net — the primary
        # mechanism is the insertion-order dependency documented in KPI_RANGES.
        if "dl_throughput_mbps" in cell_kpis and "dl_volume_gb" in cell_kpis:
            rops_per_hour = 60 // ROP_MINUTES
            recomputed = cell_kpis["dl_throughput_mbps"] * (1.0 / rops_per_hour) * 3600 / 8 / 1024
            cell_kpis["dl_volume_gb"] = np.clip(recomputed, *KPI_RANGES["dl_volume_gb"])

        # Assemble rows for this cell
        for t_idx in range(n_ts):
            row: Dict = {
                "timestamp": timestamps[t_idx],
                "cell_id": cell.cell_id,
                "site_id": cell.site_id,
            }
            for kpi in kpis:
                row[kpi] = float(cell_kpis[kpi][t_idx])
            # Anomaly columns initialised to 0; filled in the next step
            row["is_anomaly"] = 0
            row["anomaly_type"] = ""
            row["anomaly_severity"] = ""
            row["root_cause_cell_id"] = ""
            row["counter_reset_flag"] = 0
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["cell_id"] = df["cell_id"].astype("string")
    df["site_id"] = df["site_id"].astype("string")

    logger.info("Base time series shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 4. Inject anomalies
# ---------------------------------------------------------------------------

def _apply_kpi_degradation(
    df: pd.DataFrame,
    cell_ids: List[str],
    rop_indices: List[int],
    anomaly_type: str,
    severity: str,
    root_cause_cell: str,
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Apply KPI degradation to specified cells and ROPs in-place.

    Degradation patterns are anomaly-type-specific:
      - interference: RSRP stable, SINR/CQI/throughput degrade
      - hardware: availability drops, success rates drop
      - overload: PRB 100%, throughput plateau, success rates drop
      - config_error: RSRP/RSRQ stable, HO failure spike
      - backhaul: throughput collapses, availability stays OK

    Severity multiplier: low=0.6, medium=0.4, high=0.25, critical=0.1
    """
    severity_factor: Dict[str, float] = {
        "low": 0.60, "medium": 0.40, "high": 0.25, "critical": 0.10
    }
    sf = severity_factor.get(severity, 0.40)

    affected_timestamps = timestamps[rop_indices]

    # Create mask for affected rows
    mask = (
        df["cell_id"].isin(cell_ids)
        & df["timestamp"].isin(affected_timestamps)
    )

    n_affected = mask.sum()
    if n_affected == 0:
        return df

    if anomaly_type == "interference":
        df.loc[mask, "sinr_db"] *= sf
        df.loc[mask, "avg_cqi"] *= sf
        df.loc[mask, "dl_throughput_mbps"] *= sf
        df.loc[mask, "ul_throughput_mbps"] *= sf
        df.loc[mask, "rsrq_db"] *= sf  # RSRQ degrades with interference

    elif anomaly_type == "hardware":
        df.loc[mask, "cell_availability_pct"] -= (100 - 95.0) * (1 - sf) * 10
        df.loc[mask, "rrc_conn_setup_success_rate"] *= sf
        df.loc[mask, "handover_success_rate"] *= sf
        df.loc[mask, "dl_throughput_mbps"] *= sf

    elif anomaly_type == "overload":
        df.loc[mask, "dl_prb_usage_rate"] = np.minimum(
            df.loc[mask, "dl_prb_usage_rate"] * (1.0 / sf), 99.5
        )
        df.loc[mask, "dl_throughput_mbps"] *= 0.85  # throughput plateau
        df.loc[mask, "rrc_conn_setup_success_rate"] *= (sf + (1 - sf) * 0.3)
        df.loc[mask, "active_ue_count"] *= 1.5

    elif anomaly_type == "config_error":
        df.loc[mask, "handover_success_rate"] *= sf
        df.loc[mask, "rrc_conn_setup_success_rate"] *= sf
        df.loc[mask, "rsrq_db"] *= sf

    elif anomaly_type == "backhaul":
        df.loc[mask, "dl_throughput_mbps"] *= sf * 0.5
        df.loc[mask, "ul_throughput_mbps"] *= sf * 0.5
        df.loc[mask, "dl_volume_gb"] *= sf * 0.5
        # Availability stays OK (radio is up, just no backhaul)
        # Slight RRC degradation as UEs lose sessions
        df.loc[mask, "rrc_conn_setup_success_rate"] *= (sf + (1 - sf) * 0.6)

    # Clip all KPI ranges after degradation
    for kpi, (lo, hi) in KPI_RANGES.items():
        if kpi in df.columns:
            df.loc[mask, kpi] = df.loc[mask, kpi].clip(lo, hi)

    # Set anomaly label columns
    df.loc[mask, "is_anomaly"] = 1
    df.loc[mask, "anomaly_type"] = anomaly_type
    df.loc[mask, "anomaly_severity"] = severity
    df.loc[mask, "root_cause_cell_id"] = root_cause_cell

    return df


def generate_anomaly_events(
    cells: List[CellConfig],
    n_rops_total: int,
    anomaly_rate: float,
    multi_cell_prob: float,
    rng: np.random.Generator,
) -> List[AnomalyEvent]:
    """Plan anomaly events to achieve the target per-cell-ROP anomaly rate.

    Approach:
      1. Compute total anomalous cell-ROPs = n_cells × n_rops × anomaly_rate
      2. Distribute into single-cell and multi-cell events
      3. Multi-cell events are tied to a root-cause cell (typically the site
         with the highest betweenness in the NRT — approximated here by
         choosing sites with 3 sectors as more likely to be root cause)

    This produces a mix of:
      - Isolated single-cell faults (hardware, config error)
      - Multi-cell correlated faults (backhaul failure, site power issue,
        inter-cell interference)

    The multi-cell structure is what makes the Graph ML layer valuable —
    tabular models see each cell independently and miss the shared cause.
    """
    n_cells = len(cells)
    target_anomalous_cell_rops = int(n_cells * n_rops_total * anomaly_rate)

    # Group cells by site for multi-cell event generation
    site_to_cells: Dict[str, List[CellConfig]] = {}
    for cell in cells:
        site_to_cells.setdefault(cell.site_id, []).append(cell)

    anomaly_types = ["interference", "hardware", "overload", "config_error", "backhaul"]
    # Realistic distribution: interference and overload are most common
    type_weights = [0.30, 0.20, 0.25, 0.15, 0.10]
    severities = ["low", "medium", "high", "critical"]
    severity_weights = [0.50, 0.30, 0.15, 0.05]

    events: List[AnomalyEvent] = []
    total_anomalous_rops_planned = 0
    event_id = 0

    while total_anomalous_rops_planned < target_anomalous_cell_rops:
        is_multi_cell = rng.random() < multi_cell_prob
        atype = rng.choice(anomaly_types, p=type_weights)
        severity = rng.choice(severities, p=severity_weights)

        # Duration: 1 ROP (15 min) to 32 ROPs (8 hours)
        # Most events are short; heavy tail for major outages
        duration_rops = int(rng.choice(
            [1, 2, 4, 8, 12, 16, 24, 32],
            p=[0.30, 0.25, 0.20, 0.10, 0.06, 0.04, 0.03, 0.02],
        ))

        # Start time: avoid edges of simulation window
        start_rop = int(rng.integers(24, n_rops_total - duration_rops - 24))

        if is_multi_cell and atype in ("backhaul", "interference", "hardware"):
            # Pick a site and affect all its sectors
            site_id = rng.choice(list(site_to_cells.keys()))
            affected_cells = [c.cell_id for c in site_to_cells[site_id]]
            # Root cause: the site's primary sector (sector 1)
            root_cause = next(
                (c.cell_id for c in site_to_cells[site_id] if c.sector == 1),
                affected_cells[0],
            )
        else:
            # Single cell event
            root_cause_cell = cells[rng.integers(0, n_cells)]
            affected_cells = [root_cause_cell.cell_id]
            root_cause = root_cause_cell.cell_id

        events.append(AnomalyEvent(
            event_id=f"EVT_{event_id:04d}",
            anomaly_type=atype,
            root_cause_cell_id=root_cause,
            affected_cell_ids=affected_cells,
            start_rop=start_rop,
            duration_rops=duration_rops,
            severity=severity,
        ))

        total_anomalous_rops_planned += len(affected_cells) * duration_rops
        event_id += 1

    logger.info(
        "Planned %d anomaly events (%d multi-cell), targeting ~%d anomalous cell-ROPs",
        len(events),
        sum(1 for e in events if len(e.affected_cell_ids) > 1),
        target_anomalous_cell_rops,
    )
    return events


def inject_anomalies(
    df: pd.DataFrame,
    events: List[AnomalyEvent],
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Apply all planned anomaly events to the time series DataFrame."""
    for event in events:
        rop_end = min(event.start_rop + event.duration_rops, len(timestamps))
        rop_indices = list(range(event.start_rop, rop_end))

        df = _apply_kpi_degradation(
            df=df,
            cell_ids=event.affected_cell_ids,
            rop_indices=rop_indices,
            anomaly_type=event.anomaly_type,
            severity=event.severity,
            root_cause_cell=event.root_cause_cell_id,
            timestamps=timestamps,
        )

    actual_anomaly_rate = df["is_anomaly"].mean()
    logger.info(
        "Injected %d events. Actual anomaly rate: %.2f%% of cell-ROPs",
        len(events),
        actual_anomaly_rate * 100,
    )
    return df


def inject_counter_resets(
    df: pd.DataFrame,
    reset_rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject counter reset events across the dataset.

    Counter resets are independent of anomalies — they occur due to BBU
    restarts, software upgrades, or NMS polling gaps. They affect all
    cumulative counters simultaneously for a cell.

    Markers are critical for feature engineering: the 15-min delta
    calculation in Script 02 must skip resets to avoid spurious anomaly
    signals from legitimate counter wraps.
    """
    # Resets happen at the cell-ROP level — if a cell resets, all its
    # KPIs at that ROP are affected
    cell_ids = df["cell_id"].unique()
    timestamps = df["timestamp"].unique()

    # For efficiency, sample reset (cell, timestamp) pairs
    n_total_cell_rops = len(df)
    n_resets = int(n_total_cell_rops * reset_rate)

    reset_row_indices = rng.choice(df.index, size=n_resets, replace=False)

    # At reset rows, volume/count KPIs drop to near-zero
    reset_kpis = ["dl_throughput_mbps", "ul_throughput_mbps", "dl_volume_gb", "active_ue_count"]
    for kpi in reset_kpis:
        if kpi in df.columns:
            df.loc[reset_row_indices, kpi] *= rng.uniform(0.01, 0.15, size=len(reset_row_indices))

    df.loc[reset_row_indices, "counter_reset_flag"] = 1

    logger.info(
        "Injected %d counter reset events (%.2f%% of cell-ROPs)",
        n_resets, reset_rate * 100,
    )
    return df


# ---------------------------------------------------------------------------
# 5. Build anomaly labels table
# ---------------------------------------------------------------------------

def build_anomaly_labels_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract anomaly ground truth into a separate, compact labels table.

    Separating labels from PM counters allows:
      - Zero-label bootstrapping: models trained without this table
      - Retrospective labelling: labels added after training
      - Label versioning: multiple labellers can provide annotations

    This mirrors the phased labelling pipeline in Part 1 §5 and the
    whitepaper §4 (phased approach: unsupervised → operator correlation
    → engineer annotation).
    """
    anomaly_df = df[df["is_anomaly"] == 1][
        ["timestamp", "cell_id", "anomaly_type", "anomaly_severity", "root_cause_cell_id"]
    ].copy()

    anomaly_df = anomaly_df.rename(columns={
        "anomaly_type": "label_type",
        "anomaly_severity": "label_severity",
    })

    anomaly_df["label_source"] = "synthetic_injection"  # vs. "trouble_ticket", "engineer"
    anomaly_df["label_confidence"] = 1.0  # synthetic = 100% confidence

    logger.info(
        "Anomaly labels table: %d rows, %d unique cells affected",
        len(anomaly_df),
        anomaly_df["cell_id"].nunique(),
    )
    return anomaly_df


# ---------------------------------------------------------------------------
# 6. Data validation
# ---------------------------------------------------------------------------

def validate_dataset(
    pm_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    nrt_df: pd.DataFrame,
) -> bool:
    """Run basic schema and value-range validation on generated data.

    In production, this would use Great Expectations or pandera.
    Here we implement lightweight assertions to catch generation bugs
    before writing to disk.

    See Coursebook Ch. 28: Data Pipelines — validation as a pipeline gate.
    """
    logger.info("Running data validation checks...")
    passed = True

    # Schema checks
    required_kpi_cols = list(KPI_RANGES.keys())
    missing = [c for c in required_kpi_cols if c not in pm_df.columns]
    if missing:
        logger.error("PM data missing columns: %s", missing)
        passed = False

    # Value range checks
    for kpi, (lo, hi) in KPI_RANGES.items():
        if kpi not in pm_df.columns:
            continue
        col = pm_df[kpi]
        # Allow a tiny margin for floating-point clipping
        margin = abs(lo) * 0.001 + abs(hi) * 0.001 + 0.01
        if col.min() < lo - margin:
            logger.error(
                "KPI %s min=%.3f below range floor %.3f", kpi, col.min(), lo
            )
            passed = False
        if col.max() > hi + margin:
            logger.error(
                "KPI %s max=%.3f above range ceiling %.3f", kpi, col.max(), hi
            )
            passed = False

    # Anomaly rate check
    actual_rate = pm_df["is_anomaly"].mean()
    if not (0.005 < actual_rate < 0.08):
        logger.warning(
            "Anomaly rate %.3f outside expected 0.5–8%% range", actual_rate
        )

    # Cell ID format check
    cell_id_pattern = pm_df["cell_id"].str.match(r"CELL_\d{3}_\d+")
    if not cell_id_pattern.all():
        bad = pm_df["cell_id"][~cell_id_pattern].unique()[:5]
        logger.error("Non-conforming cell_ids: %s", bad)
        passed = False

    # NRT referential integrity: all source/target cell_ids exist in inventory
    inventory_cell_ids = set(inventory_df["cell_id"].tolist())
    nrt_sources = set(nrt_df["source_cell_id"].tolist())
    nrt_targets = set(nrt_df["target_cell_id"].tolist())
    orphan_sources = nrt_sources - inventory_cell_ids
    orphan_targets = nrt_targets - inventory_cell_ids
    if orphan_sources:
        logger.error("NRT source cells not in inventory: %s", orphan_sources)
        passed = False
    if orphan_targets:
        logger.error("NRT target cells not in inventory: %s", orphan_targets)
        passed = False

    # Timestamp monotonicity check (sample first cell)
    first_cell = pm_df["cell_id"].iloc[0]
    cell_ts = pm_df[pm_df["cell_id"] == first_cell]["timestamp"]
    if not cell_ts.is_monotonic_increasing:
        logger.error("Timestamps not monotonically increasing for cell %s", first_cell)
        passed = False

    # No duplicate (cell_id, timestamp) pairs
    dupes = pm_df.duplicated(subset=["cell_id", "timestamp"]).sum()
    if dupes > 0:
        logger.error("%d duplicate (cell_id, timestamp) pairs", dupes)
        passed = False

    if passed:
        logger.info("All validation checks PASSED")
    else:
        logger.warning("Some validation checks FAILED — review logs above")

    return passed


# ---------------------------------------------------------------------------
# 7. Summary statistics logging
# ---------------------------------------------------------------------------

def log_dataset_summary(
    pm_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    nrt_df: pd.DataFrame,
) -> None:
    """Log a human-readable summary of the generated dataset."""
    n_cells = pm_df["cell_id"].nunique()
    n_rops = pm_df["timestamp"].nunique()
    n_rows = len(pm_df)
    n_anomalous = pm_df["is_anomaly"].sum()
    anomaly_rate = n_anomalous / n_rows

    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info("Cells:              %d", n_cells)
    logger.info("ROPs:               %d  (%.1f days at %d-min granularity)",
                n_rops, n_rops / ROPS_PER_DAY, ROP_MINUTES)
    logger.info("Total rows:         %d", n_rows)
    logger.info("Anomalous rows:     %d  (%.2f%%)", n_anomalous, anomaly_rate * 100)
    logger.info("Counter resets:     %d  (%.2f%%)",
                pm_df["counter_reset_flag"].sum(),
                pm_df["counter_reset_flag"].mean() * 100)
    logger.info("NRT edges:          %d  (avg %.1f neighbours/cell)",
                len(nrt_df), len(nrt_df) / n_cells)

    logger.info("--- Anomaly breakdown by type ---")
    if n_anomalous > 0:
        atype_counts = pm_df[pm_df["is_anomaly"] == 1]["anomaly_type"].value_counts()
        for atype, count in atype_counts.items():
            logger.info("  %-20s %6d  (%.1f%%)", atype, count, 100 * count / n_anomalous)

    logger.info("--- KPI summary (non-anomalous rows) ---")
    normal_df = pm_df[pm_df["is_anomaly"] == 0]
    for kpi in list(KPI_RANGES.keys())[:5]:  # log first 5 for brevity
        logger.info(
            "  %-35s mean=%8.2f  std=%7.2f  min=%8.2f  max=%8.2f",
            kpi,
            normal_df[kpi].mean(),
            normal_df[kpi].std(),
            normal_df[kpi].min(),
            normal_df[kpi].max(),
        )

    logger.info("--- Environment distribution ---")
    env_counts = inventory_df["environment"].value_counts()
    for env, count in env_counts.items():
        logger.info("  %-12s %d cells", env, count)

    logger.info("--- Technology distribution ---")
    tech_counts = inventory_df["technology"].value_counts()
    for tech, count in tech_counts.items():
        logger.info("  %-6s %d cells", tech, count)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# 8. Persistence
# ---------------------------------------------------------------------------

def save_outputs(
    pm_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    nrt_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write all datasets to parquet files.

    Parquet is preferred over CSV for:
      - Schema preservation (dtype, timezone in timestamps)
      - Efficient columnar reads in downstream feature engineering
      - ~5× compression ratio for PM counter data

    In production, these would be written to an object store (S3/GCS)
    partitioned by date for efficient time-range scans.
    See Coursebook Ch. 28: Data Pipelines.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # PM counters: partition by date for production-style access patterns
    pm_path = output_dir / "pm_counters.parquet"
    pm_df.to_parquet(pm_path, index=False, engine="pyarrow", compression="snappy")
    logger.info("Wrote PM counters: %s  (%.1f MB)", pm_path, pm_path.stat().st_size / 1e6)

    inv_path = output_dir / "cell_inventory.parquet"
    inventory_df.to_parquet(inv_path, index=False, engine="pyarrow")
    logger.info("Wrote cell inventory: %s", inv_path)

    nrt_path = output_dir / "neighbour_relations.parquet"
    nrt_df.to_parquet(nrt_path, index=False, engine="pyarrow")
    logger.info("Wrote NRT: %s", nrt_path)

    labels_path = output_dir / "anomaly_labels.parquet"
    labels_df.to_parquet(labels_path, index=False, engine="pyarrow")
    logger.info("Wrote anomaly labels: %s", labels_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate synthetic data generation pipeline."""
    logger.info("Starting synthetic data generation for Telco MLOps Part 2")
    logger.info("Random seed: %d | Cells: %d | Days: %d", RANDOM_SEED, N_CELLS, N_DAYS)

    # Seeded RNG — all downstream scripts use the same seed for reproducibility.
    # Using the new numpy Generator API (not legacy np.random.*) for better
    # statistical properties and reproducibility guarantees.
    rng = np.random.default_rng(RANDOM_SEED)

    # --- Step 1: Cell topology ---
    logger.info("Step 1/7: Generating site configurations...")
    sites = generate_site_configs(N_SITES, GEO_BBOX, rng)

    logger.info("Step 2/7: Generating cell configurations...")
    cells = generate_cell_configs(sites, N_CELLS)

    logger.info("Step 3/7: Building cell inventory DataFrame...")
    inventory_df = build_cell_inventory_df(cells)

    # --- Step 2: Neighbour relation table ---
    logger.info("Step 4/7: Generating neighbour relation table...")
    nrt_df = generate_neighbour_relations(cells, max_neighbours=8, max_distance_km=5.0)

    # --- Step 3: Time series ---
    logger.info("Step 5/7: Generating PM counter time series...")
    timestamps = rop_timestamps(SIM_START, N_DAYS, ROP_MINUTES)
    n_rops = len(timestamps)
    logger.info("Timestamp range: %s → %s (%d ROPs)", timestamps[0], timestamps[-1], n_rops)

    pm_df = generate_kpi_time_series(cells, timestamps, rng)

    # --- Step 4: Inject anomalies ---
    logger.info("Step 6/7: Planning and injecting anomaly events...")
    anomaly_events = generate_anomaly_events(
        cells=cells,
        n_rops_total=n_rops,
        anomaly_rate=ANOMALY_RATE_CELL,
        multi_cell_prob=MULTI_CELL_FAULT_PROBABILITY,
        rng=rng,
    )
    pm_df = inject_anomalies(pm_df, anomaly_events, timestamps)

    # Inject counter resets (independent of anomalies)
    pm_df = inject_counter_resets(pm_df, COUNTER_RESET_RATE, rng)

    # --- Step 5: Sort by (timestamp, cell_id) for temporal correctness ---
    pm_df = pm_df.sort_values(["timestamp", "cell_id"]).reset_index(drop=True)

    # --- Step 6: Build labels table ---
    labels_df = build_anomaly_labels_df(pm_df)

    # --- Step 7: Validate and save ---
    logger.info("Step 7/7: Validating and saving outputs...")
    validation_passed = validate_dataset(pm_df, inventory_df, nrt_df)

    if not validation_passed:
        logger.error("Validation failed — outputs may be corrupt. Review errors above.")
        sys.exit(1)

    log_dataset_summary(pm_df, inventory_df, nrt_df)

    save_outputs(
        pm_df=pm_df,
        inventory_df=inventory_df,
        nrt_df=nrt_df,
        labels_df=labels_df,
        output_dir=OUTPUT_DIR,
    )

    logger.info("Synthetic data generation complete. Outputs written to: %s/", OUTPUT_DIR)
    logger.info(
        "Next step: python 02_feature_engineering.py  "
        "(reads from %s/pm_counters.parquet)", OUTPUT_DIR
    )



# ===========================================================================
# GNN node feature encoding (used by finalize_ran_topology_graph)
# These functions are optional — they require torch, which is only needed
# when the GNN module in Script 03 is active.
# ===========================================================================


def _encode_site_features(site_data: dict) -> "torch.Tensor":
    """Encode a single site's metadata as a feature tensor.

    Feature vector (5-dimensional):
        [0] site_type   — categorical: macro=0, micro=1, pico=2, indoor=3
        [1] environment — categorical: urban=0, suburban=1, rural=2
        [2] num_sectors — integer count of sectors at this site
        [3] lat         — latitude (raw; normalised downstream if needed)
        [4] lon         — longitude (raw; normalised downstream if needed)

    Args:
        site_data: dict with keys matching the NRM record schema.

    Returns:
        torch.Tensor of shape (5,), dtype float32.
    """
    import torch

    env_map = {"urban": 0, "suburban": 1, "rural": 2}
    site_type_map = {"macro": 0, "micro": 1, "pico": 2, "indoor": 3}
    return torch.tensor([
        site_type_map.get(site_data.get("site_type", "macro"), 0),
        env_map.get(site_data.get("environment", "urban"), 0),
        float(site_data.get("num_sectors", 3)),
        float(site_data.get("lat", 0.0)),
        float(site_data.get("lon", 0.0)),
    ], dtype=torch.float32)


def _encode_backhaul_features(site_data: dict) -> "torch.Tensor":
    """Encode a single backhaul node's metadata as a feature tensor.

    Feature vector (5-dimensional):
        [0] backhaul_type   — categorical: fibre=0, microwave=1, copper=2, satellite=3
        [1] capacity_norm   — capacity in Mbps, normalised by /10000
        [2] utilisation_norm — utilisation percentage, normalised by /100
        [3] latency_norm    — one-way latency in ms, normalised by /100
        [4] redundancy      — number of redundant paths (1 = no redundancy)

    Args:
        site_data: dict with keys matching the NRM backhaul record schema.

    Returns:
        torch.Tensor of shape (5,), dtype float32.
    """
    import torch

    bh_type_map = {"fibre": 0, "microwave": 1, "copper": 2, "satellite": 3}
    return torch.tensor([
        bh_type_map.get(site_data.get("backhaul_type", "fibre"), 0),
        float(site_data.get("capacity_mbps", 1000.0)) / 10000.0,
        float(site_data.get("utilisation_pct", 0.0)) / 100.0,
        float(site_data.get("latency_ms", 5.0)) / 100.0,
        float(site_data.get("redundancy", 1)),
    ], dtype=torch.float32)


def _encode_site_features_batch(nrm_records: pd.DataFrame, site_ids: List[str]) -> "torch.Tensor":
    """Encode site features for a batch of site IDs.

    Maps from the actual column names produced by Script 01's
    ``generate_pm_counters()`` (environment, is_indoor, latitude,
    longitude, cell_id) to the feature dict expected by
    ``_encode_site_features()`` (site_type, environment, num_sectors,
    lat, lon).  Also accepts the alternative column names that a
    production O1 NRM export might use (site_type, num_sectors, lat, lon)
    so the same function works in both synthetic and production contexts.

    Args:
        nrm_records: DataFrame containing NRM records with site information.
                     Expected columns from Script 01: site_id, cell_id,
                     environment, is_indoor, latitude, longitude.
        site_ids: List of site IDs to encode.

    Returns:
        torch.Tensor of shape (len(site_ids), 5) with stacked site features.
    """
    import torch

    features = []
    for site_id in site_ids:
        site_rows = nrm_records[nrm_records["site_id"] == site_id]
        if len(site_rows) == 0:
            site_data = {
                "site_type": "macro",
                "environment": "urban",
                "num_sectors": 3,
                "lat": 0.0,
                "lon": 0.0,
            }
        else:
            row = site_rows.iloc[0]
            # site_type: Script 01 uses 'is_indoor' (bool); production
            # O1 exports may use a 'site_type' string column.
            if "site_type" in site_rows.columns:
                st = str(row.get("site_type", "macro"))
            elif "is_indoor" in site_rows.columns:
                st = "indoor" if row.get("is_indoor", False) else "macro"
            else:
                st = "macro"

            # environment: same column name in Script 01 and production
            env = str(row.get("environment", "urban"))

            # num_sectors: Script 01 has per-cell rows with 'sector' column;
            # count unique cell_ids per site. Production may have 'num_sectors'.
            if "num_sectors" in site_rows.columns:
                ns = int(row.get("num_sectors", 3))
            elif "cell_id" in site_rows.columns:
                ns = int(site_rows["cell_id"].nunique())
            else:
                ns = 3
                logger.debug(
                    "Neither num_sectors nor cell_id column found; "
                    "defaulting to 3 sectors for site %s", site_id,
                )

            # lat/lon: Script 01 uses 'latitude'/'longitude';
            # production may use 'lat'/'lon'.
            lat = float(row.get("latitude", row.get("lat", 0.0)))
            lon = float(row.get("longitude", row.get("lon", 0.0)))

            site_data = {
                "site_type": st,
                "environment": env,
                "num_sectors": ns,
                "lat": lat,
                "lon": lon,
            }
        features.append(_encode_site_features(site_data))
    return torch.stack(features, dim=0)


def _encode_backhaul_features_batch(nrm_records: pd.DataFrame, bh_ids: List[str]) -> "torch.Tensor":
    """Encode backhaul features for a batch of backhaul node IDs.

    Args:
        nrm_records: DataFrame containing NRM records with backhaul information.
        bh_ids: List of backhaul node IDs to encode.

    Returns:
        torch.Tensor of shape (len(bh_ids), 5) with stacked backhaul features.
    """
    import torch

    features = []
    for bh_id in bh_ids:
        bh_rows = nrm_records[nrm_records["backhaul_id"] == bh_id] if "backhaul_id" in nrm_records.columns else pd.DataFrame()
        if len(bh_rows) == 0:
            site_data = {
                "backhaul_type": "fibre",
                "capacity_mbps": 1000.0,
                "utilisation_pct": 0.0,
                "latency_ms": 5.0,
                "redundancy": 1,
            }
        else:
            row = bh_rows.iloc[0]
            site_data = {
                "backhaul_type": row["backhaul_type"] if "backhaul_type" in bh_rows.columns else "fibre",
                "capacity_mbps": float(row["capacity_mbps"]) if "capacity_mbps" in bh_rows.columns else 1000.0,
                "utilisation_pct": float(row["utilisation_pct"]) if "utilisation_pct" in bh_rows.columns else 0.0,
                "latency_ms": float(row["latency_ms"]) if "latency_ms" in bh_rows.columns else 5.0,
                "redundancy": int(row["redundancy"]) if "redundancy" in bh_rows.columns else 1,
            }
        features.append(_encode_backhaul_features(site_data))
    return torch.stack(features, dim=0)


def finalize_ran_topology_graph(nrm_records: pd.DataFrame, site_ids: List[str], bh_ids: List[str], data=None):
    """Finalize a pre-constructed RAN topology graph with node features and reverse edges.

    This is the **synthetic data generation** variant, used only during
    ``01_synthetic_data.py`` execution. Unlike the other two implementations
    of graph construction in this codebase, this function does NOT build
    a graph from scratch — it receives an already-constructed HeteroData
    object (with cell_sector nodes and forward edges already populated)
    and adds:
      (a) site and backhaul_node feature tensors
      (b) reverse edge types for bidirectional HGTConv message passing

    The three graph construction functions in this codebase:
      - ``finalize_ran_topology_graph()`` (this file): adds features and
        reverse edges to an existing HeteroData. Synthetic data path only.
      - ``build_ran_topology_graph()`` (03_model_training.py): builds a
        complete graph from pre-processed dicts. Training-time path.
      - ``build_ran_topology_graph()`` (§8.1 CODE-1, topology.py): builds
        a complete graph from raw O1 NRT/NRM DataFrames and a Feast
        FeatureStore. Production serving path.

    All three produce the same HeteroData schema (node types, edge types,
    feature dimensions) so that models are compatible across paths.

    Uses T.ToUndirected() when PyG transforms are available; falls back
    to manual .flip(0) when they are not. The two approaches are mutually
    exclusive — combining them would double reverse edge counts.

    Args:
        nrm_records: DataFrame containing NRM records.
        site_ids: List of site node IDs.
        bh_ids: List of backhaul node IDs.
        data: Optional HeteroData object. A new one is created if not provided.

    Returns:
        Graph data object with node feature tensors and reverse edges assigned.
    """
    import torch

    if data is None:
        from torch_geometric.data import HeteroData
        data = HeteroData()
    site_x = _encode_site_features_batch(nrm_records, site_ids)
    backhaul_x = _encode_backhaul_features_batch(nrm_records, bh_ids)
    data['site'].x = site_x
    data['backhaul_node'].x = backhaul_x

    # Add reverse edges — use T.ToUndirected() when available, manual
    # fallback otherwise. NEVER combine both approaches.
    try:
        import torch_geometric.transforms as T
        data = T.ToUndirected()(data)
    except (ImportError, Exception):
        # Manual fallback: flip forward edges to create reverse types
        fwd_ss = data.get(('cell_sector', 'same_site', 'site'))
        if fwd_ss is not None and hasattr(fwd_ss, 'edge_index') and fwd_ss.edge_index.numel() > 0:
            data['site', 'rev_same_site', 'cell_sector'].edge_index = fwd_ss.edge_index.flip(0)

        fwd_bt = data.get(('backhaul_node', 'shares_transport', 'cell_sector'))
        if fwd_bt is not None and hasattr(fwd_bt, 'edge_index') and fwd_bt.edge_index.numel() > 0:
            data['cell_sector', 'rev_shares_transport', 'backhaul_node'].edge_index = fwd_bt.edge_index.flip(0)

    return data


if __name__ == "__main__":
    main()
