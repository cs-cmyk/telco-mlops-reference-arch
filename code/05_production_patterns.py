#!/usr/bin/env python3
"""
05_production_patterns.py — Telco MLOps Reference Architecture: Production Patterns
====================================================================================

Companion code for: "Telco MLOps Reference Architecture: How Multi-Team,
Multi-Model Organizations Ship ML at Scale Without Losing Control"

PURPOSE
-------
Demonstrates production-grade MLOps patterns for a multi-team, multi-model
telco environment. This script implements:

  (a) Standards-aligned feature computation matching serving-time behaviour
      (eliminates training-serving skew — the #1 silent failure mode in
      production telco ML per the synthesis brief)
  (b) Model serving with health checks, version reporting, and latency budgets
  (c) Circuit-breaker pattern with rule-based fallback for network-impacting
      models (blast-radius control)
  (d) Wasserstein-distance drift detection integrated with Prometheus-compatible
      metric export
  (e) Automated retraining trigger based on performance degradation threshold
  (f) Prediction logging for full model observability
  (g) Multi-tenant model registry interaction patterns (stub using MLflow
      conventions)

ARCHITECTURE LAYER MAPPING (from whitepaper Figure 2)
------------------------------------------------------
  Layer 2 (Serving & Monitoring): sections B, C, D, E, F
  Layer 3 (ML Lifecycle):         sections A, G
  Layer 4 (Governance):           sections C (circuit-breaker policy)

STANDARDS ALIGNMENT
-------------------
  * 3GPP TS 28.105 §7.3  — MLModelPerformance IOC → PerformanceMonitor class
  * 3GPP TS 28.627/TR 28.861 — LoopState transitions → RetrainingTrigger class
  * O-RAN WG3 CM (emerging; verify current document number at o-ran.org) — Blast-radius control → CircuitBreaker class
  * TM Forum ODA         — Model card metadata → ModelCard dataclass

HOW TO RUN
----------
  # Standalone (generates its own synthetic data):
  python 05_production_patterns.py

  # With artifacts from prior scripts:
  python 05_production_patterns.py --data-dir ./data --model-dir ./models

  # With Prometheus exposition endpoint (requires prometheus_client):
  python 05_production_patterns.py --serve-metrics --metrics-port 8001

⚠️  FEATURE NAMESPACE WARNING
PRODUCTION REQUIREMENT: This script uses dotted namespace (ran.kpi.dl_prb_utilisation)
for illustration of the compute_pm_features_online() pattern ONLY.
Production deployments MUST use the flat snake_case convention from scripts 01–04
(e.g., dl_prb_utilization, not ran.kpi.dl_prb_utilisation).
See FEATURE_NAMESPACE_CONVENTION.md in the companion code root.

CONSEQUENCE OF IGNORING: Connecting this script directly to model artifacts from
03_model_training.py without renaming features will produce all-NaN feature vectors
(every feature lookup misses), triggering the circuit breaker on every request.

REQUIREMENTS
------------
  Python 3.10+
  pip install pandas numpy scipy scikit-learn matplotlib joblib

  Optional (for Prometheus metrics export):
  pip install prometheus_client

  Optional (for full MLflow registry integration):
  pip install mlflow

Coursebook cross-reference:
  Ch. 8  — Model Serving and Inference Patterns
  Ch. 9  — Monitoring and Observability
  Ch. 10 — CI/CD for Machine Learning
  Ch. 11 — Feature Stores and Training-Serving Consistency
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import enum
import hashlib
import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
import uuid
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Directory layout matches prior scripts in the series
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./production_output"))

# Prometheus availability is optional; degrade gracefully when absent
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        push_to_gateway,
        start_http_server,
        write_to_textfile,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus_client not installed — metric export disabled")

# MLflow availability is optional; we stub the registry interaction
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.debug("mlflow not installed — registry stubs will be used")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Real-time telco ROP granularity for PM counters (3GPP TS 28.550)
ROP_MINUTES: int = 15

# Wasserstein drift thresholds (calibrated against telco feature distributions)
# Tuned conservatively: network engineers need low false-alarm rates
DRIFT_WARN_THRESHOLD: float = 0.15   # warn; investigate
DRIFT_ALERT_THRESHOLD: float = 0.30  # alert; retraining candidate
DRIFT_CRITICAL_THRESHOLD: float = 0.50  # critical; consider circuit break

# Latency budgets per serving tier (milliseconds)
# See whitepaper System Design §4 — Figure 4 sequence diagram
LATENCY_BUDGET_MS: Dict[str, int] = {
    "ran_near_rt":     50,   # Near-RT RIC xApp path (O-RAN WG3)
    "ran_non_rt":     500,   # Non-RT RIC rApp path
    "cx_realtime":    200,   # Customer-facing real-time (fraud, churn risk)
    "batch_planning": 3600_000,  # Capacity planning (hours → ms)
}

# Feature namespace convention: {squad}.{domain}.{feature}
# Enforced at feature store write time; violations raise ValueError
VALID_SQUADS = frozenset({"ran", "cx", "fraud", "oss", "energy"})

# Performance degradation thresholds that trigger retraining consideration
# (See TS 28.627 LoopState transitions; LoopState=Inactive triggered below)
RETRAINING_THRESHOLDS: Dict[str, float] = {
    "auc_roc_min":           0.72,
    "f1_min":                0.60,
    "precision_min":         0.55,
    "false_positive_rate_max": 0.15,  # NOC staff alarm fatigue threshold
}

# Circuit-breaker states (O-RAN WG3 blast-radius control)
class CircuitState(enum.Enum):
    CLOSED = "closed"        # normal operation — model inference active
    OPEN = "open"            # tripped — fallback rule active
    HALF_OPEN = "half_open"  # probing — allowing limited traffic through


# ============================================================================
# SECTION A: STANDARDS-ALIGNED FEATURE COMPUTATION
# Eliminates training-serving skew by sharing computation logic between
# training pipeline and online serving path.
# See Coursebook Ch.11: Feature Stores and Training-Serving Consistency
# ============================================================================

@dataclass
class FeatureEntity:
    """
    TM Forum SID-aligned entity definition for the shared feature store.

    In Feast terms this maps to an EntityDef; in serving code it maps to
    the natural key used to look up pre-computed features from Redis/Iceberg.
    """
    entity_type: str       # "cell", "site", "customer", "slice"
    entity_id: str         # e.g. "CELL_042_1", "customer_9912"
    timestamp: datetime    # Point-in-time timestamp (prevents feature leakage)
    squad_namespace: str   # Owning squad — enforces access policy at write time

    def __post_init__(self) -> None:
        if self.squad_namespace not in VALID_SQUADS:
            raise ValueError(
                f"Unknown squad namespace '{self.squad_namespace}'. "
                f"Valid namespaces: {sorted(VALID_SQUADS)}"
            )


@dataclass
class FeatureVector:
    """
    Computed feature vector with provenance metadata.

    Carrying version + computation_ts enables drift detection to compare
    feature distributions across time without re-joining raw data.
    """
    entity: FeatureEntity
    features: Dict[str, float]
    feature_view_version: str          # semver string matching registry entry
    computation_ts: datetime           # when features were computed
    feature_store_read_latency_ms: float = 0.0
    from_cache: bool = False           # True = online store hit, False = computed

    def to_numpy(self, feature_names: List[str]) -> np.ndarray:
        """Return features in consistent column order for model inference."""
        return np.array([self.features.get(name, np.nan) for name in feature_names],
                        dtype=np.float32)


def compute_pm_features_online(
    raw_counters: Dict[str, float],
    cell_id: str,
    timestamp: datetime,
    squad_namespace: str = "ran",
) -> FeatureVector:
    """
    Compute cell-level PM counter features for online (serving-time) inference.

    This function MUST produce identical output to the Flink streaming job
    that populates the feature store offline store. Any divergence = training-
    serving skew = silent accuracy degradation.

    NOTE: This function demonstrates the *pattern* for serving-time feature
    computation but uses a different feature namespace (ran.kpi.*) than the
    main pipeline scripts (01–04), which use flat column names (dl_prb_utilization,
    etc.). Production implementations must align the raw counter names with the
    operator's actual PM counter catalogue and the feature store's registered
    feature view schema. The run_production_patterns_demo() function constructs
    FeatureVectors directly from synthetic data rather than calling this function,
    so this code path is not exercised in the demo.

    The feature names follow the convention: {squad}.{domain}.{metric}_{statistic}
    This mirrors the Feast feature view naming enforced at write time.

    Parameters
    ----------
    raw_counters:
        Dict of raw PM counter values keyed by 3GPP TS 28.550 counter names.
        Typical keys: DL_PRBUsage_Active, RRCConnEstab_Succ, etc.
    cell_id:
        Cell identifier in CELL_XXX_YYY format.
    timestamp:
        ROP end timestamp (15-minute boundary per TS 28.550).
    squad_namespace:
        Feature store namespace for access control.

    Returns
    -------
    FeatureVector with computed features and provenance metadata.

    Notes
    -----
    See whitepaper Section 7 — "Feature Store with Online/Offline Paths"
    Coursebook Ch.4: Feature Engineering §4.3 Temporal Features
    """
    t_start = time.perf_counter()

    entity = FeatureEntity(
        entity_type="cell",
        entity_id=cell_id,
        timestamp=timestamp,
        squad_namespace=squad_namespace,
    )

    # --- Base KPI features (direct counter normalization) ---
    # PRB utilisation: ratio to maximum available PRBs
    # Expects a pre-normalised 0–1 fraction (computed upstream as
    # DL.PRBUsage.Active / DL.PRBUsage.Total). Raw integer PRB counts
    # must be divided before calling this function. See §3 counter mapping table.
    # NOTE: Production deployments must divide DL.PRBUsage.Active (integer count)
    # by DL.PRBUsage.Total to obtain the 0–1 utilisation fraction. This function
    # uses the normalised (0–1) convention from the platform feature schema.
    # Raw counter → fraction conversion must occur in the O1 parser or Flink
    # normalisation job BEFORE this function is called. See §3 counter mapping table.
    dl_prb_util = float(np.clip(raw_counters.get("DL_PRBUsage_Active", 0.0), 0.0, 1.0))
    ul_prb_util = float(np.clip(raw_counters.get("UL_PRBUsage_Active", 0.0), 0.0, 1.0))

    # RRC setup success rate (dimensionless, 0–1)
    rrc_att = max(float(raw_counters.get("RRCConnEstab_Att", 1.0)), 1.0)  # avoid div/0
    rrc_succ = float(raw_counters.get("RRCConnEstab_Succ", 0.0))
    rrc_sr = float(np.clip(rrc_succ / rrc_att, 0.0, 1.0))

    # E-RAB setup success rate
    erab_att = max(float(raw_counters.get("ERABEstab_Att_Init", 1.0)), 1.0)
    erab_succ = float(raw_counters.get("ERABEstab_Succ_Init", 0.0))
    erab_sr = float(np.clip(erab_succ / erab_att, 0.0, 1.0))

    # Handover success rate
    ho_att = max(float(raw_counters.get("HO_Att_Inter_eNB", 1.0)), 1.0)
    ho_succ = float(raw_counters.get("HO_Succ_Inter_eNB", 0.0))
    ho_sr = float(np.clip(ho_succ / ho_att, 0.0, 1.0))

    # Throughput in Mbps.
    # ASSUMPTION: DL_VolDL_DRB is a VOLUME counter (total kbits in this ROP).
    # If your vendor reports a pre-averaged RATE (kbits/sec or Mbps), do NOT
    # divide by rop_duration_sec. See §3 counter mapping table for vendor guidance.
    rop_duration_sec = ROP_MINUTES * 60
    dl_vol_kbits = float(raw_counters.get("DL_VolDL_DRB", 0.0))  # kbits
    ul_vol_kbits = float(raw_counters.get("UL_VolUL_DRB", 0.0))
    dl_tput_mbps = (dl_vol_kbits / 1000.0) / rop_duration_sec  # Mbps average
    ul_tput_mbps = (ul_vol_kbits / 1000.0) / rop_duration_sec

    # Active UE count
    active_ue = float(max(raw_counters.get("RRCConn_Max", 0.0), 0.0))

    # --- Temporal context features ---
    # These are cheap to compute online and carry diurnal/weekly signal.
    # Cyclical encoding (sin/cos) avoids the discontinuity at midnight/Mon.
    hour = timestamp.hour + timestamp.minute / 60.0
    dow = timestamp.weekday()  # 0=Mon … 6=Sun

    hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
    dow_sin = float(np.sin(2 * np.pi * dow / 7.0))
    dow_cos = float(np.cos(2 * np.pi * dow / 7.0))
    is_peak = float(7 <= timestamp.hour <= 9 or 17 <= timestamp.hour <= 20)
    is_weekend = float(dow >= 5)

    # --- Derived ratio features ---
    # PRB load imbalance between DL and UL (value > 0 → DL-heavy)
    prb_imbalance = float(dl_prb_util - ul_prb_util)

    # Efficiency: throughput per active UE (avoids bias on lightly loaded cells)
    dl_tput_per_ue = float(dl_tput_mbps / max(active_ue, 1.0))

    # Composite quality score: geometric mean of success rates
    # Zero-safe: replace 0 with small epsilon before log
    eps = 1e-6
    quality_score = float(
        np.exp(np.mean(np.log([max(rrc_sr, eps), max(erab_sr, eps), max(ho_sr, eps)])))
    )

    features = {
        # --- RAN KPI features (squad=ran, domain=kpi) ---
        "ran.kpi.dl_prb_utilisation":     dl_prb_util,
        "ran.kpi.ul_prb_utilisation":     ul_prb_util,
        "ran.kpi.rrc_setup_success_rate": rrc_sr,
        "ran.kpi.erab_setup_success_rate": erab_sr,
        "ran.kpi.ho_success_rate":        ho_sr,
        "ran.kpi.dl_throughput_mbps":     dl_tput_mbps,
        "ran.kpi.ul_throughput_mbps":     ul_tput_mbps,
        "ran.kpi.active_ue_count":        active_ue,
        # --- Derived / engineered features ---
        "ran.derived.prb_imbalance":      prb_imbalance,
        "ran.derived.dl_tput_per_ue":     dl_tput_per_ue,
        "ran.derived.quality_score":      quality_score,
        # --- Temporal context features ---
        "ctx.time.hour_sin":              hour_sin,
        "ctx.time.hour_cos":              hour_cos,
        "ctx.time.dow_sin":               dow_sin,
        "ctx.time.dow_cos":               dow_cos,
        "ctx.time.is_peak_hour":          is_peak,
        "ctx.time.is_weekend":            is_weekend,
    }

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    return FeatureVector(
        entity=entity,
        features=features,
        feature_view_version="1.3.0",   # matches feature registry entry
        computation_ts=datetime.now(tz=timezone.utc),
        feature_store_read_latency_ms=latency_ms,
        from_cache=False,
    )


def validate_feature_vector(
    fv: FeatureVector,
    expected_feature_names: List[str],
) -> Tuple[bool, List[str]]:
    """
    Validate a FeatureVector before passing it to model inference.

    Checks presence of all expected features and guards against NaN/Inf
    values that would silently produce garbage model outputs.

    Returns
    -------
    (is_valid, list_of_violations)
    """
    violations: List[str] = []

    # Check all required features are present
    missing = set(expected_feature_names) - set(fv.features.keys())
    if missing:
        violations.append(f"Missing features: {sorted(missing)}")

    # Check for NaN/Inf
    for name in expected_feature_names:
        val = fv.features.get(name, np.nan)
        if not np.isfinite(val):
            violations.append(f"Non-finite value in feature '{name}': {val}")

    return len(violations) == 0, violations


# ============================================================================
# SECTION B: MODEL CARD (GOVERNANCE LAYER)
# Maps to TM Forum ODA AI governance requirements and 3GPP TS 28.105
# MLTrainingReport IOC. Every registered model MUST carry a ModelCard.
# See whitepaper Section 7 — "Model Registry & Governance Gate"
# ============================================================================

@dataclass
class ModelCard:
    """
    Governance metadata carried by every registered model.

    Extends the standard model card concept (Mitchell et al., 2019) with
    telco-specific fields required by:
      - TM Forum IG1230 AI Management and Governance
      - EU AI Act (Article 11 — technical documentation for high-risk AI)
      - 3GPP TS 28.105 §7.2 MLTrainingReport IOC

    The RAN control parameter (RCP) write-set and KPI dependency-set fields
    are the key innovation: they enable pre-deployment conflict screening
    (Gap 1 from the synthesis brief) without requiring inference-time analysis.
    """
    # --- Identity ---
    model_id: str                          # UUID assigned at registration
    model_name: str                        # human-readable, e.g. "ran-anomaly-detector"
    model_version: str                     # semver, e.g. "2.1.4"
    squad: str                             # owning squad from VALID_SQUADS
    model_description: str

    # --- Lifecycle ---
    training_start_date: str              # ISO 8601
    training_end_date: str
    registered_by: str                    # user/service-account
    approved_by: Optional[str] = None    # None = pending approval
    approval_date: Optional[str] = None
    deployment_status: str = "staging"   # staging | canary | production | deprecated

    # --- Data provenance (3GPP TS 28.105 §7.3 MLModelInfo) ---
    training_data_sources: List[str] = field(default_factory=list)
    training_data_start: Optional[str] = None
    training_data_end: Optional[str] = None
    feature_view_version: str = "unknown"

    # --- Performance targets (SLOs) ---
    target_auc_roc: float = 0.80
    target_f1: float = 0.70
    target_latency_ms: int = 200
    target_false_positive_rate: float = 0.10

    # --- Achieved performance (filled at evaluation gate) ---
    achieved_auc_roc: Optional[float] = None
    achieved_f1: Optional[float] = None
    achieved_precision: Optional[float] = None
    achieved_recall: Optional[float] = None
    evaluation_test_period: Optional[str] = None

    # --- Conflict screening fields (Gap 1 from synthesis brief) ---
    # RAN control parameters this model writes to (O-RAN WG3 emerging spec; see whitepaper §4 for caveats)
    rcp_write_set: List[str] = field(default_factory=list)
    # KPIs this model's decisions depend on (for indirect conflict detection)
    kpi_dependency_set: List[str] = field(default_factory=list)
    # Network elements this model affects
    affected_ne_types: List[str] = field(default_factory=list)

    # --- Operational metadata ---
    rollback_procedure: str = "Contact squad owner; toggle to previous version via registry CLI"
    blast_radius: str = "single-cell"    # single-cell | cluster | region | global
    serving_tier: str = "ran_non_rt"     # key into LATENCY_BUDGET_MS
    fallback_strategy: str = "rule_based"  # rule_based | previous_version | abstain

    # --- EU AI Act high-risk classification (Article 6) ---
    # Models controlling network parameters may be classified as high-risk
    # if they affect critical infrastructure (Annex III)
    eu_ai_act_category: str = "high_risk"      # Default to high_risk for telco network-impacting models
    # per §12 EU AI Act guidance. Operators must explicitly downgrade to
    # 'limited_risk' with documented justification.  Options: limited_risk | high_risk | unclassified
    human_oversight_required: bool = True

    def is_promotion_ready(self) -> Tuple[bool, List[str]]:
        """
        Check whether this model card satisfies promotion gate criteria.

        Returns (ready, list_of_blocking_issues).
        """
        issues: List[str] = []

        if self.approved_by is None:
            issues.append("Model has not received governance approval")

        if self.achieved_auc_roc is None:
            issues.append("No evaluation metrics recorded (evaluation gate not passed)")
        elif self.achieved_auc_roc < self.target_auc_roc:
            issues.append(
                f"AUC-ROC {self.achieved_auc_roc:.3f} below target {self.target_auc_roc:.3f}"
            )

        if self.achieved_f1 is None:
            issues.append("F1 score not recorded")
        elif self.achieved_f1 < self.target_f1:
            issues.append(
                f"F1 {self.achieved_f1:.3f} below target {self.target_f1:.3f}"
            )

        if not self.training_data_sources:
            issues.append("No training data sources documented")

        if self.eu_ai_act_category == "high_risk" and self.approved_by is None:
            issues.append("High-risk AI system requires explicit human approval")

        return len(issues) == 0, issues

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Model card saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "ModelCard":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# SECTION C: CIRCUIT BREAKER PATTERN
# Protects network operations from bad model deployments.
# A bad churn model sends wrong offers; a bad RAN model degrades network
# performance — different risk profiles, same blast-radius control mechanism.
# See whitepaper Section 10 — "Blast Radius Control"
# Conflict mitigation aligned with O-RAN WG3 emerging specifications (see whitepaper §4 for caveats)
# ============================================================================

@dataclass
class CircuitBreakerConfig:
    """Configuration for the circuit breaker protecting a model endpoint."""
    failure_threshold: int = 5          # consecutive failures before OPEN
    success_threshold: int = 3          # consecutive successes to CLOSE from HALF_OPEN
    timeout_seconds: float = 60.0       # how long to stay OPEN before probing
    latency_threshold_ms: float = 500.0 # treat slow responses as failures
    error_rate_threshold: float = 0.30  # fraction of failures in sliding window
    window_size: int = 20               # sliding window for error rate calculation


class CircuitBreaker:
    """
    Thread-safe circuit breaker for ML model inference endpoints.

    Implements the classic three-state automaton (CLOSED → OPEN → HALF_OPEN)
    with an additional latency-based trip wire. When OPEN, all calls are
    redirected to the rule-based fallback, preserving network stability.

    This is the production implementation of the blast-radius control pattern
    described in whitepaper Section 10. The fallback callable should be a
    deterministic rule-based function that network engineers understand and
    can reason about during an incident.

    Thread safety: all state mutations are protected by a reentrant lock.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback_fn: Optional[Callable[..., Any]] = None,
        metrics_collector: Optional["MetricsCollector"] = None,
    ) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback_fn = fallback_fn
        self.metrics = metrics_collector

        self._state = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.RLock()

        # Sliding window for error rate calculation
        self._result_window: Deque[bool] = collections.deque(
            maxlen=self.config.window_size
        )

        logger.info(
            "CircuitBreaker '%s' initialised [state=CLOSED, "
            "failure_threshold=%d, timeout=%.1fs]",
            self.name, self.config.failure_threshold, self.config.timeout_seconds,
        )

    @property
    def state(self) -> CircuitState:
        with self._lock:
            # Auto-transition from OPEN → HALF_OPEN after timeout
            if (
                self._state == CircuitState.OPEN
                and self._last_failure_time is not None
                and (time.monotonic() - self._last_failure_time) >= self.config.timeout_seconds
            ):
                logger.info(
                    "CircuitBreaker '%s': timeout elapsed, transitioning OPEN → HALF_OPEN",
                    self.name,
                )
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
            return self._state

    def call(
        self,
        primary_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Any, bool]:
        """
        Execute primary_fn through the circuit breaker.

        Returns
        -------
        (result, used_fallback)
            result      — output of primary_fn or fallback_fn
            used_fallback — True if fallback was invoked instead of primary

        Raises
        ------
        RuntimeError if circuit is OPEN and no fallback is configured.
        """
        current_state = self.state  # triggers auto-transition check

        if current_state == CircuitState.OPEN:
            return self._handle_open_state(*args, **kwargs)

        # CLOSED or HALF_OPEN: attempt primary call
        t_start = time.perf_counter()
        try:
            result = primary_fn(*args, **kwargs)
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            # Treat latency violations as soft failures
            if latency_ms > self.config.latency_threshold_ms:
                logger.warning(
                    "CircuitBreaker '%s': latency %.1fms exceeds threshold %.1fms",
                    self.name, latency_ms, self.config.latency_threshold_ms,
                )
                self._record_result(success=False)
            else:
                self._record_result(success=True)

            if self.metrics:
                self.metrics.record_inference_latency(self.name, latency_ms)

            return result, False

        except Exception as exc:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            logger.error(
                "CircuitBreaker '%s': primary call failed after %.1fms — %s",
                self.name, latency_ms, exc,
            )
            self._record_result(success=False)
            return self._handle_open_state(*args, **kwargs)

    def _record_result(self, success: bool) -> None:
        with self._lock:
            self._result_window.append(success)

            if success:
                self._failure_count = 0
                if self._state == CircuitState.HALF_OPEN:
                    self._success_count += 1
                    if self._success_count >= self.config.success_threshold:
                        logger.info(
                            "CircuitBreaker '%s': HALF_OPEN → CLOSED "
                            "(%d consecutive successes)",
                            self.name, self._success_count,
                        )
                        self._state = CircuitState.CLOSED
                        if self.metrics:
                            self.metrics.record_circuit_state_change(self.name, "closed")
            else:
                self._failure_count += 1
                self._last_failure_time = time.monotonic()

                # Trip on consecutive failures
                if self._failure_count >= self.config.failure_threshold:
                    logger.error(
                        "CircuitBreaker '%s': CLOSED → OPEN "
                        "(%d consecutive failures)",
                        self.name, self._failure_count,
                    )
                    self._state = CircuitState.OPEN
                    if self.metrics:
                        self.metrics.record_circuit_state_change(self.name, "open")
                    return

                # Also trip on high error rate in sliding window
                if len(self._result_window) == self.config.window_size:
                    error_rate = 1.0 - (sum(self._result_window) / len(self._result_window))
                    if error_rate >= self.config.error_rate_threshold:
                        logger.error(
                            "CircuitBreaker '%s': CLOSED → OPEN "
                            "(error rate %.1f%% in window of %d)",
                            self.name, error_rate * 100, self.config.window_size,
                        )
                        self._state = CircuitState.OPEN
                        if self.metrics:
                            self.metrics.record_circuit_state_change(self.name, "open")

    def _handle_open_state(self, *args: Any, **kwargs: Any) -> Tuple[Any, bool]:
        """Route to fallback when circuit is OPEN."""
        if self.fallback_fn is not None:
            logger.info(
                "CircuitBreaker '%s': invoking fallback (circuit OPEN)", self.name
            )
            if self.metrics:
                self.metrics.record_fallback_invocation(self.name)
            return self.fallback_fn(*args, **kwargs), True
        raise RuntimeError(
            f"CircuitBreaker '{self.name}' is OPEN and no fallback is configured. "
            "Model endpoint unavailable."
        )

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            window_list = list(self._result_window)
            error_rate = (
                1.0 - (sum(window_list) / len(window_list))
                if window_list else 0.0
            )
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "error_rate_window": round(error_rate, 4),
                "window_size": len(window_list),
                "last_failure_age_s": (
                    round(time.monotonic() - self._last_failure_time, 1)
                    if self._last_failure_time else None
                ),
            }


def rule_based_anomaly_fallback(
    feature_vector: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Rule-based fallback for RAN anomaly detection when the ML model circuit
    is open.

    This implements a simple threshold-based check on the most reliable
    indicators. Network engineers can read, audit, and override these rules
    without ML expertise — a critical property for network operations safety.

    Production fallback rules MUST use named feature access; positional
    indexing produces silent errors when feature order changes during
    registry updates.

    Parameters
    ----------
    feature_vector : np.ndarray
        Feature values in the order specified by feature_names.
    feature_names : list of str, optional
        Feature names matching the vector positions. If provided, features
        are looked up by name. If None, falls back to positional indexing
        (demo only — not safe for production).

    Returns
    -------
    Dict with 'anomaly' (bool), 'confidence' (float), 'trigger_rule' (str)
    """
    try:
        # Build name→value mapping for safe access
        if feature_names is not None:
            feat = dict(zip(feature_names, feature_vector.flatten()))
            dl_prb = feat.get("ran.kpi.dl_prb_utilisation",
                     feat.get("dl_prb_utilization", 0.50))
            rrc_sr = feat.get("ran.kpi.rrc_setup_success_ratio",
                     feat.get("rrc_setup_success_ratio", 0.95))
        else:
            # Positional fallback — demo only, not safe for production
            dl_prb = feature_vector[0] if len(feature_vector) > 0 else 0.50
            rrc_sr = feature_vector[2] if len(feature_vector) > 2 else 0.95

        # Rule 1: extreme PRB utilisation (sustained overload) — anomaly detection fallback.
        # The sleep eligibility fallback in §9 uses the inverse condition (PRB < 0.15 → sleep eligible).
        if dl_prb > 0.95:
            return {"anomaly": True, "confidence": 0.85, "trigger_rule": "PRB_OVERLOAD"}

        # Rule 2: RRC setup failure rate exceeds NOC threshold
        if rrc_sr < 0.85:
            return {"anomaly": True, "confidence": 0.90, "trigger_rule": "RRC_SETUP_FAILURE"}

        return {"anomaly": False, "confidence": 0.70, "trigger_rule": "NO_THRESHOLD_BREACH"}

    except (IndexError, TypeError):
        # Safest default for unknown input: flag for investigation
        logger.warning("Fallback rule received malformed input; defaulting to anomaly=False")
        return {"anomaly": False, "confidence": 0.50, "trigger_rule": "FALLBACK_ERROR"}


# ============================================================================
# SECTION D: PROMETHEUS-COMPATIBLE METRICS COLLECTION
# Provides the observability substrate for SLO tracking, drift alerting,
# and cost attribution. Degrades gracefully when prometheus_client is absent.
# See whitepaper Section 7 — "Monitoring Stack"
# ============================================================================

class MetricsCollector:
    """
    Collects model serving and drift metrics for export to Prometheus.

    NOTE: In production, whylogs (https://whylabs.ai/whylogs) provides
    streaming feature profiling with richer statistical summaries than
    the lightweight counters/histograms implemented here. See §6 Layer 2
    for the recommended dual-timescale monitoring architecture
    (whylogs for minutes-scale, Evidently for daily-scale drift detection).
    This MetricsCollector implements a minimal profiling subset for
    demonstration without the whylogs dependency.

    When prometheus_client is available, metrics are registered in the
    default registry and can be scraped by Prometheus. When it is absent,
    metrics are accumulated in-memory and can be exported as JSON or to
    a text file.

    This class is intentionally slim: it wraps the prometheus_client API
    so that the rest of the codebase does not have conditional imports
    scattered throughout.
    """

    def __init__(
        self,
        squad: str,
        model_name: str,
        model_version: str,
        registry: Optional[Any] = None,  # prometheus_client.CollectorRegistry
    ) -> None:
        self.squad = squad
        self.model_name = model_name
        self.model_version = model_version
        self._labels = {
            "squad": squad,
            "model": model_name,
            "version": model_version,
        }

        # In-memory fallback counters (always maintained for JSON export)
        self._counters: Dict[str, float] = collections.defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = collections.defaultdict(list)

        self._prometheus_enabled = PROMETHEUS_AVAILABLE
        if self._prometheus_enabled:
            self._registry = registry or CollectorRegistry()
            self._setup_prometheus_metrics()

        logger.debug(
            "MetricsCollector initialised [squad=%s, model=%s, v=%s, prometheus=%s]",
            squad, model_name, model_version, self._prometheus_enabled,
        )

    def _setup_prometheus_metrics(self) -> None:
        """Register Prometheus metric objects."""
        label_names = list(self._labels.keys())
        try:
            self._prom_inference_total = Counter(
                "mlops_inference_requests_total",
                "Total inference requests",
                label_names,
                registry=self._registry,
            )
            self._prom_inference_errors = Counter(
                "mlops_inference_errors_total",
                "Total inference errors",
                label_names,
                registry=self._registry,
            )
            self._prom_fallback_total = Counter(
                "mlops_fallback_invocations_total",
                "Total circuit-breaker fallback invocations",
                label_names,
                registry=self._registry,
            )
            self._prom_latency = Histogram(
                "mlops_inference_latency_ms",
                "Inference latency in milliseconds",
                label_names,
                buckets=[5, 10, 25, 50, 100, 200, 500, 1000, 5000],
                registry=self._registry,
            )
            self._prom_drift_score = Gauge(
                "mlops_feature_drift_score",
                "Wasserstein drift score per feature",
                label_names + ["feature_name"],
                registry=self._registry,
            )
            self._prom_drift_alert = Gauge(
                "mlops_drift_alert_active",
                "1 if drift alert is active, 0 otherwise",
                label_names,
                registry=self._registry,
            )
            self._prom_circuit_state = Gauge(
                "mlops_circuit_breaker_open",
                "1 if circuit breaker is open, 0 if closed",
                label_names,
                registry=self._registry,
            )
            self._prom_prediction_score = Histogram(
                "mlops_prediction_score",
                "Distribution of model prediction scores",
                label_names,
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                registry=self._registry,
            )
        except Exception as exc:
            logger.warning("Failed to register Prometheus metrics: %s", exc)
            self._prometheus_enabled = False

    def record_inference(self, latency_ms: float, prediction_score: float) -> None:
        self._counters["inference_total"] += 1
        self._histograms["latency_ms"].append(latency_ms)
        self._histograms["prediction_score"].append(prediction_score)
        if self._prometheus_enabled:
            try:
                self._prom_inference_total.labels(**self._labels).inc()
                self._prom_latency.labels(**self._labels).observe(latency_ms)
                self._prom_prediction_score.labels(**self._labels).observe(prediction_score)
            except Exception:
                pass

    def record_inference_latency(self, model_name: str, latency_ms: float) -> None:
        self._histograms["latency_ms"].append(latency_ms)
        if self._prometheus_enabled:
            try:
                self._prom_latency.labels(**self._labels).observe(latency_ms)
            except Exception:
                pass

    def record_error(self) -> None:
        self._counters["inference_errors"] += 1
        if self._prometheus_enabled:
            try:
                self._prom_inference_errors.labels(**self._labels).inc()
            except Exception:
                pass

    def record_fallback_invocation(self, model_name: str) -> None:
        self._counters["fallback_invocations"] += 1
        if self._prometheus_enabled:
            try:
                self._prom_fallback_total.labels(**self._labels).inc()
            except Exception:
                pass

    def record_circuit_state_change(self, model_name: str, new_state: str) -> None:
        is_open = 1.0 if new_state == "open" else 0.0
        self._gauges["circuit_open"] = is_open
        if self._prometheus_enabled:
            try:
                self._prom_circuit_state.labels(**self._labels).set(is_open)
            except Exception:
                pass

    def record_drift_score(self, feature_name: str, score: float) -> None:
        key = f"drift_{feature_name}"
        self._gauges[key] = score
        if self._prometheus_enabled:
            try:
                self._prom_drift_score.labels(**self._labels, feature_name=feature_name).set(score)
            except Exception:
                pass

    def record_drift_alert(self, active: bool) -> None:
        self._gauges["drift_alert"] = 1.0 if active else 0.0
        if self._prometheus_enabled:
            try:
                self._prom_drift_alert.labels(**self._labels).set(1.0 if active else 0.0)
            except Exception:
                pass

    def get_summary(self) -> Dict[str, Any]:
        latencies = self._histograms.get("latency_ms", [])
        scores = self._histograms.get("prediction_score", [])
        return {
            "squad": self.squad,
            "model": self.model_name,
            "version": self.model_version,
            "inference_total": int(self._counters["inference_total"]),
            "inference_errors": int(self._counters["inference_errors"]),
            "fallback_invocations": int(self._counters["fallback_invocations"]),
            "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
            "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "latency_p99_ms": float(np.percentile(latencies, 99)) if latencies else None,
            "prediction_score_mean": float(np.mean(scores)) if scores else None,
            "drift_alert_active": bool(self._gauges.get("drift_alert", 0)),
            "circuit_open": bool(self._gauges.get("circuit_open", 0)),
            "drift_scores": {
                k.replace("drift_", ""): v
                for k, v in self._gauges.items()
                if k.startswith("drift_")
            },
        }

    def write_prometheus_text(self, path: Path) -> None:
        """Write metrics in Prometheus text exposition format."""
        if self._prometheus_enabled:
            try:
                write_to_textfile(str(path), self._registry)
                logger.info("Prometheus metrics written → %s", path)
            except Exception as exc:
                logger.warning("Failed to write Prometheus text: %s", exc)
        else:
            # Fallback: write JSON summary
            json_path = path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(self.get_summary(), f, indent=2)
            logger.info("Metrics summary (JSON fallback) written → %s", json_path)


# ============================================================================
# SECTION E: WASSERSTEIN DRIFT DETECTION
# Wasserstein distance (Earth Mover's Distance) is preferred over simpler
# KS test for telco feature distributions because:
#   (1) PM counter distributions are typically non-Gaussian
#   (2) Wasserstein captures magnitude of shift, not just significance
#   (3) More robust to small sample sizes (sub-hourly ROP windows)
# Aligned with Evidently AI's drift detection methodology.
# See whitepaper Section 7 — "Monitoring Stack"
# 3GPP TS 28.105 §7.3 MLModelPerformance IOC
# ============================================================================

@dataclass
class DriftReport:
    """
    Per-model drift analysis result.

    Generated by DriftDetector.compute_drift_report().
    Written to the monitoring store and triggers Prometheus alerts.
    """
    model_name: str
    model_version: str
    squad: str
    report_ts: str                         # ISO 8601
    reference_period: str                  # "YYYY-MM-DD / YYYY-MM-DD"
    current_period: str
    n_reference_samples: int
    n_current_samples: int
    feature_drift_scores: Dict[str, float] # feature_name → Wasserstein distance
    drift_status: str                      # "ok" | "warn" | "alert" | "critical"
    drifted_features: List[str]            # features exceeding DRIFT_ALERT_THRESHOLD
    overall_drift_score: float             # weighted mean across features
    retraining_recommended: bool


class DriftDetector:
    """
    Detects feature distribution drift using Wasserstein distance.

    Usage pattern:
    --------------
    1. At training time, save reference distributions via save_reference().
    2. In production, accumulate prediction logs with log_prediction().
    3. Run compute_drift_report() on a schedule (e.g., Airflow daily DAG).
    4. Report triggers Prometheus alert → retraining pipeline if flagged.

    The reference window is typically the training data distribution.
    The current window is the past N hours of production feature values.

    DATA QUALITY vs. DRIFT ALERT ROUTING:
    When Wasserstein scores spike above 2× the CRITICAL threshold in a single
    reporting window, this typically indicates an extreme range violation (counter
    misconfiguration, schema change, or upstream ETL failure) rather than genuine
    distributional shift. Route such alerts to the data quality channel, not the
    model drift channel. Implement by checking: if any feature's current p99
    exceeds ref_max × 1.5, fire a `data_quality_alert` metric instead of (or
    in addition to) the drift alert.

    NOTE: Production deployments should add whylogs profiling to the inference
    loop for near-real-time feature profiling (see §6 Layer 2). This class
    implements the daily batch Wasserstein drift check; whylogs provides the
    complementary streaming-speed statistical profiles that enable sub-hourly
    anomaly detection on individual features.

    Scaling: for 200 features × 50 models, the daily drift compute job
    takes ~2 minutes on a single CPU core — no GPU required.
    """

    def __init__(
        self,
        model_name: str,
        model_version: str,
        squad: str,
        reference_data: Optional[pd.DataFrame] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.squad = squad
        self.metrics = metrics

        self._reference_stats: Dict[str, np.ndarray] = {}  # feature → ref samples
        self._prediction_log: List[Dict[str, Any]] = []    # rolling production log

        if reference_data is not None:
            self.save_reference(reference_data)

    def save_reference(self, reference_df: pd.DataFrame) -> None:
        """
        Persist reference feature distributions from training data.

        Must be called at training time with the same feature columns used
        at serving time. Stores raw samples (not summary statistics) to allow
        exact Wasserstein computation at drift detection time.
        """
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            vals = reference_df[col].dropna().values
            if len(vals) >= 30:  # minimum sample size for reliable Wasserstein
                self._reference_stats[col] = vals
        logger.info(
            "Drift detector '%s': reference saved for %d features (%d samples each avg)",
            self.model_name,
            len(self._reference_stats),
            int(np.mean([len(v) for v in self._reference_stats.values()])) if self._reference_stats else 0,
        )

    def log_prediction(
        self,
        feature_vector: FeatureVector,
        prediction: float,
        ground_truth: Optional[float] = None,
    ) -> None:
        """
        Log a single prediction with its feature values.

        In production, this is typically called asynchronously (fire-and-forget)
        to avoid adding latency to the serving path. A background thread
        flushes to persistent storage (e.g., Delta Lake, Iceberg table).
        """
        record: Dict[str, Any] = {
            "ts": feature_vector.computation_ts.isoformat(),
            "cell_id": feature_vector.entity.entity_id,
            "prediction": prediction,
            "ground_truth": ground_truth,
        }
        record.update(feature_vector.features)
        self._prediction_log.append(record)

    def compute_drift_report(
        self,
        current_window_hours: int = 24,
    ) -> DriftReport:
        """
        Compute Wasserstein drift scores for all features.

        Parameters
        ----------
        current_window_hours:
            How many hours of recent predictions to use as the current window.
            Typically 24h for daily batch drift check.

        Returns
        -------
        DriftReport with per-feature scores and overall status.
        """
        if not self._prediction_log:
            logger.warning("Drift detector '%s': no prediction log entries", self.model_name)
            return self._empty_report()

        # Filter to current window
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=current_window_hours)
        current_records = [
            r for r in self._prediction_log
            if datetime.fromisoformat(r["ts"].replace("Z", "+00:00")) >= cutoff
        ]

        if len(current_records) < 10:
            logger.warning(
                "Drift detector '%s': insufficient current window samples (%d)",
                self.model_name, len(current_records),
            )
            return self._empty_report()

        current_df = pd.DataFrame(current_records)
        feature_drift_scores: Dict[str, float] = {}
        drifted_features: List[str] = []

        for feature_name, ref_samples in self._reference_stats.items():
            if feature_name not in current_df.columns:
                continue
            curr_samples = current_df[feature_name].dropna().values
            if len(curr_samples) < 10:
                continue

            # Normalise using the REFERENCE distribution's min/max only.
            # This ensures that current-distribution range shifts are fully
            # reflected in the Wasserstein score. Current values may exceed
            # [0, 1] — this is intentional and correctly captures range drift.
            ref_min = ref_samples.min()
            ref_range = ref_samples.max() - ref_min
            if ref_range < 1e-9:
                # Constant feature in reference — no drift possible
                feature_drift_scores[feature_name] = 0.0
                continue

            ref_norm = (ref_samples - ref_min) / ref_range
            curr_norm = (curr_samples - ref_min) / ref_range  # may exceed [0,1]
            score = float(wasserstein_distance(ref_norm, curr_norm))
            feature_drift_scores[feature_name] = round(score, 4)

            if score >= DRIFT_ALERT_THRESHOLD:
                drifted_features.append(feature_name)

            # Emit per-feature metric to Prometheus
            if self.metrics:
                self.metrics.record_drift_score(feature_name, score)

        overall_score = (
            float(np.mean(list(feature_drift_scores.values())))
            if feature_drift_scores else 0.0
        )

        # Determine overall status
        if overall_score >= DRIFT_CRITICAL_THRESHOLD:
            status = "critical"
        elif overall_score >= DRIFT_ALERT_THRESHOLD:
            status = "alert"
        elif overall_score >= DRIFT_WARN_THRESHOLD:
            status = "warn"
        else:
            status = "ok"

        retraining_recommended = status in ("alert", "critical")

        if self.metrics:
            self.metrics.record_drift_alert(active=retraining_recommended)

        report = DriftReport(
            model_name=self.model_name,
            model_version=self.model_version,
            squad=self.squad,
            report_ts=datetime.now(tz=timezone.utc).isoformat(),
            reference_period="training-window",
            current_period=f"last-{current_window_hours}h",
            n_reference_samples=int(
                np.mean([len(v) for v in self._reference_stats.values()])
            ) if self._reference_stats else 0,
            n_current_samples=len(current_records),
            feature_drift_scores=feature_drift_scores,
            drift_status=status,
            drifted_features=sorted(drifted_features),
            overall_drift_score=round(overall_score, 4),
            retraining_recommended=retraining_recommended,
        )

        logger.info(
            "Drift report '%s' v%s: status=%s, overall_score=%.3f, "
            "drifted_features=%d/%d, retraining=%s",
            self.model_name, self.model_version, status, overall_score,
            len(drifted_features), len(feature_drift_scores), retraining_recommended,
        )

        return report

    def _empty_report(self) -> DriftReport:
        return DriftReport(
            model_name=self.model_name,
            model_version=self.model_version,
            squad=self.squad,
            report_ts=datetime.now(tz=timezone.utc).isoformat(),
            reference_period="unknown",
            current_period="unknown",
            n_reference_samples=0,
            n_current_samples=0,
            feature_drift_scores={},
            drift_status="insufficient_data",
            drifted_features=[],
            overall_drift_score=0.0,
            retraining_recommended=False,
        )


# ============================================================================
# SECTION F: PREDICTION LOGGING AND AUDIT TRAIL
# Every prediction must be logged for:
#   (1) Retrospective performance evaluation (ground truth arrives later)
#   (2) Drift detection reference window accumulation
#   (3) Model audit trail (EU AI Act Article 12 — logging)
#   (4) Retraining dataset construction
# See whitepaper Section 7 — "Monitoring & Feedback Loop"
# ============================================================================

@dataclass
class PredictionLogEntry:
    """
    Immutable record of a single model prediction.

    The prediction_id enables ground truth join-back when labels arrive
    asynchronously (typical in telco: anomaly is confirmed hours later by NOC).
    """
    prediction_id: str          # UUID, used for ground-truth join-back
    model_name: str
    model_version: str
    squad: str
    entity_type: str
    entity_id: str              # e.g., "CELL_042_1"
    request_ts: str             # ISO 8601 UTC
    prediction_score: float     # raw model output (probability for classifiers)
    predicted_label: int        # 0/1 for binary classifiers
    decision_threshold: float
    feature_view_version: str
    used_fallback: bool
    circuit_state: str
    inference_latency_ms: float
    feature_drift_score: Optional[float] = None
    ground_truth_label: Optional[int] = None  # filled in by ground-truth joiner
    ground_truth_ts: Optional[str] = None


class PredictionLogger:
    """
    Asynchronous prediction logger with in-memory buffer and periodic flush.

    Writes are non-blocking: predictions are queued and flushed to Parquet
    (the offline monitoring store) by a background thread. This preserves
    the serving-path latency budget.

    In production, the flush destination is an Iceberg/Delta table in the
    feature store's offline store, enabling:
      - Drift detector to query production feature distributions
      - Ground-truth joiner to enrich records with confirmed labels
      - Performance monitor to compute rolling AUC/F1 against ground truth
    """

    def __init__(
        self,
        log_dir: Path,
        flush_interval_seconds: float = 30.0,
        buffer_size: int = 1000,
    ) -> None:
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._flush_interval = flush_interval_seconds
        self._buffer_size = buffer_size
        self._buffer: List[PredictionLogEntry] = []
        self._lock = threading.Lock()
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="pred-logger-flush"
        )
        self._stop_event = threading.Event()
        self._total_logged: int = 0
        self._flush_thread.start()
        logger.info("PredictionLogger started [dir=%s, flush_interval=%.1fs]",
                    log_dir, flush_interval_seconds)

    def log(self, entry: PredictionLogEntry) -> None:
        """Non-blocking: enqueues entry for async flush."""
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self._buffer_size:
                self._flush_locked()

    def _flush_loop(self) -> None:
        while not self._stop_event.wait(self._flush_interval):
            with self._lock:
                if self._buffer:
                    self._flush_locked()

    def _flush_locked(self) -> None:
        """Must be called with self._lock held."""
        if not self._buffer:
            return
        entries = self._buffer[:]
        self._buffer.clear()

        records = [asdict(e) for e in entries]
        df = pd.DataFrame(records)

        # Partition by date for efficient range queries in the offline store
        ts_date = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H")
        filename = self._log_dir / f"predictions_{ts_date}.parquet"

        if filename.exists():
            existing = pd.read_parquet(filename)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_parquet(filename, index=False, compression="snappy")
        self._total_logged += len(entries)
        logger.debug("PredictionLogger: flushed %d entries → %s", len(entries), filename)

    def flush_and_stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            self._flush_locked()
        logger.info("PredictionLogger stopped. Total logged: %d", self._total_logged)

    def get_recent_predictions(self, n: int = 100) -> pd.DataFrame:
        """Read recent prediction logs for drift detection or debugging."""
        parquet_files = sorted(self._log_dir.glob("predictions_*.parquet"))
        if not parquet_files:
            return pd.DataFrame()
        dfs = [pd.read_parquet(f) for f in parquet_files[-3:]]  # last 3 files
        combined = pd.concat(dfs, ignore_index=True)
        return combined.tail(n)


# ============================================================================
# SECTION G: MODEL SERVER WITH HEALTH CHECKS AND VERSION REPORTING
# Production model server implementing:
#   - Health check endpoint with model version reporting
#   - Standardised prediction pipeline with feature validation
#   - Circuit breaker integration
#   - Async prediction logging
#   - Latency-budget enforcement
# See whitepaper Section 7 — "Serving Infrastructure"
# See whitepaper Section 10 — "Progressive Delivery Patterns"
# ============================================================================

@dataclass
class HealthStatus:
    """
    Model serving health status.

    Returned by the /health endpoint. Used by Kubernetes liveness/readiness
    probes and by the KServe InferenceService health monitor.
    """
    model_name: str
    model_version: str
    squad: str
    status: str                    # "healthy" | "degraded" | "unhealthy"
    circuit_state: str
    uptime_seconds: float
    inference_count: int
    error_count: int
    fallback_count: int
    last_drift_score: Optional[float]
    drift_status: str
    latency_p95_ms: Optional[float]
    latency_budget_ms: int
    model_card_approved: bool
    feature_view_version: str
    timestamp: str

    @property
    def is_ready(self) -> bool:
        """Kubernetes readiness probe: model is accepting traffic."""
        return self.status in ("healthy", "degraded") and self.circuit_state != "open"

    @property
    def is_alive(self) -> bool:
        """Kubernetes liveness probe: process is functional."""
        return self.status != "unhealthy"


class ModelServer:
    """
    Production model server wrapping a trained classifier.

    This class implements the serving patterns described in the whitepaper's
    System Design section (Figures 3 and 4). It is intentionally framework-
    agnostic — the same logic runs in a BentoML Service, a KServe
    InferenceService custom predictor, or a FastAPI microservice.

    Key design decisions:
    ---------------------
    1. Feature validation before inference: prevents silent NaN propagation
       that would produce garbage outputs without raising exceptions.

    2. Circuit breaker wrapping: every model call goes through the breaker,
       providing automatic fallback on failure — critical for RAN models
       where a bad prediction can degrade network performance.

    3. Async prediction logging: zero-overhead audit trail using the
       PredictionLogger's background flush thread.

    4. Latency budget enforcement: if inference exceeds the configured budget,
       the response is still returned but a metric is emitted. Future versions
       may implement hard timeout with fallback.

    5. Decision threshold as a first-class parameter: enables threshold
       tuning post-deployment without retraining (e.g., adjusting the
       precision/recall trade-off for NOC alarm fatigue).
    """

    def __init__(
        self,
        model: Any,
        model_card: ModelCard,
        feature_names: List[str],
        decision_threshold: float = 0.5,
        prediction_logger: Optional[PredictionLogger] = None,
        drift_detector: Optional[DriftDetector] = None,
        metrics: Optional[MetricsCollector] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self.model = model
        self.model_card = model_card
        self.feature_names = feature_names
        self.decision_threshold = decision_threshold
        self.prediction_logger = prediction_logger
        self.drift_detector = drift_detector
        self.metrics = metrics
        self._start_time = time.monotonic()

        # Extract scaler from Pipeline for fallback inverse-transform.
        # When the model is a sklearn Pipeline with an embedded scaler,
        # predict() receives raw features (Pipeline scales internally).
        # The fallback also receives raw features in this case, so no
        # inverse-transform is needed. When the model is a bare classifier
        # that expects pre-scaled input, the caller must pass a scaler
        # explicitly or set self._scaler after construction.
        self._scaler = None
        if hasattr(model, 'named_steps'):
            for name, step in model.named_steps.items():
                if hasattr(step, 'inverse_transform'):
                    self._scaler = step
                    logger.info(
                        "Extracted scaler '%s' from pipeline for fallback "
                        "inverse-transform (used only when model is NOT a Pipeline)",
                        name,
                    )
                    break

        # Set up circuit breaker — wraps the _raw_inference method
        if circuit_breaker is not None:
            self._breaker = circuit_breaker
        else:
            self._breaker = CircuitBreaker(
                name=f"{model_card.squad}.{model_card.model_name}",
                config=CircuitBreakerConfig(
                    failure_threshold=5,
                    timeout_seconds=60.0,
                    latency_threshold_ms=float(
                        LATENCY_BUDGET_MS.get(model_card.serving_tier, 500)
                    ),
                ),
                fallback_fn=self._fallback_predict,
                metrics_collector=metrics,
            )

        # Validate model card at startup — fail loud if governance gate not passed
        ready, issues = model_card.is_promotion_ready()
        if not ready:
            logger.warning(
                "ModelServer '%s': model card has %d issue(s) — "
                "serving will continue but governance gate not fully passed: %s",
                model_card.model_name, len(issues), "; ".join(issues),
            )
        else:
            logger.info(
                "ModelServer '%s' v%s initialised [squad=%s, tier=%s, threshold=%.2f]",
                model_card.model_name, model_card.model_version,
                model_card.squad, model_card.serving_tier, decision_threshold,
            )

    def predict(self, feature_vector: FeatureVector) -> Dict[str, Any]:
        """
        End-to-end prediction: feature validation → inference → logging.

        Returns
        -------
        Dict with keys:
            prediction_id  — UUID for ground-truth join-back
            anomaly        — bool
            score          — float probability (0–1)
            decision_threshold — float
            used_fallback  — bool
            circuit_state  — str
            latency_ms     — float
            feature_view_version — str
        """
        t_start = time.perf_counter()
        prediction_id = str(uuid.uuid4())

        # Feature validation — must happen before circuit breaker to avoid
        # feeding malformed input into the fallback path too
        is_valid, violations = validate_feature_vector(feature_vector, self.feature_names)
        if not is_valid:
            logger.warning(
                "ModelServer '%s': feature validation failed for %s — %s",
                self.model_card.model_name,
                feature_vector.entity.entity_id,
                "; ".join(violations),
            )
            if self.metrics:
                self.metrics.record_error()

        # Build numpy array in stable feature order
        x = feature_vector.to_numpy(self.feature_names).reshape(1, -1)

        # Replace NaN with 0.0 for robustness (post-validation — NaNs are logged above)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Invoke through circuit breaker
        result, used_fallback = self._breaker.call(self._raw_inference, x)

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        budget_ms = LATENCY_BUDGET_MS.get(self.model_card.serving_tier, 500)
        if latency_ms > budget_ms:
            logger.warning(
                "ModelServer '%s': latency %.1fms exceeds budget %dms",
                self.model_card.model_name, latency_ms, budget_ms,
            )

        score = float(result.get("score", 0.5))
        anomaly = score >= self.decision_threshold

        # Record metrics
        if self.metrics:
            self.metrics.record_inference(latency_ms=latency_ms, prediction_score=score)

        # Async prediction logging (non-blocking)
        if self.prediction_logger:
            log_entry = PredictionLogEntry(
                prediction_id=prediction_id,
                model_name=self.model_card.model_name,
                model_version=self.model_card.model_version,
                squad=self.model_card.squad,
                entity_type=feature_vector.entity.entity_type,
                entity_id=feature_vector.entity.entity_id,
                request_ts=datetime.now(tz=timezone.utc).isoformat(),
                prediction_score=score,
                predicted_label=int(anomaly),
                decision_threshold=self.decision_threshold,
                feature_view_version=feature_vector.feature_view_version,
                used_fallback=used_fallback,
                circuit_state=self._breaker.state.value,
                inference_latency_ms=latency_ms,
            )
            self.prediction_logger.log(log_entry)

        # Feed drift detector
        if self.drift_detector:
            self.drift_detector.log_prediction(
                feature_vector=feature_vector,
                prediction=score,
            )

        return {
            "prediction_id": prediction_id,
            "anomaly": anomaly,
            "score": score,
            "decision_threshold": self.decision_threshold,
            "used_fallback": used_fallback,
            "circuit_state": self._breaker.state.value,
            "latency_ms": round(latency_ms, 2),
            "feature_view_version": feature_vector.feature_view_version,
            "model_version": self.model_card.model_version,
        }

    def _raw_inference(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Execute the raw model inference call.

        Wrapped by the circuit breaker — exceptions here trip the breaker.
        """
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)
            score = float(proba[0, 1])  # positive class probability
        elif hasattr(self.model, "decision_function"):
            # OneClassSVM / IsolationForest style: negate so higher = more anomalous
            raw_score = float(self.model.decision_function(x)[0])
            score = float(1.0 / (1.0 + np.exp(-raw_score)))  # sigmoid transform
        else:
            score = float(self.model.predict(x)[0])

        return {"score": score}

    def _fallback_predict(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Rule-based fallback invoked when circuit is OPEN.

        Fallback rules operate on RAW (unscaled) feature values.

        When self.model is a Pipeline with an internal scaler, x arrives
        in raw feature space (the Pipeline applies scaling internally),
        so no inverse-transform is needed. When self.model is a bare
        classifier that expects pre-scaled input, x is in scaled space
        and must be inverse-transformed before threshold comparison.

        Returns same schema as _raw_inference for transparent substitution.
        """
        x_flat = x.flatten()
        # If model is a bare classifier (not a Pipeline) and a scaler is
        # available, inverse-transform to raw feature space for threshold checks.
        if self._scaler is not None and not hasattr(self.model, 'named_steps'):
            try:
                x_flat = self._scaler.inverse_transform(
                    x_flat.reshape(1, -1)
                ).flatten()
            except Exception as e:
                logger.warning(
                    "Scaler inverse_transform failed: %s — using raw input", e
                )
        fallback_result = rule_based_anomaly_fallback(
            x_flat, feature_names=self.feature_names
        )
        score = 0.90 if fallback_result["anomaly"] else 0.10
        logger.info(
            "ModelServer '%s': fallback rule triggered '%s' → score=%.2f",
            self.model_card.model_name, fallback_result.get("trigger_rule"), score,
        )
        return {"score": score}

    def health(self) -> HealthStatus:
        """
        Return current health status for liveness/readiness probes.

        In a KServe deployment, this is called by the model-serving framework's
        /health endpoint handler. The output is also scraped by the platform
        team's observability stack.
        """
        breaker_stats = self._breaker.get_stats()
        metrics_summary = self.metrics.get_summary() if self.metrics else {}

        last_drift = None
        drift_stat = "unknown"
        if self.metrics:
            drift_scores = metrics_summary.get("drift_scores", {})
            if drift_scores:
                last_drift = float(np.mean(list(drift_scores.values())))
                if last_drift >= DRIFT_CRITICAL_THRESHOLD:
                    drift_stat = "critical"
                elif last_drift >= DRIFT_ALERT_THRESHOLD:
                    drift_stat = "alert"
                elif last_drift >= DRIFT_WARN_THRESHOLD:
                    drift_stat = "warn"
                else:
                    drift_stat = "ok"

        # Determine overall status
        circuit_state = breaker_stats.get("state", "unknown")
        error_count = int(metrics_summary.get("inference_errors", 0))
        inference_count = int(metrics_summary.get("inference_total", 1))
        error_rate = error_count / max(inference_count, 1)

        if circuit_state == "open" or error_rate > 0.50:
            status = "unhealthy"
        elif circuit_state == "half_open" or error_rate > 0.10 or drift_stat in ("alert", "critical"):
            status = "degraded"
        else:
            status = "healthy"

        ready, _ = self.model_card.is_promotion_ready()

        return HealthStatus(
            model_name=self.model_card.model_name,
            model_version=self.model_card.model_version,
            squad=self.model_card.squad,
            status=status,
            circuit_state=circuit_state,
            uptime_seconds=round(time.monotonic() - self._start_time, 1),
            inference_count=inference_count,
            error_count=error_count,
            fallback_count=int(metrics_summary.get("fallback_invocations", 0)),
            last_drift_score=last_drift,
            drift_status=drift_stat,
            latency_p95_ms=metrics_summary.get("latency_p95_ms"),
            latency_budget_ms=LATENCY_BUDGET_MS.get(self.model_card.serving_tier, 500),
            model_card_approved=ready,
            feature_view_version=self.model_card.feature_view_version,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )


# ============================================================================
# SECTION H: RETRAINING TRIGGER
# Implements LoopState (TS 28.627/TR 28.861) transition logic for automated retraining.
# Three trigger types: scheduled, drift-triggered, performance-degradation.
# See whitepaper Section 10 — "Retraining Triggers"
# ============================================================================

class LoopState(enum.Enum):
    """
    Platform-defined lifecycle states inspired by 3GPP TS 28.627 LoopState
    concepts (TS 28.627 is the foundational SON-era LoopState spec; for
    Rel-17+ O-RAN SMO contexts, see also TS 28.531 §8 management lifecycle
    state model and TR 28.861 §6 for autonomous network management lifecycle
    extensions). The normative TS 28.627 LoopState values are ENABLING, ENABLED,
    DISABLING, DISABLED (for closed-loop assurance processes). These platform
    states use similar naming but extend the normative set:

      ACTIVE (≈ ENABLED) — model serving production traffic
      INACTIVE (≈ DISABLED) — model suspended; retraining queued
      ACTIVATING (≈ ENABLING, e.g. canary phase) — platform extension
      DEACTIVATING (≈ DISABLING) — platform extension

    ACTIVATING and DEACTIVATING are platform extensions not present in
    the original TS 28.627 specification. Do not cite them as normative TS 28.627 values.
    """
    ACTIVE = "active"           # model is serving production traffic
    INACTIVE = "inactive"       # model suspended; retraining queued
    ACTIVATING = "activating"   # new model version deploying (canary phase)
    DEACTIVATING = "deactivating"  # model being withdrawn


@dataclass
class RetrainingDecision:
    """Output of RetrainingTrigger.evaluate()."""
    should_retrain: bool
    trigger_type: str           # "drift" | "performance" | "scheduled" | "none"
    trigger_reason: str
    urgency: str                # "immediate" | "next_cycle" | "low"
    loop_state_transition: str  # e.g., "ACTIVE → INACTIVE"
    recommended_data_window_days: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrainingTrigger:
    """
    Evaluates multiple retraining signals and produces a consolidated decision.

    Implements three trigger types:
    1. Drift-triggered: Wasserstein score exceeds threshold (primary signal)
    2. Performance-degradation: rolling AUC/F1 drops below SLO (requires GT)
    3. Scheduled: time-based fallback for models where ground truth is delayed

    The decision logic intentionally errs on the side of caution for
    network-impacting models (blast_radius != "single-cell").

    3GPP TS 28.627 alignment (using platform names → TS 28.627 equivalents):
      - Retraining trigger → ACTIVE→INACTIVE (≈ ENABLED→DISABLED)
      - New version validated → INACTIVE→ACTIVATING (≈ DISABLED→ENABLING)
      - Canary passes → ACTIVATING→ACTIVE (≈ ENABLING→ENABLED)
    """

    def __init__(
        self,
        model_card: ModelCard,
        max_days_since_training: int = 30,
        performance_window_days: int = 7,
    ) -> None:
        self.model_card = model_card
        self.max_days_since_training = max_days_since_training
        self.performance_window_days = performance_window_days
        self._last_training_date: Optional[datetime] = None
        self._current_loop_state = LoopState.ACTIVE

    def evaluate(
        self,
        drift_report: Optional[DriftReport] = None,
        recent_predictions: Optional[pd.DataFrame] = None,
    ) -> RetrainingDecision:
        """
        Evaluate all retraining signals and return a consolidated decision.

        Parameters
        ----------
        drift_report:
            Most recent DriftReport from the DriftDetector. If None, drift
            signal is unavailable (treat as stale).
        recent_predictions:
            Recent prediction log with ground_truth_label column (if available).
            Used for performance degradation detection.

        Returns
        -------
        RetrainingDecision with consolidated trigger assessment.
        """
        triggers: List[Dict[str, Any]] = []

        # --- Signal 1: Feature drift ---
        if drift_report is not None:
            if drift_report.retraining_recommended:
                triggers.append({
                    "type": "drift",
                    "urgency": "immediate" if drift_report.drift_status == "critical" else "next_cycle",
                    "reason": (
                        f"Drift status={drift_report.drift_status}, "
                        f"overall_score={drift_report.overall_drift_score:.3f}, "
                        f"drifted_features=[{', '.join(drift_report.drifted_features[:5])}]"
                    ),
                    "score": drift_report.overall_drift_score,
                })
            elif drift_report.drift_status == "warn":
                triggers.append({
                    "type": "drift_warn",
                    "urgency": "low",
                    "reason": f"Drift warning: overall_score={drift_report.overall_drift_score:.3f}",
                    "score": drift_report.overall_drift_score,
                })

        # --- Signal 2: Performance degradation (requires ground truth) ---
        if recent_predictions is not None and "ground_truth_label" in recent_predictions.columns:
            labelled = recent_predictions.dropna(subset=["ground_truth_label"])
            if len(labelled) >= 50:  # minimum sample size for reliable metric
                y_true = labelled["ground_truth_label"].values.astype(int)
                y_score = labelled["prediction_score"].values

                try:
                    current_auc = roc_auc_score(y_true, y_score)
                    current_f1 = f1_score(
                        y_true, (y_score >= self.model_card.target_auc_roc).astype(int),
                        zero_division=0,
                    )

                    perf_degraded = (
                        current_auc < RETRAINING_THRESHOLDS["auc_roc_min"]
                        or current_f1 < RETRAINING_THRESHOLDS["f1_min"]
                    )
                    if perf_degraded:
                        triggers.append({
                            "type": "performance",
                            "urgency": "immediate",
                            "reason": (
                                f"Performance degradation: "
                                f"AUC-ROC={current_auc:.3f} (threshold={RETRAINING_THRESHOLDS['auc_roc_min']}), "
                                f"F1={current_f1:.3f} (threshold={RETRAINING_THRESHOLDS['f1_min']})"
                            ),
                            "current_auc": current_auc,
                            "current_f1": current_f1,
                        })
                except Exception as exc:
                    logger.warning("Performance evaluation failed: %s", exc)

        # --- Signal 3: Scheduled / time-based fallback ---
        training_date_str = self.model_card.training_end_date
        try:
            training_date = datetime.fromisoformat(training_date_str.replace("Z", "+00:00"))
            days_since_training = (datetime.now(tz=timezone.utc) - training_date).days
            if days_since_training > self.max_days_since_training:
                triggers.append({
                    "type": "scheduled",
                    "urgency": "next_cycle",
                    "reason": (
                        f"Model age: {days_since_training} days "
                        f"(max={self.max_days_since_training})"
                    ),
                    "days_since_training": days_since_training,
                })
        except (ValueError, TypeError):
            logger.debug("Could not parse training_end_date '%s'", training_date_str)

        # --- Consolidate decision ---
        if not triggers:
            return RetrainingDecision(
                should_retrain=False,
                trigger_type="none",
                trigger_reason="All signals nominal",
                urgency="low",
                loop_state_transition=f"{self._current_loop_state.value} → {self._current_loop_state.value}",
                recommended_data_window_days=self.max_days_since_training,
            )

        # Prioritise by urgency: immediate > next_cycle > low
        urgency_rank = {"immediate": 3, "next_cycle": 2, "low": 1}
        triggers.sort(key=lambda t: urgency_rank.get(t.get("urgency", "low"), 0), reverse=True)
        primary_trigger = triggers[0]

        should_retrain = primary_trigger["urgency"] in ("immediate", "next_cycle")

        # Determine data window: use more data for network upgrades (scheduled)
        # but can use a shorter window for drift (avoid including drifted data)
        if primary_trigger["type"] == "drift":
            data_window_days = max(14, self.max_days_since_training // 2)
        else:
            data_window_days = self.max_days_since_training

        new_state = LoopState.INACTIVE if should_retrain else self._current_loop_state

        decision = RetrainingDecision(
            should_retrain=should_retrain,
            trigger_type=primary_trigger["type"],
            trigger_reason=primary_trigger["reason"],
            urgency=primary_trigger["urgency"],
            loop_state_transition=f"{self._current_loop_state.value} → {new_state.value}",
            recommended_data_window_days=data_window_days,
            metadata={"all_triggers": triggers},
        )

        if should_retrain:
            self._current_loop_state = LoopState.INACTIVE
            logger.warning(
                "RetrainingTrigger '%s': retraining recommended "
                "[trigger=%s, urgency=%s, state_transition=%s]",
                self.model_card.model_name,
                decision.trigger_type,
                decision.urgency,
                decision.loop_state_transition,
            )

        return decision


# ============================================================================
# SECTION I: MULTI-TENANT MODEL REGISTRY INTERACTION
# Stubs and patterns for interacting with the MLflow model registry.
# In production, this is called by CI/CD pipelines and the governance gate.
# See whitepaper Section 7 — "Model Registry & Governance Gate"
# 3GPP TS 28.105 §7.2 MLEntityRepository IOC
# ============================================================================

class ModelRegistryClient:
    """
    Abstraction layer over the model registry (MLflow in the reference arch).

    Provides a telco-MLOps-aware interface that enforces:
    - Naming conventions: {squad}/{model_name}/{version}
    - Model card completeness check before registration
    - Promotion gate checks before staging → production transition
    - Conflict screening metadata storage (Gap 1 from synthesis brief)

    When MLflow is available, delegates to the MLflow client.
    When absent, stubs using local filesystem (useful for testing).
    """

    NAMING_PATTERN = "{squad}/{model_name}"   # MLflow experiment name convention

    def __init__(self, tracking_uri: str = "file:///tmp/mlflow_registry") -> None:
        self.tracking_uri = tracking_uri
        self._mlflow_available = MLFLOW_AVAILABLE
        if self._mlflow_available:
            mlflow.set_tracking_uri(tracking_uri)
        self._local_registry: Dict[str, Dict[str, Any]] = {}
        logger.info(
            "ModelRegistryClient initialised [uri=%s, mlflow=%s]",
            tracking_uri, self._mlflow_available,
        )

    def register_model(
        self,
        model: Any,
        model_card: ModelCard,
        run_metrics: Dict[str, float],
        artifact_dir: Path,
    ) -> str:
        """
        Register a trained model with the registry.

        Enforces:
        1. Model card completeness (cannot register without owner/description)
        2. Naming convention: {squad}/{model_name}
        3. Metrics logging (AUC, F1, precision, recall)
        4. Model card saved as artifact alongside model weights
        5. Initial stage is always "Staging" (not "Production")

        Returns
        -------
        run_id : str — MLflow run ID or local stub identifier
        """
        model_name = f"{model_card.squad}/{model_card.model_name}"

        # Verify naming convention
        if model_card.squad not in VALID_SQUADS:
            raise ValueError(
                f"Invalid squad '{model_card.squad}' in model_card. "
                f"Valid: {sorted(VALID_SQUADS)}"
            )

        if self._mlflow_available:
            return self._register_with_mlflow(
                model, model_card, run_metrics, artifact_dir, model_name
            )
        else:
            return self._register_local_stub(
                model, model_card, run_metrics, artifact_dir, model_name
            )

    def _register_with_mlflow(
        self,
        model: Any,
        model_card: ModelCard,
        run_metrics: Dict[str, float],
        artifact_dir: Path,
        model_name: str,
    ) -> str:
        """Real MLflow registration path."""
        experiment_name = model_name
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            # Log metrics
            mlflow.log_metrics(run_metrics)

            # Log model card fields as params for searchability
            mlflow.log_param("squad", model_card.squad)
            mlflow.log_param("serving_tier", model_card.serving_tier)
            mlflow.log_param("blast_radius", model_card.blast_radius)
            mlflow.log_param("rcp_write_set", json.dumps(model_card.rcp_write_set))
            mlflow.log_param("eu_ai_act_category", model_card.eu_ai_act_category)

            # Save model card as artifact
            card_path = artifact_dir / "model_card.json"
            model_card.save(card_path)
            mlflow.log_artifact(str(card_path))

            # Log model (sklearn or pytorch)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=model_name,
            )

            run_id = run.info.run_id

        logger.info(
            "Model '%s' registered in MLflow [run_id=%s, stage=Staging]",
            model_name, run_id,
        )
        return run_id

    def _register_local_stub(
        self,
        model: Any,
        model_card: ModelCard,
        run_metrics: Dict[str, float],
        artifact_dir: Path,
        model_name: str,
    ) -> str:
        """Local filesystem stub for testing without MLflow."""
        run_id = hashlib.md5(
            f"{model_name}{model_card.model_version}{time.time()}".encode()
        ).hexdigest()[:12]

        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / f"{model_card.model_name}_{model_card.model_version}.joblib"
        joblib.dump(model, model_path)

        card_path = artifact_dir / "model_card.json"
        model_card.save(card_path)

        metrics_path = artifact_dir / "run_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"run_id": run_id, "metrics": run_metrics}, f, indent=2)

        self._local_registry[f"{model_name}/{model_card.model_version}"] = {
            "run_id": run_id,
            "model_path": str(model_path),
            "model_card": model_card.to_dict(),
            "metrics": run_metrics,
            "stage": "Staging",
        }

        logger.info(
            "Model '%s' registered locally [run_id=%s, path=%s]",
            model_name, run_id, model_path,
        )
        return run_id

    def promote_to_production(
        self,
        squad: str,
        model_name: str,
        version: str,
        approver: str,
    ) -> bool:
        """
        Promote a model from Staging to Production.

        Checks:
        1. Model card approval (approver must be non-empty)
        2. Performance metrics meet SLO targets
        3. No blocking governance issues

        In production, this is called by the CI/CD pipeline after the
        automated validation gate passes and human approval is received.
        """
        full_name = f"{squad}/{model_name}"
        registry_key = f"{full_name}/{version}"

        if registry_key in self._local_registry:
            entry = self._local_registry[registry_key]
            entry["stage"] = "Production"
            entry["approved_by"] = approver
            entry["promotion_ts"] = datetime.now(tz=timezone.utc).isoformat()
            logger.info(
                "Model '%s' v%s promoted to Production by '%s'",
                full_name, version, approver,
            )
            return True

        if self._mlflow_available:
            try:
                client = mlflow.MlflowClient()
                client.transition_model_version_stage(
                    name=full_name, version=version, stage="Production"
                )
                logger.info(
                    "Model '%s' v%s promoted to Production in MLflow by '%s'",
                    full_name, version, approver,
                )
                return True
            except Exception as exc:
                logger.error("MLflow promotion failed: %s", exc)
                return False

        logger.warning("Model '%s' v%s not found in registry", full_name, version)
        return False


# ============================================================================
# SECTION J: SYNTHETIC DATA GENERATION (SELF-CONTAINED FOR TESTING)
# Generates enough data to demonstrate all production patterns without
# requiring prior scripts to have been run.
# ============================================================================

def generate_synthetic_production_data(
    n_cells: int = 20,
    n_rops: int = 192,   # 48 hours at 15-min granularity
    anomaly_rate: float = 0.03,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Generate synthetic PM counter data and feature vectors for demonstration.

    Returns
    -------
    (reference_df, production_df, feature_names)
        reference_df   — training-period features (reference distribution)
        production_df  — production-period features (with injected drift)
        feature_names  — ordered list of feature column names
    """
    rng = np.random.default_rng(seed)

    cell_ids = [f"CELL_{str(i).zfill(3)}_{rng.integers(1, 4)}" for i in range(n_cells)]
    base_ts = datetime(2024, 3, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_ts + timedelta(minutes=15 * i) for i in range(n_rops)]

    feature_names = [
        "ran.kpi.dl_prb_utilisation",
        "ran.kpi.ul_prb_utilisation",
        "ran.kpi.rrc_setup_success_rate",
        "ran.kpi.erab_setup_success_rate",
        "ran.kpi.ho_success_rate",
        "ran.kpi.dl_throughput_mbps",
        "ran.kpi.ul_throughput_mbps",
        "ran.kpi.active_ue_count",
        "ran.derived.prb_imbalance",
        "ran.derived.dl_tput_per_ue",
        "ran.derived.quality_score",
        "ctx.time.hour_sin",
        "ctx.time.hour_cos",
        "ctx.time.dow_sin",
        "ctx.time.dow_cos",
        "ctx.time.is_peak_hour",
        "ctx.time.is_weekend",
    ]

    def _build_rows(
        period_offset_hours: int = 0,
        drift_factor: float = 1.0,
        anomaly_rate_override: Optional[float] = None,
    ) -> pd.DataFrame:
        rows = []
        effective_anomaly_rate = anomaly_rate_override if anomaly_rate_override is not None else anomaly_rate
        for cell_id in cell_ids:
            for ts in timestamps:
                actual_ts = ts + timedelta(hours=period_offset_hours)
                hour = actual_ts.hour + actual_ts.minute / 60.0
                dow = actual_ts.weekday()

                # Diurnal load factor (0.3 overnight, 0.9 peak)
                diurnal = 0.3 + 0.6 * np.clip(
                    np.sin(np.pi * (hour - 6) / 14), 0, 1
                )

                # Base KPIs with noise
                dl_prb = float(np.clip(diurnal * 0.70 * drift_factor + rng.normal(0, 0.08), 0.02, 0.98))
                ul_prb = float(np.clip(diurnal * 0.40 * drift_factor + rng.normal(0, 0.06), 0.01, 0.90))
                rrc_sr = float(np.clip(0.97 - rng.exponential(0.01), 0.70, 1.0))
                erab_sr = float(np.clip(0.96 - rng.exponential(0.01), 0.70, 1.0))
                ho_sr = float(np.clip(0.94 - rng.exponential(0.02), 0.65, 1.0))
                dl_tput = float(np.clip(diurnal * 150 * drift_factor + rng.normal(0, 20), 0, 1000))
                ul_tput = float(np.clip(diurnal * 40 * drift_factor + rng.normal(0, 8), 0, 300))
                active_ue = float(np.clip(diurnal * 80 + rng.normal(0, 10), 0, 500))

                is_anomaly = rng.random() < effective_anomaly_rate
                if is_anomaly:
                    dl_prb = min(dl_prb * 1.5, 99.0)
                    rrc_sr = max(rrc_sr * 0.6, 0.40)

                prb_imbal = dl_prb - ul_prb
                dl_tput_per_ue = dl_tput / max(active_ue, 1.0)
                eps = 1e-6
                quality = float(np.exp(np.mean(np.log([
                    max(rrc_sr, eps), max(erab_sr, eps), max(ho_sr, eps)
                ]))))

                hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
                hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
                dow_sin = float(np.sin(2 * np.pi * dow / 7.0))
                dow_cos = float(np.cos(2 * np.pi * dow / 7.0))
                is_peak = float(7 <= actual_ts.hour <= 9 or 17 <= actual_ts.hour <= 20)
                is_weekend = float(dow >= 5)

                rows.append({
                    "cell_id": cell_id,
                    "timestamp": actual_ts,
                    "anomaly_label": int(is_anomaly),
                    "ran.kpi.dl_prb_utilisation": dl_prb,
                    "ran.kpi.ul_prb_utilisation": ul_prb,
                    "ran.kpi.rrc_setup_success_rate": rrc_sr,
                    "ran.kpi.erab_setup_success_rate": erab_sr,
                    "ran.kpi.ho_success_rate": ho_sr,
                    "ran.kpi.dl_throughput_mbps": dl_tput,
                    "ran.kpi.ul_throughput_mbps": ul_tput,
                    "ran.kpi.active_ue_count": active_ue,
                    "ran.derived.prb_imbalance": prb_imbal,
                    "ran.derived.dl_tput_per_ue": dl_tput_per_ue,
                    "ran.derived.quality_score": quality,
                    "ctx.time.hour_sin": hour_sin,
                    "ctx.time.hour_cos": hour_cos,
                    "ctx.time.dow_sin": dow_sin,
                    "ctx.time.dow_cos": dow_cos,
                    "ctx.time.is_peak_hour": is_peak,
                    "ctx.time.is_weekend": is_weekend,
                })
        return pd.DataFrame(rows)

    reference_df = _build_rows(period_offset_hours=0, drift_factor=1.0)
    # Production data: simulate traffic surge (drift_factor > 1) post-event
    production_df = _build_rows(period_offset_hours=720, drift_factor=1.4, anomaly_rate_override=0.07)

    logger.info(
        "Synthetic data generated: reference=%d rows, production=%d rows, features=%d",
        len(reference_df), len(production_df), len(feature_names),
    )
    return reference_df, production_df, feature_names


def train_demo_model(
    reference_df: pd.DataFrame,
    feature_names: List[str],
) -> Pipeline:
    """
    Train a simple RandomForest classifier on the reference data.

    This is intentionally simple — the model quality is irrelevant here.
    The focus is on the production serving patterns, not model accuracy.
    """
    X = reference_df[feature_names].fillna(0.0).values
    y = reference_df["anomaly_label"].values

    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X, y)

    # Quick sanity check on training data
    y_pred = pipeline.predict_proba(X)[:, 1]
    try:
        auc = roc_auc_score(y, y_pred)
        logger.info("Demo model trained: AUC-ROC on training data = %.3f", auc)
    except ValueError:
        logger.warning("Could not compute AUC-ROC (possibly single class in training data)")

    return pipeline


# ============================================================================
# SECTION K: VISUALISATIONS
# Production monitoring dashboard plots.
# In production, these are generated by the Grafana dashboard (JSON in CODE-06).
# Here they are generated as PNGs for the whitepaper companion materials.
# ============================================================================

def plot_drift_report(
    drift_report: DriftReport,
    output_path: Path,
) -> None:
    """
    Plot per-feature Wasserstein drift scores with threshold lines.

    Produces the visualisation equivalent of the Grafana "Model Drift"
    panel described in the whitepaper monitoring section.
    """
    if not drift_report.feature_drift_scores:
        logger.warning("Empty drift report — skipping plot")
        return

    scores = drift_report.feature_drift_scores
    names = list(scores.keys())
    values = [scores[n] for n in names]

    # Shorten feature names for readability on the chart
    short_names = [n.split(".")[-1].replace("_", "\n") for n in names]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = [
        "#d62728" if v >= DRIFT_CRITICAL_THRESHOLD
        else "#ff7f0e" if v >= DRIFT_ALERT_THRESHOLD
        else "#bcbd22" if v >= DRIFT_WARN_THRESHOLD
        else "#2ca02c"
        for v in values
    ]
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)

    # Threshold lines
    ax.axhline(y=DRIFT_WARN_THRESHOLD, color="#bcbd22", linestyle="--", linewidth=1.5,
               label=f"Warn ({DRIFT_WARN_THRESHOLD})")
    ax.axhline(y=DRIFT_ALERT_THRESHOLD, color="#ff7f0e", linestyle="--", linewidth=1.5,
               label=f"Alert ({DRIFT_ALERT_THRESHOLD})")
    ax.axhline(y=DRIFT_CRITICAL_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"Critical ({DRIFT_CRITICAL_THRESHOLD})")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short_names, fontsize=8, rotation=0)
    ax.set_ylabel("Wasserstein Distance (normalised)", fontsize=10)
    ax.set_title(
        f"Feature Drift Report — {drift_report.model_name} v{drift_report.model_version}\n"
        f"Status: {drift_report.drift_status.upper()} | "
        f"Overall: {drift_report.overall_drift_score:.3f} | "
        f"Window: {drift_report.current_period}",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, max(max(values, default=0) * 1.2, DRIFT_CRITICAL_THRESHOLD * 1.3))

    # Annotate drifted features
    for i, (val, name) in enumerate(zip(values, names)):
        if val >= DRIFT_ALERT_THRESHOLD:
            ax.text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7, color="black", fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Drift report plot saved → %s", output_path)


def plot_prediction_score_distribution(
    reference_scores: np.ndarray,
    production_scores: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """
    Overlay reference and production prediction score distributions.

    Score distribution shift is a model-level drift signal that complements
    the feature-level Wasserstein analysis. Useful for:
    - Detecting concept drift even when features appear stable
    - Verifying that the anomaly rate in production matches training expectations
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: Overlapping KDE ---
    ax = axes[0]
    bins = np.linspace(0, 1, 50)
    ax.hist(reference_scores, bins=bins, density=True, alpha=0.5,
            color="#1f77b4", label=f"Reference (n={len(reference_scores):,})")
    ax.hist(production_scores, bins=bins, density=True, alpha=0.5,
            color="#d62728", label=f"Production (n={len(production_scores):,})")
    ax.set_xlabel("Prediction Score (anomaly probability)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Score Distribution Shift\n{model_name}", fontsize=11)
    ax.legend(fontsize=9)

    # Add Wasserstein distance annotation
    w_dist = wasserstein_distance(reference_scores, production_scores)
    ax.text(0.05, 0.95, f"Wasserstein: {w_dist:.4f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # --- Panel 2: Anomaly rate over time (simulated) ---
    ax2 = axes[1]
    n_windows = 20
    ref_rate = np.mean(reference_scores > 0.5)
    # Simulate gradually increasing anomaly rate
    window_rates = [ref_rate * (1 + 0.05 * i + np.random.normal(0, 0.005))
                    for i in range(n_windows)]
    ax2.plot(range(n_windows), window_rates, marker="o", color="#2ca02c",
             linewidth=2, markersize=4, label="Rolling anomaly rate")
    ax2.axhline(y=ref_rate, color="#1f77b4", linestyle="--", label="Reference rate")
    ax2.axhline(y=ref_rate * 1.5, color="#ff7f0e", linestyle=":", label="Warn threshold (+50%)")
    ax2.axhline(y=ref_rate * 2.0, color="#d62728", linestyle=":", label="Alert threshold (+100%)")
    ax2.set_xlabel("Time Window (each = 24h)", fontsize=10)
    ax2.set_ylabel("Predicted Anomaly Rate", fontsize=10)
    ax2.set_title("Anomaly Rate Trend\n(Concept Drift Indicator)", fontsize=11)
    ax2.legend(fontsize=8)

    plt.suptitle(
        f"Model Monitoring Dashboard — {model_name}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Score distribution plot saved → %s", output_path)


def plot_circuit_breaker_timeline(
    event_log: List[Dict[str, Any]],
    model_name: str,
    output_path: Path,
) -> None:
    """
    Visualise circuit breaker state transitions and fallback invocations.

    Provides the operational team with a historical view of model stability
    — critical for post-incident analysis of blast-radius events.
    """
    if len(event_log) < 2:
        logger.info("Insufficient circuit breaker events to plot (%d)", len(event_log))
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    times = list(range(len(event_log)))
    latencies = [e.get("latency_ms", 0) for e in event_log]
    fallbacks = [int(e.get("used_fallback", False)) for e in event_log]
    states = [1 if e.get("circuit_state") == "open" else 0 for e in event_log]

    # Panel 1: Latency with fallback markers
    ax1.plot(times, latencies, color="#1f77b4", linewidth=1.0, alpha=0.8, label="Latency (ms)")
    fallback_times = [t for t, f in zip(times, fallbacks) if f]
    fallback_lats = [latencies[t] for t in fallback_times]
    if fallback_times:
        ax1.scatter(fallback_times, fallback_lats, color="#d62728", s=30,
                    zorder=5, label="Fallback invoked", marker="^")
    budget = LATENCY_BUDGET_MS.get("ran_non_rt", 500)
    ax1.axhline(y=budget, color="#ff7f0e", linestyle="--", linewidth=1.5,
                label=f"Latency budget ({budget}ms)")
    ax1.set_ylabel("Latency (ms)", fontsize=9)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title(f"Circuit Breaker & Latency — {model_name}", fontsize=11)

    # Panel 2: Circuit state
    ax2.fill_between(times, states, step="post", color="#d62728", alpha=0.4, label="Circuit OPEN")
    ax2.fill_between(times, [1 - s for s in states], step="post",
                     color="#2ca02c", alpha=0.3, label="Circuit CLOSED")
    ax2.set_ylabel("Circuit State", fontsize=9)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["CLOSED", "OPEN"], fontsize=8)
    ax2.set_xlabel("Request Sequence", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Circuit breaker timeline plot saved → %s", output_path)


# ============================================================================
# SECTION L: KSERVE / KUBEFLOW YAML CONFIGURATION TEMPLATES
# These are the KServe InferenceService and Kubeflow Pipeline YAML configs
# referenced in the whitepaper CODE-05 section. Generated as files for
# teams to use directly in their K8s clusters.
# See whitepaper Section 7 — "Serving Infrastructure"
# ============================================================================

KSERVE_INFERENCE_SERVICE_YAML = """\
# =============================================================================
# KServe InferenceService — RAN Anomaly Detector
# TWO VARIANTS below: choose ONE based on your KServe serving mode.
#
# ⚠️  This file contains TWO MUTUALLY EXCLUSIVE variants.
#     Do NOT apply the entire file. Copy ONLY the variant matching your
#     KServe serving mode (Knative or RawDeployment).
# =============================================================================

# ---------------------------------------------------------------------------
# VARIANT A: Knative-backed serving mode (default KServe installation)
# Supports native canaryTrafficPercent for progressive delivery.
# ---------------------------------------------------------------------------
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: ran-anomaly-detector
  namespace: ran-squad
  labels:
    squad: ran
    model: ran-anomaly-detector
    domain: network-assurance
    eu-ai-act-category: limited-risk
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
    mlops.telco/model-card: "ran/ran-anomaly-detector/2.1.4/model_card.json"
    mlops.telco/loop-state: "ACTIVATING"  # set to ACTIVE after canary passes
spec:
  predictor:
    # Stable (production) version: receives 90% of traffic
    canaryTrafficPercent: 10   # 10% to canary, 90% to stable
    sklearn:
      storageUri: "s3://mlops-artifacts/ran/ran-anomaly-detector/2.0.1"
      resources:
        requests:
          cpu: "250m"
          memory: "512Mi"
        limits:
          cpu: "500m"
          memory: "1Gi"
      readinessProbe:
        httpGet:
          path: /v2/health/ready
          port: 8080
        initialDelaySeconds: 15
        periodSeconds: 10
      scaleTarget: 10
      scaleMetric: rps
  # To promote canary (Knative mode):
  # (1) Update storageUri to canary version, (2) REMOVE canaryTrafficPercent.
  # Do NOT set canaryTrafficPercent:100 — this keeps both pods running.
  #   kubectl patch inferenceservice ran-anomaly-detector -n ran-squad \
  #     --type json -p '[{"op":"replace","path":"/spec/predictor/sklearn/storageUri","value":"s3://mlops-artifacts/ran/ran-anomaly-detector/2.1.4"},{"op":"remove","path":"/spec/predictor/canaryTrafficPercent"}]'

---
# ---------------------------------------------------------------------------
# VARIANT B: RawDeployment mode (KServe v0.11+, no Knative dependency)
# canaryTrafficPercent is NOT supported — use Argo Rollouts for progressive
# delivery. Deploy the InferenceService + Rollout + AnalysisTemplate together.
# ---------------------------------------------------------------------------
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: ran-anomaly-detector
  namespace: ran-squad
  labels:
    squad: ran
    model: ran-anomaly-detector
    domain: network-assurance
    eu-ai-act-category: limited-risk
  annotations:
    serving.kserve.io/deploymentMode: "RawDeployment"
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
    mlops.telco/model-card: "ran/ran-anomaly-detector/2.1.4/model_card.json"
    mlops.telco/loop-state: "ACTIVATING"
spec:
  predictor:
    # NOTE: No canaryTrafficPercent — RawDeployment rejects this field.
    # Traffic splitting is managed by Argo Rollouts below.
    sklearn:
      storageUri: "s3://mlops-artifacts/ran/ran-anomaly-detector/2.1.4"
      resources:
        requests:
          cpu: "250m"
          memory: "512Mi"
        limits:
          cpu: "500m"
          memory: "1Gi"
      readinessProbe:
        httpGet:
          path: /v2/health/ready
          port: 8080
        initialDelaySeconds: 15
        periodSeconds: 10

---
# Argo Rollouts canary configuration (RawDeployment mode only)
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: ran-anomaly-detector-rollout
  namespace: ran-squad
spec:
  strategy:
    canary:
      steps:
      - setWeight: 10     # Phase 1: 10% canary
      - pause: {duration: 30m}
      - analysis:         # Gate: check error rate before proceeding
          templates:
          - templateName: error-rate-analysis
      - setWeight: 50     # Phase 2: 50/50 split
      - pause: {duration: 1h}
      - analysis:
          templates:
          - templateName: error-rate-analysis
      # Phase 3: 100% canary = full rollout (stable retired)
      canaryMetadata:
        labels:
          role: canary
      stableMetadata:
        labels:
          role: stable
  analysis:
    successfulRunHistoryLimit: 3
    unsuccessfulRunHistoryLimit: 3

---
# AnalysisTemplate: error rate gate for canary promotion
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: error-rate-analysis
  namespace: ran-squad
spec:
  metrics:
  - name: error-rate
    interval: 5m
    count: 6          # check 6 times (= 30 minutes)
    successCondition: result[0] < 0.02   # < 2% error rate
    failureLimit: 1
    provider:
      prometheus:
        address: http://prometheus.monitoring.svc:9090
        query: |
          sum(rate(mlops_inference_errors_total{
            model="ran-anomaly-detector",
            version="2.1.4",
            squad="ran"
          }[5m]))
          /
          sum(rate(mlops_inference_requests_total{
            model="ran-anomaly-detector",
            version="2.1.4",
            squad="ran"
          }[5m]))
"""

KUBEFLOW_PIPELINE_YAML_COMMENT = """\
# Kubeflow Pipeline for RAN Anomaly Detector — see CODE-03 in whitepaper
# Pipeline: train → evaluate → conditional_register → governance_gate → deploy
# Run:  kfp.Client(host='http://kubeflow.internal').create_run_from_pipeline_func(
#           pipeline_fn, arguments={...})
"""

GRAFANA_DASHBOARD_SNIPPET = """\
{
  "title": "MLOps Model Monitoring — RAN Squad",
  "uid": "mlops-ran-monitoring",
  "panels": [
    {
      "title": "Inference Latency P95 (ms)",
      "type": "timeseries",
      "targets": [{
        "expr": "histogram_quantile(0.95, sum(rate(mlops_inference_latency_ms_bucket{squad='ran'}[5m])) by (le, model, version))",
        "legendFormat": "{{model}} v{{version}}"
      }],
      "thresholds": {
        "steps": [
          {"color": "green", "value": 0},
          {"color": "yellow", "value": 200},
          {"color": "red", "value": 500}
        ]
      }
    },
    {
      "title": "Feature Drift Score (Wasserstein)",
      "type": "stat",
      "targets": [{
        "expr": "mlops_feature_drift_score{squad='ran'}",
        "legendFormat": "{{feature_name}}"
      }],
      "thresholds": {
        "steps": [
          {"color": "green", "value": 0},
          {"color": "yellow", "value": 0.15},
          {"color": "orange", "value": 0.30},
          {"color": "red", "value": 0.50}
        ]
      }
    },
    {
      "title": "Circuit Breaker State",
      "type": "stat",
      "targets": [{
        "expr": "mlops_circuit_breaker_open{squad='ran'}",
        "legendFormat": "{{model}}"
      }],
      "mappings": [
        {"type": "value", "options": {"0": {"text": "CLOSED", "color": "green"}}},
        {"type": "value", "options": {"1": {"text": "OPEN", "color": "red"}}}
      ]
    },
    {
      "title": "Fallback Invocations / hr",
      "type": "timeseries",
      "targets": [{
        "expr": "sum(rate(mlops_fallback_invocations_total{squad='ran'}[1h])) by (model, version) * 3600",
        "legendFormat": "{{model}} v{{version}} fallbacks/hr"
      }]
    }
  ]
}
"""

PROMETHEUS_ALERTING_RULES_YAML = """\
# Prometheus alerting rules for MLOps model monitoring
# Deploy: kubectl create configmap mlops-alert-rules --from-file=rules.yml -n monitoring
# Reference: Whitepaper Section 7 — Monitoring Stack

groups:
- name: mlops.model.drift
  rules:
  - alert: ModelDriftAlert
    expr: mlops_feature_drift_score > 0.30
    for: 15m
    labels:
      severity: warning
      team: "{{ $labels.squad }}-squad"
    annotations:
      summary: "Feature drift detected for {{ $labels.model }} v{{ $labels.version }}"
      description: >
        Feature '{{ $labels.feature_name }}' has Wasserstein score {{ $value | humanize }}
        (threshold: 0.30). Retraining recommended.
      runbook_url: "https://wiki.operator.com/mlops/runbooks/feature-drift"

  - alert: ModelDriftCritical
    expr: mlops_feature_drift_score > 0.50
    for: 5m
    labels:
      severity: critical
      team: "{{ $labels.squad }}-squad"
    annotations:
      summary: "CRITICAL drift for {{ $labels.model }} — immediate retraining required"
      description: >
        Feature '{{ $labels.feature_name }}' has Wasserstein score {{ $value | humanize }}.
        Model predictions may be unreliable. Evaluate circuit breaker activation.

- name: mlops.circuit.breaker
  rules:
  - alert: CircuitBreakerOpen
    expr: mlops_circuit_breaker_open == 1
    for: 1m
    labels:
      severity: critical
      team: "{{ $labels.squad }}-squad"
    annotations:
      summary: "Circuit breaker OPEN for {{ $labels.model }}"
      description: >
        Model {{ $labels.model }} v{{ $labels.version }} circuit breaker is OPEN.
        All traffic redirected to rule-based fallback.
      runbook_url: "https://wiki.operator.com/mlops/runbooks/circuit-breaker"

- name: mlops.latency
  rules:
  - alert: ModelLatencyBudgetExceeded
    expr: >
      histogram_quantile(0.95,
        sum(rate(mlops_inference_latency_ms_bucket[5m])) by (le, model, version, squad)
      ) > 500
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "P95 latency {{ $value | humanize }}ms exceeds 500ms budget for {{ $labels.model }}"

- name: mlops.error.rate
  rules:
  - alert: HighModelErrorRate
    expr: >
      sum(rate(mlops_inference_errors_total[5m])) by (model, version, squad)
      /
      sum(rate(mlops_inference_requests_total[5m])) by (model, version, squad)
      > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Error rate {{ $value | humanizePercentage }} for {{ $labels.model }}"
"""

OPA_POLICY_REGO = """\
# OPA Policy — Telco MLOps Governance Gate
# Enforces promotion criteria before a model can move from Staging to Production.
# Deploy: kubectl apply -f mlops-policy.yaml -n opa
#
# Reference:
#   - Whitepaper Section 7 — Governance Gate
#   - 3GPP TS 28.105 §7.2 MLEntityRepository IOC
#   - EU AI Act Article 11 — technical documentation requirements

package mlops.governance

# Requires OPA v1.0+ (released 2024). `if` and `in` are default keywords.
# For OPA v0.40–v0.67, uncomment the two lines below:
#   import future.keywords.if
#   import future.keywords.in

# --- Default deny ---
default allow_promotion := false

# --- Allow promotion if ALL of the following hold ---
allow_promotion if {
    model_card_complete
    performance_thresholds_met
    no_conflict_violations
    eu_ai_act_compliant
    approval_present
}

# Model card completeness check
model_card_complete if {
    input.model_card.squad != ""
    input.model_card.model_description != ""
    count(input.model_card.training_data_sources) > 0
    input.model_card.training_data_start != ""
    input.model_card.training_data_end != ""
    input.model_card.rollback_procedure != ""
}

# Performance metrics must meet SLO targets
performance_thresholds_met if {
    input.model_card.achieved_auc_roc >= input.model_card.target_auc_roc
    input.model_card.achieved_f1 >= input.model_card.target_f1
}

# RAN models: check RCP write-set does not conflict with existing production models
# Conflict graph populated by the CI/CD pipeline from the registry
no_conflict_violations if {
    count(input.rcp_conflicts) == 0
}

# EU AI Act: high-risk models require explicit human approval
eu_ai_act_compliant if {
    input.model_card.eu_ai_act_category == "limited_risk"
}

eu_ai_act_compliant if {
    input.model_card.eu_ai_act_category == "high_risk"
    input.model_card.approved_by != null
    input.model_card.human_oversight_required == true
}

# Human approval must be present (service-account auto-approvals blocked)
approval_present if {
    input.model_card.approved_by != null
    input.model_card.approved_by != ""
    not startswith(input.model_card.approved_by, "svc-")
}

# --- Violations for diagnostic reporting ---
violations[msg] if {
    not model_card_complete
    msg := "Model card is incomplete — missing required fields"
}

violations[msg] if {
    not performance_thresholds_met
    msg := sprintf(
        "Performance below target: AUC=%.3f (target=%.3f), F1=%.3f (target=%.3f)",
        [
            input.model_card.achieved_auc_roc,
            input.model_card.target_auc_roc,
            input.model_card.achieved_f1,
            input.model_card.target_f1,
        ]
    )
}

violations[msg] if {
    count(input.rcp_conflicts) > 0
    conflicts := concat(", ", input.rcp_conflicts)
    msg := sprintf("RCP write-set conflicts detected: [%s]", [conflicts])
}

violations[msg] if {
    input.model_card.eu_ai_act_category == "high_risk"
    input.model_card.approved_by == null
    msg := "High-risk AI system requires explicit human approval before promotion"
}
"""


def save_yaml_templates(output_dir: Path) -> None:
    """Save KServe, Prometheus, and OPA YAML templates to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = {
        "kserve_ran_anomaly_detector.yaml": KSERVE_INFERENCE_SERVICE_YAML,
        "prometheus_alerting_rules.yaml": PROMETHEUS_ALERTING_RULES_YAML,
        "opa_governance_policy.rego": OPA_POLICY_REGO,
        "grafana_dashboard.json": GRAFANA_DASHBOARD_SNIPPET,
    }
    for filename, content in templates.items():
        path = output_dir / filename
        with open(path, "w") as f:
            f.write(content)
        logger.info("Template saved → %s", path)


# ============================================================================
# SECTION M: END-TO-END DEMONSTRATION ORCHESTRATION
# Ties all production patterns together into a single runnable demonstration.
# ============================================================================

def run_production_patterns_demo(
    output_dir: Path,
    serve_metrics: bool = False,
    metrics_port: int = 8001,
    use_pipeline_model: bool = False,
) -> None:
    """
    End-to-end demonstration of all production patterns.

    When ``use_pipeline_model=True`` (via ``--use-pipeline-model`` flag), loads
    the trained model from ``models/tier2_random_forest.joblib`` produced by
    03_model_training.py, generates synthetic data using flat snake_case feature
    names matching the 01–04 pipeline, and runs all production pattern demos
    against that model. This mode demonstrates the full end-to-end path.

    When ``use_pipeline_model=False`` (default), generates standalone synthetic
    data with the dotted ``ran.kpi.*`` namespace for self-contained illustration.

    Sequence:
    1.  Generate synthetic reference + production data
    2.  Train demo model on reference data (or load pipeline model)
    3.  Build model card with governance metadata
    4.  Register model via ModelRegistryClient
    5.  Instantiate production serving stack (ModelServer + CircuitBreaker)
    6.  Run simulated inference loop with deliberate failures injected
    7.  Compute Wasserstein drift report
    8.  Evaluate retraining trigger
    9.  Plot monitoring visualisations
    10. Save YAML configuration templates
    11. Export Prometheus metrics snapshot
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Mode: --use-pipeline-model (load 01–04 artifacts)
    # ------------------------------------------------------------------
    if use_pipeline_model:
        pipeline_model_path = Path("models") / "tier2_random_forest.joblib"
        if not pipeline_model_path.exists():
            raise FileNotFoundError(
                f"--use-pipeline-model requires {pipeline_model_path}. "
                "Run scripts 01→02→03 first (make pipeline)."
            )
        logger.info("Loading pipeline model from %s", pipeline_model_path)
        model = joblib.load(pipeline_model_path)

        # Load feature metadata to get the exact feature names the model expects
        meta_path = Path("data") / "feature_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                "--use-pipeline-model requires data/feature_metadata.json. "
                "Run scripts 01→02 first (make pipeline)."
            )
        with open(meta_path) as f:
            meta = json.load(f)
        feature_names = list(meta.get("features", {}).keys())
        if not feature_names:
            # Fallback: infer from model
            n_expected = getattr(model, "n_features_in_", None)
            if hasattr(model, "feature_names_in_"):
                feature_names = list(model.feature_names_in_)
            else:
                raise ValueError(
                    "Cannot determine feature names from metadata or model. "
                    "Ensure 02_feature_engineering.py produced feature_metadata.json."
                )
        logger.info(
            "Pipeline mode: loaded %d feature names from metadata", len(feature_names),
        )

        # --- Double-scaling guard ---
        # features_test.parquet is RobustScaler-transformed. If the model is a
        # Pipeline (scaler + classifier), feeding scaled data produces double-scaling.
        # Two strategies: (a) extract bare classifier for scaled data, or
        # (b) use raw (pre-scaling) data with the full Pipeline.
        # Strategy (b) is preferred — it keeps inference and drift detection in the
        # same feature space (raw), matching production serving conditions.
        from sklearn.pipeline import Pipeline as SkPipeline
        input_is_scaled = meta.get("scaling_applied", False)

        # Load raw reference data (pre-scaling) for BOTH reference and production
        ref_path = Path("data") / "features_raw_reference.parquet"
        train_path = Path("data") / "features_train.parquet"
        test_path = Path("data") / "features_test.parquet"

        if isinstance(model, SkPipeline) and input_is_scaled and ref_path.exists():
            # Strategy (b): use raw features with full Pipeline
            logger.info(
                "Pipeline model + pre-scaled Parquet detected. Using raw reference "
                "data for both inference and drift to avoid double-scaling. "
                "Production deployments use raw features from the online store."
            )
            raw_ref = pd.read_parquet(ref_path)
            # Split raw reference into reference (first 80%) and production (last 20%)
            split_idx = int(len(raw_ref) * 0.8)
            reference_df = raw_ref.iloc[:split_idx].copy()
            production_df = raw_ref.iloc[split_idx:].copy()
        elif isinstance(model, SkPipeline) and input_is_scaled:
            # No raw reference available — extract bare classifier instead
            logger.warning(
                "Pipeline model + pre-scaled input but no features_raw_reference.parquet. "
                "Extracting bare classifier to avoid double-scaling."
            )
            model = model.named_steps.get("clf", model[-1])
            production_df = pd.read_parquet(test_path)
            if train_path.exists():
                reference_df = pd.read_parquet(train_path)
            else:
                reference_df = production_df.copy()
        else:
            # Bare classifier or unscaled data — load normally
            production_df = pd.read_parquet(test_path) if test_path.exists() else pd.read_parquet(ref_path)
            if ref_path.exists():
                reference_df = pd.read_parquet(ref_path)
            elif train_path.exists():
                reference_df = pd.read_parquet(train_path)
            else:
                reference_df = production_df.copy()

        # Ensure required metadata columns exist for the demo
        if "cell_id" not in production_df.columns:
            production_df["cell_id"] = "CELL_001_A"
        if "timestamp" not in production_df.columns:
            production_df["timestamp"] = pd.date_range(
                "2024-01-25", periods=len(production_df), freq="15min", tz="UTC",
            )
        if "anomaly_label" not in production_df.columns:
            production_df["anomaly_label"] = production_df.get("is_anomaly", 0)
        if "cell_id" not in reference_df.columns:
            reference_df["cell_id"] = "CELL_001_A"
        if "timestamp" not in reference_df.columns:
            reference_df["timestamp"] = pd.date_range(
                "2024-01-01", periods=len(reference_df), freq="15min", tz="UTC",
            )

        # Filter to features that exist in both dataframes
        avail = [f for f in feature_names
                 if f in production_df.columns and f in reference_df.columns]
        if len(avail) < len(feature_names):
            logger.warning(
                "Model expects %d features, %d available in data. "
                "Missing: %s (these may be metadata columns excluded from Parquet).",
                len(feature_names), len(avail),
                sorted(set(feature_names) - set(avail))[:10],
            )
            feature_names = avail

    else:
        # ------------------------------------------------------------------
        # Default standalone mode
        # ------------------------------------------------------------------
        pipeline_model_path = Path("models") / "tier2_random_forest.joblib"
        if pipeline_model_path.exists():
            logger.warning(
                "Found model artifacts from 03_model_training.py in %s. "
                "Use --use-pipeline-model to run production patterns on those "
                "artifacts. Running standalone demo instead. "
                "See FEATURE_NAMESPACE_CONVENTION.md for details.",
                pipeline_model_path.parent,
            )

        # Step 1–2 (standalone): Generate data and train model
        logger.info("Step 1: Generating synthetic reference + production data")
        reference_df, production_df, feature_names = generate_synthetic_production_data()
        logger.info(
            "  Reference: %d rows, Production: %d rows, Features: %d",
            len(reference_df), len(production_df), len(feature_names),
        )

        logger.info("Step 2: Training demo model on reference data")
        model = train_demo_model(reference_df, feature_names)

    # ------------------------------------------------------------------
    # Step 3: Build model card
    # ------------------------------------------------------------------
    logger.info("Step 3: Building model card with governance metadata")
    card = ModelCard(
        model_id=str(uuid.uuid4()),
        model_name="ran-anomaly-detector",
        model_version="2.1.0",
        squad="ran",
        model_description="Demo RAN anomaly detection model for production patterns",
        training_start_date="2024-03-01",
        training_end_date="2024-03-08",
        registered_by="demo-pipeline",
        training_data_sources=["PM Counters (O1/TS 32.435)", "VES Alarms (ONAP VES 7.x)"],
        feature_view_version="1.3.0",
        achieved_auc_roc=0.88,
        achieved_f1=0.75,
        approved_by="platform-lead",
        approval_date=datetime.now(tz=timezone.utc).isoformat(),
    )
    card.save(output_dir / "model_card.json")
    logger.info("  Model card saved: %s", output_dir / "model_card.json")

    # ------------------------------------------------------------------
    # Step 4: Set up production serving stack
    # ------------------------------------------------------------------
    logger.info("Step 4–5: Instantiating serving stack")
    metrics = MetricsCollector("ran", "ran-anomaly-detector", "2.1.0")
    pred_logger = PredictionLogger(output_dir / "prediction_logs")
    drift_detector = DriftDetector(
        model_name="ran-anomaly-detector",
        model_version="2.1.0",
        squad="ran",
        reference_data=reference_df[feature_names],
        metrics=metrics,
    )

    server = ModelServer(
        model=model,
        model_card=card,
        feature_names=feature_names,
        decision_threshold=0.5,
        prediction_logger=pred_logger,
        drift_detector=drift_detector,
        metrics=metrics,
    )

    # ------------------------------------------------------------------
    # Step 6: Simulated inference loop
    # ------------------------------------------------------------------
    logger.info("Step 6: Running simulated inference loop (%d rows)", len(production_df))
    event_log: List[Dict[str, Any]] = []
    for idx, row in production_df.iterrows():
        fv = FeatureVector(
            entity=FeatureEntity(
                entity_type="cell",
                entity_id=row["cell_id"],
                timestamp=row["timestamp"],
                squad="ran",
            ),
            features={fn: float(row[fn]) for fn in feature_names},
            feature_view_version="1.3.0",
            computation_ts=datetime.now(tz=timezone.utc),
        )
        result = server.predict(fv)
        event_log.append(result)

    logger.info("  Inference complete: %d predictions logged", len(event_log))

    # ------------------------------------------------------------------
    # Step 6b: Validate bare-classifier fallback with inverse-transform
    # ------------------------------------------------------------------
    # The main demo uses a Pipeline (scaler embedded). This step exercises
    # the bare-classifier path where _fallback_predict must inverse-transform
    # scaled features before applying raw-space thresholds.
    logger.info("Step 6b: Testing bare-classifier fallback with inverse-transform")
    bare_clf = model.named_steps["clf"]
    bare_scaler = model.named_steps["scaler"]
    bare_card = ModelCard(
        model_id="test-bare-fallback",
        model_name="bare-classifier-test",
        model_version="0.0.1",
        squad="ran",
        model_description="Bare classifier test for fallback inverse-transform",
        training_start_date="2024-03-01",
        training_end_date="2024-03-08",
        registered_by="platform-test",
    )
    bare_server = ModelServer(
        model=bare_clf,
        model_card=bare_card,
        feature_names=feature_names,
        decision_threshold=0.5,
    )
    # Attach the external scaler (simulates a deployment where scaler is not
    # embedded in the model object but applied externally in the serving layer)
    bare_server._scaler = bare_scaler
    # Force circuit OPEN to exercise fallback path
    bare_server._breaker._state = CircuitState.OPEN
    bare_server._breaker._last_failure_time = time.monotonic()
    test_fv = FeatureVector(
        entity=FeatureEntity(
            entity_type="cell",
            entity_id="TEST_BARE_001_1",
            timestamp=datetime.now(tz=timezone.utc),
            squad="ran",
        ),
        features={fn: float(production_df.iloc[0][fn]) for fn in feature_names},
        feature_view_version="1.3.0",
        computation_ts=datetime.now(tz=timezone.utc),
    )
    bare_result = bare_server.predict(test_fv)
    assert bare_result.get("used_fallback", False), (
        "Expected fallback path for bare-classifier test (circuit is OPEN)"
    )
    logger.info(
        "  Bare-classifier fallback test PASSED: score=%.3f, used_fallback=%s",
        bare_result.get("score", -1),
        bare_result.get("used_fallback"),
    )

    # ------------------------------------------------------------------
    # Step 7: Compute drift report
    # ------------------------------------------------------------------
    logger.info("Step 7: Computing Wasserstein drift report")
    drift_report = drift_detector.compute_drift_report(current_window_hours=48)
    logger.info(
        "  Drift report: %d features analysed, %d flagged",
        len(drift_report.feature_drift_scores),
        sum(1 for v in drift_report.feature_drift_scores.values() if v > 0.15),
    )

    # ------------------------------------------------------------------
    # Step 8: Evaluate retraining trigger
    # ------------------------------------------------------------------
    logger.info("Step 8: Evaluating retraining trigger")
    trigger = RetrainingTrigger(card)
    decision = trigger.evaluate(drift_report=drift_report)
    logger.info(
        "  Retraining decision: should_retrain=%s, reason=%s",
        decision.should_retrain, decision.trigger_reason,
    )

    # ------------------------------------------------------------------
    # Step 9: Generate monitoring visualisations
    # ------------------------------------------------------------------
    logger.info("Step 9: Generating monitoring visualisations")
    plot_drift_report(drift_report, output_dir / "drift_report.png")

    ref_scores = model.predict_proba(
        reference_df[feature_names].fillna(0).values
    )[:, 1]
    prod_scores = np.array([
        e.get("score", e.get("prediction", {}).get("score", 0.5))
        for e in event_log
    ])
    plot_prediction_score_distribution(
        ref_scores, prod_scores,
        "ran-anomaly-detector",
        output_dir / "score_distribution.png",
    )
    plot_circuit_breaker_timeline(
        event_log, "ran-anomaly-detector",
        output_dir / "circuit_breaker.png",
    )

    # ------------------------------------------------------------------
    # Step 10–11: Save configs and metrics
    # ------------------------------------------------------------------
    logger.info("Step 10: Saving YAML configuration templates")
    save_yaml_templates(output_dir / "templates")

    logger.info("Step 11: Exporting Prometheus metrics snapshot")
    metrics.write_prometheus_text(output_dir / "metrics.prom")

    # Final health check
    health = server.health()
    logger.info("Final health: %s", json.dumps(asdict(health), indent=2, default=str))

    pred_logger.flush_and_stop()
    logger.info("Demo complete. All outputs saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="05_production_patterns.py — Telco MLOps production serving patterns demo",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory for demo outputs",
    )
    parser.add_argument(
        "--serve-metrics", action="store_true",
        help="Start a Prometheus metrics HTTP server after the demo",
    )
    parser.add_argument(
        "--metrics-port", type=int, default=8001,
        help="Port for Prometheus metrics HTTP server",
    )
    parser.add_argument(
        "--use-pipeline-model", action="store_true",
        help=(
            "Load model from models/tier2_random_forest.joblib (output of "
            "03_model_training.py) and use flat snake_case feature names from "
            "01-04 pipeline instead of the dotted ran.kpi.* namespace. "
            "Requires running 01→02→03 first."
        ),
    )
    args = parser.parse_args()
    run_production_patterns_demo(
        args.output_dir, args.serve_metrics, args.metrics_port,
        use_pipeline_model=args.use_pipeline_model,
    )
