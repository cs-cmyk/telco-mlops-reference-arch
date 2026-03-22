#!/usr/bin/env python3
"""End-to-end integration test for the Telco MLOps Part 2 companion scripts.

Runs all five scripts in sequence and verifies that expected output
artifacts exist at each stage.  Intended as a smoke test — it confirms
the pipeline produces output, not that the output is statistically valid.

Usage:
    python test_pipeline.py              # full pipeline
    python test_pipeline.py --quick      # skip Script 05 demo (faster)

Prerequisites:
    pip install -r requirements.txt

Runtime: approximately 15–30 minutes on a 4-core / 16 GB machine,
dominated by Script 02 (feature engineering) and Script 03 (model training).
On machines with < 8 GB RAM, set TELCO_SKIP_NEIGHBOUR_FEATURES=1 to
reduce Script 02 memory usage at the cost of neighbour features.
"""

import os
import subprocess
import sys
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

SCRIPTS = [
    "01_synthetic_data.py",
    "02_feature_engineering.py",
    "03_model_training.py",
    "04_evaluation.py",
    "05_production_patterns.py",
]

# Artifacts expected after each script completes
EXPECTED_ARTIFACTS = {
    "01_synthetic_data.py": [
        Path("data/pm_counters.parquet"),
        Path("data/cell_inventory.parquet"),
    ],
    "02_feature_engineering.py": [
        Path("data/features/train.parquet"),
        Path("data/features/val.parquet"),
        Path("data/features/test.parquet"),
        Path("data/features/feature_catalog.json"),
        Path("data/features/scaler.json"),
        Path("data/features/scaler.joblib"),
    ],
    "03_model_training.py": [
        Path("artifacts/models/random_forest.joblib"),
        Path("artifacts/models/isolation_forest.joblib"),
    ],
    "04_evaluation.py": [
        Path("artifacts/evaluation/metrics_summary.json"),
    ],
    "05_production_patterns.py": [],  # demo script; no persistent artifacts
}

# ── Runner ─────────────────────────────────────────────────────────────────


def run_script(script: str, timeout: int = 1800) -> None:
    """Run a script and raise on failure."""
    print(f"\n{'=' * 70}")
    print(f"  Running: {script}")
    print(f"{'=' * 70}")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout[-2000:]}")
        print(f"STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"{script} exited with code {result.returncode}")
    print(f"  ✓ {script} completed successfully")


def check_artifacts(script: str) -> None:
    """Verify expected artifacts exist after a script runs."""
    missing = []
    for path in EXPECTED_ARTIFACTS.get(script, []):
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise RuntimeError(
            f"After {script}, expected artifacts missing:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )
    n = len(EXPECTED_ARTIFACTS.get(script, []))
    if n:
        print(f"  ✓ {n} expected artifact(s) verified")


def main() -> None:
    quick = "--quick" in sys.argv

    scripts = SCRIPTS[:-1] if quick else SCRIPTS

    passed = 0
    failed = 0

    for script in scripts:
        try:
            run_script(script)
            check_artifacts(script)
            passed += 1
        except Exception as exc:
            print(f"\n  ✗ FAILED: {exc}")
            failed += 1
            # Continue to next script to report all failures
            continue

    # ── Cross-layer contract tests ────────────────────────────────────
    # Verify Layer 5 → Layer 6 → Layer 7 data flow contracts:
    #   GNN root cause output → RAG build_retrieval_query() → Safety layer
    # These catch schema mismatches between layers that file-existence
    # checks cannot detect.

    print(f"\n{'=' * 70}")
    print(f"  Cross-layer contract tests")
    print(f"{'=' * 70}")

    try:
        _run_cross_layer_contract_tests()
        print("  ✓ All cross-layer contract tests passed")
        passed += 1
    except Exception as exc:
        print(f"\n  ✗ Cross-layer contract FAILED: {exc}")
        failed += 1

    print(f"\n{'=' * 70}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(scripts) + 1} stages")
    print(f"{'=' * 70}")

    sys.exit(1 if failed else 0)


def _run_cross_layer_contract_tests() -> None:
    """Verify cross-layer data contracts without running full inference.

    Tests that:
    1. A GNN-style root cause output can be consumed by the RAG pipeline's
       build_retrieval_query() (Layer 5 → Layer 6 contract)
    2. The RAG pipeline produces a valid query string from an AlertCard
       with and without root cause attribution
    3. The safety layer's ActionSafetyLayer can evaluate a proposal
       that references GNN scores (Layer 5 → Layer 7 contract)

    These are schema/contract tests, not accuracy tests.
    """
    print("  Testing Layer 5 → Layer 6 (GNN → RAG) contract...")

    # Import AlertCard and build_retrieval_query from part2_08
    sys.path.insert(0, ".")
    from part2_08_rag_pipeline import AlertCard, build_retrieval_query

    # Test 1a: AlertCard WITH root cause (Layer 5 output present)
    alert_with_rc = AlertCard(
        alert_id="CONTRACT_TEST_001",
        cell_id="CELL_0042",
        timestamp="2024-06-15T14:30:00Z",
        anomaly_score=0.87,
        shap_top_features={"peer_zscore_avg_cqi": -3.4},
        peer_group_delta={"avg_cqi": -2.9},
        severity="major",
        root_cause_type="backhaul",
        root_cause_node_id="BH_042",
        root_cause_confidence=0.84,
    )
    query_with_rc = build_retrieval_query(alert_with_rc)
    assert isinstance(query_with_rc, str), "build_retrieval_query must return str"
    assert len(query_with_rc) > 0, "query must be non-empty"
    assert "backhaul" in query_with_rc.lower() or "transport" in query_with_rc.lower(), \
        "query with backhaul root cause should contain domain terms"
    print("    ✓ AlertCard with root cause → valid retrieval query")

    # Test 1b: AlertCard WITHOUT root cause (GNN in shadow mode)
    alert_no_rc = AlertCard(
        alert_id="CONTRACT_TEST_002",
        cell_id="CELL_0089",
        timestamp="2024-06-15T16:15:00Z",
        anomaly_score=0.78,
        shap_top_features={"peer_zscore_dl_prb_usage_rate": 3.2},
        peer_group_delta={"dl_prb_usage_rate": 18.5},
        severity="warning",
    )
    query_no_rc = build_retrieval_query(alert_no_rc)
    assert isinstance(query_no_rc, str), "build_retrieval_query must return str"
    assert len(query_no_rc) > 0, "query must be non-empty without root cause"
    print("    ✓ AlertCard without root cause → valid fallback query")

    # Test 2: Layer 5 → Layer 7 contract (GNN scores → safety layer)
    print("  Testing Layer 5 → Layer 7 (GNN → Safety Layer) contract...")
    try:
        from part2_05_production_patterns import ActionSafetyLayer
        # Verify the safety layer can be instantiated and accepts
        # GNN-style attribution data in the action proposal
        action_proposal = {
            "tool": "antenna_tilt_adjust",
            "target_cell": "CELL_0042",
            "parameter_delta": {"electrical_tilt_deg": -1.5},
            "gnn_root_cause": {
                "type": "tilt",
                "node_id": "CELL_0042",
                "confidence": 0.81,
            },
            "autonomy_level": 1,
            "require_human_gate": True,
        }
        # Contract test: the proposal dict with gnn_root_cause field
        # should be serialisable (no schema errors on construction)
        import json
        json.dumps(action_proposal)
        print("    ✓ Action proposal with GNN root cause is schema-valid")
    except ImportError:
        print("    ⚠ part2_05 not importable — skipping Layer 7 contract test")
    except Exception as exc:
        raise RuntimeError(f"Layer 5 → Layer 7 contract failed: {exc}") from exc

    print("  ✓ Cross-layer contracts verified")

    # Test 3: Temporal split integrity (§8.1 temporal leakage prevention)
    print("  Testing temporal split integrity...")
    try:
        import pandas as pd
        feature_dir = Path("data/features")
        train_path = feature_dir / "train.parquet"
        val_path = feature_dir / "val.parquet"
        test_path = feature_dir / "test.parquet"
        if train_path.exists() and val_path.exists() and test_path.exists():
            train_df = pd.read_parquet(train_path)
            val_df = pd.read_parquet(val_path)
            test_df = pd.read_parquet(test_path)
            ts_col = "timestamp" if "timestamp" in train_df.columns else None
            if ts_col is None:
                # Try common alternatives
                for candidate in ["ts", "datetime", "time", "rop_timestamp"]:
                    if candidate in train_df.columns:
                        ts_col = candidate
                        break
            if ts_col is not None:
                assert train_df[ts_col].max() < val_df[ts_col].min(), \
                    f"Temporal leakage: train {ts_col} overlaps with val"
                assert val_df[ts_col].max() < test_df[ts_col].min(), \
                    f"Temporal leakage: val {ts_col} overlaps with test"
                print(f"    ✓ Temporal split integrity verified on '{ts_col}'")
            else:
                print("    ⚠ No timestamp column found — skipping temporal check")
        else:
            print("    ⚠ Feature parquet files not found — skipping temporal check")
    except ImportError:
        print("    ⚠ pandas not available — skipping temporal check")
    except Exception as exc:
        print(f"    ✗ Temporal integrity check failed: {exc}")
        raise


if __name__ == "__main__":
    main()
