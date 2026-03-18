from datetime import datetime, timezone
from feast import FeatureStore

# Initialise Feast (assumes feast_repo_path is configured)
store = FeatureStore(repo_path="./feast_repo")

# Expected feature names from the cell_ran_features FeatureView
EXPECTED_FEATURES = {
    "dl_prb_usage_mean_15m",
    "ul_throughput_p95_1h",
    "rrc_conn_estab_success_rate_15m",
    "ho_intra_freq_success_rate_1h",
    "alarm_count_critical_1h",
}


def push_cell_features(
    cell_id: str,
    event_timestamp: datetime,
    features: dict,
) -> None:
    """
    Push a single cell's computed features to the Feast online store.

    Called from a Flink sink operator after windowed aggregation.
    In production, batch multiple cells into a single push call
    for efficiency (Feast supports DataFrame-based push).

    WARNING: Push COMPUTED features (ratios, aggregates), not raw
    PM counter values. Pushing raw counters produces training-serving
    skew because the training path computes features from raw counters
    via the offline store.
    """
    # Schema alignment check — fail loud if feature keys don't match
    actual_keys = set(features.keys())
    missing = EXPECTED_FEATURES - actual_keys
    extra = actual_keys - EXPECTED_FEATURES
    assert not missing, (
        f"Missing features for push: {missing}. "
        f"FeatureView schema and Flink output are misaligned."
    )
    if extra:
        import logging
        logging.warning("Extra features ignored by push: %s", extra)

    # Construct the push DataFrame
    import pandas as pd
    push_df = pd.DataFrame([{
        "cell_id": cell_id,
        "event_timestamp": event_timestamp,
        **{k: features[k] for k in EXPECTED_FEATURES},
    }])

    # Feast FeatureView schema expects Float32 — Flink/Pandas default to Float64.
    # Explicit cast avoids silent coercion or materialisation failures.
    push_df[list(EXPECTED_FEATURES)] = push_df[list(EXPECTED_FEATURES)].astype("float32")

    # PushSource name must match PushSource(name=...) in the Feast registry
    # definition (see CODE-02 in whitepaper §7).
    store.push("pm_counters_push", push_df)


# Example invocation (would be called from Flink's ProcessFunction)
if __name__ == "__main__":
    push_cell_features(
        cell_id="CELL_001_A",
        event_timestamp=datetime.now(timezone.utc),
        features={
            "dl_prb_usage_mean_15m": 0.73,
            "ul_throughput_p95_1h": 45.2,
            "rrc_conn_estab_success_rate_15m": 0.987,
            "ho_intra_freq_success_rate_1h": 0.965,
            "alarm_count_critical_1h": 0,
        },
    )
    print("Push successful")

