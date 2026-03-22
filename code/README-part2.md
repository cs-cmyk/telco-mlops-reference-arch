# Telco MLOps Reference Architecture — Part 2: Companion Code

Companion scripts for the whitepaper *"Telco MLOps Reference Architecture — Part 2: Extending the Platform to Graph ML, LLMs, Agentic Systems, and Beyond"*.

**Whitepaper:** [whitepaper.md](../whitepaper.md)  
**Part 1:** [github.com/cs-cmyk/telco-mlops-reference-arch](https://github.com/cs-cmyk/telco-mlops-reference-arch/blob/main/whitepaper.md)  
**RAN paper:** [github.com/cs-cmyk/ran-kpi-anomaly-detection](https://github.com/cs-cmyk/ran-kpi-anomaly-detection)

## Quick Start

```bash
# Clone
git clone https://github.com/cs-cmyk/telco-mlops-reference-arch.git
cd telco-mlops-reference-arch/code

# Install core dependencies (Python 3.10+)
pip install -r requirements.txt

# Run the full pipeline
python part2_01_synthetic_data.py
python part2_02_feature_engineering.py
python part2_03_model_training.py
python part2_04_evaluation.py
python part2_05_production_patterns.py

# Run extended scripts (independent of each other)
python part2_06_model_compression.py   # requires part2_03 outputs
python part2_07_digital_twin.py        # requires part2_01 outputs
python part2_08_rag_pipeline.py        # standalone
python part2_09_fraud_detection.py     # standalone
```

## Scripts

| Script | What it does | Inputs | Runtime |
|---|---|---|---|
| `part2_01` | Generates synthetic RAN KPI data (200 cells, 30 days, injected anomalies) | None | 2–5 min |
| `part2_02` | Engineers ~150 features (rolling stats, z-scores, temporal encodings) | `data/` from 01 | 3–8 min |
| `part2_03` | Trains IF → RF → LSTM-AE ensemble + optional GNN root cause classifier | `data/features/` from 02 | 5–30 min |
| `part2_04` | Three-tier evaluation: model metrics, KPI impact, business outcome proxies | `artifacts/` from 03 | 2–5 min |
| `part2_05` | Demonstrates 10 production patterns (BentoML, drift detection, safety layer) | `artifacts/models/` from 03 | <1 min |
| `part2_06` | INT8 quantisation of RF model with F1 ≥ 0.82 governance gate | `artifacts/models/` from 03, `data/features/` from 02 | 1–3 min |
| `part2_07` | Per-cell digital twin with what-if simulation and validation | `data/` from 01 | 2–5 min |
| `part2_08` | RAG pipeline: ingest → chunk → embed → retrieve → evaluate | None (sample corpus) | 1–3 min |
| `part2_09` | CDR fraud ring detection with heterogeneous GNN | None (synthetic CDRs) | 3–10 min |

## Dependencies

**Core** (required by all scripts): `pandas numpy scipy pyarrow scikit-learn joblib matplotlib seaborn shap`

**Optional** (scripts degrade gracefully without these):

| Package | Used by | What happens without it |
|---|---|---|
| `torch` | 03 (LSTM-AE), 06, 09 | LSTM-AE skipped; compression uses synthetic ONNX; fraud uses logistic regression |
| `torch_geometric` | 03 (GNN), 09 | GNN skipped; fraud uses logistic regression baseline |
| `onnx`, `onnxruntime`, `skl2onnx` | 06 | FP32 copy used as INT8 placeholder |
| `sentence-transformers` | 08 | Random embeddings used |
| `qdrant-client` | 08 | Numpy-based cosine search |
| `bentoml`, `prometheus_client` | 05 | Required for this script |

See `requirements.txt` for version constraints and installation instructions.

## Directory Structure

After running all scripts:

```
code/
├── requirements.txt
├── README.md
├── part2_01_synthetic_data.py
├── part2_02_feature_engineering.py
├── part2_03_model_training.py
├── part2_04_evaluation.py
├── part2_05_production_patterns.py
├── part2_06_model_compression.py
├── part2_07_digital_twin.py
├── part2_08_rag_pipeline.py
├── part2_09_fraud_detection.py
├── part2_test_pipeline.py
│
├── data/                              # Created by 01, extended by 02
│   ├── pm_counters.parquet
│   ├── cell_inventory.parquet
│   ├── neighbour_relations.parquet
│   ├── anomaly_labels.parquet
│   └── features/                      # Created by 02
│       ├── train.parquet
│       ├── val.parquet
│       ├── test.parquet
│       ├── feature_catalog.json
│       └── split_metadata.json
│
└── artifacts/                         # Created by 03, extended by 04, 06
    ├── anomaly_scores.npy
    ├── models/
    │   ├── isolation_forest.joblib
    │   ├── random_forest.joblib
    │   ├── lstm_autoencoder.pt
    │   ├── random_forest_fp32.onnx    # Created by 06
    │   ├── random_forest_int8.onnx    # Created by 06
    │   ├── scaler.joblib
    │   └── ensemble_thresholds.json
    ├── training_metadata.json
    ├── compression_report.json        # Created by 06
    ├── digital_twin/                  # Created by 07
    ├── rag/                           # Created by 08
    ├── fraud_detection/               # Created by 09
    └── evaluation/                    # Created by 04
        ├── metrics_summary.json
        ├── confusion_matrices/
        ├── roc_curves/
        ├── score_distributions/
        └── operational_report.md
```

## Governance Gates

Each script that produces a deployable artefact enforces a governance gate:

| Script | Gate | Threshold | On failure |
|---|---|---|---|
| `part2_03` | GNN node AUROC | ≥ 0.80 | Warning logged; GNN not promoted |
| `part2_06` | INT8 model F1 | ≥ 0.82 | Script exits with error; artefact not written |
| `part2_08` | Faithfulness / Precision | ≥ 0.85 / ≥ 0.80 | Warning logged in evaluation report |
| `part2_09` | Precision / Ring detection | ≥ 0.90 / ≥ 70% | Warning logged in evaluation report |

## Whitepaper Cross-Reference

| Whitepaper Section | Script(s) |
|---|---|
| §3 Three-Tier Measurement | part2_04 |
| §6 Proposed Approach | part2_03 |
| §8.1 GNN Root Cause | part2_03 |
| §8.2 RAG Pipeline | part2_08 |
| §8.4 Edge AI | part2_06 |
| §8.5 Digital Twin | part2_07 |
| §8.6 Fraud Detection | part2_09 |
| §10 Evaluation | part2_04 |
| §12 Agentic Systems | part2_05 (safety layer prototype) |
| §13 FinOps | part2_05 (cost tracking) |
| §16 Getting Started | All scripts in order |

## Testing

```bash
python part2_test_pipeline.py
```

Runs the full pipeline end-to-end and validates outputs at each stage.

## Notes

- All scripts are **deterministic** (seeded random state) — re-running produces identical outputs.
- All scripts are **illustrative** — they demonstrate architectural patterns, not production-hardened code. TLS, authentication, secret management, and Kubernetes manifests are out of scope.
- Scripts 08 and 09 are **standalone** — they generate their own data and do not depend on scripts 01–05.
- Script 06 depends on script 03 outputs (model artefacts) and script 02 outputs (test features).
- Script 07 depends on script 01 outputs (PM counters).

## License

See the repository root for license information.
