# Telco MLOps Reference Architecture

**How multi-team, multi-model organisations ship ML at scale without losing control.**

**Author:** Chirag Shinde — chirag.m.shinde@gmail.com

A standards-traceable, open-source reference architecture for operating ML at telecommunications scale — designed for organisations running 20+ models across multiple squads (RAN optimisation, network assurance, CX/churn, fraud detection) that need both team autonomy and platform-level governance.

---

## Three Architectural Pillars

1. **Streaming-first feature store** — Kafka + Flink + Feast eliminating training-serving skew across shared cell-level data
2. **Namespace-isolated multi-tenant ML lifecycle** — Kubeflow + MLflow giving each squad independence without shadow infrastructure
3. **Conflict-aware model registry** — OPA governance gate with SHAP-DAG conflict detection preventing xApp control parameter collisions before production

---

## Repository Structure

``` 
| File | Description |
|---|---|
| `LICENSE` | CC BY-NC-SA 4.0 International |
| `whitepaper.md` | Full whitepaper |
| `code/01_synthetic_data.py` | Generate realistic telco PM counter data |
| `code/02_feature_engineering.py` | Windowed aggregations, point-in-time features |
| `code/03_model_training.py` | XGBoost + Random Forest with MLflow tracking |
| `code/04_evaluation.py` | Temporal holdout, threshold gates, drift baseline |
| `code/05_production_patterns.py` | OPA policy check, inference, drift detection |
| `code/flink_feast_push_stub.py` | Streaming feature materialisation pattern |
| `code/ves_parser_stub.py` | VES event parsing pattern |
| `code/Makefile` | Pipeline orchestration (`make pipeline`, `make production`) |
| `code/requirements.txt` | Python dependencies |
| `code/FEATURE_NAMESPACE_CONVENTION.md` | Feature naming convention guide |
```

---

## Running the Companion Code

**Prerequisites:** Python 3.10+, pip

```bash
cd code
pip install -r requirements.txt

# Run the full pipeline (scripts 01–04)
make pipeline

# Run production inference pattern (script 05)
make production
```

The pipeline generates synthetic telco data, engineers features, trains models with MLflow tracking, evaluates against temporal holdouts, and produces drift baselines. Script 05 demonstrates production patterns including OPA policy validation, inference with drift detection, and the pipeline-model serving approach.

Each script is self-contained and produces artifacts consumed by the next. See the Makefile for the full dependency chain.

---

## Whitepaper Sections

| # | Section | Key Content |
|---|---|---|
| — | Executive Summary | Three pillars, problem statement, deployment velocity thesis |
| — | How to Read This Whitepaper | Role-based reading guide with estimated reading times |
| 1 | Business Case | Cost of fragmented ML infrastructure, worked ROI example |
| 2 | Problem Statement | Scaling break point, why telco is harder than enterprise |
| 3 | Data Requirements | PM counter landscape, E2SM-KPM, NWDAF, multi-team access |
| 4 | Background and Prior Art | 3GPP/O-RAN/TM Forum standards, Big Tech lessons, maturity assessment |
| 5 | Proposed Approach | Platform-as-product philosophy, architecture comparison |
| 6 | System Design | Four-layer architecture: data platform, serving, ML lifecycle, governance |
| 7 | Implementation Walkthrough | Squad onboarding, feature engineering, training, serving, monitoring |
| 8 | Evaluation and Operational Impact | Model evaluation framework, 3–5× velocity improvement derivation |
| 9 | Production Considerations | Progressive delivery, multi-region DR, security, cost, Day 2 ops |
| 10 | Platform Testing Strategy | Four-layer test suite: feature store, serving, lifecycle, governance |
| 11 | Reinforcement Learning Extensions | Offline RL (CQL), bootstrapping, OPE, reward hacking, conflict detection |
| 12 | Data Privacy Implementation | Privacy classes, k-anonymity, differential privacy, EU AI Act mapping |
| 13 | Build-vs-Buy Analysis | Per-component decision framework, cost scaling, migration risk |
| 14 | Operational Observability | Distributed tracing, burn-rate alerting, runbooks |
| 15 | Limitations | Scope boundaries, deferred topics, honest assessment |
| 16 | Getting Started | Readiness checklist, four-phase implementation roadmap |
| 17 | Coursebook Cross-Reference | Prerequisite reading, extension points, paired exercises |
| — | Glossary | Telco and MLOps terminology reference |
| 18 | Further Reading | Operator case studies, academic papers, standards documents, open-source projects |

---

## Standards Alignment

Every architecture component maps to a governing standard, with gaps explicitly identified:

| Standard | Coverage |
|---|---|
| 3GPP TS 28.105 | ML entity repository, training report, model lifecycle |
| 3GPP TS 28.627 | LoopState machine, retraining triggers, closed-loop automation |
| 3GPP TS 28.550/552 | PM data collection architecture and NR counter definitions |
| O-RAN WG2 | AI/ML workflow, rApp model hosting, SMO integration |
| O-RAN WG3 | xApp deployment, E2SM-KPM, conflict mitigation (emerging) |
| TM Forum IG1230 | Autonomous networks governance, L0–L5 maturity scale |
| ETSI GS ENI 005 | Closed-loop intelligence architecture |
| EU AI Act | High-risk AI system classification, model card requirements |

See Figure 8 (Standards Traceability Map) and §4 for the detailed mapping.

---

## Part 2 (Forthcoming)

A follow-up paper extending the architecture to:

- Graph ML for network-native use cases (topology anomaly detection, fraud ring detection)
- Network performance measurement framework (the evaluation layer connecting model metrics to network KPIs to business outcomes)
- LLMs and foundation models (customer service, NOC queries, configuration generation, RAG over standards)
- Deep anomaly detection (autoencoders, VAEs, transformer-based detectors)
- Digital twins and simulation (synthetic environments for RL pre-training)
- Edge AI and on-device inference (model compression, edge deployment lifecycle)
- Optimisation hybrids (MILP + ML surrogate models for network planning)
- Online learning and federated learning across operator boundaries
- Agents and agentic systems for autonomous network operations (agent taxonomy, tool registry, action safety layer, progressive autonomy, multi-agent coordination)

---

## Licence

This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](LICENSE). You may share and adapt the material with attribution, but not for commercial purposes. Adaptations must be distributed under the same licence.

---

## Citation

If referencing this work, please cite as:

> Shinde, C. *Telco MLOps Reference Architecture: How Multi-Team, Multi-Model Organisations Ship ML at Scale Without Losing Control.* 2025. https://github.com/cs-cmyk/telco-mlops-reference-arch
