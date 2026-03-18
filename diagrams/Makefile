# Telco MLOps Reference Architecture — Companion Code
# Pipeline integration test: verifies 01→02→03→04 produce expected outputs.
#
# Usage:
#   make pipeline          # Run full integration test (includes LSTM)
#   make pipeline-quick    # Skip LSTM training (faster, recommended for CI)
#   make clean             # Remove all generated artifacts
#   make lint              # Syntax check all Python files

.PHONY: pipeline pipeline-quick production clean lint

DATA_DIR   := data
MODEL_DIR  := models
EVAL_DIR   := eval_output
PROD_DIR   := production_output

# Shared build steps, parameterised by TRAIN_FLAGS
define RUN_PIPELINE
	@echo "=== Step 1/4: Synthetic data generation ==="
	python 01_synthetic_data.py
	@test -f $(DATA_DIR)/pm_counters.parquet || (echo "FAIL: 01 did not produce pm_counters.parquet" && exit 1)
	@test -f $(DATA_DIR)/cell_topology.parquet || (echo "FAIL: 01 did not produce cell_topology.parquet" && exit 1)
	@echo "  ✓ 01_synthetic_data.py outputs verified"
	@echo ""
	@echo "=== Step 2/4: Feature engineering ==="
	python 02_feature_engineering.py
	@test -f $(DATA_DIR)/features_train.parquet || (echo "FAIL: 02 did not produce features_train.parquet" && exit 1)
	@test -f $(DATA_DIR)/features_val.parquet || (echo "FAIL: 02 did not produce features_val.parquet" && exit 1)
	@test -f $(DATA_DIR)/features_test.parquet || (echo "FAIL: 02 did not produce features_test.parquet" && exit 1)
	@echo "  ✓ 02_feature_engineering.py outputs verified"
	@echo ""
	@echo "=== Step 3/4: Model training $(1) ==="
	python 03_model_training.py $(1)
	@test -f $(MODEL_DIR)/tier2_random_forest.joblib || (echo "FAIL: 03 did not produce tier2_random_forest.joblib" && exit 1)
	@test -f $(MODEL_DIR)/tier1_isolation_forest.joblib || (echo "FAIL: 03 did not produce tier1_isolation_forest.joblib" && exit 1)
	@echo "  ✓ 03_model_training.py outputs verified"
	@echo ""
	@echo "=== Step 4/4: Evaluation ==="
	python 04_evaluation.py
	@test -f $(EVAL_DIR)/metrics_summary.json || (echo "FAIL: 04 did not produce metrics_summary.json" && exit 1)
	@echo "  ✓ 04_evaluation.py outputs verified"
	@echo ""
	@echo "========================================="
	@echo "  Pipeline integration test PASSED"
	@echo "========================================="
endef

pipeline: clean
	$(call RUN_PIPELINE,)

pipeline-quick: clean
	$(call RUN_PIPELINE,--skip-lstm)

production: pipeline-quick
	@echo ""
	@echo "=== Step 5/5: Production patterns (pipeline model) ==="
	python 05_production_patterns.py --use-pipeline-model --output-dir $(PROD_DIR)
	@test -d $(PROD_DIR) || (echo "FAIL: 05 did not produce production_output/" && exit 1)
	@python -c "\
import pathlib, json; \
d = pathlib.Path('$(PROD_DIR)'); \
jsons = list(d.glob('*.json')); \
assert len(jsons) > 0, 'FAIL: no JSON outputs in $(PROD_DIR)'; \
print('  ✓ Production outputs: %d JSON files' % len(jsons))"
	@echo "  ✓ 05_production_patterns.py (pipeline model) outputs verified"
	@echo ""
	@echo "========================================="
	@echo "  Full production integration test PASSED"
	@echo "========================================="

clean:
	rm -rf $(DATA_DIR)/ $(MODEL_DIR)/ $(EVAL_DIR)/ production_output/
	@echo "Cleaned all generated artifacts"

lint:
	@echo "=== Lint check ==="
	python -m py_compile 01_synthetic_data.py
	python -m py_compile 02_feature_engineering.py
	python -m py_compile 03_model_training.py
	python -m py_compile 04_evaluation.py
	python -m py_compile 05_production_patterns.py
	python -m py_compile flink_feast_push_stub.py
	python -m py_compile ves_parser_stub.py
	@echo "  ✓ All files compile cleanly"
	@echo ""
	@echo "For full lint, install ruff: pip install ruff && ruff check *.py"
