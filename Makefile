# =============================================================================
# Makefile – Common project operations
# =============================================================================
# Usage:
#   make help          – print available targets
#   make setup         – create venv and install dependencies
#   make fetch-data    – download the Pima dataset into data/raw/
#   make preprocess    – run the full preprocessing pipeline
#   make train         – train all models and run cross-validation
#   make evaluate      – evaluate the best model on the held-out test set
#   make all           – run everything (fetch → preprocess → train → evaluate)
#   make clean         – remove generated artefacts (figures, reports, models)
# =============================================================================

PYTHON   := python3
VENV     := .venv
PIP      := $(VENV)/bin/pip
PYTHON_V := $(VENV)/bin/python
SRC      := src

.PHONY: help setup fetch-data preprocess train evaluate test all clean

help:
	@echo ""
	@echo "  diabetes-risk-prediction"
	@echo "  ─────────────────────────────────────"
	@echo "  make setup          Create venv and install deps"
	@echo "  make fetch-data     Download Pima dataset"
	@echo "  make preprocess     Run preprocessing pipeline"
	@echo "  make train          Train, tune, and evaluate all models"
	@echo "  make test           Run the pytest suite"
	@echo "  make all            fetch-data → preprocess → train"
	@echo "  make clean          Remove generated artefacts"
	@echo ""

setup:
	@echo "▸ Creating virtual environment ..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt -q
	@echo "✓ Done. Activate with:  source $(VENV)/bin/activate"

fetch-data:
	@echo "▸ Downloading raw data ..."
	$(PYTHON_V) -c "from src.data import download_raw_dataset; download_raw_dataset()"

preprocess:
	@echo "▸ Preprocessing data ..."
	$(PYTHON_V) -c "from src.data import process_data; process_data()"

train:
	@echo "▸ Training, tuning, and evaluating models (this may take a few minutes) ..."
	$(PYTHON_V) -m src.scripts.run_pipeline

test:
	@echo "▸ Running pytest suite ..."
	$(PYTHON_V) -m pytest tests/ -v

all: fetch-data preprocess train
	@echo ""
	@echo "✓ Full pipeline completed successfully."
	@echo "  Reports  → reports/"
	@echo "  Figures  → reports/figures/"
	@echo "  Models   → models/"

clean:
	@echo "▸ Cleaning generated artefacts ..."
	rm -rf reports/figures/*.png
	rm -rf reports/*.csv
	rm -rf models/*.joblib models/*.pkl
	rm -rf data/processed/*.csv
	rm -rf __pycache__ src/__pycache__ src/scripts/__pycache__
	@echo "✓ Done."
