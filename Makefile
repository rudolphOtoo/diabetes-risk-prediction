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

.PHONY: help setup install-dev fetch-data preprocess train test lint format check all clean

help:
	@echo ""
	@echo "  diabetes-risk-prediction"
	@echo "  ─────────────────────────────────────"
	@echo "  make setup          Create venv and install deps"
	@echo "  make install-dev    Install dev deps (ruff) into venv"
	@echo "  make fetch-data     Download Pima dataset"
	@echo "  make preprocess     Run preprocessing pipeline"
	@echo "  make train          Train, tune, and evaluate all models"
	@echo "  make test           Run the pytest suite"
	@echo "  make lint           Run ruff lint checks"
	@echo "  make format         Auto-format source with ruff"
	@echo "  make check          Lint + format-report + tests (CI-equivalent)"
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

install-dev:
	@echo "▸ Installing development dependencies (ruff) ..."
	$(PIP) install -q ruff
	@echo "✓ Done."

lint:
	@echo "▸ Running ruff linter ..."
	$(PYTHON_V) -m ruff check src/ tests/

format:
	@echo "▸ Formatting source with ruff ..."
	$(PYTHON_V) -m ruff format src/ tests/

check: lint
	@echo "▸ Verifying formatting ..."
	$(PYTHON_V) -m ruff format --check src/ tests/
	@echo "▸ Running pytest with coverage gate ..."
	$(PYTHON_V) -m pytest tests/ -q --cov=src --cov-fail-under=60

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
