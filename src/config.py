"""Central configuration shared across the research notebooks and scripts.

Path conventions and modelling hyperparameters are defined here so that every
part of the project references a single source of truth. This guarantees that
the ``data/raw``, ``data/processed``, ``models/`` and ``reports/figures``
directories are created consistently regardless of the entry point invoked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root and data locations
# ---------------------------------------------------------------------------

# ``Path(__file__).resolve().parent`` -> .../src
# ``Path(__file__).resolve().parent.parent`` -> repository root
REPO_ROOT: Path = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    """Canonical on-disk locations for every artefact of the project.

    Attributes
    ----------
    raw_data: Path
        Directory holding the original, unmodified dataset file.
    processed_data: Path
        Directory holding cleaned, feature-engineered datasets.
    models: Path
        Directory for persisted fitted model objects and encoders.
    figures: Path
        Directory for exported matplotlib / seaborn figures.
    reports: Path
        Directory for machine-readable metric tables (CSV).
    """

    raw_data: Path = REPO_ROOT / "data" / "raw"
    processed_data: Path = REPO_ROOT / "data" / "processed"
    models: Path = REPO_ROOT / "models"
    figures: Path = REPO_ROOT / "reports" / "figures"
    reports: Path = REPO_ROOT / "reports"


# ---------------------------------------------------------------------------
# Data provenance
# ---------------------------------------------------------------------------

#: UCI-hosted mirror of the Pima Indians Diabetes Database.
DATA_URL: str = (
    "https://raw.githubusercontent.com/"
    "plotly/datasets/master/diabetes.csv"
)

#: Name of the raw CSV once downloaded.
RAW_FILENAME: str = "diabetes_raw.csv"

#: Name of the fully processed CSV (features + engineered columns + label).
PROCESSED_FILENAME: str = "diabetes_processed.csv"

#: Integer 0 stands for absence of diabetes (Pima label semantics).
NEGATIVE_CLASS_LABEL: int = 0

#: Integer 1 stands for diabetes present (Pima label semantics).
POSITIVE_CLASS_LABEL: int = 1


# ---------------------------------------------------------------------------
# Modelling settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Hyperparameters and reproducibility controls.

    Attributes
    ----------
    test_size : float
        Fraction of the data reserved for the held-out test set.
    val_size : float
        Fraction of the treated data reserved for validation (dev) set.
    random_seed : int
        Master seed used to derive every child seed, guaranteeing fully
        deterministic train/validation/test splits and model fitting.
    n_splits : int
        Number of folds in the stratified K-fold cross-validation.
    scoring : str
        Primary scikit-learn metric used to select the best model.
    """

    test_size: float = 0.20
    val_size: float = 0.20
    random_seed: int = 42
    n_splits: int = 5
    scoring: str = "roc_auc"