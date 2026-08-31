"""Data acquisition and preprocessing.

Responsibilities
---------------
* ``download_raw_dataset`` : fetch the Pima Indians Diabetes Database from its
  public mirror and store a verbatim copy under ``data/raw``.
* ``load_raw_frame``        : read the raw CSV into a :class:`pandas.DataFrame`.
* ``process_data``          : apply the full preprocessing and feature-
  engineering recipe and persist the result under ``data/processed``.

Reproducibility note
--------------------
The raw file is only downloaded when it is not already present on disk. This
makes repeated runs deterministic (no re-hits to the network) and lets a
reviewer clone the repository and run the pipeline entirely offline after a
single fetch.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_URL, PROCESSED_FILENAME, RAW_FILENAME, Paths

#: Ordered list of the ten biological/clinical predictor columns (Pima schema).
FEATURE_COLUMNS: list[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

#: The binary outcome column. 1 == diabetes present, 0 == absent.
TARGET_COLUMN: str = "Outcome"


def download_raw_dataset(paths: Paths | None = None) -> Path:
    """Fetch the raw Pima dataset to ``data/raw`` if it is not already present.

    Parameters
    ----------
    paths : Paths | None
        Path configuration. Defaults to the project-level canonical ``Paths``.

    Returns
    -------
    Path
        The absolute path to the (now existing) raw CSV file.
    """
    paths = paths or Paths()
    paths.raw_data.mkdir(parents=True, exist_ok=True)
    destination: Path = paths.raw_data / RAW_FILENAME
    if destination.exists():
        return destination

    urllib.request.urlretrieve(DATA_URL, destination)
    return destination


def load_raw_frame(paths: Paths | None = None) -> pd.DataFrame:
    """Read the raw Pima dataset into a tidy :class:`pandas.DataFrame`.

    Parameters
    ----------
    paths : Paths | None
        Path configuration. Defaults to the project-level canonical ``Paths``.

    Returns
    -------
    pandas.DataFrame
        The untouched raw data with 768 rows and 9 columns.
    """
    paths = paths or Paths()
    raw_file: Path = download_raw_dataset(paths)
    return pd.read_csv(raw_file)


def process_data(paths: Paths | None = None) -> pd.DataFrame:
    """Apply the raw-data cleaning recipe without any model-facing statistics.

    This step is deliberately **free of imputation**. It only (1) coerces the
    clinical columns to numeric, and (2) re-encodes physiologically impossible
    zero measurements (e.g., ``BloodPressure = 0``) as missing (``NaN``). The
    repository of missing values is then passed to the modelling stage, where
    median imputation is performed *inside each scikit-learn ``Pipeline``* on
    the training folds only — this keeps all preprocessing contained within the
    cross-validation folds and prevents any test statistics from leaking into
    training (see ``src/models.py``).

    Parameters
    ----------
    paths : Paths | None
        Path configuration. Defaults to the project-level canonical ``Paths``.

    Returns
    -------
    pandas.DataFrame
        The cleaned frame with 768 rows and 9 columns; clinical columns are
        numeric and may still contain ``NaN`` where zero measurements were
        repaired.
    """
    paths = paths or Paths()
    frame: pd.DataFrame = load_raw_frame(paths)

    # 1. Type coercion with error tolerance.
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    # 2. Replace physiologically impossible zeros with NaNs. Blood Pressure,
    #    SkinThickness, Insulin and BMI cannot physically be zero.
    structurally_zero = ["BloodPressure", "SkinThickness", "Insulin", "BMI"]
    frame[structurally_zero] = frame[structurally_zero].replace(0.0, np.nan)

    # 3. Persist the cleaned frame for downstream reuse. Missing entries are
    #    intentionally retained here; imputation happens at model fit time.
    paths.processed_data.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.processed_data / PROCESSED_FILENAME, index=False)

    return frame
