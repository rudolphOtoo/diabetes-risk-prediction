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


def process_data(paths: Paths | None = None, impute_mode: str = "median") -> pd.DataFrame:
    """Apply the preprocessing and feature-engineering recipe end-to-end.

    Steps performed
    ---------------
    1. Load the raw CSV.
    2. Coerce the clinical columns to ``float`` (catching malformed entries).
    3. Repair physiologically impossible zero values (a common occurrence in
       this dataset) : variables that cannot legitimately be zero are
       re-encoded as :data:`numpy.nan`.
    4. Impute the missing clinical values (median by default).
    5. Persist the processed frame under ``data/processed``.

    Parameters
    ----------
    paths : Paths | None
        Path configuration. Defaults to the project-level canonical ``Paths``.
    impute_mode : str
        Imputation strategy passed through to :func:`pandas.Series.fillna`.
        Supported: ``"median"`` (robust) or ``"mean"``.

    Returns
    -------
    pandas.DataFrame
        The cleaned, imputed frame with 768 rows and 9 columns.
    """
    paths = paths or Paths()
    frame: pd.DataFrame = load_raw_frame(paths)

    # 1. Type coercion with error tolerance.
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    # 2. Replace physiologically impossible zeros with NaNs. Blood Pressure,
    #    SkinThickness, Insulin and BMI cannot physically be zero.
    structurally_zero = ["BloodPressure", "SkinThickness", "Insulin", "BMI"]
    frame[structurally_zero] = frame[structurally_zero].replace(0.0, np.nan)

    # 3. Impute missing clinical measurements.
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].fillna(
        frame[FEATURE_COLUMNS].median() if impute_mode == "median" else frame[FEATURE_COLUMNS].mean()
    )

    # 4. Persist the processed frame for downstream reuse.
    paths.processed_data.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.processed_data / PROCESSED_FILENAME, index=False)

    return frame