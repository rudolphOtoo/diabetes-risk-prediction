"""Feature splitting, scaling, and cross-validation utilities.

This module owns the transition from a raw-ish tabular frame to a
train/validation/test structure suitable for model training. All random seeds
are propagated from a single master seed, guaranteeing full reproducibility.

Important design decision
-------------------------
A **three-way split** is applied (train / validation / held-out test).

* The held-out ``test`` set is touched **once** during final evaluation.
* The ``validation`` set is used exclusively during hyperparameter tuning
  to guard against selection bias and overfitting to training performance.

A reviewer reproducing this pipeline should expect the exact same train,
validation, and test membership lists when the master seed is unchanged.
"""

from __future__ import annotations

import zlib

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import Settings
from .data import FEATURE_COLUMNS, TARGET_COLUMN


def _extract_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the frame into a feature matrix and a binary target vector.

    Parameters
    ----------
    frame : pandas.DataFrame
        Fully processed frame containing the clinical feature columns and
        the ``Outcome`` target column.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        ``(X, y)`` with matched row indices.

    Raises
    ------
    KeyError
        If the expected feature or target columns are not present.
    """
    missing_features = [col for col in FEATURE_COLUMNS if col not in frame.columns]
    if missing_features:
        raise KeyError(f"Missing required feature columns in the input frame: {missing_features}")
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found. "
            "Did you forget to run data.process_data() first?"
        )
    X: pd.DataFrame = frame[FEATURE_COLUMNS].copy()
    y: pd.Series = frame[TARGET_COLUMN].astype(int).copy()
    return X, y


def _derived_random_state(master_seed: int, context_label: str = "default") -> int:
    """Derive a child seed deterministically from the master seed.

    This guarantees that multiple independent operations (e.g., three different
    ``train_test_split`` calls) never alias the same RNG state, while still
    being fully reproducible from a single externally visible integer.

    Parameters
    ----------
    master_seed : int
        The global project-level seed.
    context_label : str
        A human-readable disambiguator (e.g., ``"train_test"``).

    Returns
    -------
    int
        A deterministic 32-bit integer seed suitable for scikit-learn's
        ``random_state`` parameter.
    """
    # NOTE: Python's built-in `hash()` is salted per-process (hash
    # randomization via PYTHONHASHSEED), so it MUST NOT be used here -- it would
    # produce different splits across interpreters/machines. `zlib.crc32` over
    # a stable byte encoding is a deterministic, cross-process 32-bit hash.
    key: bytes = f"{master_seed}:{context_label}".encode()
    return zlib.crc32(key) & 0xFFFFFFFF


def split_features_target(
    frame: pd.DataFrame,
    *,
    settings: Settings | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Produce a stratified, reproducible train/validation/test split.

    Parameters
    ----------
    frame : pandas.DataFrame
        Fully processed tabular frame with the nine Pima features and
        the binary ``Outcome`` column.
    settings : Settings | None
        Global project settings controlling the split ratios and random
        seed. Uses defaults when not supplied.

    Returns
    -------
    dict[str, pandas.DataFrame | pandas.Series]
        Keys: ``X_train, X_val, X_test, y_train, y_val, y_test``.

    Notes
    -----
    Stratification on the target preserves the baseline prevalence of diabetes
    across all three partitions, mitigating a particularly insidious class of
    data leakage.
    """
    settings = settings or Settings()
    X, y = _extract_xy(frame)

    # --- Step 1: carve out the held-out test set (20 %) ---
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        stratify=y,
        random_state=_derived_random_state(settings.random_seed, "train_test"),
    )

    # --- Step 2: split development into train + validation ---
    relative_val_ratio: float = settings.val_size / (1.0 - settings.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=relative_val_ratio,
        stratify=y_dev,
        random_state=_derived_random_state(settings.random_seed, "train_val"),
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def get_kfold(
    settings: Settings | None = None,
) -> StratifiedKFold:
    """Return a stratified K-fold generator for cross-validation.

    Parameters
    ----------
    settings : Settings | None
        Global project settings controlling the number of folds.

    Returns
    -------
    sklearn.model_selection.StratifiedKFold
        An iterable of ``(train_idx, val_idx)`` tuples.
    """
    settings = settings or Settings()
    return StratifiedKFold(
        n_splits=settings.n_splits,
        shuffle=True,
        random_state=_derived_random_state(settings.random_seed, "kfold"),
    )
