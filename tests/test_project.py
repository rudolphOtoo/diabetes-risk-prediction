"""Unit tests for the project's data and modelling utilities.

Run from the repository root with::

    python -m pytest tests/ -v

These tests validate the most failure-prone invariants:
* preprocessing preserves the expected shape and class balance,
* the train/validation/test split is stratified,
* deterministic seeds actually reproduce identical splits,
* every registered model trains and produces a valid ROC-AUC on a toy set.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import Settings
from src.data import FEATURE_COLUMNS, TARGET_COLUMN, process_data
from src.evaluate import evaluate_model
from src.features import split_features_target
from src.models import build_pipeline, list_models


@pytest.fixture(scope="module")
def processed_frame() -> pd.DataFrame:
    """Provide a single processed frame reused across tests."""
    return process_data()


def test_process_data_shape(processed_frame: pd.DataFrame) -> None:
    """The Pima dataset has 768 rows and 9 columns (8 features + target)."""
    assert processed_frame.shape == (768, 9)


def test_preprocessing_feature_columns(processed_frame: pd.DataFrame) -> None:
    """Every canonical feature column must be present and numeric."""
    assert set(FEATURE_COLUMNS).issubset(processed_frame.columns)
    assert all(pd.api.types.is_numeric_dtype(t) for t in processed_frame[FEATURE_COLUMNS].dtypes)


def test_no_missing_values_after_imputation(processed_frame: pd.DataFrame) -> None:
    """Median imputation of pathological zeros leaves no NaNs."""
    assert not processed_frame[FEATURE_COLUMNS].isna().any().any()


def test_class_balance_preserved_after_imputation(processed_frame: pd.DataFrame) -> None:
    """Patch that imputation does not alter the target counts."""
    counts = processed_frame[TARGET_COLUMN].value_counts().sort_index()
    assert counts[0] == 500
    assert counts[1] == 268


def test_split_is_stratified() -> None:
    """Each partition must retain roughly the population prevalence (0.35)."""
    frame = process_data()
    splits = split_features_target(frame)
    for key in ["y_train", "y_val", "y_test"]:
        prevalence = float(splits[key].mean())
        assert 0.30 < prevalence < 0.40, f"{key} prevalence out of range: {prevalence:.3f}"


def test_split_is_reproducible() -> None:
    """The same seed must produce byte-identical index memberships."""
    frame = process_data()
    settings = Settings()
    split_a = split_features_target(frame, settings=settings)
    split_b = split_features_target(frame, settings=settings)

    for key in ["X_train", "X_val", "X_test"]:
        assert (split_a[key].index == split_b[key].index).all(), f"{key} indices differ"


@pytest.mark.parametrize("model_name", list_models())
def test_each_model_trains_and_scores(model_name: str) -> None:
    """Every registered pipeline must fit and yield a finite evaluation profile."""
    frame = process_data()
    splits = split_features_target(frame)
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    pipe: Pipeline = build_pipeline(model_name, random_seed=42)
    pipe.fit(X_train, y_train)
    metrics = evaluate_model(pipe, X_test, y_test, model_name=model_name)

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_build_pipeline_rejects_unknown_name() -> None:
    """Unknown model identifiers must raise a descriptive ValueError."""
    with pytest.raises(ValueError):
        build_pipeline("not_a_real_model")
