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

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import Settings
from src.data import FEATURE_COLUMNS, TARGET_COLUMN, process_data
from src.evaluate import aggregate_benchmark_tables, evaluate_model
from src.features import _derived_random_state, split_features_target
from src.models import build_pipeline, list_models
from src.tune import tune_model


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


def test_process_data_retains_missing_values(processed_frame: pd.DataFrame) -> None:
    """Cleaning repairs pathological zeros to NaN but does NOT impute them.

    Imputation is intentionally deferred to each model's ``Pipeline`` (a
    ``SimpleImputer`` leading stage) so that imputation statistics are fitted on
    training folds only — the whole point of the zero-leakage design.
    """
    assert processed_frame[FEATURE_COLUMNS].isna().any().any()


def test_pipeline_imputes_missing_values(processed_frame: pd.DataFrame) -> None:
    """The Pipeline's leading SimpleImputer must yield a finite feature matrix."""
    splits = split_features_target(processed_frame)
    X_train = splits["X_train"]

    pipe: Pipeline = build_pipeline("logistic_regression", random_seed=42)
    imputer = pipe.named_steps["imputer"]
    imputed = imputer.fit_transform(X_train)

    assert not pd.isna(imputed).any()


def test_class_balance_preserved(processed_frame: pd.DataFrame) -> None:
    """Cleaning must not alter the outcome target counts."""
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


def test_derived_seed_is_stable_across_processes() -> None:
    """Derived seeds must NOT depend on Python's per-process hash salt.

    A regression to ``hash((seed, context))`` would pass an in-process test
    yet silently break reproducibility across machines (PYTHONHASHSEED
    randomization). Asserting the literal CRC32 value pins it to a fixed,
    cross-process-stable derivation.
    """
    assert _derived_random_state(42, "train_test") == 2826799802
    assert _derived_random_state(42, "train_val") == _derived_random_state(42, "train_val")


def test_split_is_reproducible_across_processes() -> None:
    """Re-running the pipeline in a fresh interpreter must yield identical splits."""
    code = (
        "from src.data import process_data;"
        "from src.features import split_features_target;"
        "from src.config import Settings;"
        "s=split_features_target(process_data(), settings=Settings());"
        "print(list(s['X_test'].index))"
    )
    first = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    second = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert first == second


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


def test_tune_model_returns_fitted_pipeline_and_params() -> None:
    """Tuning must return a fitted pipeline plus a (possibly empty) param dict."""
    frame = process_data()
    splits = split_features_target(frame)
    X_train, y_train = splits["X_train"], splits["y_train"]

    fitted, params = tune_model("logistic_regression", X_train, y_train, settings=Settings())

    assert isinstance(fitted, Pipeline)
    assert isinstance(params, dict)
    assert fitted.named_steps["model"] is not None


def test_tune_model_dummy_has_no_params() -> None:
    """The dummy baseline is hyperparameter-free and should return {}."""
    frame = process_data()
    splits = split_features_target(frame)
    X_train, y_train = splits["X_train"], splits["y_train"]

    _, params = tune_model("dummy", X_train, y_train, settings=Settings())
    assert params == {}


def test_aggregate_benchmark_tables_sorts_by_roc_auc() -> None:
    """The aggregate table must sort models descending by ROC-AUC."""
    table = aggregate_benchmark_tables(
        {
            "model_a": {"roc_auc": 0.50},
            "model_b": {"roc_auc": 0.90},
            "model_c": {"roc_auc": 0.70},
        }
    )
    # Sorted descending → best ROC-AUC first.
    assert table.index.tolist() == ["model_b", "model_c", "model_a"]
    assert table.index.name == "model_name"


def test_evaluate_model_exports_csv(tmp_path: Path) -> None:
    """Evaluation should persist a metric CSV when an export path is given."""
    frame = process_data()
    splits = split_features_target(frame)
    X_test, y_test = splits["X_test"], splits["y_test"]
    pipe = build_pipeline("logistic_regression", random_seed=42)
    pipe.fit(splits["X_train"], splits["y_train"])

    out = tmp_path / "metrics.csv"
    metrics = evaluate_model(pipe, X_test, y_test, model_name="logistic", export_path=out)

    assert out.exists()
    assert "roc_auc" in metrics
