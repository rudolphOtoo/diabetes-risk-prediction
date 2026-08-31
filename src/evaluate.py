"""Model evaluation and metric reporting.

This module implements the complete evaluation suite used throughout the
project. Every metric is computed *only* on data the model has never seen,
enforcing the strict train-validation-test protocol.

Rationale for the chosen metrics
--------------------------------
* **ROC-AUC** (primary): measures the trade-off between the true positive
  rate (sensitivity) and the false positive rate (1 − specificity) across
  *all* classification thresholds. It is invariant to class distribution,
  making it the gold standard for medical screening problems where class
  prevalence varies across populations.

* **F1-score** (secondary): the harmonic mean of precision and recall. It is
  more sensitive than ROC-AUC to the chosen threshold and to class imbalance,
  and therefore functions as an important reliability check: if a model scores
  well on ROC-AUC but poorly on F1, this is evidence of threshold sensitivity
  rather than genuine discriminative ability.

* **Accuracy** (supplementary): reported for transparency, but never used as
  the primary criterion, because a naïve model that always predicts the
  majority class can achieve > 65 % accuracy on this dataset while having
  zero clinical utility.

* **Confusion matrix**: explicitly surfaces false negatives (missed diabetics)
  and false positives, both of which carry direct clinical consequences.

All metrics are exported to CSV for downstream comparison.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def evaluate_model(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    model_name: str = "model",
    export_path: Path | None = None,
) -> dict[str, float]:
    """Compute and optionally persist a complete evaluation profile.

    Parameters
    ----------
    fitted_pipeline : sklearn.pipeline.Pipeline
        A fully trained pipeline (including a scaler stage).
    X_test : pandas.DataFrame
        Held-out test feature matrix.
    y_test : pandas.Series
        Held-out test labels.
    model_name : str
        Human-readable identifier (used only in print output and CSV exports).
    export_path : Path | None
        If given, the metrics are saved as a two-column CSV here.

    Returns
    -------
    dict[str, float]
        Metric name → value mapping.
    """
    # Determine the estimator name (last stage) to call predict_proba.
    model_stage = fitted_pipeline.named_steps["model"]
    supports_proba = hasattr(model_stage, "predict_proba")

    y_pred: np.ndarray = fitted_pipeline.predict(X_test)

    metrics: dict[str, float] = {
        "model_name": model_name,  # kept for CSV readability
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "matthews_corrcoef": round(matthews_corrcoef(y_test, y_pred), 4),
    }

    if supports_proba:
        y_proba: np.ndarray = fitted_pipeline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)
    else:
        metrics["roc_auc"] = float("nan")

    # --- Console report ---------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Evaluation profile : {model_name}")
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        if k != "model_name":
            print(f"  {k:<24} {v:.4f}")
    print(f"{'=' * 60}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print()

    # --- Persistence -------------------------------------------------------
    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        row_df = pd.DataFrame([metrics])
        row_df.to_csv(export_path, index=False)

    return metrics


def aggregate_benchmark_tables(
    results: dict[str, dict[str, float]],
    export_path: Path | None = None,
) -> pd.DataFrame:
    """Turn a dict of per-model metric dicts into a tidy comparison table.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        ``{model_name: {metric_name: value, ...}, ...}`` as returned
        by repeated calls to :func:`evaluate_model`.
    export_path : Path | None
        If given, save the table as a CSV.

    Returns
    -------
    pandas.DataFrame
        Models as rows, metrics as columns; sorted descending by ``roc_auc``.
    """
    rows: list[dict[str, float]] = []
    for model_name, metric_dict in results.items():
        row = metric_dict.copy()
        row["model_name"] = model_name
        rows.append(row)

    table: pd.DataFrame = pd.DataFrame(rows).set_index("model_name")
    table = table.sort_values(by="roc_auc", ascending=False)

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(export_path)

    return table
