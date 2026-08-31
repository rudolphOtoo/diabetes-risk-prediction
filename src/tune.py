"""Hyperparameter tuning with stratified cross-validation.

This module provides :func:`tune_model`, which performs a grid search with
early stopping semantics (stratified K-fold CV) and refits the best
hyperparameters on the full training set.

Strict data-hygiene protocol
----------------------------
Hyperparameter optimisation is always performed on the **training** split
only. The validation (dev) set is held aside exclusively for this purpose.
The held-out **test** set is reserved exclusively for the final, irrevocable
evaluation reported in the manuscript.

This strict partitioning eliminates two of the most common methodological
pitfalls in published medical ML studies:

1. **Information leakage via re-fitting**: if the model is retrained on data
   that has already been seen, the reported test performance is optimistically
   biased and cannot be generalised to unseen patients.
2. **Selection bias from peeking**: if hyperparameters are selected based on
   test-set performance, the reported results are an unbiased estimate of
   optimisation performance, not generalisation performance.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .config import Settings
from .features import get_kfold
from .models import build_pipeline, get_param_grid


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    settings: Settings | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune a named model via GridSearchCV with stratified cross-validation.

    Parameters
    ----------
    model_name : str
        One of the identifiers returned by :func:`src.models.list_models`.
    X_train : pandas.DataFrame
        Training feature matrix.
    y_train : pandas.Series
        Training labels.
    settings : Settings | None
        Global project settings. The ``scoring`` attribute must be a valid
        scikit-learn scorer string.

    Returns
    -------
    tuple[sklearn.pipeline.Pipeline, dict[str, Any]]
        A fitted pipeline whose internal ``model`` step has been refitted with
        the best cross-validation hyperparameters, and a dict of the best
        parameters found.
    """
    settings = settings or Settings()
    pipeline: Pipeline = build_pipeline(model_name, random_seed=settings.random_seed)
    param_grid: dict[str, list] = get_param_grid(model_name, random_seed=settings.random_seed)
    cv_splitter = get_kfold(settings)

    if param_grid:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_splitter,
            scoring=settings.scoring,
            refit=True,
            n_jobs=-1,
            verbose=0,
        )
    else:
        # No hyperparameters to tune; just fit the pipeline.
        search = GridSearchCV(
            estimator=pipeline,
            param_grid={},  # trivial grid; single point
            cv=cv_splitter,
            scoring=settings.scoring,
            refit=True,
            n_jobs=-1,
            verbose=0,
        )

    search.fit(X_train, y_train)
    best_pipeline: Pipeline = search.best_estimator_
    best_params: dict[str, Any] = search.best_params_

    print(
        f"[tune_model] {model_name}: best CV {settings.scoring} = "
        f"{search.best_score_:.4f} | best params = {best_params}"
    )
    return best_pipeline, best_params
