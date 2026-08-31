"""Model specification and ``sklearn.pipeline.Pipeline`` construction.

Every model in this project is wrapped in a scikit-learn ``Pipeline``. This
design guarantees that *all* preprocessing — median imputation
(``SimpleImputer``) **and** standardization (``StandardScaler``) — is applied
*within* each fold of cross-validation, preventing test statistics from leaking
into training and eliminating a particularly common and insidious form of data
leakage.

Model family rationale
----------------------
* **Baseline (Dummy / LogisticRegression)** : any thesis reviewer expects
  comparison against a naïve or linear baseline.
* **Ensemble (RandomForest, GradientBoosting)** : provides the non-linear
  capacity that typically surpasses logistic regression in tabular medical data.
* **LogisticRegression(L2)** : regularised generalised linear model; still the
  workhorse of epidemiological modelling and an interpretable gold standard.

Extensibility
-------------
Adding a new model requires only:
1. Define its name in ``list_models``.
2. Return a new ``Pipeline`` from ``build_pipeline``.
No changes to the evaluation or tuning modules are required.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def list_models() -> list[str]:
    """Return the ordered list of model names used in this experiment.

    Returns
    -------
    list[str]
        Four model identifiers: ``"dummy"``, ``"logistic_regression"``,
        ``"random_forest"``, ``"gradient_boosting"``.
    """
    return ["dummy", "logistic_regression", "random_forest", "gradient_boosting"]


def build_pipeline(model_name: str, random_seed: int = 42) -> Pipeline:
    """Construct a fully contained, leakage-free scikit-learn ``Pipeline``.

    Imputation (``SimpleImputer``) and standardization (``StandardScaler``)
    both live *inside* the pipeline as leading stages, so every preprocessing
    statistic — median, and scaling mean/scale — is fit on training folds only.
    This makes the pipeline strict about data hygiene: no test-set information
    is observed by any preprocessing step or estimator.

    Parameters
    ----------
    model_name : str
        One of the identifiers returned by :func:`list_models`.
    random_seed : int
        Master deterministic seed forwarded to all estimator initialisers that
        are stochastic.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline with three stages: ``"imputer"`` (median), ``"preprocessor"``
        (scaler), and ``"model"`` (the chosen estimator).

    Raises
    ------
    ValueError
        If ``model_name`` does not match any registered identifier.
    """
    leading_stages: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("preprocessor", StandardScaler()),
    ]

    estimators: dict[str, Pipeline] = {
        "dummy": Pipeline(
            [
                *leading_stages,
                ("model", DummyClassifier(strategy="most_frequent", random_state=random_seed)),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                *leading_stages,
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                *leading_stages,
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_split=5,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                *leading_stages,
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.8,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
    }

    if model_name not in estimators:
        available = ", ".join(sorted(estimators.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
    return estimators[model_name]


def get_param_grid(model_name: str) -> dict[str, list]:
    """Return the hyperparameter search space for a given model.

    All keys are prefixed with ``"model__"`` to match the ``Pipeline`` naming
    convention, ensuring the scikit-learn ``Pipeline`` parameter syntax
    (``<stage_name>__<parameter_name>``) is automatically respected.

    Parameters
    ----------
    model_name : str
        One of the identifiers returned by :func:`list_models`.

    Returns
    -------
    dict[str, list]
        Grid for :class:`sklearn.model_selection.GridSearchCV`.
    """
    grid: dict[str, list] = {
        "dummy": {},  # Hyperparameter-free baseline — nothing to tune.
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "random_forest": {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [5, 10, None],
            "model__min_samples_split": [2, 5, 10],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 5],
        },
    }
    if model_name not in grid:
        raise ValueError(f"Unknown model '{model_name}'.")
    return grid[model_name]
