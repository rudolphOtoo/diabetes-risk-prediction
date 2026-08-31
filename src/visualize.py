"""Publication-quality visualization helpers.

All figure generation is delegated to explicit functions in this module rather
than being buried inside a notebook. This guarantees that the exact same plot
can be reproduced from any entry point (notebook, script, or pytest).

Design principle
----------------
Each function receives only data and configuration; side effects are limited
to writing files and optionally calling ``matplotlib.pyplot.show()``. This
allows the functions to be unit-tested and called from any context without
implicit state leakage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline

from .data import FEATURE_COLUMNS


def set_global_style() -> None:
    """Apply a consistent, publication-ready seaborn/matplotlib aesthetic.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.figsize": (7, 5),
        }
    )


def plot_target_distribution(
    y: pd.Series,
    *,
    export_path: Path | None = None,
    show: bool = False,
) -> None:
    """Bar chart of the binary class distribution.

    Parameters
    ----------
    y : pandas.Series
        Binary target vector (0 = absence, 1 = diabetes present).
    export_path : Path | None
        File path for saving the figure. None = skip saving.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts().sort_index()
    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        ax=ax,
        hue=counts.index.astype(str),
        palette="viridis",
        legend=False,
    )
    ax.set_xlabel("Outcome (0 = No diabetes, 1 = Diabetes)")
    ax.set_ylabel("Number of patients")
    ax.set_title("Class Distribution in Pima Indians Diabetes Dataset")

    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_feature_distributions(
    X: pd.DataFrame,
    *,
    export_path: Path | None = None,
    show: bool = False,
) -> None:
    """Histogram grid of every predictor feature.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    export_path : Path | None
        File path for saving the figure. None = skip saving.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    None
    """
    n_features = len(X.columns)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(X.columns):
        sns.histplot(X[col], kde=True, ax=axes[i], color="steelblue", edgecolor="white")
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution of Clinical Predictor Variables", fontsize=13, y=1.01)
    fig.tight_layout()

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_correlation_heatmap(
    frame: pd.DataFrame,
    *,
    export_path: Path | None = None,
    show: bool = False,
) -> None:
    """Spearman correlation heatmap between all clinical features + target.

    Parameters
    ----------
    frame : pandas.DataFrame
        The fully processed tabular frame.
    export_path : Path | None
        File path for saving the figure. None = skip saving.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    None
    """
    corr = frame.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Spearman Correlation Matrix (Features + Target)")

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_roc_curves(
    fitted_pipelines: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    export_path: Path | None = None,
    show: bool = False,
) -> None:
    """Overlayed ROC curves for every fitted model on the held-out test set.

    Parameters
    ----------
    fitted_pipelines : dict[str, Pipeline]
        ``{model_name: fitted_pipeline, ...}``.
    X_test : pandas.DataFrame
        Held-out test features.
    y_test : pandas.Series
        Held-out test labels.
    export_path : Path | None
        File path for saving the figure.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for name, pipe in fitted_pipelines.items():
        model_stage = pipe.named_steps["model"]
        if hasattr(model_stage, "predict_proba"):
            RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)

    ax.set_title("Receiver Operating Characteristic (ROC) Curves — Held-out Test Set")
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrices(
    fitted_pipelines: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    export_path: Path | None = None,
    show: bool = False,
) -> None:
    """Side-by-side confusion matrices for all models.

    Parameters
    ----------
    fitted_pipelines : dict[str, Pipeline]
        ``{model_name: fitted_pipeline, ...}``.
    X_test : pandas.DataFrame
        Held-out test features.
    y_test : pandas.Series
        Held-out test labels.
    export_path : Path | None
        File path for saving the figure.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    None
    """
    n_models = len(fitted_pipelines)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, pipe) in zip(axes, fitted_pipelines.items(), strict=True):
        ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax, cmap="Blues")
        ax.set_title(name, fontsize=11, fontweight="bold")

    fig.suptitle("Confusion Matrices on Held-out Test Set", fontsize=13, y=1.03)
    fig.tight_layout()

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_ode_trajectories(
    traj: dict[str, np.ndarray],
    *,
    model_name: str = "ODE",
    export_path: Path | None = None,
    show: bool = False,
) -> None:
    """Stacked time-series trajectories of every compartment in an ODE system.

    Parameters
    ----------
    traj : dict[str, numpy.ndarray]
        Mapping compartment name to a trajectory, plus the ``"t"`` grid.
    model_name : str
        Label used in the title (e.g. ``"SIR"`` or ``"SEIRD"``).
    export_path : Path | None
        File path for saving the figure.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    None
    """
    t = traj["t"]
    compartment_names = [k for k in traj if k != "t"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in compartment_names:
        ax.plot(t, traj[name], lw=2, label=name)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Compartment population")
    ax.set_title(f"{model_name.upper()} Compartment Trajectories")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_ode_phase_portrait(
    traj: dict[str, np.ndarray],
    *,
    model_name: str = "ODE",
    x_name: str = "S",
    y_name: str = "I",
    export_path: Path | None = None,
    show: bool = False,
) -> None:
    """Phase-space portrait of two compartments for an ODE system.

    Parameters
    ----------
    traj : dict[str, numpy.ndarray]
        Mapping compartment name to a trajectory, plus the ``"t"`` grid.
    model_name : str
        Label used in the title (e.g. ``"SIR"`` or ``"SEIRD"``).
    x_name : str
        Compartment plotted on the horizontal axis.
    y_name : str
        Compartment plotted on the vertical axis.
    export_path : Path | None
        File path for saving the figure.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(traj[x_name], traj[y_name], color="crimson", lw=2)
    ax.scatter(
        traj[x_name][0],
        traj[y_name][0],
        color="darkblue",
        s=60,
        zorder=5,
        label="Start",
    )
    ax.scatter(
        traj[x_name][-1],
        traj[y_name][-1],
        color="seagreen",
        s=60,
        zorder=5,
        label="End",
    )
    ax.set_xlabel(f"{x_name} population")
    ax.set_ylabel(f"{y_name} population")
    ax.set_title(f"{model_name.upper()} Phase Portrait  ({x_name}–{y_name} plane)")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_feature_importance(
    fitted_pipeline: Pipeline,
    feature_names: list[str] | None = None,
    *,
    export_path: Path | None = None,
    show: bool = False,
) -> None:
    """Horizontal bar chart of feature importances (tree-based models only).

    Parameters
    ----------
    fitted_pipeline : sklearn.pipeline.Pipeline
        A fitted pipeline whose ``model`` step has a ``feature_importances_``
        attribute (e.g., ``RandomForestClassifier``).
    feature_names : list[str] | None
        Feature labels. Defaults to the canonical Pima feature names.
    export_path : Path | None
        File path for saving the figure.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    None

    Raises
    ------
    AttributeError
        If the fitted estimator does not expose ``feature_importances_``.
    """
    model_stage = fitted_pipeline.named_steps["model"]
    if not hasattr(model_stage, "feature_importances_"):
        raise AttributeError(
            "The fitted estimator does not expose a 'feature_importances_' "
            "attribute. Use this function only with tree-based models."
        )

    feature_names = feature_names or FEATURE_COLUMNS
    importances = model_stage.feature_importances_
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(feature_names)[order], importances[order], color="teal")
    ax.set_xlabel("Feature Importance (Gini / impurity-based)")
    ax.set_title("Top Predictors of Diabetes Status")

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path)
    if show:
        plt.show()
    plt.close(fig)
