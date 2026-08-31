#!/usr/bin/env python3
"""End-to-end reproducible ML pipeline for diabetes risk classification.

Usage
-----
From the repository root::

    python -m src.scripts.run_pipeline

Or equivalently::

    python src/scripts/run_pipeline.py

This script:
    1. Downloads the Pima Indians Diabetes Database (if not already cached).
    2. Preprocesses and feature-engineers the data.
    3. Splits into train / validation / test (stratified, deterministic).
    4. Trains and cross-validates four models: Dummy, LogisticRegression,
       RandomForest, GradientBoosting.
    5. Evaluates every model on the held-out test set.
    6. Saves all metrics, figures, and the tuned final model.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a module or directly as a script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.config import Paths, Settings
from src.data import process_data
from src.evaluate import aggregate_benchmark_tables, evaluate_model
from src.features import split_features_target
from src.models import list_models
from src.tune import tune_model
from src.visualize import (
    plot_confusion_matrices,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_feature_importance,
    plot_roc_curves,
    plot_target_distribution,
    set_global_style,
)


def main() -> None:
    """Execute the full reproducible pipeline."""
    settings = Settings()
    paths = Paths()
    set_global_style()

    # ── Stage 1 ──────────────────────────────────────────────────────────
    print("▸ Stage 1/5: Acquiring and preprocessing data ...")
    frame: pd.DataFrame = process_data(paths)
    print(f"  Shape after preprocessing: {frame.shape}")

    # ── Stage 2 ──────────────────────────────────────────────────────────
    print("\n▸ Stage 2/5: Splitting data (train / val / test) ...")
    splits: dict[str, pd.DataFrame | pd.Series] = split_features_target(frame, settings=settings)
    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]

    print(f"  Train : {X_train.shape[0]:>4d} samples  | class 1 prevalence = {y_train.mean():.3f}")
    print(f"  Val   : {X_val.shape[0]:>4d} samples  | class 1 prevalence = {y_val.mean():.3f}")
    print(f"  Test  : {X_test.shape[0]:>4d} samples  | class 1 prevalence = {y_test.mean():.3f}")

    # ── Stage 3 ──────────────────────────────────────────────────────────
    print("\n▸ Stage 3/5: Training and tuning models ...")
    tuned_pipelines: dict[str, pd.Series] = {}
    best_params_map: dict[str, dict] = {}

    for model_name in list_models():
        fitted, params = tune_model(
            model_name,
            X_train,
            y_train,
            settings=settings,
        )
        tuned_pipelines[model_name] = fitted
        best_params_map[model_name] = params

    # ── Stage 4 ──────────────────────────────────────────────────────────
    print("\n▸ Stage 4/5: Evaluating on held-out test set ...")
    results: dict[str, dict[str, float]] = {}
    for model_name, fitted_pipe in tuned_pipelines.items():
        results[model_name] = evaluate_model(
            fitted_pipe,
            X_test,
            y_test,
            model_name=model_name,
            export_path=paths.reports / f"{model_name}_test_metrics.csv",
        )

    comparison_table: pd.DataFrame = aggregate_benchmark_tables(
        results,
        export_path=paths.reports / "benchmark_comparison.csv",
    )
    print("\n  ── Benchmark comparison ──")
    print(comparison_table.to_string())

    # ── Stage 5 ──────────────────────────────────────────────────────────
    print("\n▸ Stage 5/5: Generating figures ...")
    fig_dir: Path = paths.figures

    plot_target_distribution(y_train, export_path=fig_dir / "01_target_distribution.png")
    plot_correlation_heatmap(frame, export_path=fig_dir / "02_correlation_heatmap.png")
    plot_feature_distributions(X_train, export_path=fig_dir / "03_feature_distributions.png")
    plot_roc_curves(
        tuned_pipelines,
        X_test,
        y_test,
        export_path=fig_dir / "04_roc_curves.png",
    )
    plot_confusion_matrices(
        tuned_pipelines,
        X_test,
        y_test,
        export_path=fig_dir / "05_confusion_matrices.png",
    )
    plot_feature_importance(
        tuned_pipelines["gradient_boosting"],
        export_path=fig_dir / "06_feature_importance.png",
    )

    # Persist the tuned Gradient Boosting pipeline (best model by default)
    import joblib

    joblib.dump(
        tuned_pipelines["gradient_boosting"],
        paths.models / "best_gradient_boosting_pipeline.joblib",
    )
    print("  ✓ Best model saved to models/best_gradient_boosting_pipeline.joblib")
    print("\nDone.")


if __name__ == "__main__":
    main()
