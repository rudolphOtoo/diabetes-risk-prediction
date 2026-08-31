"""Mathematical models for the diabetes risk project.

Two complementary modelling families are provided under ``src.models``:

* ``src.models.ode`` — mechanistic compartmental (ordinary-differential-
  equation) epidemiology supporting simulation, conservation-law
  verification, numerical integration, and parameter estimation from
  synthetic time series.
* ``src.models.classifiers`` — statistical detection of diabetes on the
  cross-sectional Pima cohort (the original scikit-learn pipeline), kept for
  backward compatibility and re-exported at package level so legacy imports
  such as ``from src.models import build_pipeline`` continue to work.
"""

from __future__ import annotations

from .classifiers import build_pipeline, get_param_grid, list_models
from .ode import ODEModelConfig, ParamTable, solve_seird, solve_sir

__all__: list[str] = [
    "build_pipeline",
    "get_param_grid",
    "list_models",
    "ODEModelConfig",
    "ParamTable",
    "solve_seird",
    "solve_sir",
]
