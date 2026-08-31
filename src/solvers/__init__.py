"""Numerical routines (ODE integration and parameter estimation).

Public API
----------
* :func:`integrate_model` — deterministic :func:`scipy.integrate.solve_ivp`
  wrapper with conservation & non-negativity verification.
* :func:`seed_rng` — reproducible NumPy RNG factory.
* :func:`fit_model_to_data` — least-squares parameter recovery from a
  synthetic time series (synthetic-data recoverability test support).
"""

from __future__ import annotations

from .ode import fit_model_to_data, integrate_model, seed_rng

__all__: list[str] = ["fit_model_to_data", "integrate_model", "seed_rng"]
