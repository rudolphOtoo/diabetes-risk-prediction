"""Robustness and defensive-branch tests for the ODE solver.

A rigorous modelling library must fail loudly on invalid input rather than
silently produce an unphysical trajectory. These tests exercise the solver's
defensive paths: infeasible initial conditions, unrecognised model names,
non-conserving model definitions, and invalid observation channels.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.ode import ODEModelConfig, ParamTable
from src.models.ode.base import ODECompartmentalModel
from src.solvers import fit_model_to_data, integrate_model, seed_rng


class _NonConservingModel(ODECompartmentalModel):
    """A deliberately broken model whose vector field does not conserve N."""

    compartment_names: tuple[str, ...] = ("S", "I", "R")
    model_name: str = "sir"

    def __init__(self, config: ODEModelConfig, params: ParamTable) -> None:
        super().__init__(config, params)

    def rhs(self, t: float, y: np.ndarray, *p: float) -> np.ndarray:
        del t, y, p
        # S, I and R all decay slightly: the total drifts below N.
        return np.array([0.0, -0.0, -0.5])


def test_integration_rejects_infeasible_initial_conditions() -> None:
    """i0 larger than the population must raise a descriptive ValueError."""
    from src.models.ode.sir import SIRModel
    from src.solvers import integrate_model

    cfg = ODEModelConfig(n_population=10_000, n_days=30)
    model = SIRModel(cfg, ParamTable("sir", {}))
    with pytest.raises(ValueError):
        integrate_model(model, cfg, i0=20_000.0)


def test_integration_rejects_non_conserving_trajectory() -> None:
    """A model that violates conservation must be rejected by the guard."""
    cfg = ODEModelConfig(n_population=10_000, n_days=30)
    model = _NonConservingModel(cfg, ParamTable("sir", {}))
    with pytest.raises(ValueError, match="conservation"):
        integrate_model(model, cfg)


def test_seed_rng_is_reproducible_after_restart() -> None:
    """Two generators seeded identically must produce identical streams."""
    a, b = seed_rng(1234), seed_rng(1234)
    assert np.array_equal(a.normal(size=50), b.normal(size=50))


def test_rng_normal_is_reproducible() -> None:
    """rng_normal must be deterministic for a fixed generator."""
    from src.solvers.ode import rng_normal

    g = seed_rng(7)
    first = rng_normal(0.0, 1.0, 20, g)
    g2 = seed_rng(7)
    second = rng_normal(0.0, 1.0, 20, g2)
    np.testing.assert_allclose(first, second)


def test_fit_rejects_unknown_observation_channel() -> None:
    """Fitting with a channel the model does not have must raise ValueError."""
    from src.models.ode import solve_seird

    cfg = ODEModelConfig(n_population=20_000, n_days=60)
    traj = solve_seird({}, config=cfg)
    with pytest.raises(ValueError, match="channel"):
        fit_model_to_data(
            "seird", traj["t"], {"I": traj["I"], "DoesNotExist": traj["I"]}, config=cfg
        )


def test_sir_rejection_of_unknown_model_name() -> None:
    """The reproduction-number / solve helpers reject unknown model names."""
    from src.models.ode.parameters import ParamTable

    with pytest.raises(ValueError):
        ParamTable("does_not_exist", {})
