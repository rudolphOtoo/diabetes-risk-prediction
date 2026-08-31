"""Conservation-law and non-negativity tests for the compartmental models.

A closed-population compartmental system must conserve total population
:math:`\\sum_c x_c(t) = N` at every time point, and each compartment must stay
non-negative. These structural invariants are the numerical-analogue core of
the modelling pillar.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.ode import ODEModelConfig, solve_seird, solve_sir


@pytest.fixture(scope="module")
def cfg() -> ODEModelConfig:
    """A small population and short horizon to keep the test suite fast."""
    return ODEModelConfig(n_population=50_000, n_days=120)


@pytest.mark.parametrize(
    "rates",
    [
        {"beta": 0.2, "gamma": 0.1},
        {"beta": 0.35, "gamma": 1 / 7},
        {"beta": 1.5, "gamma": 0.5},
    ],
)
def test_sir_conserves_population(cfg: ODEModelConfig, rates: dict[str, float]) -> None:
    """S + I + R = N must hold to tight tolerance at every time point."""
    traj = solve_sir(rates, config=cfg)
    total = traj["S"] + traj["I"] + traj["R"]
    np.testing.assert_allclose(total, cfg.n_population, rtol=1e-6, atol=1e-6)


def test_sir_states_are_non_negative(cfg: ODEModelConfig) -> None:
    """Each SIR compartment must remain >= 0 across the simulation."""
    traj = solve_sir({"beta": 0.8, "gamma": 0.2}, config=cfg)
    for key in ("S", "I", "R"):
        assert np.all(traj[key] >= 0.0), f"compartment {key} went negative"


def test_sir_initial_conditions(cfg: ODEModelConfig) -> None:
    """Trajectories must start at the configured initial conditions."""
    traj = solve_sir({"beta": 0.35, "gamma": 0.143}, config=cfg)
    n = cfg.n_population
    np.testing.assert_allclose(traj["S"][0], n - 0.01 * n, atol=1e-6)
    np.testing.assert_allclose(traj["I"][0], 0.01 * n, atol=1e-6)


@pytest.mark.parametrize(
    "rates",
    [
        {"beta": 0.4, "sigma": 0.2, "gamma": 0.1, "mu": 0.02},
        {"beta": 0.6, "sigma": 0.3, "gamma": 0.15, "mu": 0.05},
    ],
)
def test_seird_conserves_population(cfg: ODEModelConfig, rates: dict[str, float]) -> None:
    """S + E + I + R + D = N must hold to tight tolerance at every time."""
    traj = solve_seird(rates, config=cfg)
    total = traj["S"] + traj["E"] + traj["I"] + traj["R"] + traj["D"]
    np.testing.assert_allclose(total, cfg.n_population, rtol=1e-6, atol=1e-6)


def test_seird_states_are_non_negative(cfg: ODEModelConfig) -> None:
    """Each SEIRD compartment must remain >= 0 across the simulation."""
    traj = solve_seird({"beta": 0.6, "sigma": 0.3, "gamma": 0.1, "mu": 0.03}, config=cfg)
    for key in ("S", "E", "I", "R", "D"):
        assert np.all(traj[key] >= 0.0), f"compartment {key} went negative"


def test_sir_reproduction_number_threshold(cfg: ODEModelConfig) -> None:
    """R0 = beta/gamma; sub-critical runs do not grow, super-critical do."""
    from src.models.ode import ParamTable
    from src.models.ode.sir import SIRModel

    model = SIRModel(cfg, ParamTable("sir", {"beta": 0.1, "gamma": 0.2}))
    assert model.reproduction_number() < 1.0

    super_critical = SIRModel(cfg, ParamTable("sir", {"beta": 0.4, "gamma": 0.1}))
    assert super_critical.reproduction_number() > 1.0
