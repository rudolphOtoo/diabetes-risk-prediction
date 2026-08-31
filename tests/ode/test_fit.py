"""Synthetic-data recoverability tests for the compartmental layer.

The strongest test of a mechanistic model is that its parameters can be
*correctly re-estimated* from data the model itself generated. Here we
simulate trajectories at known rates, add reproducible Gaussian noise, and
confirm the least-squares fit recovers the generating rates within tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.ode import ODEModelConfig, solve_seird, solve_sir
from src.solvers import fit_model_to_data, seed_rng


@pytest.fixture(scope="module")
def cfg() -> ODEModelConfig:
    """A compact config to keep the fitting tests fast."""
    return ODEModelConfig(n_population=50_000, n_days=150, n_points=150)


def test_sir_parameters_recoverable_from_noiseless_data(cfg: ODEModelConfig) -> None:
    """With no noise, least squares must recover beta and gamma exactly."""
    truth = {"beta": 0.25, "gamma": 0.12}
    traj = solve_sir(truth, config=cfg)
    fitted, rmse = fit_model_to_data(
        "sir", traj["t"], traj["I"], config=cfg, initial_guess={"beta": 0.1, "gamma": 0.02}
    )
    assert fitted["beta"] == pytest.approx(truth["beta"], rel=1e-2)
    assert fitted["gamma"] == pytest.approx(truth["gamma"], rel=1e-2)
    assert rmse < 1e-3


def test_sir_recoverable_under_mild_noise(cfg: ODEModelConfig) -> None:
    """With mild reproducible noise, recovery stays within a few percent."""
    rng = seed_rng(7)
    truth = {"beta": 0.3, "gamma": 0.1}
    traj = solve_sir(truth, config=cfg)
    scale = 0.02 * float(np.max(traj["I"]))
    obs = traj["I"] + rng.normal(0.0, scale, size=len(traj["I"]))
    obs = np.clip(obs, 0.0, None)

    fitted, _ = fit_model_to_data(
        "sir", traj["t"], obs, config=cfg, initial_guess={"beta": 0.1, "gamma": 0.02}
    )
    assert fitted["beta"] == pytest.approx(truth["beta"], rel=0.10)
    assert fitted["gamma"] == pytest.approx(truth["gamma"], rel=0.10)


def test_fitted_parameters_respect_bounds() -> None:
    """The optimiser must always return parameters within admissible bounds."""
    cfg = ODEModelConfig(n_population=20_000, n_days=100)
    traj = solve_sir({"beta": 0.4, "gamma": 0.18}, config=cfg)
    fitted, _ = fit_model_to_data("sir", traj["t"], traj["I"], config=cfg)
    for key, value in fitted.items():
        assert 0.0 <= value < 6.0, f"{key} outside admissible range: {value}"


def test_seird_parameters_recoverable_noiseless(cfg: ODEModelConfig) -> None:
    """SEIRD rates are recoverable when the *full* surveillance state is
    observed. Fitting all four channels (I, R, D, E) jointly identifies all
    rates; the infected channel alone (or even I+R) leaves the latent-stage
    rates confounded — a documented identifiability property."""
    truth = {"beta": 0.45, "sigma": 0.25, "gamma": 0.12, "mu": 0.02}
    traj = solve_seird(truth, config=cfg)
    obs_channels = {"I": traj["I"], "R": traj["R"], "D": traj["D"], "E": traj["E"]}
    fitted, _ = fit_model_to_data(
        "seird",
        traj["t"],
        obs_channels,
        config=cfg,
        initial_guess={"beta": 0.1, "sigma": 0.1, "gamma": 0.03, "mu": 0.001},
    )
    for key, true in truth.items():
        assert fitted[key] == pytest.approx(true, rel=5e-2), key
