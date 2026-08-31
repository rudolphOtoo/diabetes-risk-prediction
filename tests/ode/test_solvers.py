"""Numerical-convergence tests for the ODE solver.

The integrated trajectories must be robust to the requested solver
tolerances: tightening ``rtol``/``atol`` should leave the solution essentially
unchanged (to the looser tolerance), demonstrating that the discretisation
error is controlled and that conservation is not an artefact of a coarse grid.
"""

from __future__ import annotations

import numpy as np

from src.models.ode import ODEModelConfig, solve_sir
from src.solvers import seed_rng


def test_integration_is_stable_under_tolerance_refinement() -> None:
    """Crude vs. refined tolerance runs must agree within tight bounds."""
    coarse = ODEModelConfig(n_population=50_000, n_days=120, rtol=1e-4, atol=1e-6)
    fine = ODEModelConfig(n_population=50_000, n_days=120, rtol=1e-10, atol=1e-12)

    traj_coarse = solve_sir({"beta": 0.35, "gamma": 0.143}, config=coarse)
    traj_fine = solve_sir({"beta": 0.35, "gamma": 0.143}, config=fine)

    # Peak infected must agree within 0.5% between coarse and fine runs.
    peak_coarse = float(traj_coarse["I"].max())
    peak_fine = float(traj_fine["I"].max())
    assert abs(peak_coarse - peak_fine) / peak_fine < 5e-3


def test_conservation_holds_even_with_coarse_tolerance() -> None:
    """Even a coarse solver must not break the structural invariant."""
    cfg = ODEModelConfig(n_population=50_000, n_days=120, rtol=1e-4, atol=1e-6)
    traj = solve_sir({"beta": 0.5, "gamma": 0.15}, config=cfg)
    total = traj["S"] + traj["I"] + traj["R"]
    np.testing.assert_allclose(total, cfg.n_population, rtol=1e-4, atol=1e-4)


def test_seeded_rng_is_reproducible() -> None:
    """Two generators from the same seed must produce identical draws."""
    g1 = seed_rng(42)
    g2 = seed_rng(42)
    assert np.array_equal(g1.normal(size=100), g2.normal(size=100))
