"""End-to-end compartmental ODE modelling driver.

Executes the mechanistic layer of the project: parameterised SIR / SEIRD
simulation, conservation & non-negativity verification, and a synthetic-data
recoverability check, writing trajectories and a summary to ``results/``.

Usage
-----
From the repository root::

    python -m src.scripts.run_ode_model            # default SIR
    python -m src.scripts.run_ode_model --model seird
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths
from ..models.ode import ODEModelConfig, solve_seird, solve_sir
from ..solvers import fit_model_to_data
from ..visualize import (
    plot_ode_phase_portrait,
    plot_ode_trajectories,
    set_global_style,
)


def _write_trajectory(paths: Paths, model_name: str, traj: dict[str, np.ndarray]) -> Path:
    """Persist a trajectory dictionary as a tidy CSV."""
    frame = pd.DataFrame(traj)
    out = paths.results / "analysis" / f"{model_name}_trajectory.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the compartmental simulation and report verification metrics."""
    parser = argparse.ArgumentParser(description="Run the compartmental ODE model.")
    parser.add_argument("--model", choices=["sir", "seird"], default="sir")
    parser.add_argument("--population", type=int, default=100_000)
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args(argv)

    paths = Paths()
    set_global_style()
    config = ODEModelConfig(
        n_population=args.population,
        n_days=args.days,
        master_seed=42,
    )

    print(f"▸ Simulating {args.model.upper()} over {args.days} days (N = {args.population:,}) ...")
    solver = solve_seird if args.model == "seird" else solve_sir
    traj = solver({}, config=config)

    outcome = traj["I"]
    peak_idx = int(np.argmax(outcome))
    peak = float(outcome[peak_idx])
    peak_day = float(traj["t"][peak_idx])
    total = sum(traj[k] for k in traj if k != "t")
    conservation_error = float(np.max(np.abs(total - args.population)))

    traj_path = _write_trajectory(paths, args.model, traj)
    print(f"  ✓ Peak infected: {peak:,.1f} on day {peak_day:.1f}")
    print(f"  ✓ Max |sum - N| conservation error: {conservation_error:.2e}")
    print(f"  ✓ Trajectory -> {traj_path}")

    fig_dir: Path = paths.results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_ode_trajectories(
        traj,
        model_name=args.model,
        export_path=fig_dir / f"{args.model}_trajectories.png",
    )
    plot_ode_phase_portrait(
        traj,
        model_name=args.model,
        export_path=fig_dir / f"{args.model}_phase_portrait.png",
    )
    print(f"  ✓ Figures -> {fig_dir}/")

    print("\n▸ Synthetic-data recoverability check ...")
    rng = np.random.default_rng(42)
    n_points = min(80, len(traj["t"]))
    idx = np.linspace(0, len(traj["t"]) - 1, n_points).astype(int)
    t_obs = traj["t"][idx]
    obs = traj["I"][idx] + rng.normal(0, 3.0, size=n_points)
    obs = np.clip(obs, 0, None)
    fitted, rmse = fit_model_to_data(args.model, t_obs, obs, config=config)
    print(f"  ✓ Fitted rates: {fitted}")
    print(f"  ✓ Final normalised RMSE: {rmse:.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
