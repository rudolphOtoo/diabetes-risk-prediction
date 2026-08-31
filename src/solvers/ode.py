"""Deterministic ODE integration and least-squares parameter recovery.

This module owns the numerical core of the compartmental layer:

* :func:`integrate_model` wraps :func:`scipy.integrate.solve_ivp` with the
  tolerances given in the model config and **verifies** the two structural
  invariants — closed-population conservation and non-negativity — on every
  integrated trajectory before returning it.
* :func:`fit_model_to_data` recovers model rates from a synthetic time series
  by non-linear least squares, the basis of the synthetic-data recoverability
  tests.

The RNG is seeded deterministically from a master seed, so any stochastic
component (e.g. observational noise) is reproducible.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.stats import norm

from ..models.ode.base import ODECompartmentalModel
from ..models.ode.parameters import ODEModelConfig


def seed_rng(master_seed: int) -> np.random.Generator:
    """Return a deterministic :class:`numpy.random.Generator`.

    Parameters
    ----------
    master_seed : int
        Non-negative integer used to seed the generator.

    Returns
    -------
    numpy.random.Generator
        A reproducible PCG64-backed generator.
    """
    master_seed = int(master_seed) & 0xFFFFFFFF
    return np.random.default_rng(master_seed)


def integrate_model(
    model: ODECompartmentalModel,
    config: ODEModelConfig | None = None,
    **initial_extra: float,
) -> dict[str, np.ndarray]:
    """Integrate a compartmental model over the configured time grid.

    Parameters
    ----------
    model : ODECompartmentalModel
        A model exposing ``rhs`` and ``compartment_names``.
    config : ODEModelConfig | None
        Numerical / population settings (defaults to a standard config).
    **initial_extra : float
        Extra initial-state keys (e.g. ``e0``, ``i0``) forwarded as initial
        conditions.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping compartment name to a ``(n_points,)`` trajectory, plus a
        ``"t"`` key holding the time grid.

    Raises
    ------
    ValueError
        If the integrated trajectory violates conservation or non-negativity.
    """
    config = config or ODEModelConfig()
    n = int(config.n_population)

    i0 = float(initial_extra.pop("i0", 0.01 * n))
    e0 = float(initial_extra.pop("e0", 0.0))
    deaths0 = float(initial_extra.pop("deaths", 0.0))

    susceptible0 = n - i0 - e0 - deaths0
    if susceptible0 < 0:
        raise ValueError("Initial conditions exceed the population size.")

    if model.model_name == "sir":
        y0 = np.array([susceptible0, i0, 0.0], dtype=float)
    elif model.model_name == "seird":
        y0 = np.array([susceptible0, e0, i0, 0.0, deaths0], dtype=float)
    else:
        raise ValueError(f"Unknown model '{model.model_name}'.")

    t_span = (0.0, float(config.n_days))
    t_eval = config.time_grid

    kw: dict[str, float] = {"rtol": config.rtol, "atol": config.atol}
    if config.max_step is not None:
        kw["max_step"] = config.max_step

    sol = solve_ivp(
        model.rhs,
        t_span,
        y0,
        t_eval=t_eval,
        method="LSODA",
        **kw,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    y = sol.y
    if y.shape[1] != len(t_eval):
        dense = solve_ivp(
            model.rhs,
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            **kw,
        )
        y = dense.y

    total = y.sum(axis=0)
    if not np.allclose(total, n, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "Numerical integration violated population conservation: "
            f"max |sum - N| = {float(np.max(np.abs(total - n))):.3e}."
        )
    if np.any(y < -1e-8):
        raise ValueError("Numerical integration produced a negative compartment.")

    traj: dict[str, np.ndarray] = {"t": t_eval}
    for name, vec in zip(model.compartment_names, y, strict=True):
        traj[name] = vec
    return traj


def _model_factory(
    model_name: str,
) -> tuple[ODEModelConfig, Callable[..., ODECompartmentalModel]]:
    """Return a config and model factory for the given ODE model name."""
    from ..models.ode.parameters import ODEModelConfig as _C
    from ..models.ode.seird import SEIRDModel
    from ..models.ode.sir import SIRModel

    registry: dict[str, Callable[..., ODECompartmentalModel]] = {
        "sir": SIRModel,
        "seird": SEIRDModel,
    }
    if model_name not in registry:
        raise ValueError(f"Unknown ODE model '{model_name}'.")
    return _C(), registry[model_name]


def fit_model_to_data(
    model_name: str,
    t_obs: np.ndarray,
    obs: np.ndarray | dict[str, np.ndarray],
    *,
    initial_guess: dict[str, float] | None = None,
    config: ODEModelConfig | None = None,
) -> tuple[dict[str, float], float]:
    """Recover model rates from a synthetic time series by least squares.

    The objective is the residual between the integrated compartment
    trajectories and the observed data, summed across time and — when a
    ``dict`` of channels is supplied — across all observed compartments:

    .. math::

        \\min_{\\theta} \\; \\sum_{c \\in \\mathcal{C}} \\sum_i
        \\left( y^{c}_{\\mathrm{obs}}(t_i) -
        y^{c}_{\\mathrm{model}}(t_i; \\theta) \\right)^2,

    subject to ``theta`` staying within the admissible parameter bounds.
    Fitting *multiple* channels jointly (e.g. infected **and** recovered)
    is essential for identifiability in models with a latent stage (SEIRD),
    where the infected channel alone under-determines the rates.

    Parameters
    ----------
    model_name : str
        ``"sir"`` or ``"seird"``.
    t_obs : numpy.ndarray
        Observation time points.
    obs : numpy.ndarray | dict[str, numpy.ndarray]
        Array of infected counts, or ``{compartment: array}`` of one or more
        observed channels.
    initial_guess : dict[str, float] | None
        Starting rates for the optimiser (defaults to the parameter table
        defaults).
    config : ODEModelConfig | None
        Population and solver settings.

    Returns
    -------
    tuple[dict[str, float], float]
        ``(fitted_rates, final_rmse)`` where ``fitted_rates`` maps each
        parameter name to its least-squares estimate and ``final_rmse`` is the
        root-mean-square residual of the fitted trajectory (normalised per
        channel).

    Raises
    ------
    ValueError
        If an observed channel is not a compartment of the model.
    """
    from ..models.ode.parameters import ODE_MODELS, ParamTable

    config = config or ODEModelConfig()
    definitions = ODE_MODELS[model_name]
    param_names = list(definitions.keys())
    lower = [definitions[k].bounds.lo for k in param_names]
    upper = [definitions[k].bounds.hi for k in param_names]

    if isinstance(obs, dict):
        channels = dict(obs)
    else:
        channels = {"I": obs}

    if initial_guess is None:
        x0 = [definitions[k].default for k in param_names]
    else:
        x0 = [initial_guess.get(k, definitions[k].default) for k in param_names]
    x0 = np.clip(x0, np.array(lower), np.array(upper))

    _ = ParamTable(model_name, dict(zip(param_names, x0, strict=True)))
    _, factory = _model_factory(model_name)

    def residuals(theta: np.ndarray) -> np.ndarray:
        table = ParamTable(model_name, dict(zip(param_names, theta, strict=True)))
        mod = factory(config, table)
        traj = integrate_model(mod, config, i0=float(list(channels.values())[0][0]))
        parts: list[np.ndarray] = []
        for channel, data in channels.items():
            if channel not in traj:
                raise ValueError(
                    f"Unknown observation channel '{channel}'. Model "
                    f"'{model_name}' has compartments {tuple(traj)[:-1]}."
                )
            pred = np.interp(t_obs, traj["t"], traj[channel])
            scale = max(1.0, float(np.max(np.abs(data))))
            parts.append((pred - data) / scale)
        return np.concatenate(parts)

    result = least_squares(
        residuals,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=2000,
    )

    fitted: dict[str, float] = dict(zip(param_names, result.x, strict=True))
    rmse = float(np.sqrt(np.mean(residuals(result.x) ** 2)))
    return fitted, rmse


def rng_normal(mean: float, std: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw reproducible Gaussian noise via the seeded generator."""
    return norm.rvs(loc=mean, scale=std, size=size, random_state=rng)
