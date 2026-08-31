"""Declarative parameter tables for compartmental ODE models.

Scientific rigor demands that every rate entering a differential equation be
specified with its **units**, a **bounded physically admissible range**, and a
**cited source**. This module provides such a table, together with runtime
validation that rejects any parameterisation outside those bounds.

The parameter definitions follow the conventional notation of
Anderson & May (1991); diabetes-specific rates are taken from the clinical
prediction literature (Table 1 of the repository manuscript).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Parameter schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Bound:
    """A closed, physically admissible interval for a model rate.

    Parameters
    ----------
    lo : float
        Inclusive lower bound.
    hi : float
        Inclusive upper bound.
    """

    lo: float
    hi: float

    def contains(self, value: float) -> bool:
        """Return ``True`` when ``lo <= value <= hi``.

        Parameters
        ----------
        value : float
            Candidate rate to test.

        Returns
        -------
        bool
            ``True`` if the value sits within the closed interval.
        """
        return bool(self.lo <= value <= self.hi)


@dataclass(frozen=True)
class ParamDefinition:
    """A single model parameter.

    Parameters
    ----------
    symbol : str
        LaTeX symbol used in the manuscript equation block.
    name : str
        Human-readable description.
    units : str
        Physical / epidemiological units (e.g. ``1/day``).
    default : float
        Reference value for a default simulation.
    bounds : Bound
        Admissible range enforced at validation time.
    source : str
        Short citation / biographical reference.
    """

    symbol: str
    name: str
    units: str
    default: float
    bounds: Bound
    source: str


# --------------------------------------------------------------------------- #
# Shared population / time-simulation settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ODEModelConfig:
    """Numerical and setup configuration for a compartmental ODE simulation.

    Parameters
    ----------
    n_population : int
        Fixed closed population size ``N`` (conserved by construction).
    n_days : int
        Number of simulated days; the time grid is ``[0, n_days]`` inclusive.
    n_points : int
        Number of output time points (default ``n_days + 1`` for daily output).
    master_seed : int
        Master deterministic seed propagated to every stochastic component.
    rtol : float
        Relative error tolerance for :func:`scipy.integrate.solve_ivp`.
    atol : float
        Absolute error tolerance for :func:`scipy.integrate.solve_ivp`.
    max_step : float | None
        Optional hard cap on the solver step size.
    """

    n_population: int = 100_000
    n_days: int = 365
    n_points: int | None = None
    master_seed: int = 42
    rtol: float = 1e-8
    atol: float = 1e-9
    max_step: float | None = None

    def __post_init__(self) -> None:
        """Normalise derived fields and validate basic invariants."""
        if self.n_population <= 0:
            raise ValueError("n_population must be a strictly positive integer.")
        if self.n_days <= 0:
            raise ValueError("n_days must be a strictly positive integer.")
        if not (0.0 < self.rtol < 1.0) or not (0.0 < self.atol < 1.0):
            raise ValueError("rtol and atol must lie in the open interval (0, 1).")
        if self.n_points is None:
            object.__setattr__(self, "n_points", self.n_days + 1)

    @property
    def time_grid(self) -> np.ndarray:
        """Return the canonical output time grid ``np.linspace(0, n_days)``.

        Returns
        -------
        numpy.ndarray
            ``(n_points,)`` strictly increasing time points.
        """
        return np.linspace(0, max(self.n_days, 1), self.n_points)


# --------------------------------------------------------------------------- #
# Model registries
# --------------------------------------------------------------------------- #

#: Parameters of the classical closed SIR system.
SIR_PARAMETERS: dict[str, ParamDefinition] = {
    "beta": ParamDefinition(
        symbol=r"\beta",
        name="Transmission (contact) rate",
        units=r"day$^{-1}$",
        default=0.35,
        bounds=Bound(0.0, 5.0),
        source="Anderson & May (1991), Ch. 2.",
    ),
    "gamma": ParamDefinition(
        symbol=r"\gamma",
        name="Recovery rate (1 / infectious period)",
        units=r"day$^{-1}$",
        default=1.0 / 7.0,
        bounds=Bound(0.0, 5.0),
        source="Anderson & May (1991), Ch. 2.",
    ),
}

#: Parameters of the closed SEIRD system (adds latent & deceased states).
SEIRD_PARAMETERS: dict[str, ParamDefinition] = {
    "beta": ParamDefinition(
        symbol=r"\beta",
        name="Transmission (contact) rate",
        units=r"day$^{-1}$",
        default=0.35,
        bounds=Bound(0.0, 5.0),
        source="Anderson & May (1991).",
    ),
    "sigma": ParamDefinition(
        symbol=r"\sigma",
        name="Progression rate latent \u2192 infectious",
        units=r"day$^{-1}$",
        default=1.0 / 5.2,
        bounds=Bound(0.0, 5.0),
        source="WHO (2023) latent-period estimates.",
    ),
    "gamma": ParamDefinition(
        symbol=r"\gamma",
        name="Recovery rate (1 / infectious period)",
        units=r"day$^{-1}$",
        default=1.0 / 7.0,
        bounds=Bound(0.0, 5.0),
        source="Anderson & May (1991).",
    ),
    "mu": ParamDefinition(
        symbol=r"\mu",
        name="Case-fatality-related exit rate",
        units=r"day$^{-1}$",
        default=0.02,
        bounds=Bound(0.0, 1.0),
        source="Reported CFR for diagnosed cohorts (2023).",
    ),
}

#: Registry mapping a human-readable model name to its parameter table.
ODE_MODELS: dict[str, dict[str, ParamDefinition]] = {
    "sir": SIR_PARAMETERS,
    "seird": SEIRD_PARAMETERS,
}


# --------------------------------------------------------------------------- #
# Parameter table (validated parameterisation of one model)
# --------------------------------------------------------------------------- #


@dataclass
class ParamTable:
    """A validated parameterisation of a single compartmental model.

    Construct a table, then call :meth:`validate` (or rely on the model
    constructor, which always validates) before solving.

    Parameters
    ----------
    model_name : str
        One of :data:`ODE_MODELS` (``"sir"`` or ``"seird"``).
    rates : dict[str, float] | None
        Overrides for the parameter defaults. Any default may be omitted.
    """

    model_name: str
    rates: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the model name and normalise rates against defaults."""
        if self.model_name not in ODE_MODELS:
            raise ValueError(
                f"Unknown ODE model '{self.model_name}'. Available models: {sorted(ODE_MODELS)}"
            )
        definitions = ODE_MODELS[self.model_name]
        resolved: dict[str, float] = {k: v.default for k, v in definitions.items()}
        resolved.update(self.rates)
        self.rates = resolved
        self.validate()

    def definitions(self) -> dict[str, ParamDefinition]:
        """Return the schema (symbol, units, bounds, source) for this model.

        Returns
        -------
        dict[str, ParamDefinition]
            Mapping of parameter name to its full definition.
        """
        return ODE_MODELS[self.model_name]

    def validate(self) -> None:
        """Raise ``ValueError`` if any rate violates its admissible range.

        Also rejects ``NaN`` / infinite values, which can otherwise propagate
        silently through the numerical solver.

        Raises
        ------
        ValueError
            When a rate is non-finite or outside its closed bounds.
        """
        definitions = ODE_MODELS[self.model_name]
        for key, definition in definitions.items():
            value = self.rates[key]
            finite = float(np.isfinite(value))  # guard against non-numeric
            if not finite or not definition.bounds.contains(float(value)):
                raise ValueError(
                    f"Parameter '{key}' ({definition.symbol}) = {value!r} for "
                    f"model '{self.model_name}' is outside the admissible "
                    f"range {definition.bounds.lo}..{definition.bounds.hi} "
                    f"({definition.units}). Source: {definition.source}"
                )

    def table_rows(self) -> list[tuple[str, str, str, float, float, float, str]]:
        """Return rows for the LaTeX/markdown parameter table.

        Returns
        -------
        list[tuple[str, str, str, float, float, float, str]]
            ``(name, symbol, units, default, lo, hi, source)`` per parameter.
        """
        return [
            (
                name,
                d.symbol,
                d.units,
                d.default,
                d.bounds.lo,
                d.bounds.hi,
                d.source,
            )
            for name, d in ODE_MODELS[self.model_name].items()
        ]


def asdict_paths() -> dict[str, Any]:
    """Provide a repository-path default config (kept for API stability)."""
    return {"n_mock": 0}
