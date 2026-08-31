"""The classical closed **SIR** compartmental model.

Mathematical formulation
------------------------
For a fixed closed population :math:`N` partitioned into susceptible
(:math:`S`), infectious (:math:`I`) and recovered (:math:`R`) individuals, the
system of ordinary differential equations is

.. math::

    \\frac{dS}{dt} &= -\\frac{\\beta S I}{N} \\\\
    \\frac{dI}{dt} &=  \\frac{\\beta S I}{N} - \\gamma I \\\\
    \\frac{dR}{dt} &=  \\gamma I

with transmission rate :math:`\\beta \\ge 0` (``day``:math:`^{-1}`) and
recovery rate :math:`\\gamma \\ge 0`. The population is closed, so the mass
action term conserves the total:

.. math::

    \\frac{d}{dt}\\big(S + I + R\\big) = 0
    \\quad\\Longrightarrow\\quad
    S(t) + I(t) + R(t) = N \\quad \\forall\\, t \\ge 0.

The disease-free equilibrium (DFE) at :math:`(N, 0, 0)` is unstable when the
basic reproduction number :math:`\\mathcal{R}_0 = \\beta / \\gamma > 1`, which
is the standard epidemic threshold. Non-negativity of each compartment follows
from the quasi-positivity of the vector field.
"""

from __future__ import annotations

import numpy as np

from .base import ODECompartmentalModel
from .parameters import ODEModelConfig, ParamTable


class SIRModel(ODECompartmentalModel):
    """Closed SIR model (susceptible / infectious / recovered)."""

    #: Compartment ordering of the ODE state vector.
    compartment_names: tuple[str, ...] = ("S", "I", "R")
    model_name: str = "sir"

    def rhs(self, t: float, y: np.ndarray, *params: float) -> np.ndarray:
        """Right-hand side ``dy/dt = f(t, y)`` of the SIR system.

        Parameters
        ----------
        t : float
            Current simulation time (days).
        y : numpy.ndarray
            State vector ``[S, I, R]``.
        params : float
            Unused positional slot for scipy compatibility.

        Returns
        -------
        numpy.ndarray
            Time derivative ``[dS/dt, dI/dt, dR/dt]``.
        """
        del t, params  # time-invariant autonomous system
        n = self.config.n_population
        s, i, r = y
        beta = float(self.params.rates["beta"])
        gamma = float(self.params.rates["gamma"])
        force = beta * s * i / n
        return np.array([-force, force - gamma * i, gamma * i])


def solve_sir(
    rates: dict[str, float],
    *,
    config: ODEModelConfig | None = None,
    i0: float | None = None,
) -> dict[str, np.ndarray]:
    """Integrate the SIR system and return the full state trajectories.

    Parameters
    ----------
    rates : dict[str, float]
        ``beta`` and/or ``gamma`` values (defaults used for omitted keys).
    config : ODEModelConfig | None
        Population, time horizon and solver tolerances.
    i0 : float | None
        Initial number of infectious individuals (defaults to ``N * 0.01``).

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping compartment name to a ``(n_points,)`` trajectory, plus a
        ``"t"`` key holding the time grid.

    Raises
    ------
    ValueError
        If a rate violates its admissible bounds.
    """
    from ...solvers import integrate_model  # deferred to avoid import cycle

    config = config or ODEModelConfig()
    params = ParamTable("sir", rates)
    model = SIRModel(config, params)
    i0 = i0 if i0 is not None else float(config.n_population) * 0.01
    res = integrate_model(model, config, i0=i0)
    return res
