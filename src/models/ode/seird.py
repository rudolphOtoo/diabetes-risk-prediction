"""The closed **SEIRD** model (susceptible / exposed / infectious /
recovered / deceased).

Mathematical formulation
------------------------
The SEIRD system extends SIR with a latent (exposed) compartment :math:`E` and
a disease-related exit compartment :math:`D`:

.. math::

    \\frac{dS}{dt} &= -\\frac{\\beta S I}{N} \\\\
    \\frac{dE}{dt} &=  \\frac{\\beta S I}{N} - \\sigma E \\\\
    \\frac{dI}{dt} &=  \\sigma E - (\\gamma + \\mu) I \\\\
    \\frac{dR}{dt} &=  \\gamma I \\\\
    \\frac{dD}{dt} &=  \\mu I

with

* :math:`\\beta` — transmission rate (``day``:math:`^{-1}`),
* :math:`\\sigma` — rate of progression latent \u2192 infectious
  (``day``:math:`^{-1}`),
* :math:`\\gamma` — recovery rate (``day``:math:`^{-1}`),
* :math:`\\mu` — case-fatality-related exit rate (``day``:math:`^{-1}`).

The closed-population mass is conserved and each state is non-negative:

.. math::

    S(t) + E(t) + I(t) + R(t) + D(t) = N
    \\quad\\forall\\, t \\ge 0.
"""

from __future__ import annotations

import numpy as np

from .base import ODECompartmentalModel
from .parameters import ODEModelConfig, ParamTable


class SEIRDModel(ODECompartmentalModel):
    """Closed SEIRD model (susceptible / exposed / infectious / recovered /
    deceased)."""

    #: Compartment ordering of the ODE state vector.
    compartment_names: tuple[str, ...] = ("S", "E", "I", "R", "D")
    model_name: str = "seird"

    def rhs(self, t: float, y: np.ndarray, *params: float) -> np.ndarray:
        """Right-hand side ``dy/dt = f(t, y)`` of the SEIRD system.

        Parameters
        ----------
        t : float
            Current simulation time (days).
        y : numpy.ndarray
            State vector ``[S, E, I, R, D]``.
        params : float
            Unused positional slot for scipy compatibility.

        Returns
        -------
        numpy.ndarray
            Time derivative ``[dS/dt, dE/dt, dI/dt, dR/dt, dD/dt]``.
        """
        del t, params
        n = self.config.n_population
        s, e, i, r, d = y
        beta = float(self.params.rates["beta"])
        sigma = float(self.params.rates["sigma"])
        gamma = float(self.params.rates["gamma"])
        mu = float(self.params.rates["mu"])
        force = beta * s * i / n
        return np.array(
            [
                -force,
                force - sigma * e,
                sigma * e - (gamma + mu) * i,
                gamma * i,
                mu * i,
            ]
        )


def solve_seird(
    rates: dict[str, float],
    *,
    config: ODEModelConfig | None = None,
    i0: float | None = None,
    e0: float | None = None,
) -> dict[str, np.ndarray]:
    """Integrate the SEIRD system and return the state trajectories.

    Parameters
    ----------
    rates : dict[str, float]
        ``beta``, ``sigma``, ``gamma`` and/or ``mu`` (defaults used otherwise).
    config : ODEModelConfig | None
        Population, time horizon and solver tolerances.
    i0 : float | None
        Initial infectious count (defaults to ``N * 0.005``).
    e0 : float | None
        Initial latent count (defaults to a small deterministic ``0.2 * i0``).

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping compartment name to a trajectory, plus the ``"t"`` grid.

    Raises
    ------
    ValueError
        If a rate violates its admissible bounds.
    """
    from ...solvers import integrate_model

    config = config or ODEModelConfig()
    params = ParamTable("seird", rates)
    model = SEIRDModel(config, params)
    i0 = i0 if i0 is not None else float(config.n_population) * 0.005
    e0 = e0 if e0 is not None else 0.2 * i0
    res = integrate_model(model, config, i0=i0, e0=e0)
    return res
