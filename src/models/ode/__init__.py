"""Mechanistic compartmental (ODE) epidemiology models.

This subpackage implements explicit systems of ordinary differential
equations describing the flow of individuals through epidemiological
compartments (e.g. *susceptible → exposed → infected → recovered*), together
with the mathematical scaffolding required for a rigorous, reproducible
modelling study:

* a declarative :class:`ParamTable` binding every rate to units, a bounded
  physically admissible range, and a peer-reviewed source;
* an :class:`ODECompartmentalModel` base class giving a consistent
  ``.rhs()`` interface to the numerical solver;
* reference implementations of the **SIR** and **SEIRD** systems.

All rates are validated against their admissible ranges before integration,
so an invalid parameterisation fails loudly at construction time rather than
silently producing an unphysical trajectory.
"""

from __future__ import annotations

from .parameters import ODE_MODELS, ODEModelConfig, ParamTable
from .seird import SEIRDModel, solve_seird
from .sir import SIRModel, solve_sir

__all__: list[str] = [
    "ODE_MODELS",
    "ParamTable",
    "ODEModelConfig",
    "SIRModel",
    "solve_sir",
    "SEIRDModel",
    "solve_seird",
]
