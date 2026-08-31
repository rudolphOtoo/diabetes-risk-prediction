"""Shared scaffolding for compartmental ODE models.

This module defines the :class:`ODECompartmentalModel` base class exposing a
uniform right-hand-side (``rhs``) interface consumed by the numerical solver,
plus the basic reproduction number :math:`\\mathcal{R}_0` derived from the
threshold analysis of the compartmental system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .parameters import ODEModelConfig, ParamTable


class ODECompartmentalModel(ABC):
    """Abstract compartmental model described by a system of ODEs.

    Subclasses must provide:

    * :attr:`compartment_names` — ordered list of compartment labels matching
      the ordering of the ODE state vector;
    * a ``rhs(t, y, params)`` implementing ``dy/dt = f(y, t)``.
    """

    #: Ordered compartment labels, aligned with the ODE state vector order.
    compartment_names: tuple[str, ...]
    #: Name used to key into :data:`ODE_MODELS`.
    model_name: str

    def __init__(
        self,
        config: ODEModelConfig,
        params: ParamTable,
    ) -> None:
        """Bind a configuration and a (validated) parameter table.

        Parameters
        ----------
        config : ODEModelConfig
            Population size, time horizon and solver tolerances.
        params : ParamTable
            Validated rate parameterisation for this model.

        Raises
        ------
        ValueError
            If the parameter table does not match the model family.
        """
        if params.model_name != self.model_name:
            raise ValueError(
                f"ParamTable model '{params.model_name}' does not match the "
                f"'{self.model_name}' model."
            )
        # Re-validate to fail fast on malformed rates.
        params.validate()
        self.config = config
        self.params = params

    @property
    def n_compartments(self) -> int:
        """Number of compartments (== length of the ODE state vector)."""
        return len(self.compartment_names)

    @abstractmethod
    def rhs(self, t: float, y: np.ndarray, *params: float) -> np.ndarray:
        """Evaluate the right-hand side ``dy/dt = f(t, y)``."""

    def reproduction_number(self) -> float:
        """Compute the basic reproduction number :math:`\\mathcal{R}_0`.

        For the closed SIR family with negligible mortality this reduces to the
        classic threshold ratio

        .. math::

            \\mathcal{R}_0 = \\frac{\\beta}{\\gamma}.

        Subclasses may override this with a full next-generation-matrix
        expression when the system warrants it.

        Returns
        -------
        float
            :math:`\\mathcal{R}_0`.
        """
        beta = float(self.params.rates["beta"])
        gamma = float(self.params.rates["gamma"])
        return beta / gamma if gamma > 0 else float("inf")
