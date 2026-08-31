"""Parameter-table validation tests for the compartmental ODE layer.

These assert the mathematical contract that every rate entering a differential
equation lies within a physically admissible, cited range, and that the
parameter table rejects out-of-bounds or non-finite values at construction
time.
"""

from __future__ import annotations

import pytest

from src.models.ode import ODE_MODELS, ODEModelConfig, ParamTable


def test_known_models_are_registered() -> None:
    """The registry must expose the two implemented model families."""
    assert set(ODE_MODELS) == {"sir", "seird"}


@pytest.mark.parametrize("model_name", ["sir", "seird"])
def test_default_rates_are_in_admissible_range(model_name: str) -> None:
    """Every default rate must pass validation without overrides."""
    table = ParamTable(model_name, {})
    table.validate()  # no exception => defaults are admissible


def test_parameter_symbols_and_units() -> None:
    """Each SIR rate must carry a LaTeX symbol, units, and a source."""
    for name, definition in ODE_MODELS["sir"].items():
        assert definition.symbol.startswith("\\")
        assert definition.units
        assert definition.source
        assert name in {"beta", "gamma"}
        assert definition.bounds.lo >= 0.0


@pytest.mark.parametrize(
    ("model_name", "bad"),
    [
        ("sir", {"beta": -0.1}),
        ("sir", {"gamma": 6.0}),
        ("seird", {"mu": 2.0}),
        ("seird", {"sigma": -1.0}),
    ],
)
def test_out_of_bounds_rates_raise(model_name: str, bad: dict[str, float]) -> None:
    """Rates outside their admissible closed interval must raise ValueError."""
    with pytest.raises(ValueError):
        ParamTable(model_name, bad)


@pytest.mark.parametrize("model_name", ["sir", "seird"])
def test_nonfinite_rates_raise(model_name: str) -> None:
    """NaN / infinite rates must be rejected (they would poison the solver)."""
    with pytest.raises(ValueError):
        ParamTable(model_name, {"beta": float("nan")})
    with pytest.raises(ValueError):
        ParamTable(model_name, {"beta": float("inf")})


def test_unknown_model_raises() -> None:
    """An unrecognised model name must raise a descriptive ValueError."""
    with pytest.raises(ValueError):
        ParamTable("not_a_model", {})


def test_config_validates_population_and_tolerances() -> None:
    """ODEModelConfig must reject non-positive populations / invalid tolerances."""
    with pytest.raises(ValueError):
        ODEModelConfig(n_population=0)
    with pytest.raises(ValueError):
        ODEModelConfig(n_days=0)
    with pytest.raises(ValueError):
        ODEModelConfig(rtol=1.5)
    with pytest.raises(ValueError):
        ODEModelConfig(atol=0.0)
