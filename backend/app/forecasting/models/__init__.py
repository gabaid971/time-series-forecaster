"""Model registry: API model type -> Forecaster class."""

from typing import Any, Dict

from pydantic import ValidationError

from app.forecasting.models.arima import ArimaForecaster
from app.forecasting.models.base import Block, ForecastContext, Forecaster
from app.forecasting.models.prophet import ProphetForecaster
from app.forecasting.models.tabular import LagForecaster, LinearRegressionForecaster, XGBoostForecaster

MODELS: Dict[str, type[Forecaster]] = {
    "LAG": LagForecaster,
    "LINEAR_REGRESSION": LinearRegressionForecaster,
    "XGBOOST": XGBoostForecaster,
    "ARIMA": ArimaForecaster,
    "PROPHET": ProphetForecaster,
}


def create_forecaster(model_type: str, params: Dict[str, Any], ctx: ForecastContext) -> Forecaster:
    """Instantiate a model, validating its parameters (ValueError with a readable message)."""
    cls = MODELS.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown model type: {model_type}")
    try:
        validated = cls.Params.model_validate(params)
    except ValidationError as e:
        details = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        raise ValueError(f"Invalid parameters: {details}")
    return cls(validated, ctx)


__all__ = ["Block", "ForecastContext", "Forecaster", "MODELS", "create_forecaster"]
