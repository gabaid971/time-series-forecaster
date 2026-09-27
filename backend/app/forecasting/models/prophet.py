"""Prophet model: trend + seasonality decomposition, with optional lag regressors."""

import logging
from typing import Annotated, Any, Dict, List, Literal, Union

import numpy as np
import pandas as pd
import polars as pl
from pydantic import Field, field_validator

from app.forecasting.features import TargetLag
from app.forecasting.models.base import Block, ForecastContext, Forecaster, ModelParams

# Prophet is optional
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetParams(ModelParams):
    daily_seasonality: bool = False
    weekly_seasonality: bool = True
    yearly_seasonality: bool = True
    seasonality_mode: Literal["additive", "multiplicative"] = "additive"
    # Disabled by default: lags shorter than the horizon would leak future values
    use_lag_regressors: bool = False
    lag_regressors: Annotated[List[TargetLag], Field(max_length=20)] = []

    @field_validator("lag_regressors", mode="before")
    @classmethod
    def parse_lags_string(cls, value: Union[str, List[int]]):
        if isinstance(value, str):
            return [int(x.strip()) for x in value.split(",") if x.strip()]
        return value


class ProphetForecaster(Forecaster):
    """
    Prophet is trained once and does not update with recent observations: its
    forecast of a date does not depend on the block origin (except through lag
    regressors, which must be >= horizon to stay known at forecast time).
    """

    Params = ProphetParams

    def __init__(self, params: ProphetParams, ctx: ForecastContext):
        if not PROPHET_AVAILABLE:
            raise ValueError("Prophet is not installed. Please use Linear Regression, XGBoost, or ARIMA instead.")
        super().__init__(params, ctx)
        self.regressor_lags = params.lag_regressors if params.use_lag_regressors else []
        short_lags = sorted(lag for lag in self.regressor_lags if lag < ctx.horizon)
        if short_lags:
            raise ValueError(
                f"Prophet lag regressor(s) {short_lags} < horizon {ctx.horizon}: these values are not "
                f"known at forecast time. Use lags >= {ctx.horizon}."
            )
        self.regressor_names = [f"lag_{lag}" for lag in self.regressor_lags]

    def _prophet_frame(self, df: pl.DataFrame, rows: np.ndarray, with_target: bool) -> pd.DataFrame:
        frame = df.with_columns(
            pl.col(self.ctx.target_col).shift(lag).alias(name)
            for lag, name in zip(self.regressor_lags, self.regressor_names)
        )[rows].with_row_index("__position")
        frame = frame.drop_nulls(subset=self.regressor_names) if self.regressor_names else frame
        columns = [pl.col("__position"), pl.col(self.ctx.date_col).alias("ds")]
        if with_target:
            columns.append(pl.col(self.ctx.target_col).alias("y"))
        pdf = frame.select(columns + [pl.col(name) for name in self.regressor_names]).to_pandas()
        # Prophet mishandles datetime64[us]: force nanosecond precision for proper seasonality
        pdf["ds"] = pd.to_datetime(pdf["ds"]).astype("datetime64[ns]")
        return pdf

    def fit(self, df: pl.DataFrame, train_rows: np.ndarray) -> None:
        train = self._prophet_frame(df, train_rows, with_target=True)
        if train.empty:
            raise ValueError("No training data after filtering by date ranges")
        self.model = Prophet(
            daily_seasonality=self.params.daily_seasonality,
            weekly_seasonality=self.params.weekly_seasonality,
            yearly_seasonality=self.params.yearly_seasonality,
            seasonality_mode=self.params.seasonality_mode
        )
        for name in self.regressor_names:
            self.model.add_regressor(name)
        self.model.fit(train.drop(columns="__position"))

    def predict_blocks(self, df: pl.DataFrame, blocks: List[Block]) -> List[np.ndarray]:
        rows = np.concatenate([np.arange(b.start, b.end) for b in blocks])
        future = self._prophet_frame(df, rows, with_target=False)
        values = np.full(len(rows), np.nan)
        if not future.empty:
            values[future["__position"].to_numpy()] = self.model.predict(future.drop(columns="__position"))["yhat"].to_numpy()
        return np.split(values, np.cumsum([len(b) for b in blocks])[:-1])

    def explain(self) -> Dict[str, Any]:
        if not self.regressor_names:
            return {}
        # Approximation: Prophet does not expose comparable regressor importances
        weight = 1.0 / len(self.regressor_names)
        return {"feature_importance": [{"feature": name, "importance": weight} for name in self.regressor_names]}
