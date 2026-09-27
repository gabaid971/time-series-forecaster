"""
Models on tabular features (lags, calendar, exogenous): Lag baseline, Linear Regression, XGBoost.

They share the feature pipeline and the recursive forecasting engine: within a
block, features are recomputed from the actual history plus the predictions of the
previous steps, so everything derived from the target (lags, derived features,
residual base) is recursive by construction.
"""

from abc import abstractmethod
from typing import Annotated, Any, ClassVar, Dict, List, Literal, Optional, Union

import numpy as np
import polars as pl
from pydantic import Field, field_validator

from app.forecasting.features import (
    FeatureConfig,
    LagList,
    TargetLag,
    build_features,
    parse_feature_config,
    validate_no_future_leakage,
)
from app.forecasting.models.base import Block, ForecastContext, Forecaster, ModelParams

GROUP_COL = "__block"


class TabularParams(ModelParams):
    lags: Optional[LagList] = None
    feature_config: Optional[Dict[str, Any]] = None
    target_mode: Literal["raw", "residual"] = "raw"
    residual_lag: TargetLag = 1

    @field_validator("lags", mode="before")
    @classmethod
    def parse_lags_string(cls, value: Union[str, List[int], None]):
        if isinstance(value, str):
            return [int(x.strip()) for x in value.split(",") if x.strip()]
        return value


class TabularForecaster(Forecaster):
    """Base class: feature building, training data and recursive block forecasting."""

    default_lags: ClassVar[List[int]] = [1, 7]

    def __init__(self, params: ModelParams, ctx: ForecastContext):
        super().__init__(params, ctx)
        self.features = self.feature_config()
        validate_no_future_leakage(self.features, ctx.target_col, ctx.horizon)
        self.residual_lag = self.residual_lag_or_none()
        self.lookback = max(self.features.lookback(), self.residual_lag or 0)
        self.feature_names: List[str] = []

    # --- Configuration hooks -------------------------------------------------

    def feature_config(self) -> FeatureConfig:
        return parse_feature_config(self.params.model_dump(), self.default_lags)

    def residual_lag_or_none(self) -> Optional[int]:
        """In residual mode the model predicts y(t) - y(t - residual_lag)."""
        if getattr(self.params, "target_mode", "raw") == "residual":
            return self.params.residual_lag
        return None

    # --- Estimator hooks -----------------------------------------------------

    @abstractmethod
    def fit_estimator(self, X: np.ndarray, y: np.ndarray, train_frame: pl.DataFrame) -> None:
        """Fit on features X and (raw or residual) target y."""

    @abstractmethod
    def predict_estimator(self, X: np.ndarray) -> np.ndarray:
        ...

    # --- Training ------------------------------------------------------------

    def _with_features(self, df: pl.DataFrame, over: Optional[str] = None) -> pl.DataFrame:
        df, self.feature_names = build_features(df, self.ctx.date_col, self.ctx.target_col, self.features, over=over)
        if self.residual_lag:
            base = pl.col(self.ctx.target_col).shift(self.residual_lag)
            df = df.with_columns((base.over(over) if over else base).alias("__residual_base"))
        return df

    def fit(self, df: pl.DataFrame, train_rows: np.ndarray) -> None:
        target = self.ctx.target_col
        frame = self._with_features(df)[train_rows]
        if self.residual_lag:
            frame = frame.with_columns((pl.col(target) - pl.col("__residual_base")).alias("__y"))
        else:
            frame = frame.with_columns(pl.col(target).alias("__y"))
        frame = frame.drop_nulls(subset=self.feature_names + ["__y"])
        if frame.height == 0:
            raise ValueError("Not enough data after creating features")

        X = frame.select(self.feature_names).to_numpy()
        y = frame["__y"].to_numpy()
        self.fit_estimator(X, y, frame)

    # --- Recursive block forecasting -----------------------------------------

    def predict_blocks(self, df: pl.DataFrame, blocks: List[Block]) -> List[np.ndarray]:
        """
        Step s = 1..horizon: for every block at once, build a window made of the
        `lookback` rows before the block (actuals) and the first s rows of the block
        (predictions of steps < s, unknown value at step s), compute the features of
        the last row of each window, and predict it.
        """
        target = self.ctx.target_col
        y = df[target].cast(pl.Float64).to_numpy()
        predictions = [np.full(len(b), np.nan) for b in blocks]
        unknown_exog = [c for c in self.features.unknown_exogenous_columns() if c in df.columns]

        for step in range(1, max(len(b) for b in blocks) + 1):
            active = [j for j, b in enumerate(blocks) if len(b) >= step]
            windows = [np.arange(max(0, blocks[j].start - self.lookback), blocks[j].start + step) for j in active]
            rows = np.concatenate(windows)
            group = np.concatenate([np.full(len(w), j) for j, w in zip(active, windows)])
            block_start = np.array([blocks[j].start for j in group])
            step_in_block = rows - block_start + 1  # <= 0: history

            # Target: actual history, then predictions of previous steps, unknown at current step
            y_window = y[rows].copy()
            future = step_in_block >= 1
            y_window[future] = [
                predictions[j][s - 1] for j, s in zip(group[future], step_in_block[future])
            ]
            window = df[rows].with_columns(
                pl.Series(target, y_window).fill_nan(None),
                pl.Series(GROUP_COL, group),
                pl.Series("__future", future),
            )
            # Exogenous values not known in advance are hidden inside the block
            window = window.with_columns(
                pl.when(pl.col("__future")).then(None).otherwise(pl.col(c)).alias(c) for c in unknown_exog
            )

            current = self._with_features(window, over=GROUP_COL).filter(pl.Series(step_in_block == step))
            X = current.select(self.feature_names).to_numpy().astype(float)
            valid = ~np.isnan(X).any(axis=1)
            base = None
            if self.residual_lag:
                base = current["__residual_base"].cast(pl.Float64).to_numpy()
                valid &= ~np.isnan(base)
            if not valid.any():
                continue

            values = self.predict_estimator(X[valid])
            if base is not None:
                values = values + base[valid]
            for j, value in zip(np.array(active)[valid], values):
                predictions[j][step - 1] = value

        return predictions


class LagParams(ModelParams):
    lag: TargetLag = 1


class LagForecaster(TabularForecaster):
    """Baseline: the prediction is the value `lag` steps before (recursively within a block)."""

    Params = LagParams

    def feature_config(self) -> FeatureConfig:
        return FeatureConfig(target_lags=[self.params.lag])

    def fit_estimator(self, X, y, train_frame) -> None:
        pass  # Nothing to learn

    def predict_estimator(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0]


class LinearRegressionParams(TabularParams):
    standardize: bool = False


class LinearRegressionForecaster(TabularForecaster):
    Params = LinearRegressionParams

    def fit_estimator(self, X, y, train_frame) -> None:
        from sklearn.linear_model import LinearRegression

        self.means = self.stds = None
        if self.params.standardize:
            self.means = X.mean(axis=0)
            self.stds = X.std(axis=0)
            self.stds[self.stds == 0] = 1
            X = (X - self.means) / self.stds
        self.model = LinearRegression().fit(X, y)

    def predict_estimator(self, X: np.ndarray) -> np.ndarray:
        if self.means is not None:
            X = (X - self.means) / self.stds
        return self.model.predict(X)

    def explain(self) -> Dict[str, Any]:
        # Importance = normalized absolute coefficients
        abs_coefs = np.abs(self.model.coef_)
        total = abs_coefs.sum() or 1.0
        importance = [
            {"feature": name, "importance": float(c / total)} for name, c in zip(self.feature_names, abs_coefs)
        ]
        return {"feature_importance": sorted(importance, key=lambda x: x["importance"], reverse=True)}


class XGBoostParams(TabularParams):
    n_estimators: Annotated[int, Field(ge=1, le=1000)] = 100
    max_depth: Annotated[int, Field(ge=1, le=12)] = 6
    learning_rate: Annotated[float, Field(gt=0, le=1)] = 0.1


class XGBoostForecaster(TabularForecaster):
    Params = XGBoostParams
    default_lags = [1, 7, 14, 30]

    def fit_estimator(self, X, y, train_frame) -> None:
        import xgboost as xgb

        self.model = xgb.XGBRegressor(
            n_estimators=self.params.n_estimators,
            max_depth=self.params.max_depth,
            learning_rate=self.params.learning_rate,
            random_state=42,
            n_jobs=1  # Small tabular data: multithreading is ~300x slower (thread contention)
        )
        self.model.fit(X, y)
        self.X_train = X
        self.train_frame = train_frame

    def predict_estimator(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def explain(self) -> Dict[str, Any]:
        from app.forecasting.shap_analysis import compute_shap_values

        importance = [
            {"feature": name, "importance": float(imp)}
            for name, imp in zip(self.feature_names, self.model.feature_importances_)
        ]
        return {
            "feature_importance": sorted(importance, key=lambda x: x["importance"], reverse=True),
            "shap_analysis": compute_shap_values(
                model=self.model,
                X=self.X_train,
                feature_names=self.feature_names,
                df_aligned=self.train_frame.select([self.ctx.date_col] + self.feature_names),
                date_col=self.ctx.date_col,
                feature_config=self.features,
            ),
        }
