"""Tests for forecasting models and the evaluation engine."""

import time
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from app.forecasting.backtest import make_blocks, run_backtest, training_rows
from app.forecasting.models import Block, ForecastContext, create_forecaster
from app.forecasting.models.arima import ArimaForecaster, ArimaParams
from app.forecasting.models.prophet import PROPHET_AVAILABLE
from app.forecasting.models.tabular import LinearRegressionForecaster, LinearRegressionParams
from app.forecasting.training import train_models


class DateRange:
    """Mock DateRange for tests."""
    def __init__(self, start, end):
        self.start = start
        self.end = end


TRAINING = [DateRange("2023-01-01", "2023-05-01")]
PREDICTION = [DateRange("2023-05-01", "2023-06-01")]


def evaluate(df, model_type, params, horizon=1, training=TRAINING, prediction=PREDICTION):
    """Fit a model and evaluate it like the /train endpoint does."""
    ctx = ForecastContext(date_col="date", target_col="value", horizon=horizon)
    model = create_forecaster(model_type, params, ctx)
    model.fit(df, training_rows(df, "date", training))
    result = run_backtest(model, df, make_blocks(df, "date", prediction, horizon), horizon)
    return {**result, **model.explain()}


@pytest.fixture
def large_daily_df():
    """Create a larger daily DataFrame for training."""
    np.random.seed(42)
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(200)]
    # Create predictable pattern: trend + seasonality + noise
    values = [
        100 + 0.1 * i + 10 * np.sin(2 * np.pi * i / 7) + np.random.randn() * 2
        for i in range(200)
    ]
    return pl.DataFrame({"date": dates, "value": values})


class TestLag:
    """Tests for the LAG baseline."""

    def test_basic_training(self, large_daily_df):
        result = evaluate(large_daily_df, "LAG", {"lag": 1})
        assert result["metrics"]["rmse"] >= 0
        assert result["metrics"]["mae"] >= 0
        assert len(result["forecast"]) > 0

    def test_prediction_is_lagged_value(self, large_daily_df):
        result = evaluate(large_daily_df, "LAG", {"lag": 7})
        values = large_daily_df["value"].to_list()
        first = result["forecast"][0]
        row = large_daily_df["date"].dt.strftime("%Y-%m-%dT%H:%M:%S").to_list().index(first["date"])
        assert first["prediction"] == pytest.approx(values[row - 7])

    def test_multi_horizon(self, large_daily_df):
        """Within a block, the lag model repeats its own predictions."""
        result = evaluate(large_daily_df, "LAG", {"lag": 1}, horizon=7)
        assert len(result["metrics_by_horizon"]) == 7
        first_block = [f["prediction"] for f in result["forecast"][:7]]
        assert len(set(first_block)) == 1


class TestLinearRegression:
    """Tests for Linear Regression."""

    def test_basic_training(self, large_daily_df):
        result = evaluate(large_daily_df, "LINEAR_REGRESSION", {"lags": [1, 7]})
        assert len(result["forecast"]) > 0
        assert len(result["feature_importance"]) == 2

    def test_with_temporal_features(self, large_daily_df):
        result = evaluate(large_daily_df, "LINEAR_REGRESSION", {
            "feature_config": {"target_lags": [1, 7], "temporal": {"day_of_week": True, "month": True}}
        })
        feature_names = [f["feature"] for f in result["feature_importance"]]
        assert any("dow" in f or "month" in f for f in feature_names)

    def test_residual_mode(self, large_daily_df):
        result = evaluate(large_daily_df, "LINEAR_REGRESSION", {"lags": [1, 7], "target_mode": "residual", "residual_lag": 1})
        assert len(result["forecast"]) > 0

    def test_standardization_does_not_change_predictions(self, large_daily_df):
        raw = evaluate(large_daily_df, "LINEAR_REGRESSION", {"lags": [1, 7]})
        std = evaluate(large_daily_df, "LINEAR_REGRESSION", {"lags": [1, 7], "standardize": True})
        assert [f["prediction"] for f in std["forecast"]] == pytest.approx([f["prediction"] for f in raw["forecast"]])

    def test_multi_horizon_with_metrics(self, large_daily_df):
        result = evaluate(large_daily_df, "LINEAR_REGRESSION", {"lags": [1, 7]}, horizon=7)
        assert len(result["metrics_by_horizon"]) == 7


class TestXGBoost:
    def test_training_and_shap(self, large_daily_df):
        result = evaluate(large_daily_df, "XGBOOST", {
            "lags": [1, 7], "n_estimators": 20, "feature_config": {"temporal": {"day_of_week": True}}
        })
        assert len(result["forecast"]) > 0
        assert "day_of_week" in result["shap_analysis"]["temporal"]


class TestRecursiveEngine:
    """Guarantees of the recursive forecasting engine of tabular models."""

    @staticmethod
    def recording_model(df, params, horizon):
        """Linear regression that records the features it is asked to predict on."""
        ctx = ForecastContext(date_col="date", target_col="value", horizon=horizon)
        model = LinearRegressionForecaster(LinearRegressionParams.model_validate(params), ctx)
        model.fit(df, training_rows(df, "date", TRAINING))
        model.seen = []
        predict = model.predict_estimator
        model.predict_estimator = lambda X: (model.seen.append(X.copy()), predict(X))[1]
        return model

    def test_derived_features_are_recursive(self, large_daily_df):
        """A derived feature built on a target lag uses the predictions within the block."""
        params = {"lags": [1], "feature_config": {"derived": [
            {"operation": "sum", "feature_a": "target_lag_1", "feature_b": "target_lag_1", "alias": "double"}
        ]}}
        model = self.recording_model(large_daily_df, params, horizon=5)
        model.predict_blocks(large_daily_df, [Block(150, 155)])
        X = np.vstack(model.seen)
        assert np.allclose(X[:, 1], 2 * X[:, 0])

    def test_skipped_step_propagates(self):
        """A step that cannot be predicted must not make later steps fall back to actual values."""
        n = 120
        df = pl.DataFrame({
            "date": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)],
            "value": [float(i % 10) for i in range(n)],
            "promo": [1.0] * 101 + [None] + [1.0] * (n - 102),  # missing at row 101 -> step 2 skipped
        })
        params = {"lags": [1], "feature_config": {"exogenous": [
            {"column": "promo", "use_actual": True, "known_in_advance": True}
        ]}}
        ctx = ForecastContext(date_col="date", target_col="value", horizon=3)
        model = create_forecaster("LINEAR_REGRESSION", params, ctx)
        model.fit(df, training_rows(df, "date", [DateRange("2023-01-01", "2023-04-01")]))

        predictions = model.predict_blocks(df, [Block(100, 103)])[0]
        # Step 1 predicted, step 2 has no feature, step 3 needs the prediction of step 2
        assert not np.isnan(predictions[0])
        assert np.isnan(predictions[1]) and np.isnan(predictions[2])

    def test_unknown_exogenous_hidden_inside_block(self):
        """Even beyond the validated horizon, unknown exogenous values of the block are masked."""
        n = 120
        df = pl.DataFrame({
            "date": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)],
            "value": [float(i % 10) for i in range(n)],
            "sensor": [float(i) for i in range(n)],
        })
        params = {"lags": [1], "feature_config": {"exogenous": [{"column": "sensor", "lags": [1]}]}}
        ctx = ForecastContext(date_col="date", target_col="value", horizon=1)
        model = create_forecaster("LINEAR_REGRESSION", params, ctx)
        model.fit(df, training_rows(df, "date", [DateRange("2023-01-01", "2023-04-01")]))

        # Block of 3 rows while validated for horizon 1: sensor lag 1 of steps 2-3 is in the future
        predictions = model.predict_blocks(df, [Block(100, 103)])[0]
        assert not np.isnan(predictions[0])
        assert np.isnan(predictions[1]) and np.isnan(predictions[2])


class TestARIMA:
    """Tests for ARIMA."""

    def test_basic_training(self, large_daily_df):
        result = evaluate(large_daily_df, "ARIMA", {"p": 1, "d": 1, "q": 1})
        assert len(result["forecast"]) > 0
        assert "feature_importance" not in result  # ARIMA has no features

    def test_different_orders(self, large_daily_df):
        for p, d, q in [(1, 0, 0), (0, 1, 1), (2, 1, 2)]:
            result = evaluate(large_daily_df, "ARIMA", {"p": p, "d": d, "q": q})
            assert len(result["forecast"]) > 0

    def test_multi_horizon(self, large_daily_df):
        result = evaluate(large_daily_df, "ARIMA", {"p": 1, "d": 1, "q": 1}, horizon=7)
        assert {f["horizon_step"] for f in result["forecast"]} == set(range(1, 8))
        assert len(result["metrics_by_horizon"]) == 7

    def test_insufficient_data_raises(self, large_daily_df):
        with pytest.raises(ValueError, match="Not enough"):
            evaluate(large_daily_df, "ARIMA", {"p": 5, "d": 1, "q": 5}, training=[DateRange("2023-01-01", "2023-01-05")])

    def test_one_step_fast_path_matches_block_updates(self, large_daily_df):
        """Horizon 1 uses one filtering pass: same results as updating the state block by block."""
        ctx = ForecastContext(date_col="date", target_col="value", horizon=1)
        model = ArimaForecaster(ArimaParams(p=2, d=1, q=1), ctx)
        model.fit(large_daily_df, training_rows(large_daily_df, "date", TRAINING))
        y = large_daily_df["value"].to_numpy()
        blocks = [Block(i, i + 1) for i in range(120, 150)] + [Block(i, i + 1) for i in range(170, 180)]
        fast = np.concatenate(model._one_step_ahead(y, blocks))
        slow = np.concatenate(model._multi_step(y, blocks))
        assert np.allclose(fast, slow)

    def test_gap_between_train_and_prediction(self):
        """ARIMA must use actual observations between training end and prediction start."""
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(300)]
        df = pl.DataFrame({"date": dates, "value": [float(i) for i in range(300)]})

        result = evaluate(df, "ARIMA", {"p": 1, "d": 1, "q": 0}, horizon=5,
                          training=[DateRange("2023-01-01", "2023-06-01")],
                          prediction=[DateRange("2023-09-01", "2023-09-10")])

        first = result["forecast"][0]
        # Linear series: the first prediction must be close to the actual value (243),
        # not to the end of training (~151)
        assert abs(first["prediction"] - first["value"]) < 5


@pytest.mark.skipif(not PROPHET_AVAILABLE, reason="Prophet not installed")
class TestProphet:
    """Prophet reports metrics by horizon like the other models."""

    def test_horizon_steps(self, large_daily_df):
        result = evaluate(large_daily_df, "PROPHET", {}, horizon=7)
        assert [f["horizon_step"] for f in result["forecast"][:8]] == [1, 2, 3, 4, 5, 6, 7, 1]
        assert len(result["metrics_by_horizon"]) == 7

    def test_short_lag_regressor_rejected(self, large_daily_df):
        with pytest.raises(ValueError, match="< horizon 7"):
            evaluate(large_daily_df, "PROPHET", {"use_lag_regressors": True, "lag_regressors": [1, 7]}, horizon=7)


class TestRegistry:
    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            create_forecaster("NBEATS", {}, ForecastContext("date", "value", 1))

    def test_invalid_params_are_readable(self):
        with pytest.raises(ValueError, match="n_estimators"):
            create_forecaster("XGBOOST", {"n_estimators": 100000}, ForecastContext("date", "value", 1))

    def test_unknown_params_ignored(self):
        """The UI sends extra keys (e.g. lags for Prophet): they must not fail."""
        create_forecaster("ARIMA", {"p": 1, "lags": [1, 7]}, ForecastContext("date", "value", 1))


class TestTrainingService:
    def test_metrics_format(self, large_daily_df):
        """All models return the same metric keys and forecast format."""
        models = [type("M", (), {"id": t, "name": t, "type": t, "params": p})
                  for t, p in [("LAG", {"lag": 1}), ("LINEAR_REGRESSION", {"lags": [1]}), ("ARIMA", {})]]
        results = train_models(large_daily_df, ForecastContext("date", "value", 1), TRAINING, PREDICTION,
                               models, deadline=time.monotonic() + 60)
        for result in results:
            assert set(result["metrics"]) == {"rmse", "mae", "mape", "r2", "msle", "execution_time"}
            for f in result["forecast"]:
                assert set(f) == {"date", "prediction", "value", "horizon_step"}
                assert isinstance(f["prediction"], float)

    def test_time_budget(self, large_daily_df):
        models = [type("M", (), {"id": "lag", "name": "Lag", "type": "LAG", "params": {}})]
        results = train_models(large_daily_df, ForecastContext("date", "value", 1), TRAINING, PREDICTION,
                               models, deadline=time.monotonic() - 1)
        assert "time budget" in results[0]["error"]

    def test_empty_prediction_range_is_a_request_error(self, large_daily_df):
        with pytest.raises(ValueError, match="prediction ranges"):
            train_models(large_daily_df, ForecastContext("date", "value", 1), TRAINING,
                         [DateRange("2030-01-01", "2030-02-01")], [], deadline=time.monotonic() + 60)
