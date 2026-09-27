"""Tests for forecasting beyond the data (/forecast)."""

import time
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from app.forecasting.future import divergence_warning, forecast_models, future_dates
from app.forecasting.models import Block, ForecastContext, create_forecaster
from app.main import app


def model(model_type, params=None, model_id=None):
    return type("M", (), {"id": model_id or model_type, "name": model_id or model_type, "type": model_type, "params": params or {}})


@pytest.fixture
def daily_df():
    np.random.seed(0)
    n = 300
    return pl.DataFrame({
        "date": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)],
        "value": [20 + 5 * np.sin(2 * np.pi * i / 7) + np.random.randn() for i in range(n)],
        "sensor": [float(i % 13) for i in range(n)],
    })


class TestFutureDates:
    def test_daily(self, daily_df):
        dates = future_dates(daily_df, "date", "D", 3)
        assert dates == [datetime(2023, 10, 28), datetime(2023, 10, 29), datetime(2023, 10, 30)]

    def test_monthly_keeps_month_ends(self):
        df = pl.DataFrame({"date": [datetime(2023, m, 1) for m in range(1, 13)] + [datetime(2024, 1, 31)]})
        assert future_dates(df, "date", "M", 2) == [datetime(2024, 2, 29), datetime(2024, 3, 31)]

    def test_minute(self):
        df = pl.DataFrame({"date": [datetime(2024, 3, 1) + timedelta(minutes=i) for i in range(10)]})
        assert future_dates(df, "date", "min", 2) == [datetime(2024, 3, 1, 0, 10), datetime(2024, 3, 1, 0, 11)]


    def test_duplicated_dates_do_not_give_a_zero_interval(self):
        base = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        df = pl.DataFrame({"date": sorted(base + base)})  # Every date twice
        assert future_dates(df, "date", "D", 2) == [datetime(2024, 1, 11), datetime(2024, 1, 12)]

    def test_single_date_is_an_error(self):
        with pytest.raises(ValueError, match="two distinct dates"):
            future_dates(pl.DataFrame({"date": [datetime(2024, 1, 1)] * 3}), "date", "D", 2)


class TestDivergence:
    def test_stable_forecast_has_no_warning(self):
        assert divergence_warning(np.array([10.0, 12.0, 11.0]), np.array([5.0, 15.0])) is None

    def test_exploding_forecast_is_flagged(self):
        assert "diverges" in divergence_warning(np.array([10.0, 1e9]), np.array([5.0, 15.0]))

    def test_unstable_recursive_model_is_flagged(self):
        """
        With an exogenous variable, this linear regression learns an autoregressive
        coefficient above 1: feeding on its own forecasts for 365 steps, it explodes.
        """
        np.random.seed(0)
        n = 24 * 40
        values = [20 + 5 * np.sin(2 * np.pi * i / 24) + 0.05 * i + np.random.randn() for i in range(n)]
        df = pl.DataFrame({
            "date": [datetime(2020, 1, 1) + i * timedelta(hours=1) for i in range(n)],
            "value": values,
            "sensor": [v + np.random.randn() for v in values],
        })
        unstable = model("LINEAR_REGRESSION", {"lags": [1], "feature_config": {"exogenous": [{"column": "sensor", "lags": [365, 366]}]}})
        output = forecast_models(df, "date", "value", [unstable], 365, time.monotonic() + 60)
        assert "diverges" in output["results"][0]["warning"]


class TestForecastModels:
    @pytest.mark.parametrize("model_type,params", [
        ("LINEAR_REGRESSION", {"lags": [1, 7], "feature_config": {"temporal": {"day_of_week": True}}}),
        ("XGBOOST", {"lags": [1, 7], "n_estimators": 20}),
        ("ARIMA", {"p": 1, "d": 0, "q": 1}),
        ("LAG", {"lag": 1}),
    ])
    def test_same_as_evaluation_engine(self, daily_df, model_type, params):
        """
        Forecasting the future from data cut at T gives exactly the forecast of the
        evaluation engine for a block starting at T: both paths share the model code.
        """
        cut, steps = 250, 10
        output = forecast_models(daily_df.head(cut), "date", "value", [model(model_type, params)], steps, time.monotonic() + 60)
        future = [p["prediction"] for p in output["results"][0]["forecast"]]

        forecaster = create_forecaster(model_type, params, ForecastContext("date", "value", steps))
        forecaster.fit(daily_df, np.arange(cut))
        evaluated = forecaster.predict_blocks(daily_df, [Block(cut, cut + steps)])[0]

        assert future == pytest.approx(list(evaluated), rel=1e-6)

    def test_unknown_exogenous_needs_long_enough_lags(self, daily_df):
        short = model("LINEAR_REGRESSION", {"lags": [1], "feature_config": {"exogenous": [{"column": "sensor", "lags": [3]}]}})
        long = model("LINEAR_REGRESSION", {"lags": [1], "feature_config": {"exogenous": [{"column": "sensor", "lags": [14]}]}}, "long")
        output = forecast_models(daily_df, "date", "value", [short, long], 14, time.monotonic() + 60)
        assert "< horizon 14" in output["results"][0]["error"]
        assert len(output["results"][1]["forecast"]) == 14

    def test_lag_longer_than_history_is_explained(self, daily_df):
        too_long = model("LINEAR_REGRESSION", {"lags": [1], "feature_config": {"exogenous": [{"column": "sensor", "lags": [400]}]}})
        output = forecast_models(daily_df, "date", "value", [too_long], 400, time.monotonic() + 60)
        assert "longest lag (400)" in output["results"][0]["error"]

    def test_known_in_advance_not_supported_yet(self, daily_df):
        known = model("LINEAR_REGRESSION", {"lags": [1], "feature_config": {"exogenous": [{"column": "sensor", "lags": [0], "known_in_advance": True}]}})
        output = forecast_models(daily_df, "date", "value", [known], 5, time.monotonic() + 60)
        assert "future values" in output["results"][0]["error"]


class TestForecastEndpoint:
    def test_forecast(self, daily_df):
        data = [{"date": d.strftime("%Y-%m-%d"), "value": v} for d, v in zip(daily_df["date"], daily_df["value"])]
        response = TestClient(app).post("/forecast", json={
            "data": data, "date_column": "date", "target_column": "value", "steps": 5,
            "models": [{"id": "lr", "type": "LINEAR_REGRESSION", "name": "LR", "params": {"lags": [1, 7]}}],
        })
        assert response.status_code == 200
        body = response.json()
        assert body["frequency"] == "D"
        assert body["last_date"].startswith("2023-10-27")
        forecast = body["results"][0]["forecast"]
        assert [p["step"] for p in forecast] == [1, 2, 3, 4, 5]
        assert forecast[0]["date"].startswith("2023-10-28")

    def test_steps_bounds(self, daily_df):
        response = TestClient(app).post("/forecast", json={
            "data": [{"date": "2023-01-01", "value": 1}], "date_column": "date", "target_column": "value", "steps": 0,
            "models": [{"id": "lag", "type": "LAG", "name": "Lag", "params": {}}],
        })
        assert response.status_code == 422
