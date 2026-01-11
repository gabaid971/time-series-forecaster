"""Tests for model training functions."""

import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/home/gabaid/workspace/time-series-forecaster/backend')

from models.lag import train_lag
from models.linear_regression import train_linear_regression
from models.arima import train_arima


class DateRange:
    """Mock DateRange for tests."""
    def __init__(self, start, end):
        self.start = start
        self.end = end


class ForecastStrategy:
    """Mock ForecastStrategy for tests."""
    def __init__(self, horizon=1):
        self.horizon = horizon


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
    
    return pl.DataFrame({
        "date": dates,
        "value": values
    })


class TestTrainLag:
    """Tests for LAG model training."""
    
    def test_basic_training(self, large_daily_df):
        """Should train and return valid metrics."""
        result = train_lag(
            df=large_daily_df,
            date_col="date",
            target_col="value",
            training_ranges=[DateRange("2023-01-01", "2023-05-01")],
            prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
            params={"lag": 1}
        )
        
        assert "metrics" in result
        assert "forecast" in result
        assert result["metrics"]["rmse"] >= 0
        assert result["metrics"]["mae"] >= 0
        assert len(result["forecast"]) > 0
    
    def test_different_lags(self, large_daily_df):
        """Different lag values should work."""
        for lag in [1, 7, 14]:
            result = train_lag(
                df=large_daily_df,
                date_col="date",
                target_col="value",
                training_ranges=[DateRange("2023-01-01", "2023-05-01")],
                prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
                params={"lag": lag}
            )
            
            assert len(result["forecast"]) > 0
    
    def test_multi_horizon(self, large_daily_df):
        """Multi-horizon should add horizon_step to forecasts."""
        result = train_lag(
            df=large_daily_df,
            date_col="date",
            target_col="value",
            training_ranges=[DateRange("2023-01-01", "2023-05-01")],
            prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
            params={"lag": 1},
            forecast_strategy=ForecastStrategy(horizon=7)
        )
        
        assert "metrics_by_horizon" in result
        assert len(result["metrics_by_horizon"]) > 0
        
        # Check horizon steps exist in forecasts
        horizon_steps = set(f.get("horizon_step", 1) for f in result["forecast"])
        assert len(horizon_steps) > 1


class TestTrainLinearRegression:
    """Tests for Linear Regression model training."""
    
    def test_basic_training(self, large_daily_df):
        """Should train and return valid metrics."""
        result = train_linear_regression(
            df=large_daily_df,
            date_col="date",
            target_col="value",
            training_ranges=[DateRange("2023-01-01", "2023-05-01")],
            prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
            params={"lags": [1, 7]}
        )
        
        assert "metrics" in result
        assert "forecast" in result
        assert "feature_importance" in result
        assert len(result["feature_importance"]) > 0
    
    def test_with_temporal_features(self, large_daily_df):
        """Should work with temporal features."""
        result = train_linear_regression(
            df=large_daily_df,
            date_col="date",
            target_col="value",
            training_ranges=[DateRange("2023-01-01", "2023-05-01")],
            prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
            params={
                "feature_config": {
                    "target_lags": [1, 7],
                    "temporal": {
                        "day_of_week": True,
                        "month": True
                    }
                }
            }
        )
        
        # Should have temporal features in importance
        feature_names = [f["feature"] for f in result["feature_importance"]]
        assert any("dow" in f or "month" in f for f in feature_names)
    
    def test_residual_mode(self, large_daily_df):
        """Residual mode should work."""
        result = train_linear_regression(
            df=large_daily_df,
            date_col="date",
            target_col="value",
            training_ranges=[DateRange("2023-01-01", "2023-05-01")],
            prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
            params={
                "lags": [1, 7],
                "target_mode": "residual",
                "residual_lag": 1
            }
        )
        
        assert len(result["forecast"]) > 0
    
    def test_standardization(self, large_daily_df):
        """Standardization option should work."""
        result = train_linear_regression(
            df=large_daily_df,
            date_col="date",
            target_col="value",
            training_ranges=[DateRange("2023-01-01", "2023-05-01")],
            prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
            params={
                "lags": [1, 7],
                "standardize": True
            }
        )
        
        assert len(result["forecast"]) > 0
    
    def test_multi_horizon_with_metrics(self, large_daily_df):
        """Multi-horizon should provide per-horizon metrics."""
        result = train_linear_regression(
            df=large_daily_df,
            date_col="date",
            target_col="value",
            training_ranges=[DateRange("2023-01-01", "2023-05-01")],
            prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
            params={"lags": [1, 7]},
            forecast_strategy=ForecastStrategy(horizon=7)
        )
        
        assert result["metrics_by_horizon"] is not None
        assert len(result["metrics_by_horizon"]) == 7


class TestTrainARIMA:
    """Tests for ARIMA model training."""
    
    def test_basic_training(self, large_daily_df):
        """Should train ARIMA and return valid metrics."""
        result = train_arima(
            df=large_daily_df,
            date_col="date",
            target_col="value",
            training_ranges=[DateRange("2023-01-01", "2023-05-01")],
            prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
            params={"p": 1, "d": 1, "q": 1},
            horizon=1
        )
        
        assert "metrics" in result
        assert "forecast" in result
        assert result["feature_importance"] is None  # ARIMA has no features
    
    def test_different_orders(self, large_daily_df):
        """Different ARIMA orders should work."""
        orders = [(1, 0, 0), (0, 1, 1), (2, 1, 2)]
        
        for p, d, q in orders:
            result = train_arima(
                df=large_daily_df,
                date_col="date",
                target_col="value",
                training_ranges=[DateRange("2023-01-01", "2023-05-01")],
                prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
                params={"p": p, "d": d, "q": q},
                horizon=1
            )
            
            assert len(result["forecast"]) > 0
    
    def test_multi_horizon(self, large_daily_df):
        """Multi-horizon should track horizon steps."""
        result = train_arima(
            df=large_daily_df,
            date_col="date",
            target_col="value",
            training_ranges=[DateRange("2023-01-01", "2023-05-01")],
            prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
            params={"p": 1, "d": 1, "q": 1},
            horizon=7
        )
        
        assert len(result["metrics_by_horizon"]) > 0
        
        # Should have predictions for each horizon step
        horizon_steps = set(f["horizon_step"] for f in result["forecast"])
        assert len(horizon_steps) == 7
    
    def test_insufficient_data_raises(self, large_daily_df):
        """Should raise error with insufficient training data."""
        small_df = large_daily_df.head(5)
        
        with pytest.raises(ValueError, match="Not enough"):
            train_arima(
                df=small_df,
                date_col="date",
                target_col="value",
                training_ranges=[DateRange("2023-01-01", "2023-01-05")],
                prediction_ranges=[DateRange("2023-01-05", "2023-01-06")],
                params={"p": 5, "d": 1, "q": 5},
                horizon=1
            )


class TestModelMetricsFormat:
    """Tests to ensure all models return consistent metric formats."""
    
    def test_all_metrics_present(self, large_daily_df):
        """All models should return same metric keys."""
        required_metrics = {"rmse", "mae", "mape", "r2", "msle", "execution_time"}
        
        # LAG
        lag_result = train_lag(
            df=large_daily_df,
            date_col="date", target_col="value",
            training_ranges=[DateRange("2023-01-01", "2023-05-01")],
            prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
            params={"lag": 1}
        )
        assert required_metrics.issubset(set(lag_result["metrics"].keys()))
        
        # Linear Regression
        lr_result = train_linear_regression(
            df=large_daily_df,
            date_col="date", target_col="value",
            training_ranges=[DateRange("2023-01-01", "2023-05-01")],
            prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
            params={"lags": [1]}
        )
        assert required_metrics.issubset(set(lr_result["metrics"].keys()))
        
        # ARIMA
        arima_result = train_arima(
            df=large_daily_df,
            date_col="date", target_col="value",
            training_ranges=[DateRange("2023-01-01", "2023-05-01")],
            prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
            params={"p": 1, "d": 1, "q": 1},
            horizon=1
        )
        assert required_metrics.issubset(set(arima_result["metrics"].keys()))
    
    def test_forecast_format(self, large_daily_df):
        """Forecasts should have consistent format."""
        result = train_linear_regression(
            df=large_daily_df,
            date_col="date", target_col="value",
            training_ranges=[DateRange("2023-01-01", "2023-05-01")],
            prediction_ranges=[DateRange("2023-05-01", "2023-06-01")],
            params={"lags": [1]}
        )
        
        for forecast in result["forecast"]:
            assert "date" in forecast
            assert "prediction" in forecast
            assert "value" in forecast
            assert isinstance(forecast["prediction"], float)
