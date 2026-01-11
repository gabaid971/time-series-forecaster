"""Tests for metrics utilities."""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/gabaid/workspace/time-series-forecaster/backend')

from utils.metrics import calculate_metrics, calculate_metrics_by_horizon


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""
    
    def test_perfect_prediction(self):
        """Perfect predictions should give RMSE=0, MAE=0, R2=1."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0
        assert metrics["msle"] == 0.0
    
    def test_rmse_calculation(self):
        """RMSE should be sqrt(mean(squared errors))."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])  # Error = 1 for each
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics["rmse"] == pytest.approx(1.0)
        assert metrics["mae"] == pytest.approx(1.0)
    
    def test_mape_avoids_division_by_zero(self):
        """MAPE should handle zeros in y_true."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 1, 2])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # MAPE should be calculated only on non-zero values
        assert metrics["mape"] == pytest.approx(0.0)
    
    def test_msle_positive_values_only(self):
        """MSLE should only use positive values."""
        y_true = np.array([-1, 1, 2])
        y_pred = np.array([0, 1, 2])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # MSLE calculated only for positive pairs
        assert metrics["msle"] == 0.0
    
    def test_r2_worst_case(self):
        """R2 should be negative for very bad predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([5, 4, 3, 2, 1])  # Opposite trend
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics["r2"] < 0
    
    def test_handles_flat_input(self):
        """Should handle 2D arrays by flattening."""
        y_true = np.array([[1, 2], [3, 4]])
        y_pred = np.array([[1, 2], [3, 4]])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics["rmse"] == 0.0


class TestCalculateMetricsByHorizon:
    """Tests for calculate_metrics_by_horizon function."""
    
    def test_single_horizon(self):
        """Single horizon should return one entry."""
        forecasts = [
            {"date": "2023-01-01", "prediction": 1.0, "value": 1.0, "horizon_step": 1},
            {"date": "2023-01-02", "prediction": 2.0, "value": 2.0, "horizon_step": 1},
        ]
        
        result = calculate_metrics_by_horizon(forecasts, "value")
        
        assert len(result) == 1
        assert result[0]["horizon_step"] == 1
        assert result[0]["rmse"] == 0.0
        assert result[0]["count"] == 2
    
    def test_multiple_horizons(self):
        """Multiple horizons should be grouped correctly."""
        forecasts = [
            {"date": "2023-01-01", "prediction": 1.0, "value": 1.0, "horizon_step": 1},
            {"date": "2023-01-02", "prediction": 2.0, "value": 2.5, "horizon_step": 2},
            {"date": "2023-01-03", "prediction": 3.0, "value": 4.0, "horizon_step": 3},
            {"date": "2023-01-04", "prediction": 1.5, "value": 1.5, "horizon_step": 1},
        ]
        
        result = calculate_metrics_by_horizon(forecasts, "value")
        
        assert len(result) == 3
        horizons = {r["horizon_step"]: r for r in result}
        
        assert horizons[1]["count"] == 2
        assert horizons[2]["count"] == 1
        assert horizons[3]["count"] == 1
    
    def test_empty_forecasts(self):
        """Empty forecasts should return empty list."""
        result = calculate_metrics_by_horizon([], "value")
        assert result == []
    
    def test_missing_horizon_step_defaults_to_1(self):
        """Missing horizon_step should default to 1."""
        forecasts = [
            {"date": "2023-01-01", "prediction": 1.0, "value": 1.0},
        ]
        
        result = calculate_metrics_by_horizon(forecasts, "value")
        
        assert len(result) == 1
        assert result[0]["horizon_step"] == 1
    
    def test_sorted_by_horizon(self):
        """Results should be sorted by horizon step."""
        forecasts = [
            {"date": "2023-01-01", "prediction": 1.0, "value": 1.0, "horizon_step": 3},
            {"date": "2023-01-02", "prediction": 1.0, "value": 1.0, "horizon_step": 1},
            {"date": "2023-01-03", "prediction": 1.0, "value": 1.0, "horizon_step": 2},
        ]
        
        result = calculate_metrics_by_horizon(forecasts, "value")
        
        assert result[0]["horizon_step"] == 1
        assert result[1]["horizon_step"] == 2
        assert result[2]["horizon_step"] == 3
