"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import numpy as np
import main
from main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_data():
    """Generate sample time series data."""
    np.random.seed(42)
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    values = [100 + 10 * np.sin(2 * np.pi * i / 7) + np.random.randn() * 2 for i in range(100)]
    
    return [
        {"date": d.strftime("%Y-%m-%d"), "value": v}
        for d, v in zip(dates, values)
    ]


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_root(self, client):
        """Root endpoint should return OK."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_health(self, client):
        """Health endpoint should return healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestAnalyzeEndpoint:
    """Tests for /analyze endpoint."""
    
    def test_analyze_valid_data(self, client, sample_data):
        """Should analyze valid data successfully."""
        response = client.post(
            "/analyze",
            json={
                "data": sample_data,
                "date_column": "date",
                "target_column": "value"
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert result["stats"] is not None
        assert result["stats"]["total_rows"] == 100
        assert result["stats"]["frequency"] == "D"
    
    def test_analyze_returns_normalized_data(self, client, sample_data):
        """Should return normalized data for frontend."""
        response = client.post(
            "/analyze",
            json={
                "data": sample_data,
                "date_column": "date",
                "target_column": "value"
            }
        )
        
        result = response.json()
        assert result["normalized_data"] is not None
        assert len(result["normalized_data"]) == 100


class TestTrainEndpoint:
    """Tests for /train endpoint."""
    
    def test_train_lag_model(self, client, sample_data):
        """Should train LAG model successfully."""
        response = client.post(
            "/train",
            json={
                "data": sample_data,
                "data_config": {
                    "target_column": "value",
                    "date_column": "date",
                    "frequency": "D",
                    "training_ranges": [{"start": "2023-01-01", "end": "2023-03-01"}],
                    "prediction_ranges": [{"start": "2023-03-01", "end": "2023-04-01"}],
                    "forecast_strategy": {"horizon": 1}
                },
                "models": [
                    {"id": "lag-1", "type": "LAG", "name": "LAG-1", "params": {"lag": 1}}
                ]
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert len(result["results"]) == 1
        assert result["results"][0]["model_id"] == "lag-1"
    
    def test_train_linear_regression(self, client, sample_data):
        """Should train Linear Regression model successfully."""
        response = client.post(
            "/train",
            json={
                "data": sample_data,
                "data_config": {
                    "target_column": "value",
                    "date_column": "date",
                    "frequency": "D",
                    "training_ranges": [{"start": "2023-01-01", "end": "2023-03-01"}],
                    "prediction_ranges": [{"start": "2023-03-01", "end": "2023-04-01"}],
                    "forecast_strategy": {"horizon": 1}
                },
                "models": [
                    {"id": "lr-1", "type": "LINEAR_REGRESSION", "name": "LR", "params": {"lags": [1, 7]}}
                ]
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert result["results"][0]["feature_importance"] is not None
    
    def test_train_multiple_models(self, client, sample_data):
        """Should train multiple models in one request."""
        response = client.post(
            "/train",
            json={
                "data": sample_data,
                "data_config": {
                    "target_column": "value",
                    "date_column": "date",
                    "frequency": "D",
                    "training_ranges": [{"start": "2023-01-01", "end": "2023-03-01"}],
                    "prediction_ranges": [{"start": "2023-03-01", "end": "2023-04-01"}],
                    "forecast_strategy": {"horizon": 1}
                },
                "models": [
                    {"id": "lag-1", "type": "LAG", "name": "LAG-1", "params": {"lag": 1}},
                    {"id": "lr-1", "type": "LINEAR_REGRESSION", "name": "LR", "params": {"lags": [1, 7]}},
                    {"id": "arima-1", "type": "ARIMA", "name": "ARIMA", "params": {"p": 1, "d": 1, "q": 1}}
                ]
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert len(result["results"]) == 3
    
    def test_train_with_multi_horizon(self, client, sample_data):
        """Should return metrics by horizon for multi-step forecast."""
        response = client.post(
            "/train",
            json={
                "data": sample_data,
                "data_config": {
                    "target_column": "value",
                    "date_column": "date",
                    "frequency": "D",
                    "training_ranges": [{"start": "2023-01-01", "end": "2023-03-01"}],
                    "prediction_ranges": [{"start": "2023-03-01", "end": "2023-04-01"}],
                    "forecast_strategy": {"horizon": 7}
                },
                "models": [
                    {"id": "lr-1", "type": "LINEAR_REGRESSION", "name": "LR", "params": {"lags": [1, 7]}}
                ]
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["results"][0]["metrics_by_horizon"] is not None
        assert len(result["results"][0]["metrics_by_horizon"]) == 7
    
    def test_train_handles_model_error(self, client):
        """Should handle model training errors gracefully."""
        # Very small dataset that will cause issues
        small_data = [
            {"date": "2023-01-01", "value": 1},
            {"date": "2023-01-02", "value": 2}
        ]
        
        response = client.post(
            "/train",
            json={
                "data": small_data,
                "data_config": {
                    "target_column": "value",
                    "date_column": "date",
                    "frequency": "D",
                    "training_ranges": [{"start": "2023-01-01", "end": "2023-01-02"}],
                    "prediction_ranges": [{"start": "2023-01-02", "end": "2023-01-03"}],
                    "forecast_strategy": {"horizon": 1}
                },
                "models": [
                    {"id": "lr-1", "type": "LINEAR_REGRESSION", "name": "LR", "params": {"lags": [1, 7, 14]}}
                ]
            }
        )
        
        # Should still return 200 with error in result
        assert response.status_code == 200
        result = response.json()
        assert result["results"][0]["error"] is not None


def _train_payload(data, models, horizon=1):
    return {
        "data": data,
        "data_config": {
            "target_column": "value",
            "date_column": "date",
            "frequency": "D",
            "training_ranges": [{"start": "2023-01-01", "end": "2023-03-01"}],
            "prediction_ranges": [{"start": "2023-03-01", "end": "2023-04-01"}],
            "forecast_strategy": {"horizon": horizon}
        },
        "models": models
    }


class TestLimits:
    """The API is public: oversized requests must be rejected."""

    def test_too_many_rows(self, client, sample_data, monkeypatch):
        monkeypatch.setattr(main, "MAX_ROWS", 50)
        response = client.post("/analyze", json={
            "data": sample_data, "date_column": "date", "target_column": "value"
        })
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]

    def test_too_many_models(self, client, sample_data, monkeypatch):
        monkeypatch.setattr(main, "MAX_MODELS", 1)
        lag = {"id": "l", "type": "LAG", "name": "L", "params": {"lag": 1}}
        response = client.post("/train", json=_train_payload(sample_data, [lag, lag]))
        assert response.status_code == 413

    def test_body_too_large(self, client, monkeypatch):
        monkeypatch.setattr(main, "MAX_BODY_MB", 0.001)
        response = client.post("/analyze", json={
            "data": [{"date": "2023-01-01", "value": i} for i in range(100)],
            "date_column": "date", "target_column": "value"
        })
        assert response.status_code == 413


class TestCors:
    """Only configured origins are allowed by browsers."""

    def test_allowed_origin(self, client):
        response = client.options("/train", headers={
            "Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"
        })
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_other_origin_rejected(self, client):
        response = client.options("/train", headers={
            "Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"
        })
        assert "access-control-allow-origin" not in response.headers
