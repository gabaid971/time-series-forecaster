"""Tests for time series analysis utilities."""

import pytest
import numpy as np
from utils.analysis import (
    compute_acf,
    compute_pacf,
    suggest_lags,
    detect_outliers,
    detect_trend,
    compute_stationarity_indicators
)


class TestACF:
    """Tests for ACF computation."""
    
    def test_acf_lag_zero_is_one(self):
        """ACF at lag 0 should always be 1."""
        series = np.random.randn(100)
        acf = compute_acf(series, max_lag=10)
        assert acf[0] == 1.0
    
    def test_acf_constant_series(self):
        """ACF of constant series should be 1 at lag 0, ~0 elsewhere."""
        series = np.ones(100)
        acf = compute_acf(series, max_lag=10)
        assert acf[0] == 1.0
        # Other values should be 0 for constant
        for v in acf[1:]:
            assert v == 0.0
    
    def test_acf_seasonal_series(self):
        """ACF should detect weekly seasonality."""
        n = 365
        series = np.sin(2 * np.pi * np.arange(n) / 7)
        acf = compute_acf(series, max_lag=14)
        
        # ACF at lag 7 should be high (peak)
        assert acf[7] > 0.5
    
    def test_acf_short_series(self):
        """ACF on short series should not crash."""
        series = np.array([1, 2, 3])
        acf = compute_acf(series, max_lag=10)
        assert len(acf) == 1  # Only lag 0


class TestPACF:
    """Tests for PACF computation."""
    
    def test_pacf_lag_zero_is_one(self):
        """PACF at lag 0 should always be 1."""
        series = np.random.randn(100)
        pacf = compute_pacf(series, max_lag=10)
        assert pacf[0] == 1.0
    
    def test_pacf_ar1_process(self):
        """PACF of AR(1) should have significant first lag only."""
        np.random.seed(42)
        n = 500
        phi = 0.8
        series = np.zeros(n)
        for i in range(1, n):
            series[i] = phi * series[i-1] + np.random.randn()
        
        pacf = compute_pacf(series, max_lag=10)
        
        # First lag should be close to phi
        assert abs(pacf[1] - phi) < 0.2
        # Subsequent lags should be small
        assert abs(pacf[5]) < 0.2


class TestSuggestLags:
    """Tests for lag suggestion functionality."""
    
    def test_suggest_lags_returns_expected_structure(self):
        """Should return all expected keys."""
        series = np.random.randn(100)
        result = suggest_lags(series, frequency='D', max_lags=10)
        
        assert 'suggested_lags' in result
        assert 'acf' in result
        assert 'pacf' in result
        assert 'confidence_interval' in result
        assert 'significant_lags' in result
        assert 'seasonality' in result
        assert 'n_observations' in result
    
    def test_suggest_lags_includes_lag_one(self):
        """Lag 1 should always be included."""
        series = np.random.randn(100)
        result = suggest_lags(series, frequency='D', max_lags=10)
        
        assert 1 in result['suggested_lags']
    
    def test_suggest_lags_weekly_seasonality(self):
        """Should detect weekly seasonality for daily data."""
        n = 365
        series = 50 + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + np.random.randn(n)
        result = suggest_lags(series, frequency='D', max_lags=20)
        
        # Should detect seasonality
        assert result['seasonality']['detected'] == True
        # Should suggest lag 7 for weekly
        assert 7 in result['suggested_lags']
    
    def test_suggest_lags_hourly_data(self):
        """Should add lag 24 for hourly data."""
        n = 24 * 30  # 30 days of hourly data
        series = 50 + 5 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.randn(n)
        result = suggest_lags(series, frequency='H', max_lags=30)
        
        # Should include 24 for daily pattern
        assert 24 in result['suggested_lags'] or result['seasonality'].get('period') == 24


class TestDetectOutliers:
    """Tests for outlier detection."""
    
    def test_detect_outliers_no_outliers(self):
        """Normal distribution should have few outliers."""
        np.random.seed(42)
        series = np.random.randn(100)
        result = detect_outliers(series, method='iqr')
        
        # Few or no outliers expected
        assert result['count'] < 10
    
    def test_detect_outliers_with_extreme_values(self):
        """Should detect extreme values."""
        series = np.concatenate([
            np.random.randn(95),
            np.array([100, -100, 200, -200, 500])  # Extreme values
        ])
        result = detect_outliers(series, method='iqr')
        
        # Should detect at least the extreme values
        assert result['count'] >= 5
    
    def test_detect_outliers_zscore_method(self):
        """Z-score method should work."""
        series = np.concatenate([
            np.random.randn(95),
            np.array([10, -10])  # Outliers (>3 sigma)
        ])
        result = detect_outliers(series, method='zscore')
        
        assert result['method'] == 'zscore'
        assert result['count'] >= 2


class TestDetectTrend:
    """Tests for trend detection."""
    
    def test_detect_trend_upward(self):
        """Should detect upward trend."""
        series = np.linspace(0, 100, 100) + np.random.randn(100) * 5
        result = detect_trend(series)
        
        assert result['detected'] == True
        assert result['direction'] == 'upward'
    
    def test_detect_trend_downward(self):
        """Should detect downward trend."""
        series = np.linspace(100, 0, 100) + np.random.randn(100) * 5
        result = detect_trend(series)
        
        assert result['detected'] == True
        assert result['direction'] == 'downward'
    
    def test_detect_trend_stationary(self):
        """Should not detect trend in stationary series."""
        np.random.seed(42)
        series = 50 + np.random.randn(100)
        result = detect_trend(series)
        
        assert result['detected'] == False
        assert result['direction'] == 'none'


class TestStationarityIndicators:
    """Tests for stationarity indicators."""
    
    def test_stationarity_stationary_series(self):
        """Stationary series should be detected as likely stationary."""
        np.random.seed(42)
        series = 50 + np.random.randn(200)
        result = compute_stationarity_indicators(series)
        
        assert result['likely_stationary'] == True
    
    def test_stationarity_trending_series(self):
        """Trending series should be detected as non-stationary."""
        series = np.linspace(0, 100, 200) + np.random.randn(200)
        result = compute_stationarity_indicators(series)
        
        assert result['likely_stationary'] == False
    
    def test_stationarity_short_series(self):
        """Short series should return appropriate message."""
        series = np.random.randn(10)
        result = compute_stationarity_indicators(series)
        
        assert result['likely_stationary'] is None
        assert 'Not enough data' in result['message']
