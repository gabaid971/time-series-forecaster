"""Utility functions for time series forecasting."""

from utils.metrics import calculate_metrics, calculate_metrics_by_horizon
from utils.features import build_features, FeatureConfig, TemporalFeatureConfig, ExogenousFeatureConfig, DerivedFeatureConfig
from utils.date_utils import detect_frequency, parse_dates_flexible, filter_by_date_range
from utils.validation import block_recursive_forecast
from utils.analysis import (
    compute_acf,
    compute_pacf,
    suggest_lags,
    detect_outliers,
    detect_trend,
    compute_stationarity_indicators
)

__all__ = [
    # Metrics
    "calculate_metrics",
    "calculate_metrics_by_horizon",
    # Features
    "build_features",
    "FeatureConfig",
    "TemporalFeatureConfig",
    "ExogenousFeatureConfig",
    "DerivedFeatureConfig",
    # Date utilities
    "detect_frequency",
    "parse_dates_flexible",
    "filter_by_date_range",
    # Validation
    "block_recursive_forecast",
    # Analysis
    "compute_acf",
    "compute_pacf",
    "suggest_lags",
    "detect_outliers",
    "detect_trend",
    "compute_stationarity_indicators",
]
