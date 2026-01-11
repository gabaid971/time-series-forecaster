"""Shared fixtures for tests."""

import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_daily_df():
    """Create a sample daily time series DataFrame."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    values = [100 + 10 * np.sin(2 * np.pi * i / 7) + np.random.randn() * 2 for i in range(100)]
    
    return pl.DataFrame({
        "date": dates,
        "value": values
    })


@pytest.fixture
def sample_hourly_df():
    """Create a sample hourly time series DataFrame."""
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(240)]
    values = [50 + 20 * np.sin(2 * np.pi * i / 24) + np.random.randn() * 3 for i in range(240)]
    
    return pl.DataFrame({
        "date": dates,
        "value": values
    })


@pytest.fixture
def sample_df_with_exogenous():
    """Create a DataFrame with exogenous variables."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    values = [100 + 10 * np.sin(2 * np.pi * i / 7) for i in range(100)]
    temperature = [20 + 5 * np.sin(2 * np.pi * i / 365) for i in range(100)]
    is_weekend = [(datetime(2023, 1, 1) + timedelta(days=i)).weekday() >= 5 for i in range(100)]
    
    return pl.DataFrame({
        "date": dates,
        "value": values,
        "temperature": temperature,
        "is_weekend": is_weekend
    })


@pytest.fixture
def training_range():
    """Simple training range."""
    class DateRange:
        def __init__(self, start, end):
            self.start = start
            self.end = end
    
    return DateRange("2023-01-01", "2023-03-01")


@pytest.fixture
def prediction_range():
    """Simple prediction range."""
    class DateRange:
        def __init__(self, start, end):
            self.start = start
            self.end = end
    
    return DateRange("2023-03-01", "2023-04-01")
