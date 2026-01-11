"""Tests for date utilities."""

import pytest
import polars as pl
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/home/gabaid/workspace/time-series-forecaster/backend')

from utils.date_utils import detect_frequency, parse_dates_flexible, filter_by_date_range


class TestDetectFrequency:
    """Tests for detect_frequency function."""
    
    def test_daily_frequency(self, sample_daily_df):
        """Should detect daily frequency."""
        freq_code, freq_label, missing = detect_frequency(sample_daily_df, "date")
        
        assert freq_code == "D"
        assert freq_label == "Daily"
        assert missing == 0
    
    def test_hourly_frequency(self, sample_hourly_df):
        """Should detect hourly frequency."""
        freq_code, freq_label, missing = detect_frequency(sample_hourly_df, "date")
        
        assert freq_code == "H"
        assert freq_label == "Hourly"
    
    def test_minutely_frequency(self):
        """Should detect minutely frequency."""
        dates = [datetime(2023, 1, 1) + timedelta(minutes=i) for i in range(60)]
        df = pl.DataFrame({"date": dates, "value": range(60)})
        
        freq_code, freq_label, _ = detect_frequency(df, "date")
        
        assert freq_code == "min"
        assert freq_label == "Minutely"
    
    def test_weekly_frequency(self):
        """Should detect weekly frequency."""
        dates = [datetime(2023, 1, 1) + timedelta(weeks=i) for i in range(10)]
        df = pl.DataFrame({"date": dates, "value": range(10)})
        
        freq_code, freq_label, _ = detect_frequency(df, "date")
        
        assert freq_code == "W"
        assert freq_label == "Weekly"
    
    def test_missing_dates_detected(self):
        """Should count missing dates."""
        # Create daily series with a gap
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
        dates.pop(5)  # Remove one date to create gap
        df = pl.DataFrame({"date": dates, "value": range(9)})
        
        _, _, missing = detect_frequency(df, "date")
        
        assert missing >= 1
    
    def test_single_row_returns_unknown(self):
        """Single row should return unknown frequency."""
        df = pl.DataFrame({"date": [datetime(2023, 1, 1)], "value": [1]})
        
        freq_code, freq_label, _ = detect_frequency(df, "date")
        
        assert freq_code == "unknown"


class TestParseDatesFlexible:
    """Tests for parse_dates_flexible function."""
    
    def test_iso_format(self):
        """Should parse ISO format dates."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "value": [1, 2, 3]
        })
        
        result = parse_dates_flexible(df, "date")
        
        assert result["date"].dtype == pl.Datetime
        assert result.height == 3
    
    def test_us_format(self):
        """Should parse US format (M/D/Y)."""
        df = pl.DataFrame({
            "date": ["1/15/2023", "2/20/2023", "3/25/2023"],
            "value": [1, 2, 3]
        })
        
        result = parse_dates_flexible(df, "date")
        
        assert result["date"].dtype == pl.Datetime
        assert result.height == 3
    
    def test_european_format(self):
        """Should parse European format (D/M/Y) when day > 12."""
        df = pl.DataFrame({
            "date": ["15/01/2023", "20/02/2023", "25/03/2023"],
            "value": [1, 2, 3]
        })
        
        result = parse_dates_flexible(df, "date")
        
        assert result["date"].dtype == pl.Datetime
    
    def test_raises_on_invalid_dates(self):
        """Should raise ValueError on completely invalid dates."""
        df = pl.DataFrame({
            "date": ["not_a_date", "also_not", "nope"],
            "value": [1, 2, 3]
        })
        
        with pytest.raises(ValueError):
            parse_dates_flexible(df, "date")


class TestFilterByDateRange:
    """Tests for filter_by_date_range function."""
    
    def test_inclusive_end(self, sample_daily_df):
        """Should include end date when inclusive_end=True."""
        result = filter_by_date_range(
            sample_daily_df, "date", 
            "2023-01-10", "2023-01-20",
            inclusive_end=True
        )
        
        max_date = result["date"].max()
        assert max_date.date() == datetime(2023, 1, 20).date()
    
    def test_exclusive_end(self, sample_daily_df):
        """Should exclude end date when inclusive_end=False."""
        result = filter_by_date_range(
            sample_daily_df, "date",
            "2023-01-10", "2023-01-20",
            inclusive_end=False
        )
        
        max_date = result["date"].max()
        assert max_date.date() < datetime(2023, 1, 20).date()
    
    def test_correct_row_count(self, sample_daily_df):
        """Should return correct number of rows."""
        result = filter_by_date_range(
            sample_daily_df, "date",
            "2023-01-10", "2023-01-20",
            inclusive_end=True
        )
        
        assert result.height == 11  # 10 to 20 inclusive = 11 days
    
    def test_empty_result_for_invalid_range(self, sample_daily_df):
        """Should return empty DataFrame for range outside data."""
        result = filter_by_date_range(
            sample_daily_df, "date",
            "2024-01-01", "2024-01-10"
        )
        
        assert result.height == 0
