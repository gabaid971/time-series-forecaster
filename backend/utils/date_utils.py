"""Date parsing and filtering utilities."""

import polars as pl
import numpy as np
from typing import Tuple


def detect_frequency(df: pl.DataFrame, date_col: str) -> Tuple[str, str, int]:
    """
    Detect the frequency of a time series.
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
    
    Returns:
        (frequency_code, frequency_label, missing_dates_count)
    """
    if df.height < 2:
        return "unknown", "Unknown", 0
    
    # Get sorted dates
    dates = df.sort(date_col).select(date_col).to_series()
    
    # Calculate differences between consecutive dates
    diffs = dates.diff().drop_nulls()
    
    if len(diffs) == 0:
        return "unknown", "Unknown", 0
    
    # Convert to total seconds for comparison
    diff_seconds = [d.total_seconds() for d in diffs.to_list()]
    
    # Find the median difference (more robust than mode)
    median_diff = np.median(diff_seconds)
    
    # Classify frequency
    MINUTE = 60
    HOUR = 3600
    DAY = 86400
    WEEK = 7 * DAY
    MONTH = 30 * DAY
    
    if median_diff < MINUTE:
        freq_code, freq_label = "s", "Secondly"
        expected_diff = median_diff
    elif median_diff < HOUR:
        freq_code, freq_label = "min", "Minutely"
        expected_diff = round(median_diff / MINUTE) * MINUTE
    elif median_diff < DAY:
        freq_code, freq_label = "H", "Hourly"
        expected_diff = round(median_diff / HOUR) * HOUR
    elif median_diff < WEEK:
        freq_code, freq_label = "D", "Daily"
        expected_diff = DAY
    elif median_diff < MONTH:
        freq_code, freq_label = "W", "Weekly"
        expected_diff = WEEK
    else:
        freq_code, freq_label = "M", "Monthly"
        expected_diff = MONTH
    
    # Count ACTUAL missing dates/periods (not just gaps)
    # For each gap, calculate how many expected periods are missing
    missing_count = 0
    for d in diff_seconds:
        if d > expected_diff * 1.5:  # There's a gap
            # How many periods are missing in this gap?
            missing_periods = int(round(d / expected_diff)) - 1
            missing_count += max(0, missing_periods)
    
    return freq_code, freq_label, missing_count


def parse_dates_flexible(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    """
    Parse dates with intelligent format detection.
    Analyzes the data to determine if format is M/D/Y or D/M/Y.
    
    Args:
        df: DataFrame with date column as string
        date_col: Name of date column
    
    Returns:
        DataFrame with parsed datetime column
    """
    original_len = len(df)
    
    # Get sample of date strings for analysis
    date_strings = df.select(pl.col(date_col).cast(pl.Utf8)).to_series().to_list()[:100]
    
    def detect_date_format(samples: list) -> str:
        """Detect whether dates are M/D/Y or D/M/Y."""
        first_parts = []
        second_parts = []
        
        for s in samples:
            if not s:
                continue
            s = str(s).strip()
            
            for sep in ['/', '-', '.']:
                if sep in s:
                    parts = s.split(sep)
                    if len(parts) >= 2:
                        try:
                            first_parts.append(int(parts[0]))
                            second_parts.append(int(parts[1]))
                        except ValueError:
                            pass
                    break
        
        if not first_parts or not second_parts:
            return None
        
        max_first = max(first_parts)
        max_second = max(second_parts)
        
        if max_first > 12:
            return "DMY"
        if max_second > 12:
            return "MDY"
        return "MDY"  # Default
    
    detected_format = detect_date_format(date_strings)
    print(f"Detected date format: {detected_format}")
    
    # Define format lists based on detection
    if detected_format == "DMY":
        date_formats = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d"]
    else:
        date_formats = ["%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%d/%m/%Y"]
    
    best_df = None
    best_success_count = 0
    best_format = None
    
    for fmt in date_formats:
        try:
            test_df = df.with_columns(
                pl.col(date_col).cast(pl.Utf8).str.strip_chars().str.to_date(fmt, strict=False).cast(pl.Datetime)
            )
            success_count = original_len - test_df[date_col].null_count()
            
            print(f"Format {fmt}: {success_count}/{original_len} rows parsed")
            
            if success_count > best_success_count:
                best_success_count = success_count
                best_df = test_df
                best_format = fmt
        except Exception as e:
            print(f"Format {fmt} failed: {e}")
            continue
    
    # Try automatic parsing
    try:
        test_df = df.with_columns(pl.col(date_col).str.to_datetime(strict=False))
        success_count = original_len - test_df[date_col].null_count()
        print(f"Auto parsing: {success_count}/{original_len} rows parsed")
        
        if success_count > best_success_count:
            best_success_count = success_count
            best_df = test_df
            best_format = "auto"
    except Exception:
        pass
    
    if best_df is None or best_success_count == 0:
        raise ValueError(f"Could not parse date column '{date_col}'")
    
    print(f"Best format: {best_format} with {best_success_count}/{original_len} rows")
    return best_df


def filter_by_date_range(
    df: pl.DataFrame, 
    date_col: str, 
    start: str, 
    end: str, 
    inclusive_end: bool = True
) -> pl.DataFrame:
    """
    Filter dataframe by date range.
    
    Args:
        df: DataFrame to filter
        date_col: Name of date column
        start: Start date string
        end: End date string
        inclusive_end: If True, uses <= for end. If False, uses < for end.
    
    Returns:
        Filtered DataFrame
    """
    if inclusive_end:
        return df.filter(
            (pl.col(date_col) >= pl.lit(start).str.to_datetime()) &
            (pl.col(date_col) <= pl.lit(end).str.to_datetime())
        )
    else:
        return df.filter(
            (pl.col(date_col) >= pl.lit(start).str.to_datetime()) &
            (pl.col(date_col) < pl.lit(end).str.to_datetime())
        )
