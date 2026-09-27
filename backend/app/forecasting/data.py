"""Loading, cleaning and filtering of time series data."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

NUMERIC_DTYPES = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)


def load_frame(records: List[Dict[str, Any]], date_col: str, target_col: str) -> pl.DataFrame:
    """
    Build a clean DataFrame from raw records (CSV rows as dicts).

    Dates are parsed (format auto-detected), rows without date are dropped, the
    target is converted to float (null when not parseable) and rows are sorted by date.
    """
    if not records:
        raise ValueError("No data rows")
    df = pl.DataFrame(records, infer_schema_length=None)
    for col in (date_col, target_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data")

    df = parse_dates_flexible(df, date_col)
    df = df.filter(pl.col(date_col).is_not_null())
    df = df.with_columns(to_float(df[target_col]).alias(target_col))
    df = df.sort(date_col)

    if df.height == 0:
        raise ValueError("No valid data after parsing dates")
    return df


def to_float(series: pl.Series) -> pl.Series:
    """Convert a column to float, tolerating stray characters (e.g. '?0.2' -> 0.2)."""
    try:
        return (
            series.cast(pl.Utf8)
            .str.strip_chars()
            .str.replace(r"^\?", "")
            .str.replace(r"[^\d\.-]", "")
            .cast(pl.Float64)
        )
    except Exception:
        return series.cast(pl.Float64, strict=False)


def prepare_for_training(df: pl.DataFrame, date_col: str, target_col: str) -> pl.DataFrame:
    """Drop rows without target and convert the other columns to numeric."""
    df = df.filter(pl.col(target_col).is_not_null())
    for col in df.columns:
        if col in (date_col, target_col) or df[col].dtype in NUMERIC_DTYPES:
            continue
        try:
            df = df.with_columns(pl.col(col).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False))
        except Exception:
            pass
    if df.height == 0:
        raise ValueError("No valid data rows after cleaning")
    return df


def detect_frequency(df: pl.DataFrame, date_col: str) -> Tuple[str, str, int]:
    """
    Detect the frequency of a time series.

    Returns:
        (frequency_code, frequency_label, missing_dates_count)
        frequency_code is one of: s, min, H, D, W, M (or "unknown")
    """
    if df.height < 2:
        return "unknown", "Unknown", 0

    diffs = df.sort(date_col).select(date_col).to_series().diff().drop_nulls()
    if len(diffs) == 0:
        return "unknown", "Unknown", 0

    diff_seconds = [d.total_seconds() for d in diffs.to_list()]
    # Median difference is more robust than mode
    median_diff = np.median(diff_seconds)

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

    # Count missing periods (not just gaps): a gap of 3 periods means 2 missing dates
    missing_count = 0
    for d in diff_seconds:
        if d > expected_diff * 1.5:
            missing_count += max(0, int(round(d / expected_diff)) - 1)

    return freq_code, freq_label, missing_count


def _detect_date_format(samples: list) -> Optional[str]:
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
    if max(first_parts) > 12:
        return "DMY"
    if max(second_parts) > 12:
        return "MDY"
    return "MDY"  # Default


def parse_dates_flexible(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    """
    Parse dates with format detection (M/D/Y vs D/M/Y, ISO, with or without time).
    Keeps the format that parses the most rows.
    """
    original_len = len(df)
    date_strings = df.select(pl.col(date_col).cast(pl.Utf8)).to_series().to_list()[:100]
    detected_format = _detect_date_format(date_strings)

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
            if success_count > best_success_count:
                best_success_count, best_df, best_format = success_count, test_df, fmt
        except Exception:
            continue

    # Automatic parsing (ISO with time, etc.)
    try:
        test_df = df.with_columns(pl.col(date_col).str.to_datetime(strict=False))
        success_count = original_len - test_df[date_col].null_count()
        if success_count > best_success_count:
            best_success_count, best_df, best_format = success_count, test_df, "auto"
    except Exception:
        pass

    if best_df is None or best_success_count == 0:
        raise ValueError(f"Could not parse date column '{date_col}'")

    logger.debug("Date column '%s': format %s parsed %d/%d rows", date_col, best_format, best_success_count, original_len)
    return best_df


def filter_by_date_range(
    df: pl.DataFrame,
    date_col: str,
    start: str,
    end: str,
    inclusive_end: bool = True
) -> pl.DataFrame:
    """Filter rows with start <= date <= end (or < end if inclusive_end is False)."""
    upper = pl.col(date_col) <= pl.lit(end).str.to_datetime() if inclusive_end else pl.col(date_col) < pl.lit(end).str.to_datetime()
    return df.filter((pl.col(date_col) >= pl.lit(start).str.to_datetime()) & upper)


def row_indices_in_range(df: pl.DataFrame, date_col: str, start: str, end: str, inclusive_end: bool) -> np.ndarray:
    """Row positions (in df) whose date is in the range."""
    return (
        df.with_row_index("__row")
        .pipe(filter_by_date_range, date_col, start, end, inclusive_end)
        ["__row"].to_numpy()
    )
