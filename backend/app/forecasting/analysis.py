"""
Dataset analysis: statistics, ACF/PACF and lag suggestions, data quality alerts.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import polars as pl

from app.forecasting.data import NUMERIC_DTYPES, detect_frequency

logger = logging.getLogger(__name__)


def to_native(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types (JSON serialization)."""
    if isinstance(obj, np.ndarray):
        return [to_native(x) for x in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(x) for x in obj]
    return obj


def _isoformat(value: Any) -> str:
    return value.isoformat() if hasattr(value, 'isoformat') else str(value)


def analyze_dataset(df: pl.DataFrame, date_col: str, target_col: str) -> Dict[str, Any]:
    """
    Analyze a loaded dataset (see data.load_frame).

    Returns stats, the cleaned target series, the other columns (candidate exogenous
    variables), ACF/PACF lag suggestions and data quality alerts.
    """
    freq_code, freq_label, missing_dates = detect_frequency(df, date_col)
    target_series = df[target_col]
    missing_target = target_series.null_count()
    valid_target = target_series.drop_nulls()
    has_values = len(valid_target) > 0

    stats = {
        "date_min": _isoformat(df[date_col].min()),
        "date_max": _isoformat(df[date_col].max()),
        "total_rows": df.height,
        "frequency": freq_code,
        "frequency_label": freq_label,
        "missing_dates": missing_dates,
        "missing_values_target": missing_target,
        "value_min": float(valid_target.min()) if has_values else 0.0,
        "value_max": float(valid_target.max()) if has_values else 0.0,
        "value_mean": float(valid_target.mean()) if has_values else 0.0,
    }

    clean_df = df.filter(pl.col(target_col).is_not_null())
    normalized_data = [
        {"date": _isoformat(d), "value": float(v)}
        for d, v in zip(clean_df[date_col].to_list(), clean_df[target_col].to_list())
    ]

    available_columns = []
    for col_name in df.columns:
        if col_name in (date_col, target_col):
            continue
        col_series = df[col_name]
        dtype = col_series.dtype
        if dtype in NUMERIC_DTYPES:
            dtype_str = "numeric"
        elif dtype == pl.Boolean:
            dtype_str = "boolean"
        elif dtype in (pl.Datetime, pl.Date):
            dtype_str = "date"
        else:
            dtype_str = "string"
        available_columns.append({
            "name": col_name,
            "dtype": dtype_str,
            "missing_count": col_series.null_count(),
            "sample_values": col_series.drop_nulls().head(5).to_list(),
        })

    alerts: List[Dict[str, Any]] = []
    lag_analysis = None
    target_values = valid_target.to_numpy()

    if len(target_values) >= 20:
        # Each analysis is best-effort: a failure must not prevent the others
        try:
            lag_analysis = to_native(suggest_lags(target_values, frequency=freq_code, max_lags=20))
            seasonality = lag_analysis["seasonality"]
            if seasonality.get("detected"):
                alerts.append({
                    "type": "info",
                    "category": "seasonality",
                    "message": f"Seasonality detected: {seasonality.get('period_label', '')} pattern "
                               f"(strength: {seasonality.get('strength', 0):.2f})",
                    "details": seasonality,
                })
        except Exception:
            logger.exception("Lag analysis failed")

        try:
            outliers = to_native(detect_outliers(target_values, method="iqr"))
            if outliers["count"] > 0:
                pct = outliers["percentage"]
                alerts.append({
                    "type": "warning" if pct > 5 else "info",
                    "category": "outliers",
                    "message": f"{outliers['count']} outliers detected ({pct:.1f}% of data)",
                    "details": outliers,
                })
        except Exception:
            logger.exception("Outlier detection failed")

        try:
            trend = to_native(detect_trend(target_values))
            if trend["detected"]:
                alerts.append({
                    "type": "info",
                    "category": "trend",
                    "message": f"{trend['strength'].capitalize()} {trend['direction']} trend detected",
                    "details": trend,
                })
        except Exception:
            logger.exception("Trend detection failed")

        try:
            stationarity = to_native(compute_stationarity_indicators(target_values))
            if stationarity.get("likely_stationary") is False:
                alerts.append({
                    "type": "warning",
                    "category": "stationarity",
                    "message": "Series may be non-stationary. Consider differencing for ARIMA.",
                    "details": stationarity,
                })
        except Exception:
            logger.exception("Stationarity check failed")

    if missing_dates > 0:
        pct = (missing_dates / df.height) * 100
        alerts.append({
            "type": "warning" if pct > 10 else "info",
            "category": "missing",
            "message": f"{missing_dates} missing dates detected ({pct:.1f}%)",
            "details": {"missing_count": missing_dates, "percentage": pct},
        })

    if missing_target > 0:
        pct = (missing_target / df.height) * 100
        alerts.append({
            "type": "warning" if pct > 5 else "info",
            "category": "missing",
            "message": f"{missing_target} missing target values ({pct:.1f}%)",
            "details": {"missing_count": missing_target, "percentage": pct},
        })

    return {
        "stats": stats,
        "normalized_data": normalized_data,
        "available_columns": available_columns,
        "lag_analysis": lag_analysis,
        "alerts": alerts or None,
    }


def compute_acf(series: np.ndarray, max_lag: int = 40) -> List[float]:
    """
    Compute Autocorrelation Function (ACF).
    
    Args:
        series: Time series values (1D array)
        max_lag: Maximum lag to compute
        
    Returns:
        List of ACF values for lags 0 to max_lag
    """
    n = len(series)
    if n < 10:
        return [1.0]
    
    max_lag = min(max_lag, n // 2)
    mean = np.mean(series)
    var = np.var(series)
    
    if var == 0:
        return [1.0] + [0.0] * max_lag
    
    acf_values = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_values.append(1.0)
        else:
            cov = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / n
            acf_values.append(cov / var)
    
    return acf_values


def compute_pacf(series: np.ndarray, max_lag: int = 40) -> List[float]:
    """
    Compute Partial Autocorrelation Function (PACF) using Durbin-Levinson algorithm.
    
    Args:
        series: Time series values (1D array)
        max_lag: Maximum lag to compute
        
    Returns:
        List of PACF values for lags 0 to max_lag
    """
    n = len(series)
    if n < 10:
        return [1.0]
    
    max_lag = min(max_lag, n // 3)
    
    # Get ACF first
    acf = compute_acf(series, max_lag)
    
    pacf = [1.0]  # PACF at lag 0 is always 1
    
    # Durbin-Levinson algorithm
    phi = np.zeros((max_lag + 1, max_lag + 1))
    
    for k in range(1, max_lag + 1):
        # Compute phi[k,k]
        if k == 1:
            phi[1, 1] = acf[1]
        else:
            num = acf[k] - sum(phi[k-1, j] * acf[k-j] for j in range(1, k))
            den = 1 - sum(phi[k-1, j] * acf[j] for j in range(1, k))
            
            if abs(den) < 1e-10:
                phi[k, k] = 0.0
            else:
                phi[k, k] = num / den
            
            # Update other phi values
            for j in range(1, k):
                phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]
        
        pacf.append(float(phi[k, k]))
    
    return pacf


# Seasonal cycles worth testing for each frequency: (period in rows, label)
SEASONAL_CANDIDATES: Dict[str, List[Tuple[int, str]]] = {
    "s": [(60, "Minutely")],
    "min": [(60, "Hourly"), (1440, "Daily")],
    "H": [(24, "Daily"), (168, "Weekly")],
    "D": [(7, "Weekly"), (365, "Yearly")],
    "W": [(52, "Yearly")],
    "M": [(12, "Yearly")],
}

# Calendar feature that captures a seasonal cycle (better than a very long lag)
SEASONAL_FEATURES: Dict[Tuple[str, int], str] = {
    ("min", 1440): "minute_of_day",
    ("H", 24): "hour_of_day",
    ("H", 168): "day_of_week",
    ("D", 7): "day_of_week",
    ("D", 365): "month",
    ("W", 52): "week_of_year",
    ("M", 12): "month",
}

MIN_SEASONAL_ACF = 0.2      # Autocorrelation at the period to call it a cycle
MAX_SEASONAL_LAG = 60       # Longer cycles are suggested as calendar features, not lags
MIN_PACF_FOR_LAG = 0.1      # With many points everything is "significant": keep useful lags only


def _acf_fft(series: np.ndarray, max_lag: int) -> np.ndarray:
    """ACF up to max_lag in O(n log n) (same estimator as compute_acf)."""
    x = np.asarray(series, dtype=float) - np.mean(series)
    n = len(x)
    var = np.dot(x, x) / n
    if var == 0:
        return np.concatenate([[1.0], np.zeros(max_lag)])
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, size)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), size)[: max_lag + 1] / n
    return acov / var


def detect_seasonalities(series: np.ndarray, frequency: str) -> List[Dict[str, Any]]:
    """
    Seasonal cycles among the plausible ones for the frequency (e.g. weekly and yearly
    for daily data). A cycle is detected when the ACF has a clear peak at its period.
    A multiple of a detected cycle (weekly = 7 x daily) only counts if it is stronger.
    """
    n = len(series)
    candidates = [(p, label) for p, label in SEASONAL_CANDIDATES.get(frequency, []) if n >= 2 * p + 1]
    if not candidates:
        return []
    max_period = max(p for p, _ in candidates)
    acf = _acf_fft(series, min(n - 1, max_period + max_period // 10 + 1))

    detected: List[Dict[str, Any]] = []
    for period, label in sorted(candidates):
        # Tolerance for calendar irregularities (365 vs 366 days)
        window = max(1, period // 30)
        around = acf[max(1, period - window): min(len(acf), period + window + 1)]
        strength = float(np.max(around))
        half = float(acf[period // 2])
        is_peak = strength >= MIN_SEASONAL_ACF and strength > half + 0.1
        if not is_peak:
            continue
        harmonic_of = [d for d in detected if period % d["period"] == 0]
        if harmonic_of and strength <= max(d["strength"] for d in harmonic_of) + 0.1:
            continue
        detected.append({
            "period": period,
            "period_label": label,
            "strength": round(strength, 4),
            "suggested_feature": SEASONAL_FEATURES.get((frequency, period)),
        })
    return detected


def suggest_lags(
    series: np.ndarray,
    frequency: str = "D",
    max_lags: int = 20
) -> Dict[str, Any]:
    """
    Analyze a time series: ACF/PACF, seasonal cycles, suggested lags and calendar features.

    Args:
        series: Time series values
        frequency: Detected frequency code (s, min, H, D, W, M), as returned by detect_frequency
        max_lags: Maximum number of lags to analyze

    Returns:
        Dictionary with suggested lags and analysis
    """
    n = len(series)
    seasonalities = detect_seasonalities(series, frequency)
    short_periods = [s["period"] for s in seasonalities if s["period"] <= MAX_SEASONAL_LAG]

    # Look far enough to see short seasonal lags (e.g. 60 for minute data)
    max_lag_compute = min(max([max_lags * 2] + [p + 1 for p in short_periods]), n // 3)
    acf_values = compute_acf(series, max_lag_compute)
    pacf_values = compute_pacf(series, max_lag_compute)

    # Confidence interval (approximate 95%)
    confidence = 1.96 / np.sqrt(n)

    significant_lags = [
        {"lag": lag, "pacf": round(pacf_values[lag], 4), "significant": True}
        for lag in range(1, len(pacf_values))
        if abs(pacf_values[lag]) > confidence
    ]
    significant_lags.sort(key=lambda x: abs(x["pacf"]), reverse=True)

    # Strongest useful PACF lags, lag 1, and the short seasonal periods
    suggested = [x["lag"] for x in significant_lags if abs(x["pacf"]) >= MIN_PACF_FOR_LAG][:5]
    suggested = sorted(set(suggested + [1] + short_periods))[:7]

    # ACF long enough to show the longest detected cycle, PACF up to short seasonal lags
    n_acf = min(max([max_lags] + [s["period"] + s["period"] // 10 for s in seasonalities]) + 1, n // 3)
    acf_display = _acf_fft(series, n_acf - 1) if n_acf > len(acf_values) else np.array(acf_values[:n_acf])
    n_pacf = max([max_lags] + short_periods) + 1

    main = max(seasonalities, key=lambda s: s["strength"]) if seasonalities else None
    return {
        "suggested_lags": suggested,
        "suggested_temporal": sorted({s["suggested_feature"] for s in seasonalities if s["suggested_feature"]}),
        "acf": [round(float(v), 4) for v in acf_display],
        "pacf": [round(v, 4) for v in pacf_values[:n_pacf]],
        "confidence_interval": round(confidence, 4),
        "significant_lags": significant_lags[:10],  # Top 10
        "seasonality": {"detected": True, **main} if main else {"detected": False},
        "seasonalities": seasonalities,
        "n_observations": n
    }


def detect_outliers(series: np.ndarray, method: str = "iqr") -> Dict[str, Any]:
    """
    Detect outliers in time series.
    
    Args:
        series: Time series values
        method: Detection method ('iqr' or 'zscore')
        
    Returns:
        Dictionary with outlier information
    """
    clean_series = series[~np.isnan(series)]
    
    if len(clean_series) < 10:
        return {"count": 0, "indices": [], "method": method}
    
    if method == "iqr":
        q1 = np.percentile(clean_series, 25)
        q3 = np.percentile(clean_series, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        outlier_mask = (series < lower) | (series > upper)
    else:  # zscore
        mean = np.mean(clean_series)
        std = np.std(clean_series)
        if std == 0:
            return {"count": 0, "indices": [], "method": method}
        
        z_scores = np.abs((series - mean) / std)
        outlier_mask = z_scores > 3
    
    outlier_indices = np.where(outlier_mask)[0].tolist()
    
    return {
        "count": len(outlier_indices),
        "indices": outlier_indices[:50],  # Limit to first 50
        "percentage": round(len(outlier_indices) / len(series) * 100, 2),
        "method": method
    }


def detect_trend(series: np.ndarray) -> Dict[str, Any]:
    """
    Detect trend in time series using linear regression.
    
    Args:
        series: Time series values
        
    Returns:
        Dictionary with trend information
    """
    clean_series = series[~np.isnan(series)]
    n = len(clean_series)
    
    if n < 10:
        return {"detected": False, "direction": "none", "strength": 0}
    
    # Simple linear regression
    x = np.arange(n)
    slope = np.cov(x, clean_series)[0, 1] / np.var(x)
    
    # Normalize slope by series range
    value_range = np.max(clean_series) - np.min(clean_series)
    if value_range > 0:
        normalized_slope = slope * n / value_range
    else:
        normalized_slope = 0
    
    # Determine direction and strength
    if abs(normalized_slope) < 0.1:
        direction = "none"
        strength = "none"
    elif normalized_slope > 0:
        direction = "upward"
        strength = "strong" if abs(normalized_slope) > 0.5 else "moderate" if abs(normalized_slope) > 0.2 else "weak"
    else:
        direction = "downward"
        strength = "strong" if abs(normalized_slope) > 0.5 else "moderate" if abs(normalized_slope) > 0.2 else "weak"
    
    return {
        "detected": abs(normalized_slope) >= 0.1,
        "direction": direction,
        "strength": strength,
        "slope": round(slope, 6),
        "normalized_slope": round(normalized_slope, 4)
    }


def compute_stationarity_indicators(series: np.ndarray) -> Dict[str, Any]:
    """
    Compute basic stationarity indicators (without statsmodels ADF test).
    
    Args:
        series: Time series values
        
    Returns:
        Dictionary with stationarity indicators
    """
    clean_series = series[~np.isnan(series)]
    n = len(clean_series)
    
    if n < 20:
        return {"likely_stationary": None, "message": "Not enough data"}
    
    # Split into halves and compare statistics
    half = n // 2
    first_half = clean_series[:half]
    second_half = clean_series[half:]
    
    mean_change = abs(np.mean(second_half) - np.mean(first_half)) / (np.std(clean_series) + 1e-10)
    var_change = np.std(second_half) / (np.std(first_half) + 1e-10)
    
    # Simple heuristic
    likely_stationary = mean_change < 0.5 and 0.5 < var_change < 2.0
    
    return {
        "likely_stationary": likely_stationary,
        "mean_shift": round(mean_change, 4),
        "variance_ratio": round(var_change, 4),
        "recommendation": "Consider differencing" if not likely_stationary else "Series appears stationary"
    }
