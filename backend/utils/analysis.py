"""
Advanced time series analysis utilities.
ACF, PACF computation and lag suggestions.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import polars as pl


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


def suggest_lags(
    series: np.ndarray, 
    frequency: str = "D",
    max_lags: int = 20
) -> Dict[str, Any]:
    """
    Analyze time series and suggest optimal lags based on PACF.
    
    Args:
        series: Time series values
        frequency: Detected frequency code (D, H, T, W, M, etc.)
        max_lags: Maximum number of lags to analyze
        
    Returns:
        Dictionary with suggested lags and analysis
    """
    n = len(series)
    max_lag_compute = min(max_lags * 2, n // 3, 50)
    
    # Compute ACF and PACF
    acf_values = compute_acf(series, max_lag_compute)
    pacf_values = compute_pacf(series, max_lag_compute)
    
    # Confidence interval (approximate 95%)
    confidence = 1.96 / np.sqrt(n)
    
    # Find significant lags from PACF
    significant_lags = []
    for lag in range(1, len(pacf_values)):
        if abs(pacf_values[lag]) > confidence:
            significant_lags.append({
                "lag": lag,
                "pacf": round(pacf_values[lag], 4),
                "significant": True
            })
    
    # Sort by absolute PACF value
    significant_lags.sort(key=lambda x: abs(x["pacf"]), reverse=True)
    
    # Suggest top lags (max 5)
    suggested = [x["lag"] for x in significant_lags[:5]]
    
    # Add seasonal lag based on frequency
    seasonal_lag = _get_seasonal_lag(frequency)
    if seasonal_lag and seasonal_lag not in suggested and seasonal_lag <= max_lag_compute:
        suggested.append(seasonal_lag)
    
    # Always include lag 1 if significant or if no significant lags found
    if 1 not in suggested:
        suggested.insert(0, 1)
    
    suggested = sorted(set(suggested))[:7]  # Max 7 lags
    
    # Detect seasonality from ACF peaks
    seasonality = _detect_seasonality(acf_values, frequency)
    
    return {
        "suggested_lags": suggested,
        "acf": [round(v, 4) for v in acf_values[:max_lags + 1]],
        "pacf": [round(v, 4) for v in pacf_values[:max_lags + 1]],
        "confidence_interval": round(confidence, 4),
        "significant_lags": significant_lags[:10],  # Top 10
        "seasonality": seasonality,
        "n_observations": n
    }


def _get_seasonal_lag(frequency: str) -> Optional[int]:
    """Get expected seasonal lag based on frequency."""
    seasonal_map = {
        "T": 60,      # Minute -> hourly pattern
        "H": 24,      # Hourly -> daily pattern
        "D": 7,       # Daily -> weekly pattern
        "W": 52,      # Weekly -> yearly pattern (too long usually)
        "M": 12,      # Monthly -> yearly pattern
    }
    return seasonal_map.get(frequency)


def _detect_seasonality(acf_values: List[float], frequency: str) -> Dict[str, Any]:
    """Detect seasonality patterns from ACF."""
    if len(acf_values) < 5:
        return {"detected": False}
    
    # Find local maxima in ACF
    peaks = []
    for i in range(2, len(acf_values) - 1):
        if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
            if acf_values[i] > 0.1:  # Minimum threshold
                peaks.append({"lag": i, "acf": round(acf_values[i], 4)})
    
    if not peaks:
        return {"detected": False}
    
    # Sort by ACF value
    peaks.sort(key=lambda x: x["acf"], reverse=True)
    
    # Check for periodic pattern
    main_peak = peaks[0] if peaks else None
    
    if main_peak and main_peak["acf"] > 0.3:
        period = main_peak["lag"]
        
        # Map to human-readable
        period_label = _period_to_label(period, frequency)
        
        return {
            "detected": True,
            "period": period,
            "period_label": period_label,
            "strength": main_peak["acf"],
            "peaks": peaks[:5]
        }
    
    return {"detected": False, "peaks": peaks[:5]}


def _period_to_label(period: int, frequency: str) -> str:
    """Convert period to human-readable label."""
    if frequency == "D":
        if period == 7:
            return "Weekly"
        elif period == 30 or period == 31:
            return "Monthly"
        elif period == 365 or period == 364:
            return "Yearly"
    elif frequency == "H":
        if period == 24:
            return "Daily"
        elif period == 168:
            return "Weekly"
    elif frequency == "T":
        if period == 60:
            return "Hourly"
        elif period == 1440:
            return "Daily"
    elif frequency == "M":
        if period == 12:
            return "Yearly"
    
    return f"{period}-period cycle"


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
