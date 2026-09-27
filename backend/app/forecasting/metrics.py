"""Metrics calculation for time series forecasting."""

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Returns:
        Dict with rmse, mae, mape (as a fraction), r2, msle
    """
    y_true = np.array(y_true, dtype=float).flatten()
    y_pred = np.array(y_pred, dtype=float).flatten()
    errors = y_true - y_pred

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    # MAPE (avoid division by zero)
    mask = y_true != 0
    mape = float(np.mean(np.abs(errors[mask] / y_true[mask]))) if mask.sum() > 0 else 0.0

    # R2
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # MSLE (Mean Squared Log Error) - only for positive values
    mask_positive = (y_true > 0) & (y_pred > 0)
    if mask_positive.sum() > 0:
        msle = float(np.mean((np.log1p(y_true[mask_positive]) - np.log1p(y_pred[mask_positive])) ** 2))
    else:
        msle = 0.0

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "msle": msle}


def calculate_metrics_by_horizon(forecasts: List[Dict[str, Any]], target_col: str) -> List[Dict[str, Any]]:
    """
    Calculate metrics grouped by horizon step.

    Args:
        forecasts: List of forecast dicts with 'prediction', target_col, and 'horizon_step'
        target_col: Name of the actual value column

    Returns:
        List of {horizon_step, rmse, mae, mape, msle, count} dicts
    """
    by_horizon = defaultdict(list)
    for f in forecasts:
        actual = f.get(target_col)
        pred = f.get("prediction")
        if actual is not None and pred is not None:
            by_horizon[f.get("horizon_step", 1)].append((float(actual), float(pred)))

    metrics_list = []
    for h in sorted(by_horizon):
        y_true, y_pred = np.array(by_horizon[h]).T
        metrics = calculate_metrics(y_true, y_pred)
        metrics_list.append({
            "horizon_step": h,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "mape": metrics["mape"],
            "msle": metrics["msle"],
            "count": len(y_true)
        })

    return metrics_list
