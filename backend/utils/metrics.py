"""Metrics calculation for time series forecasting."""

import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Dict with rmse, mae, mape, r2, msle
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # RMSE
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
    else:
        mape = 0.0
    
    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
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
    # Group by horizon step
    by_horizon = defaultdict(list)
    for f in forecasts:
        h = f.get("horizon_step", 1)
        actual = f.get(target_col)
        pred = f.get("prediction")
        if actual is not None and pred is not None:
            by_horizon[h].append((float(actual), float(pred)))
    
    metrics_list = []
    for h in sorted(by_horizon.keys()):
        pairs = by_horizon[h]
        if len(pairs) == 0:
            continue
        
        y_true = np.array([p[0] for p in pairs])
        y_pred = np.array([p[1] for p in pairs])
        
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        
        # MAPE
        mask = y_true != 0
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
        else:
            mape = 0.0
        
        # MSLE
        mask_positive = (y_true > 0) & (y_pred > 0)
        if mask_positive.sum() > 0:
            msle = float(np.mean((np.log1p(y_true[mask_positive]) - np.log1p(y_pred[mask_positive])) ** 2))
        else:
            msle = 0.0
        
        metrics_list.append({
            "horizon_step": h,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "msle": msle,
            "count": len(pairs)
        })
    
    return metrics_list
