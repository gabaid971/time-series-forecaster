"""ARIMA model for time series forecasting."""

import time
import numpy as np
import polars as pl
from typing import Dict, Any, List
from statsmodels.tsa.arima.model import ARIMA

from utils.metrics import calculate_metrics, calculate_metrics_by_horizon
from utils.date_utils import filter_by_date_range


def train_arima(
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    training_ranges: List[Any],
    prediction_ranges: List[Any],
    params: Dict[str, Any],
    horizon: int = 1
) -> Dict[str, Any]:
    """
    Train an ARIMA model with block-wise forecasting for fair horizon comparison.
    ARIMA is a univariate model - it only uses the target variable's history.
    
    Block-wise strategy:
    - Split prediction range into blocks of size `horizon`
    - For each block, forecast h steps ahead
    - Assign horizon_step 1..h to each prediction
    - Update model between blocks using append(refit=False) for efficiency
    """
    start_time = time.time()
    
    # Get ARIMA params (p, d, q)
    p = params.get("p", 1)  # AR order
    d = params.get("d", 1)  # Differencing order
    q = params.get("q", 1)  # MA order
    
    # Build training data
    train_dfs = []
    for tr in training_ranges:
        chunk = filter_by_date_range(df, date_col, tr.start, tr.end, inclusive_end=False)
        train_dfs.append(chunk)
    
    if not train_dfs:
        raise ValueError("No training data found")
    
    train_df = pl.concat(train_dfs).sort(date_col)
    y_train = train_df.select(target_col).to_numpy().flatten()
    
    if len(y_train) < p + d + q + 5:
        raise ValueError(f"Not enough training data for ARIMA({p},{d},{q})")
    
    # Fit ARIMA on training data
    try:
        model = ARIMA(y_train, order=(p, d, q))
        fitted = model.fit()
    except Exception as e:
        raise ValueError(f"ARIMA fitting failed: {e}")
    
    # Predict on prediction ranges with block-wise horizon tracking
    all_predictions = []
    all_actuals = []
    forecast_output = []
    
    for pr in prediction_ranges:
        pred_df = filter_by_date_range(df, date_col, pr.start, pr.end).sort(date_col)
        
        if pred_df.height == 0:
            continue
        
        y_actual = pred_df.select(target_col).to_numpy().flatten()
        dates = pred_df.select(date_col).to_series().to_list()
        n_total = len(y_actual)
        
        # Block-wise forecasting
        current_fitted = fitted
        
        for block_start in range(0, n_total, horizon):
            block_end = min(block_start + horizon, n_total)
            block_size = block_end - block_start
            
            try:
                block_forecast = current_fitted.forecast(steps=block_size)
                y_pred_block = np.array(block_forecast)
            except Exception:
                # Fallback: use last known value from training
                y_pred_block = np.full(block_size, y_train[-1])
            
            # Add predictions with horizon_step
            for i in range(block_size):
                idx = block_start + i
                horizon_step = i + 1  # 1-indexed within block
                
                all_predictions.append(y_pred_block[i])
                all_actuals.append(y_actual[idx])
                
                forecast_output.append({
                    date_col: dates[idx].isoformat() if hasattr(dates[idx], 'isoformat') else str(dates[idx]),
                    "prediction": float(y_pred_block[i]),
                    target_col: float(y_actual[idx]),
                    "horizon_step": horizon_step
                })
            
            # Update model with actual observations for next block (fast update, no refit)
            try:
                current_fitted = current_fitted.append(y_actual[block_start:block_end], refit=False)
            except Exception:
                pass
    
    # Metrics
    metrics = calculate_metrics(np.array(all_actuals), np.array(all_predictions)) if all_actuals else {"rmse": 0, "mae": 0, "mape": 0, "r2": 0, "msle": 0}
    metrics["execution_time"] = time.time() - start_time
    
    # Metrics by horizon
    metrics_by_horizon = calculate_metrics_by_horizon(forecast_output, target_col) if forecast_output else []
    
    return {
        "metrics": metrics,
        "metrics_by_horizon": metrics_by_horizon,
        "forecast": forecast_output,
        "feature_importance": None  # ARIMA doesn't have feature importance
    }
