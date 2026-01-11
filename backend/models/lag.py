"""LAG baseline model for time series forecasting."""

import time
import numpy as np
import polars as pl
from typing import Dict, Any, List, Optional

from utils.metrics import calculate_metrics, calculate_metrics_by_horizon
from utils.date_utils import filter_by_date_range


class DateRange:
    """Date range specification."""
    def __init__(self, start: str, end: str):
        self.start = start
        self.end = end


class ForecastStrategyConfig:
    """Forecast strategy configuration."""
    def __init__(self, horizon: int = 1, **kwargs):
        self.horizon = horizon


def train_lag(
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    training_ranges: List[Any],
    prediction_ranges: List[Any],
    params: Dict[str, Any],
    forecast_strategy: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Baseline model: predict using a simple lag.
    
    Params:
    - lag: int - which lag to use for prediction (default: 1)
    
    With horizon > 1, uses block-wise recursive:
    - Block 1: pred(t+1) = y(t₀), pred(t+2) = pred(t+1), ...
    - Block 2: reset to actual values, then recursive again
    
    Returns metrics and predictions.
    """
    start_time = time.time()
    
    # Get lag parameter
    lag = params.get("lag", 1)
    
    # Determine horizon
    horizon = 1
    if forecast_strategy is not None:
        horizon = getattr(forecast_strategy, 'horizon', 1)
    
    # Create lagged column
    df_lagged = df.with_columns(
        pl.col(target_col).shift(lag).alias(f"lag_{lag}")
    )
    
    # Build training data
    train_dfs = []
    for tr in training_ranges:
        range_df = filter_by_date_range(df_lagged, date_col, tr.start, tr.end, inclusive_end=False)
        train_dfs.append(range_df)
    
    train_df = pl.concat(train_dfs) if train_dfs else df_lagged.head(0)
    train_df = train_df.drop_nulls(subset=[target_col, f"lag_{lag}"])
    
    if train_df.height == 0:
        raise ValueError(f"No training data available after applying lag {lag}")
    
    # Calculate metrics on training data (still using naive lag for training metrics)
    y_true = train_df[target_col].to_numpy()
    y_pred = train_df[f"lag_{lag}"].to_numpy()
    
    training_metrics = calculate_metrics(y_true, y_pred)
    
    # Generate predictions for all prediction ranges
    all_forecasts = []
    all_actuals = []
    all_predictions = []
    
    for pr in prediction_ranges:
        pred_df = filter_by_date_range(df_lagged, date_col, pr.start, pr.end)
        pred_df = pred_df.drop_nulls(subset=[target_col, f"lag_{lag}"])
        
        if pred_df.height == 0:
            continue
        
        if horizon > 1:
            # Block-wise recursive for lag model
            rows = list(pred_df.iter_rows(named=True))
            block_num = 0
            idx = 0
            
            while idx < len(rows):
                block_num += 1
                block_predictions = {}  # step_in_block -> predicted value
                block_end = min(idx + horizon, len(rows))
                
                for step, row_idx in enumerate(range(idx, block_end), start=1):
                    row = rows[row_idx]
                    date_value = row[date_col]
                    actual = float(row[target_col])
                    
                    # For lag model: which step does the lag refer to?
                    lag_refers_to_step = step - lag
                    
                    if lag_refers_to_step > 0 and lag_refers_to_step in block_predictions:
                        # Use prediction from earlier in THIS block
                        prediction = block_predictions[lag_refers_to_step]
                    else:
                        # Use actual value from dataframe
                        prediction = float(row[f"lag_{lag}"])
                    
                    block_predictions[step] = prediction
                    
                    all_forecasts.append({
                        date_col: date_value.isoformat() if hasattr(date_value, 'isoformat') else str(date_value),
                        "prediction": prediction,
                        target_col: actual,
                        "block_num": block_num,
                        "step_in_block": step,
                        "horizon_step": step
                    })
                    all_actuals.append(actual)
                    all_predictions.append(prediction)
                
                idx = block_end
        else:
            # Standard 1-step prediction
            for row in pred_df.iter_rows(named=True):
                date_value = row[date_col]
                prediction = float(row[f"lag_{lag}"])
                actual = float(row[target_col])
                
                all_forecasts.append({
                    date_col: date_value.isoformat() if hasattr(date_value, 'isoformat') else str(date_value),
                    "prediction": prediction,
                    target_col: actual
                })
                all_actuals.append(actual)
                all_predictions.append(prediction)
    
    # Calculate validation metrics
    if all_actuals and all_predictions:
        validation_metrics = calculate_metrics(
            np.array(all_actuals),
            np.array(all_predictions)
        )
    else:
        validation_metrics = training_metrics
    
    execution_time = time.time() - start_time
    
    result = {
        "metrics": {
            **validation_metrics,
            "execution_time": execution_time
        },
        "forecast": all_forecasts,
        "feature_importance": None
    }
    
    # Add metrics by horizon if horizon > 1
    if horizon > 1 and all_forecasts:
        result["metrics_by_horizon"] = calculate_metrics_by_horizon(all_forecasts, target_col)
    
    return result
