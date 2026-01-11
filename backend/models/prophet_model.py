"""Prophet model for time series forecasting."""

import time
import numpy as np
import polars as pl
import pandas as pd
from typing import Dict, Any, List

from utils.metrics import calculate_metrics
from utils.date_utils import filter_by_date_range

# Prophet is optional
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


def train_prophet(
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    training_ranges: List[Any],
    prediction_ranges: List[Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train a Prophet model with optional lag regressors.
    Prophet decomposes series into trend + seasonality.
    
    Note: Prophet has a bug with datetime64[us] (microseconds) precision.
    We force conversion to datetime64[ns] (nanoseconds) for proper seasonality detection.
    """
    if not PROPHET_AVAILABLE:
        raise ValueError("Prophet is not installed. Please use Linear Regression, XGBoost, or ARIMA instead.")
    
    start_time = time.time()
    
    # Get Prophet params
    daily_seasonality = params.get("daily_seasonality", False)
    weekly_seasonality = params.get("weekly_seasonality", True)
    yearly_seasonality = params.get("yearly_seasonality", True)
    seasonality_mode = params.get("seasonality_mode", "additive")
    
    # Get lag regressors (disabled by default - can cause data leakage)
    use_lag_regressors = params.get("use_lag_regressors", False)
    lag_regressors = params.get("lag_regressors", [])
    if isinstance(lag_regressors, str):
        lag_regressors = [int(x.strip()) for x in lag_regressors.split(",")]
    
    # Add lag columns to dataframe
    df_with_lags = df.clone().sort(date_col)
    regressor_names = []
    
    if use_lag_regressors:
        for lag in lag_regressors:
            col_name = f"lag_{lag}"
            df_with_lags = df_with_lags.with_columns(
                pl.col(target_col).shift(lag).alias(col_name)
            )
            regressor_names.append(col_name)
    
    # Build training data
    train_dfs = []
    for tr in training_ranges:
        chunk = filter_by_date_range(df_with_lags, date_col, tr.start, tr.end, inclusive_end=False)
        train_dfs.append(chunk)
    
    if not train_dfs:
        raise ValueError("No training data found")
    
    train_df = pl.concat(train_dfs).sort(date_col)
    
    # Drop nulls from lag columns
    if regressor_names:
        train_df = train_df.drop_nulls(subset=regressor_names)
    
    if train_df.height == 0:
        raise ValueError("No training data after filtering by date ranges")
    
    # Build pandas dataframe for Prophet
    cols_to_select = [pl.col(date_col).alias("ds"), pl.col(target_col).alias("y")]
    for reg_name in regressor_names:
        cols_to_select.append(pl.col(reg_name))
    
    prophet_train = train_df.select(cols_to_select).to_pandas()
    
    # CRITICAL: Ensure ds is datetime with nanosecond precision
    # Prophet has issues with microseconds precision (datetime64[us])
    prophet_train['ds'] = pd.to_datetime(prophet_train['ds']).astype('datetime64[ns]')
    
    # Initialize Prophet
    model = Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        seasonality_mode=seasonality_mode
    )
    
    # Add lag regressors
    for reg_name in regressor_names:
        model.add_regressor(reg_name)
    
    model.fit(prophet_train)
    
    # Predict on prediction ranges
    all_predictions = []
    all_actuals = []
    forecast_output = []
    
    for pr in prediction_ranges:
        pred_df = filter_by_date_range(df_with_lags, date_col, pr.start, pr.end).sort(date_col)
        
        if regressor_names:
            pred_df = pred_df.drop_nulls(subset=regressor_names)
        
        if pred_df.height == 0:
            continue
        
        y_actual = pred_df.select(target_col).to_numpy().flatten()
        dates = pred_df.select(date_col).to_series().to_list()
        
        # Create future dataframe for Prophet
        cols_for_future = [pl.col(date_col).alias("ds")]
        for reg_name in regressor_names:
            cols_for_future.append(pl.col(reg_name))
        
        future = pred_df.select(cols_for_future).to_pandas()
        
        # CRITICAL: Ensure ds is datetime with nanosecond precision
        future['ds'] = pd.to_datetime(future['ds']).astype('datetime64[ns]')
        
        forecast = model.predict(future)
        y_pred = forecast["yhat"].values
        
        all_predictions.extend(y_pred)
        all_actuals.extend(y_actual)
        
        for date, pred_val, actual_val in zip(dates, y_pred, y_actual):
            forecast_output.append({
                date_col: date.isoformat() if hasattr(date, 'isoformat') else str(date),
                "prediction": float(pred_val),
                target_col: float(actual_val)
            })
    
    # Metrics
    metrics = calculate_metrics(np.array(all_actuals), np.array(all_predictions)) if all_actuals else {"rmse": 0, "mae": 0, "mape": 0, "r2": 0, "msle": 0}
    metrics["execution_time"] = time.time() - start_time
    
    # Feature importance approximation
    feature_importance = None
    if regressor_names:
        feature_importance = [
            {"feature": reg_name, "importance": 1.0 / len(regressor_names)}
            for reg_name in regressor_names
        ]
    
    return {
        "metrics": metrics,
        "forecast": forecast_output,
        "feature_importance": feature_importance
    }
