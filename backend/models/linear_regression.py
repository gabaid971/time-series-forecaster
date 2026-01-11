"""Linear Regression model for time series forecasting."""

import time
import numpy as np
import polars as pl
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LinearRegression

from utils.metrics import calculate_metrics, calculate_metrics_by_horizon
from utils.date_utils import filter_by_date_range
from utils.features import (
    build_features, 
    FeatureConfig, 
    TemporalFeatureConfig, 
    ExogenousFeatureConfig, 
    DerivedFeatureConfig
)
from utils.validation import block_recursive_forecast


def train_linear_regression(
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    training_ranges: List[Any],
    prediction_ranges: List[Any],
    params: Dict[str, Any],
    forecast_strategy: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Train a Linear Regression model with configurable features.
    
    Params:
    - lags: List[int] - target lags (legacy, used if feature_config not provided)
    - target_mode: "raw" or "residual"
    - residual_lag: int - which lag to subtract in residual mode
    - standardize: bool - standardize features
    - feature_config: dict - full feature configuration (temporal, exogenous)
    
    Returns metrics and predictions.
    """
    start_time = time.time()
    
    # Get params
    target_mode = params.get("target_mode", "raw")
    residual_lag = params.get("residual_lag", 1)
    standardize = params.get("standardize", False)
    
    # Build feature config from params
    if "feature_config" in params:
        fc = params["feature_config"]
        feature_config = FeatureConfig(
            target_lags=fc.get("target_lags", [1, 7]),
            temporal=TemporalFeatureConfig(**fc.get("temporal", {})),
            exogenous=[ExogenousFeatureConfig(**e) for e in fc.get("exogenous", [])],
            derived=[DerivedFeatureConfig(**d) for d in fc.get("derived", [])]
        )
    else:
        # Legacy mode: just use lags param
        lags = params.get("lags", [1, 7])
        if isinstance(lags, str):
            lags = [int(x.strip()) for x in lags.split(",")]
        feature_config = FeatureConfig(target_lags=lags)
    
    # Build features on full dataset
    df_features, feature_names = build_features(df.clone(), date_col, target_col, feature_config)
    
    # If residual mode, create the residual target
    if target_mode == "residual":
        residual_col = f"target_lag_{residual_lag}"
        if residual_col not in df_features.columns:
            df_features = df_features.with_columns(
                pl.col(target_col).shift(residual_lag).alias(residual_col)
            )
        
        df_features = df_features.with_columns(
            (pl.col(target_col) - pl.col(residual_col)).alias("target_residual")
        )
        effective_target = "target_residual"
    else:
        effective_target = target_col
    
    # Build training data from all training ranges
    train_dfs = []
    for tr in training_ranges:
        chunk = filter_by_date_range(df_features, date_col, tr.start, tr.end, inclusive_end=False)
        train_dfs.append(chunk)
    
    if not train_dfs:
        raise ValueError("No training data found in specified ranges")
    
    train_df = pl.concat(train_dfs)
    
    # Drop rows with NaN
    cols_to_check = feature_names.copy()
    if target_mode == "residual":
        cols_to_check.append(effective_target)
    train_df = train_df.drop_nulls(subset=cols_to_check)
    
    if train_df.height == 0:
        raise ValueError("Not enough data after creating features")
    
    # Prepare X and y for training
    X_train = train_df.select(feature_names).to_numpy()
    y_train = train_df.select(effective_target).to_numpy().flatten()
    
    # Standardization
    feature_means = None
    feature_stds = None
    if standardize:
        feature_means = X_train.mean(axis=0)
        feature_stds = X_train.std(axis=0)
        feature_stds[feature_stds == 0] = 1
        X_train = (X_train - feature_means) / feature_stds
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Determine forecast horizon
    horizon = 1
    if forecast_strategy is not None:
        horizon = getattr(forecast_strategy, 'horizon', 1)
    
    # Predict on prediction ranges
    all_predictions = []
    all_actuals = []
    forecast_output = []
    
    for pr in prediction_ranges:
        pred_df = filter_by_date_range(df_features, date_col, pr.start, pr.end)
        
        cols_to_check_pred = feature_names.copy()
        if target_mode == "residual":
            residual_col = f"target_lag_{residual_lag}"
            if residual_col not in cols_to_check_pred:
                cols_to_check_pred.append(residual_col)
        pred_df = pred_df.drop_nulls(subset=cols_to_check_pred)
        
        if pred_df.height == 0:
            continue
        
        if horizon > 1:
            # Block-wise recursive forecasting
            pred_dates = pred_df.select(date_col).to_series().to_list()
            first_date = pred_dates[0]
            last_date = pred_dates[-1]
            
            all_dates = df_features.select(date_col).to_series().to_list()
            try:
                pred_start_idx = next(i for i, d in enumerate(all_dates) if d >= first_date)
                pred_end_idx = next(i for i, d in enumerate(all_dates) if d >= last_date)
            except StopIteration:
                continue
            
            forecasts = block_recursive_forecast(
                model=model,
                df=df_features,
                date_col=date_col,
                target_col=target_col,
                feature_names=feature_names,
                feature_config=feature_config,
                horizon=horizon,
                pred_start_idx=pred_start_idx,
                pred_end_idx=pred_end_idx,
                target_mode=target_mode,
                residual_lag=residual_lag,
                standardize=standardize,
                feature_means=feature_means,
                feature_stds=feature_stds
            )
            
            for f in forecasts:
                forecast_output.append(f)
                actual = f.get(target_col)
                pred = f.get("prediction")
                if actual is not None and pred is not None:
                    all_actuals.append(actual)
                    all_predictions.append(pred)
        else:
            # Standard 1-step prediction
            X_pred = pred_df.select(feature_names).to_numpy()
            y_actual_original = pred_df.select(target_col).to_numpy().flatten()
            dates = pred_df.select(date_col).to_series().to_list()
            
            if standardize and feature_means is not None:
                X_pred = (X_pred - feature_means) / feature_stds
            
            y_pred_raw = model.predict(X_pred)
            
            if target_mode == "residual":
                y_lag_values = pred_df.select(f"target_lag_{residual_lag}").to_numpy().flatten()
                y_pred = y_pred_raw + y_lag_values
            else:
                y_pred = y_pred_raw
            
            all_predictions.extend(y_pred)
            all_actuals.extend(y_actual_original)
            
            for date, pred_val, actual_val in zip(dates, y_pred, y_actual_original):
                forecast_output.append({
                    date_col: date.isoformat() if hasattr(date, 'isoformat') else str(date),
                    "prediction": float(pred_val),
                    target_col: float(actual_val),
                    "horizon_step": 1
                })
    
    # Calculate metrics
    if len(all_actuals) > 0:
        metrics = calculate_metrics(np.array(all_actuals), np.array(all_predictions))
    else:
        metrics = {"rmse": 0, "mae": 0, "mape": 0, "r2": 0, "msle": 0}
    
    execution_time = time.time() - start_time
    metrics["execution_time"] = execution_time
    
    # Calculate metrics by horizon step
    metrics_by_horizon = None
    if horizon > 1 and len(forecast_output) > 0:
        metrics_by_horizon = calculate_metrics_by_horizon(forecast_output, target_col)
    
    # Feature importance (normalized coefficients)
    feature_importance = []
    abs_coefs = [abs(coef) for coef in model.coef_]
    total_abs = sum(abs_coefs) if sum(abs_coefs) > 0 else 1.0
    
    for feat_name, abs_coef in zip(feature_names, abs_coefs):
        feature_importance.append({
            "feature": feat_name,
            "importance": float(abs_coef / total_abs)
        })
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    
    return {
        "metrics": metrics,
        "forecast": forecast_output,
        "feature_importance": feature_importance,
        "metrics_by_horizon": metrics_by_horizon
    }
