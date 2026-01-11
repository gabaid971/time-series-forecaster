"""XGBoost model for time series forecasting with SHAP analysis."""

import time
import numpy as np
import polars as pl
import pandas as pd
import xgboost as xgb
import shap
from typing import Dict, Any, List, Optional

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


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: List[str],
    df_aligned,
    date_col: str,
    feature_config: FeatureConfig
) -> Dict[str, Any]:
    """
    Compute SHAP values for XGBoost and aggregate them into
    human-interpretable temporal and exogenous effects.
    """
    # Compute raw SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # Convert df_aligned to pandas if needed
    if not isinstance(df_aligned, pd.DataFrame):
        df_aligned = df_aligned.to_pandas()

    assert len(df_aligned) == X.shape[0], "SHAP / data misalignment"

    output: Dict[str, Any] = {
        "temporal": {},
        "exogenous": {},
    }

    # Temporal features (cyclical aggregation)
    temporal_groups = {
        "hour_of_day": {
            "enabled": feature_config.temporal.hour_of_day,
            "features": ["hour_sin", "hour_cos"],
            "values": df_aligned[date_col].dt.hour,
            "range": range(24),
        },
        "day_of_week": {
            "enabled": feature_config.temporal.day_of_week,
            "features": ["dow_sin", "dow_cos"],
            "values": df_aligned[date_col].dt.weekday,
            "range": range(7),
        },
        "month": {
            "enabled": feature_config.temporal.month,
            "features": ["month_sin", "month_cos"],
            "values": df_aligned[date_col].dt.month,
            "range": range(1, 13),
        },
        "minute_of_day": {
            "enabled": feature_config.temporal.minute_of_day,
            "features": ["minute_of_day_sin", "minute_of_day_cos"],
            "values": (
                df_aligned[date_col].dt.hour * 60
                + df_aligned[date_col].dt.minute
            ),
            "range": range(1440),
        },
    }

    for name, cfg in temporal_groups.items():
        if not cfg["enabled"]:
            continue

        f1, f2 = cfg["features"]
        if f1 not in shap_df.columns or f2 not in shap_df.columns:
            continue

        shap_sum = shap_df[f1] + shap_df[f2]
        temp_df = pd.DataFrame({
            "value": cfg["values"].values,
            "shap": shap_sum.values,
        })

        grouped = temp_df.groupby("value", as_index=False).agg(
            shap=("shap", "mean"),
            count=("shap", "size"),
        )

        # Fill missing values
        existing = set(grouped["value"].tolist())
        full_rows = []
        for v in cfg["range"]:
            if v in existing:
                row = grouped[grouped["value"] == v].iloc[0]
                full_rows.append({
                    "value": int(v),
                    "shap": float(row["shap"]),
                    "count": int(row["count"]),
                })
            else:
                full_rows.append({
                    "value": int(v),
                    "shap": 0.0,
                    "count": 0,
                })

        # Normalization
        max_abs = max(abs(r["shap"]) for r in full_rows) or 1.0
        for r in full_rows:
            r["shap_norm"] = r["shap"] / max_abs

        output["temporal"][name] = full_rows

    # Exogenous features
    for exog in feature_config.exogenous:
        base_col = exog.column
        related_features = [
            f for f in feature_names
            if f.startswith(base_col + "_") or f == f"{base_col}_actual"
        ]

        if not related_features:
            continue

        mean_abs_shap = float(np.abs(shap_df[related_features].values).mean())
        mean_shap = float(shap_df[related_features].values.mean())

        output["exogenous"][base_col] = {
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": mean_shap,
            "direction": "positive" if mean_shap > 0 else "negative" if mean_shap < 0 else "neutral",
            "features": related_features,
        }

    return output


def train_xgboost(
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    training_ranges: List[Any],
    prediction_ranges: List[Any],
    params: Dict[str, Any],
    forecast_strategy: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Train an XGBoost model with configurable features.
    Uses the same feature engineering as Linear Regression.
    Supports target_mode (raw/residual).
    """
    start_time = time.time()
    
    # Get XGBoost specific params
    n_estimators = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", 6)
    learning_rate = params.get("learning_rate", 0.1)
    
    # Get target mode params
    target_mode = params.get("target_mode", "raw")
    residual_lag = params.get("residual_lag", 1)
    
    # Build feature config
    if "feature_config" in params:
        fc = params["feature_config"]
        feature_config = FeatureConfig(
            target_lags=fc.get("target_lags", [1, 7]),
            temporal=TemporalFeatureConfig(**fc.get("temporal", {})),
            exogenous=[ExogenousFeatureConfig(**e) for e in fc.get("exogenous", [])],
            derived=[DerivedFeatureConfig(**d) for d in fc.get("derived", [])]
        )
    else:
        lags = params.get("lags", [1, 7, 14, 30])
        if isinstance(lags, str):
            lags = [int(x.strip()) for x in lags.split(",")]
        feature_config = FeatureConfig(target_lags=lags)
    
    # Build features
    df_features, feature_names = build_features(df.clone(), date_col, target_col, feature_config)
    
    # Residual mode
    effective_target = target_col
    if target_mode == "residual":
        residual_col = f"target_lag_{residual_lag}"
        if residual_col not in df_features.columns:
            df_features = df_features.with_columns(
                pl.col(target_col).shift(residual_lag).alias(residual_col)
            )
        
        df_features = df_features.with_columns(
            (pl.col(target_col) - pl.col(residual_col)).alias("_target_residual")
        )
        effective_target = "_target_residual"
    
    # Build training data
    train_dfs = []
    for tr in training_ranges:
        chunk = filter_by_date_range(df_features, date_col, tr.start, tr.end, inclusive_end=False)
        train_dfs.append(chunk)
    
    if not train_dfs:
        raise ValueError("No training data found")
    
    cols_to_check = feature_names.copy()
    if target_mode == "residual":
        cols_to_check.append(effective_target)
    train_df = pl.concat(train_dfs).drop_nulls(subset=cols_to_check)
    
    if train_df.height == 0:
        raise ValueError("Not enough data after creating features")
    
    X_train = train_df.select(feature_names).to_numpy()
    y_train = train_df.select(effective_target).to_numpy().flatten()
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Determine forecast horizon
    horizon = 1
    if forecast_strategy is not None:
        horizon = getattr(forecast_strategy, 'horizon', 1)
    
    # Predict
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
                standardize=False,
                feature_means=None,
                feature_stds=None
            )
            
            for f in forecasts:
                forecast_output.append(f)
                actual = f.get(target_col)
                pred = f.get("prediction")
                if actual is not None and pred is not None:
                    all_actuals.append(actual)
                    all_predictions.append(pred)
        else:
            X_pred = pred_df.select(feature_names).to_numpy()
            y_actual_original = pred_df.select(target_col).to_numpy().flatten()
            dates = pred_df.select(date_col).to_series().to_list()
            
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
    
    # Metrics
    metrics = calculate_metrics(np.array(all_actuals), np.array(all_predictions)) if all_actuals else {"rmse": 0, "mae": 0, "mape": 0, "r2": 0, "msle": 0}
    metrics["execution_time"] = time.time() - start_time
    
    # Metrics by horizon
    metrics_by_horizon = None
    if horizon > 1 and len(forecast_output) > 0:
        metrics_by_horizon = calculate_metrics_by_horizon(forecast_output, target_col)
    
    # Feature importance
    feature_importance = [
        {"feature": name, "importance": float(imp)}
        for name, imp in zip(feature_names, model.feature_importances_)
    ]
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    
    # SHAP analysis
    shap_analysis = compute_shap_values(
        model=model,
        X=X_train,
        feature_names=feature_names,
        df_aligned=train_df.select([date_col] + feature_names),
        date_col=date_col,
        feature_config=feature_config,
    )

    return {
        "metrics": metrics,
        "forecast": forecast_output,
        "feature_importance": feature_importance,
        "shap_analysis": shap_analysis,
        "metrics_by_horizon": metrics_by_horizon
    }
