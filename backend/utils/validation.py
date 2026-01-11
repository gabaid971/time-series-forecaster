"""Validation and forecasting utilities."""

import polars as pl
import numpy as np
from typing import List, Dict, Any, Optional

from utils.features import FeatureConfig


def block_recursive_forecast(
    model,
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    feature_names: List[str],
    feature_config: FeatureConfig,
    horizon: int,
    pred_start_idx: int,
    pred_end_idx: int,
    target_mode: str = "raw",
    residual_lag: int = 1,
    standardize: bool = False,
    feature_means: Optional[np.ndarray] = None,
    feature_stds: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Generate forecasts using block-wise recursive prediction.
    
    Strategy:
    - Model trained ONCE on [0, pred_start_idx - 1]
    - Prediction range [pred_start_idx, pred_end_idx] split into blocks of size `horizon`
    - Within each block: recursive prediction (use predictions for lags)
    - Between blocks: reset to actual values for lags
    
    Block 1: [pred_start_idx, pred_start_idx + horizon - 1]
      - Step 1: use actual historical lags
      - Step 2: use prediction from step 1 for lag_1, actual for other lags
      - Step h: fully recursive within block
      
    Block 2: [pred_start_idx + horizon, pred_start_idx + 2*horizon - 1]
      - RESET: use actual values (including block 1 actuals) for lags
      - Then recursive within block
    
    Args:
        model: Trained sklearn-compatible model
        df: Full dataframe with features already computed (sorted by date)
        date_col, target_col: Column names
        feature_names: Features used by model
        feature_config: For lag information
        horizon: Block size (h steps per block)
        pred_start_idx: Index where prediction starts
        pred_end_idx: Index where prediction ends (inclusive)
        target_mode: "raw" or "residual"
        residual_lag: Lag for residual reconstruction
        standardize, feature_means, feature_stds: Standardization params
    
    Returns:
        List of forecast dicts: {date, prediction, actual, block_num, step_in_block, horizon_step}
    """
    target_lags = feature_config.target_lags if feature_config else [1]
    forecasts = []
    
    # Process blocks
    block_num = 0
    current_idx = pred_start_idx
    
    while current_idx <= pred_end_idx:
        block_num += 1
        block_end_idx = min(current_idx + horizon - 1, pred_end_idx)
        block_predictions = {}  # step_in_block -> predicted value
        
        for step, idx in enumerate(range(current_idx, block_end_idx + 1), start=1):
            if idx >= df.height:
                break
            
            row = df.row(idx, named=True)
            
            # Build feature vector
            feature_values = []
            for feat_name in feature_names:
                if feat_name.startswith("target_lag_"):
                    lag = int(feat_name.split("_")[-1])
                    
                    # Which step in this block does this lag refer to?
                    lag_refers_to_step = step - lag
                    
                    if lag_refers_to_step > 0 and lag_refers_to_step in block_predictions:
                        # Use prediction from earlier in THIS block (recursive)
                        feature_values.append(block_predictions[lag_refers_to_step])
                    else:
                        # Use actual value from dataframe (historical or previous blocks)
                        feature_values.append(row.get(feat_name, 0))
                else:
                    feature_values.append(row.get(feat_name, 0))
            
            # Skip if any None
            if any(v is None for v in feature_values):
                continue
            
            X = np.array([feature_values])
            
            # Standardization
            if standardize and feature_means is not None and feature_stds is not None:
                X = (X - feature_means) / feature_stds
            
            # Predict
            y_pred_raw = float(model.predict(X)[0])
            
            # Residual mode reconstruction
            if target_mode == "residual":
                residual_feat = f"target_lag_{residual_lag}"
                lag_refers_to_step = step - residual_lag
                
                if lag_refers_to_step > 0 and lag_refers_to_step in block_predictions:
                    y_lag = block_predictions[lag_refers_to_step]
                else:
                    y_lag = row.get(residual_feat, 0)
                
                y_pred = y_pred_raw + y_lag if y_lag is not None else y_pred_raw
            else:
                y_pred = y_pred_raw
            
            # Store for recursive use within block
            block_predictions[step] = y_pred
            
            # Get actual and date
            actual = row.get(target_col)
            date_value = row.get(date_col)
            
            forecasts.append({
                date_col: date_value.isoformat() if hasattr(date_value, 'isoformat') else str(date_value),
                "prediction": float(y_pred),
                target_col: float(actual) if actual is not None else None,
                "block_num": block_num,
                "step_in_block": step,
                "horizon_step": step  # For compatibility with metrics calculation
            })
        
        # Move to next block
        current_idx = block_end_idx + 1
    
    return forecasts
