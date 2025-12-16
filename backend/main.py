"""
Time Series Forecaster Backend
Simple, readable API with Polars + scikit-learn
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
import time
from datetime import datetime

# ============================================================================
# SCHEMAS (Pydantic models for API request/response)
# ============================================================================

class DateRange(BaseModel):
    start: str
    end: str

class DataConfig(BaseModel):
    target_column: str
    date_column: str
    frequency: str
    training_ranges: List[DateRange]
    prediction_ranges: List[DateRange]

class ModelConfig(BaseModel):
    id: str
    type: str  # LINEAR_REGRESSION, XGBOOST, ARIMA, etc.
    name: str
    params: Dict[str, Any]

class TrainingRequest(BaseModel):
    data: List[Dict[str, Any]]
    data_config: DataConfig
    models: List[ModelConfig]

class ModelMetrics(BaseModel):
    rmse: float
    mae: float
    mape: float
    r2: float
    execution_time: float

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class ModelResult(BaseModel):
    model_id: str
    model_name: str
    metrics: ModelMetrics
    forecast: List[Dict[str, Any]]
    feature_importance: Optional[List[FeatureImportance]] = None

class TrainingResponse(BaseModel):
    status: str
    results: List[ModelResult]
    message: Optional[str] = None

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Time Series Forecaster API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_lag_features(df: pl.DataFrame, target_col: str, lags: List[int]) -> pl.DataFrame:
    """Create lag features for time series prediction."""
    for lag in lags:
        df = df.with_columns(
            pl.col(target_col).shift(lag).alias(f"lag_{lag}")
        )
    return df

def filter_by_date_range(df: pl.DataFrame, date_col: str, start: str, end: str) -> pl.DataFrame:
    """Filter dataframe by date range."""
    return df.filter(
        (pl.col(date_col) >= pl.lit(start).str.to_datetime()) &
        (pl.col(date_col) <= pl.lit(end).str.to_datetime())
    )

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    else:
        mape = 0.0
    
    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

# ============================================================================
# MODEL TRAINERS
# ============================================================================

def train_linear_regression(
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    training_ranges: List[DateRange],
    prediction_ranges: List[DateRange],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train a Linear Regression model with lag features.
    
    Supports:
    - target_mode: "raw" (predict y directly) or "residual" (predict y - y_lag)
    - residual_lag: which lag to subtract when using residual mode (default: 1)
    - standardize: whether to standardize features (useful for regularized models)
    
    Returns metrics and predictions.
    """
    start_time = time.time()
    
    # Get params
    lags = params.get("lags", [1, 7])
    if isinstance(lags, str):
        lags = [int(x.strip()) for x in lags.split(",")]
    
    target_mode = params.get("target_mode", "raw")  # "raw" or "residual"
    residual_lag = params.get("residual_lag", 1)    # which lag to subtract
    standardize = params.get("standardize", False)   # standardize features
    
    # Create lag features on full dataset
    df_features = create_lag_features(df.clone(), target_col, lags)
    
    # If residual mode, create the residual target
    if target_mode == "residual":
        # Ensure residual_lag is in lags (we need it for reconstruction)
        if residual_lag not in lags:
            df_features = df_features.with_columns(
                pl.col(target_col).shift(residual_lag).alias(f"lag_{residual_lag}")
            )
        
        # Create residual target: y_residual = y - y_lag
        df_features = df_features.with_columns(
            (pl.col(target_col) - pl.col(f"lag_{residual_lag}")).alias("target_residual")
        )
        effective_target = "target_residual"
    else:
        effective_target = target_col
    
    # Build training data from all training ranges
    train_dfs = []
    for tr in training_ranges:
        chunk = filter_by_date_range(df_features, date_col, tr.start, tr.end)
        train_dfs.append(chunk)
    
    if not train_dfs:
        raise ValueError("No training data found in specified ranges")
    
    train_df = pl.concat(train_dfs)
    
    # Drop rows with NaN (from lag creation)
    lag_cols = [f"lag_{lag}" for lag in lags]
    cols_to_check = lag_cols + ([effective_target] if target_mode == "residual" else [])
    train_df = train_df.drop_nulls(subset=cols_to_check)
    
    if train_df.height == 0:
        raise ValueError("Not enough data after creating lag features")
    
    # Prepare X and y for training
    X_train = train_df.select(lag_cols).to_numpy()
    y_train = train_df.select(effective_target).to_numpy().flatten()
    
    # Standardization
    feature_means = None
    feature_stds = None
    if standardize:
        feature_means = X_train.mean(axis=0)
        feature_stds = X_train.std(axis=0)
        feature_stds[feature_stds == 0] = 1  # Avoid division by zero
        X_train = (X_train - feature_means) / feature_stds
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on prediction ranges
    all_predictions = []
    all_actuals = []
    forecast_output = []
    
    for pr in prediction_ranges:
        pred_df = filter_by_date_range(df_features, date_col, pr.start, pr.end)
        cols_to_check_pred = lag_cols + ([effective_target, f"lag_{residual_lag}"] if target_mode == "residual" else [])
        pred_df = pred_df.drop_nulls(subset=cols_to_check_pred)
        
        if pred_df.height == 0:
            continue
        
        X_pred = pred_df.select(lag_cols).to_numpy()
        y_actual_original = pred_df.select(target_col).to_numpy().flatten()
        dates = pred_df.select(date_col).to_series().to_list()
        
        # Standardize prediction features if needed
        if standardize and feature_means is not None:
            X_pred = (X_pred - feature_means) / feature_stds
        
        y_pred_raw = model.predict(X_pred)
        
        # If residual mode, reconstruct original scale: y_pred = y_pred_residual + y_lag
        if target_mode == "residual":
            y_lag_values = pred_df.select(f"lag_{residual_lag}").to_numpy().flatten()
            y_pred = y_pred_raw + y_lag_values
        else:
            y_pred = y_pred_raw
        
        all_predictions.extend(y_pred)
        all_actuals.extend(y_actual_original)
        
        # Build forecast output for frontend
        for date, pred_val, actual_val in zip(dates, y_pred, y_actual_original):
            forecast_output.append({
                date_col: date.isoformat() if hasattr(date, 'isoformat') else str(date),
                "prediction": float(pred_val),
                target_col: float(actual_val)
            })
    
    # Calculate metrics (on original scale)
    if len(all_actuals) > 0:
        metrics = calculate_metrics(np.array(all_actuals), np.array(all_predictions))
    else:
        metrics = {"rmse": 0, "mae": 0, "mape": 0, "r2": 0}
    
    execution_time = time.time() - start_time
    metrics["execution_time"] = execution_time
    
    # Extract feature importance (coefficients for Linear Regression)
    feature_importance = []
    for feat_name, coef in zip(lag_cols, model.coef_):
        feature_importance.append({
            "feature": feat_name,
            "importance": float(abs(coef))  # Use absolute value for importance
        })
    # Sort by importance descending
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    
    return {
        "metrics": metrics,
        "forecast": forecast_output,
        "feature_importance": feature_importance
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Time Series Forecaster API is running"}

@app.get("/health")
async def health():
    """Health check for monitoring."""
    return {"status": "healthy"}

@app.post("/train", response_model=TrainingResponse)
async def train_models(request: TrainingRequest):
    """
    Main training endpoint.
    Receives data + config, trains requested models, returns predictions + metrics.
    """
    try:
        # 1. Convert data to Polars DataFrame
        df = pl.DataFrame(request.data)
        date_col = request.data_config.date_column
        target_col = request.data_config.target_column
        
        # Ensure date column is datetime
        df = df.with_columns(
            pl.col(date_col).str.to_datetime()
        )
        
        # Sort by date
        df = df.sort(date_col)
        
        results = []
        
        # 2. Train each requested model
        for model_config in request.models:
            try:
                if model_config.type == "LINEAR_REGRESSION":
                    result = train_linear_regression(
                        df=df,
                        date_col=date_col,
                        target_col=target_col,
                        training_ranges=request.data_config.training_ranges,
                        prediction_ranges=request.data_config.prediction_ranges,
                        params=model_config.params
                    )
                else:
                    # For unimplemented models, return a placeholder
                    result = {
                        "metrics": {
                            "rmse": 0, "mae": 0, "mape": 0, "r2": 0, "execution_time": 0
                        },
                        "forecast": [],
                        "feature_importance": None
                    }
                
                results.append(ModelResult(
                    model_id=model_config.id,
                    model_name=model_config.name,
                    metrics=ModelMetrics(**result["metrics"]),
                    forecast=result["forecast"],
                    feature_importance=[FeatureImportance(**fi) for fi in result.get("feature_importance", [])] if result.get("feature_importance") else None
                ))
                
            except Exception as e:
                print(f"Error training {model_config.name}: {e}")
                # Return empty result for failed model
                results.append(ModelResult(
                    model_id=model_config.id,
                    model_name=model_config.name,
                    metrics=ModelMetrics(rmse=0, mae=0, mape=0, r2=0, execution_time=0),
                    forecast=[]
                ))
        
        return TrainingResponse(
            status="success",
            results=results,
            message=f"Trained {len(results)} model(s) successfully"
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Time Series Forecaster Backend...")
    print("📍 API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
