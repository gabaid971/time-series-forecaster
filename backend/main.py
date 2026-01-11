"""
Time Series Forecaster Backend
Refactored: modular structure with clean separation of concerns.
"""

import os
import secrets
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import polars as pl
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# Import modular components
from utils.date_utils import detect_frequency, parse_dates_flexible, filter_by_date_range
from utils.features import (
    FeatureConfig, 
    TemporalFeatureConfig, 
    ExogenousFeatureConfig, 
    DerivedFeatureConfig
)
from models import (
    train_lag,
    train_linear_regression,
    train_xgboost,
    train_arima,
    train_prophet
)

# Check Prophet availability
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not installed. Prophet model will be unavailable.")

API_KEY = os.environ.get("API_KEY", "")

def verify_api_key(x_api_key: str = Header(...)):
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API Key")

# ============================================================================
# SCHEMAS (Pydantic models for API request/response)
# ============================================================================

class DateRange(BaseModel):
    start: str
    end: str

class ForecastStrategyConfig(BaseModel):
    """Configuration for multi-step forecasting."""
    horizon: int = 1
    mode: str = "direct"
    sliding_window: Optional[bool] = False
    window_size: Optional[int] = None

class DataConfig(BaseModel):
    target_column: str
    date_column: str
    frequency: str
    training_ranges: List[DateRange]
    prediction_ranges: List[DateRange]
    forecast_strategy: Optional[ForecastStrategyConfig] = None

class ModelConfig(BaseModel):
    id: str
    type: str
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
    msle: float
    execution_time: float

class HorizonMetrics(BaseModel):
    """Metrics for a specific forecast horizon step."""
    horizon_step: int
    rmse: float
    mae: float
    mape: float
    msle: float
    count: int

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class ModelResult(BaseModel):
    model_id: str
    model_name: str
    metrics: ModelMetrics
    forecast: List[Dict[str, Any]]
    metrics_by_horizon: Optional[List[HorizonMetrics]] = None
    feature_importance: Optional[List[FeatureImportance]] = None
    shap_analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class TrainingResponse(BaseModel):
    status: str
    results: List[ModelResult]
    message: Optional[str] = None

# Dataset analysis schemas
class DatasetAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_column: str
    target_column: str

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    missing_count: int
    sample_values: List[Any]

class DatasetStats(BaseModel):
    date_min: str
    date_max: str
    total_rows: int
    frequency: str
    frequency_label: str
    missing_dates: int
    missing_values_target: int
    value_min: float
    value_max: float
    value_mean: float

class NormalizedDataPoint(BaseModel):
    date: str
    value: float

class DatasetAnalysisResponse(BaseModel):
    status: str
    stats: Optional[DatasetStats] = None
    normalized_data: Optional[List[NormalizedDataPoint]] = None
    available_columns: Optional[List[ColumnInfo]] = None
    message: Optional[str] = None

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Time Series Forecaster API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/analyze", response_model=DatasetAnalysisResponse)
async def analyze_dataset(request: DatasetAnalysisRequest, _: None = Depends(verify_api_key)):
    """
    Analyze a dataset and return statistics.
    Auto-detects frequency, missing values, date range, etc.
    """
    try:
        df = pl.DataFrame(request.data, infer_schema_length=None)
        date_col = request.date_column
        target_col = request.target_column
        
        # Parse dates
        df = parse_dates_flexible(df, date_col)
        df = df.filter(pl.col(date_col).is_not_null())
        
        # Convert target to float
        try:
            df = df.with_columns(
                pl.col(target_col)
                  .cast(pl.Utf8)
                  .str.strip_chars()
                  .str.replace(r"^\?", "")
                  .str.replace(r"[^\d\.-]", "")
                  .cast(pl.Float64)
            )
        except Exception:
            df = df.with_columns(
                pl.col(target_col).cast(pl.Float64, strict=False)
            )
        
        df = df.sort(date_col)
        
        if df.height == 0:
            return DatasetAnalysisResponse(status="error", message="No valid data after parsing")
        
        # Get stats
        date_min = df[date_col].min()
        date_max = df[date_col].max()
        freq_code, freq_label, missing_dates = detect_frequency(df, date_col)
        
        target_series = df[target_col]
        missing_target = target_series.null_count()
        valid_target = target_series.drop_nulls()
        
        stats = DatasetStats(
            date_min=date_min.isoformat() if hasattr(date_min, 'isoformat') else str(date_min),
            date_max=date_max.isoformat() if hasattr(date_max, 'isoformat') else str(date_max),
            total_rows=df.height,
            frequency=freq_code,
            frequency_label=freq_label,
            missing_dates=missing_dates,
            missing_values_target=missing_target,
            value_min=float(valid_target.min()) if len(valid_target) > 0 else 0.0,
            value_max=float(valid_target.max()) if len(valid_target) > 0 else 0.0,
            value_mean=float(valid_target.mean()) if len(valid_target) > 0 else 0.0,
        )
        
        # Normalized data for frontend
        clean_df = df.filter(pl.col(target_col).is_not_null())
        normalized_data = [
            NormalizedDataPoint(
                date=row[date_col].isoformat() if hasattr(row[date_col], 'isoformat') else str(row[date_col]),
                value=float(row[target_col])
            )
            for row in clean_df.iter_rows(named=True)
        ]
        
        # Available columns info
        available_columns = []
        for col_name in df.columns:
            if col_name in [date_col, target_col]:
                continue
            
            col_series = df[col_name]
            dtype = col_series.dtype
            
            if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                dtype_str = "numeric"
            elif dtype == pl.Boolean:
                dtype_str = "boolean"
            elif dtype in [pl.Datetime, pl.Date]:
                dtype_str = "date"
            else:
                dtype_str = "string"
            
            sample_vals = col_series.drop_nulls().head(5).to_list()
            available_columns.append(ColumnInfo(
                name=col_name,
                dtype=dtype_str,
                missing_count=col_series.null_count(),
                sample_values=sample_vals
            ))
        
        return DatasetAnalysisResponse(
            status="success",
            stats=stats,
            normalized_data=normalized_data,
            available_columns=available_columns
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return DatasetAnalysisResponse(status="error", message=str(e))


@app.post("/train", response_model=TrainingResponse)
async def train_models(request: TrainingRequest, _: None = Depends(verify_api_key)):
    """
    Main training endpoint.
    Receives data + config, trains requested models, returns predictions + metrics.
    """
    try:
        df = pl.DataFrame(request.data, infer_schema_length=None)
        
        date_col = request.data_config.date_column
        target_col = request.data_config.target_column
        
        # Parse dates
        df = parse_dates_flexible(df, date_col)
        df = df.filter(pl.col(date_col).is_not_null())
        
        # Handle target column
        try:
            df = df.with_columns(
                pl.col(target_col)
                  .cast(pl.Utf8)
                  .str.strip_chars()
                  .str.replace(r"^\?", "")
                  .str.replace(r"[^\d\.-]", "")
                  .cast(pl.Float64)
            )
        except Exception:
            df = df.with_columns(
                pl.col(target_col).cast(pl.Float64, strict=False)
            )
        
        df = df.filter(pl.col(target_col).is_not_null())
        
        # Convert exogenous columns to numeric
        for col_name in df.columns:
            if col_name in [date_col, target_col]:
                continue
            col_dtype = df[col_name].dtype
            if col_dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                try:
                    df = df.with_columns(
                        pl.col(col_name).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)
                    )
                except Exception:
                    pass
        
        df = df.sort(date_col)
        
        if df.height == 0:
            raise ValueError("No valid data rows after cleaning")
        
        results = []
        
        for model_config in request.models:
            try:
                if model_config.type == "LAG":
                    result = train_lag(
                        df=df,
                        date_col=date_col,
                        target_col=target_col,
                        training_ranges=request.data_config.training_ranges,
                        prediction_ranges=request.data_config.prediction_ranges,
                        params=model_config.params,
                        forecast_strategy=request.data_config.forecast_strategy
                    )
                elif model_config.type == "LINEAR_REGRESSION":
                    result = train_linear_regression(
                        df=df,
                        date_col=date_col,
                        target_col=target_col,
                        training_ranges=request.data_config.training_ranges,
                        prediction_ranges=request.data_config.prediction_ranges,
                        params=model_config.params,
                        forecast_strategy=request.data_config.forecast_strategy
                    )
                elif model_config.type == "XGBOOST":
                    result = train_xgboost(
                        df=df,
                        date_col=date_col,
                        target_col=target_col,
                        training_ranges=request.data_config.training_ranges,
                        prediction_ranges=request.data_config.prediction_ranges,
                        params=model_config.params,
                        forecast_strategy=request.data_config.forecast_strategy
                    )
                elif model_config.type == "ARIMA":
                    horizon = request.data_config.forecast_strategy.horizon if request.data_config.forecast_strategy else 1
                    result = train_arima(
                        df=df,
                        date_col=date_col,
                        target_col=target_col,
                        training_ranges=request.data_config.training_ranges,
                        prediction_ranges=request.data_config.prediction_ranges,
                        params=model_config.params,
                        horizon=horizon
                    )
                elif model_config.type == "PROPHET":
                    if not PROPHET_AVAILABLE:
                        raise ValueError("Prophet is not installed. Use Linear Regression, XGBoost, or ARIMA instead.")
                    result = train_prophet(
                        df=df,
                        date_col=date_col,
                        target_col=target_col,
                        training_ranges=request.data_config.training_ranges,
                        prediction_ranges=request.data_config.prediction_ranges,
                        params=model_config.params
                    )
                else:
                    result = {
                        "metrics": {"rmse": 0, "mae": 0, "mape": 0, "r2": 0, "msle": 0, "execution_time": 0},
                        "forecast": [],
                        "feature_importance": None
                    }
                
                results.append(ModelResult(
                    model_id=model_config.id,
                    model_name=model_config.name,
                    metrics=ModelMetrics(**result["metrics"]),
                    forecast=result["forecast"],
                    metrics_by_horizon=[HorizonMetrics(**hm) for hm in result.get("metrics_by_horizon", [])] if result.get("metrics_by_horizon") else None,
                    feature_importance=[FeatureImportance(**fi) for fi in result.get("feature_importance", [])] if result.get("feature_importance") else None,
                    shap_analysis=result.get("shap_analysis")
                ))
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error training {model_config.name}: {error_msg}")
                results.append(ModelResult(
                    model_id=model_config.id,
                    model_name=model_config.name,
                    metrics=ModelMetrics(rmse=0, mae=0, mape=0, r2=0, msle=0, execution_time=0),
                    forecast=[],
                    error=error_msg
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
