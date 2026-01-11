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

# Helper function to convert numpy types to native Python types for JSON serialization
def numpy_to_native(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return [numpy_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_native(x) for x in obj]
    else:
        return obj

# Import modular components
from utils.date_utils import detect_frequency, parse_dates_flexible, filter_by_date_range
from utils.features import (
    FeatureConfig, 
    TemporalFeatureConfig, 
    ExogenousFeatureConfig, 
    DerivedFeatureConfig
)
from utils.analysis import (
    suggest_lags,
    detect_outliers,
    detect_trend,
    compute_stationarity_indicators
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

# Advanced analysis schemas
class LagSuggestion(BaseModel):
    suggested_lags: List[int]
    acf: List[float]
    pacf: List[float]
    confidence_interval: float
    significant_lags: List[Dict[str, Any]]
    seasonality: Dict[str, Any]
    n_observations: int

class DataAlert(BaseModel):
    type: str  # 'warning', 'info', 'error'
    category: str  # 'outliers', 'missing', 'trend', 'stationarity'
    message: str
    details: Optional[Dict[str, Any]] = None

class DatasetAnalysisResponse(BaseModel):
    status: str
    stats: Optional[DatasetStats] = None
    normalized_data: Optional[List[NormalizedDataPoint]] = None
    available_columns: Optional[List[ColumnInfo]] = None
    lag_analysis: Optional[LagSuggestion] = None
    alerts: Optional[List[DataAlert]] = None
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
        
        # ====================================================================
        # ADVANCED ANALYSIS: ACF/PACF, Outliers, Trend, Stationarity
        # ====================================================================
        alerts: List[DataAlert] = []
        lag_analysis = None
        
        # Get clean numpy array for analysis
        target_values = valid_target.to_numpy()
        
        if len(target_values) >= 20:
            # Lag suggestion with ACF/PACF
            try:
                lag_result = suggest_lags(target_values, frequency=freq_code, max_lags=20)
                # Convert numpy types to native Python for JSON serialization
                lag_result = numpy_to_native(lag_result)
                lag_analysis = LagSuggestion(
                    suggested_lags=lag_result["suggested_lags"],
                    acf=lag_result["acf"],
                    pacf=lag_result["pacf"],
                    confidence_interval=lag_result["confidence_interval"],
                    significant_lags=lag_result["significant_lags"],
                    seasonality=lag_result["seasonality"],
                    n_observations=lag_result["n_observations"]
                )
                
                # Seasonality alert
                if lag_result["seasonality"].get("detected"):
                    period_label = lag_result["seasonality"].get("period_label", "")
                    strength = lag_result["seasonality"].get("strength", 0)
                    alerts.append(DataAlert(
                        type="info",
                        category="seasonality",
                        message=f"Seasonality detected: {period_label} pattern (strength: {strength:.2f})",
                        details=lag_result["seasonality"]
                    ))
            except Exception as e:
                print(f"Lag analysis failed: {e}")
            
            # Outlier detection
            try:
                outlier_result = numpy_to_native(detect_outliers(target_values, method="iqr"))
                if outlier_result["count"] > 0:
                    pct = outlier_result["percentage"]
                    alert_type = "warning" if pct > 5 else "info"
                    alerts.append(DataAlert(
                        type=alert_type,
                        category="outliers",
                        message=f"{outlier_result['count']} outliers detected ({pct:.1f}% of data)",
                        details=outlier_result
                    ))
            except Exception as e:
                print(f"Outlier detection failed: {e}")
            
            # Trend detection
            try:
                trend_result = numpy_to_native(detect_trend(target_values))
                if trend_result["detected"]:
                    direction = trend_result["direction"]
                    strength = trend_result["strength"]
                    alerts.append(DataAlert(
                        type="info",
                        category="trend",
                        message=f"{strength.capitalize()} {direction} trend detected",
                        details=trend_result
                    ))
            except Exception as e:
                print(f"Trend detection failed: {e}")
            
            # Stationarity check
            try:
                stat_result = numpy_to_native(compute_stationarity_indicators(target_values))
                if stat_result.get("likely_stationary") is False:
                    alerts.append(DataAlert(
                        type="warning",
                        category="stationarity",
                        message="Series may be non-stationary. Consider differencing for ARIMA.",
                        details=stat_result
                    ))
            except Exception as e:
                print(f"Stationarity check failed: {e}")
        
        # Missing data alerts
        if missing_dates > 0:
            pct = (missing_dates / df.height) * 100
            alert_type = "warning" if pct > 10 else "info"
            alerts.append(DataAlert(
                type=alert_type,
                category="missing",
                message=f"{missing_dates} missing dates detected ({pct:.1f}%)",
                details={"missing_count": missing_dates, "percentage": pct}
            ))
        
        if missing_target > 0:
            pct = (missing_target / df.height) * 100
            alert_type = "warning" if pct > 5 else "info"
            alerts.append(DataAlert(
                type=alert_type,
                category="missing",
                message=f"{missing_target} missing target values ({pct:.1f}%)",
                details={"missing_count": missing_target, "percentage": pct}
            ))
        
        return DatasetAnalysisResponse(
            status="success",
            stats=stats,
            normalized_data=normalized_data,
            available_columns=available_columns,
            lag_analysis=lag_analysis,
            alerts=alerts if alerts else None
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
