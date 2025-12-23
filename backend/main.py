"""
Time Series Forecaster Backend
Simple, readable API with Polars + scikit-learn
"""

import os
import secrets
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')  # Suppress Prophet/statsmodels warnings

# Optional Prophet import (heavy dependency)
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
    error: Optional[str] = None  # Error message if model training failed

class TrainingResponse(BaseModel):
    status: str
    results: List[ModelResult]
    message: Optional[str] = None

# Schema for dataset analysis
class DatasetAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_column: str
    target_column: str

class ColumnInfo(BaseModel):
    """Information about a column in the dataset."""
    name: str
    dtype: str  # "numeric", "string", "date", "boolean"
    missing_count: int
    sample_values: List[Any]

class DatasetStats(BaseModel):
    date_min: str
    date_max: str
    total_rows: int
    frequency: str  # "D", "W", "M", "H", "min", "irregular"
    frequency_label: str  # Human readable: "Daily", "Weekly", etc.
    missing_dates: int
    missing_values_target: int
    value_min: float
    value_max: float
    value_mean: float

class NormalizedDataPoint(BaseModel):
    """A single data point with normalized date (ISO format) and value."""
    date: str  # ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
    value: float

class DatasetAnalysisResponse(BaseModel):
    status: str
    stats: Optional[DatasetStats] = None
    normalized_data: Optional[List[NormalizedDataPoint]] = None  # Clean data for frontend
    available_columns: Optional[List[ColumnInfo]] = None  # Exogenous columns info
    message: Optional[str] = None

# Schema for feature configuration
class ExogenousFeatureConfig(BaseModel):
    """Configuration for a single exogenous feature."""
    column: str
    lags: List[int] = []           # Lag values to create (e.g., [1, 7])
    use_actual: bool = False       # Use actual value at prediction time
    delta_lag: Optional[int] = None  # Compute delta vs this lag
    pct_change_lag: Optional[int] = None  # Compute % change vs this lag

class DerivedFeatureConfig(BaseModel):
    """Configuration for derived features (operations between columns)."""
    operation: str  # "sum", "product", "ratio", "difference"
    feature_a: str  # Name of first feature (column name)
    feature_b: str  # Name of second feature (column name)
    alias: Optional[str] = None

class TemporalFeatureConfig(BaseModel):
    """Configuration for temporal features extracted from date."""
    month: bool = False        # Month (1-12), cyclical encoded
    day_of_week: bool = False  # Day of week (0-6), cyclical encoded
    day_of_month: bool = False # Day of month (1-31)
    week_of_year: bool = False # Week of year (1-52)
    year: bool = False         # Year as numeric
    hour_of_day: bool = False  # Hour of day (0-23), cyclical encoded
    minute_of_day: bool = False # Minute of day (0-1439), cyclical encoded

class FeatureConfig(BaseModel):
    """Complete feature configuration for model training."""
    target_lags: List[int] = [1, 7]  # Lags of target variable
    temporal: TemporalFeatureConfig = TemporalFeatureConfig()
    exogenous: List[ExogenousFeatureConfig] = []
    derived: List[DerivedFeatureConfig] = []

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Time Series Forecaster API", version="1.0.0")

# CORS : origines autorisées (séparées par virgule dans la variable d'environnement)
# Ex: ALLOWED_ORIGINS="https://time-series-forecaster.vercel.app,http://localhost:3000"
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_env == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_frequency(df: pl.DataFrame, date_col: str) -> tuple[str, str, int]:
    """
    Detect the frequency of a time series.
    Returns: (frequency_code, frequency_label, missing_dates_count)
    """
    if df.height < 2:
        return "unknown", "Unknown", 0
    
    # Get sorted dates
    dates = df.sort(date_col).select(date_col).to_series()
    
    # Calculate differences between consecutive dates
    diffs = dates.diff().drop_nulls()
    
    if len(diffs) == 0:
        return "unknown", "Unknown", 0
    
    # Get the most common difference (mode)
    # Convert to total seconds for comparison
    diff_seconds = [d.total_seconds() for d in diffs.to_list()]
    
    # Find the median difference (more robust than mode for irregular data)
    median_diff = np.median(diff_seconds)
    
    # Classify frequency based on median difference
    MINUTE = 60
    HOUR = 3600
    DAY = 86400
    WEEK = 7 * DAY
    MONTH = 30 * DAY  # Approximate
    
    if median_diff < MINUTE:
        freq_code, freq_label = "s", "Secondly"
        expected_diff = median_diff
    elif median_diff < HOUR:
        freq_code, freq_label = "min", "Minutely"
        expected_diff = round(median_diff / MINUTE) * MINUTE
    elif median_diff < DAY:
        freq_code, freq_label = "H", "Hourly"
        expected_diff = round(median_diff / HOUR) * HOUR
    elif median_diff < WEEK:
        freq_code, freq_label = "D", "Daily"
        expected_diff = DAY
    elif median_diff < MONTH:
        freq_code, freq_label = "W", "Weekly"
        expected_diff = WEEK
    else:
        freq_code, freq_label = "M", "Monthly"
        expected_diff = MONTH
    
    # Count missing dates (gaps larger than expected)
    tolerance = expected_diff * 1.5
    missing_count = sum(1 for d in diff_seconds if d > tolerance)
    
    return freq_code, freq_label, missing_count


def parse_dates_flexible(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    """
    Parse dates with intelligent format detection.
    Analyzes the data to determine if format is M/D/Y or D/M/Y.
    """
    original_len = len(df)
    
    # Get sample of date strings for analysis
    date_strings = df.select(pl.col(date_col).cast(pl.Utf8)).to_series().to_list()[:100]
    
    # Analyze to detect format
    def detect_date_format(samples: list) -> str:
        """
        Detect whether dates are M/D/Y or D/M/Y by looking for values > 12.
        If first part ever > 12, it must be day (D/M/Y).
        If second part ever > 12, it must be day (M/D/Y).
        """
        first_parts = []
        second_parts = []
        
        for s in samples:
            if not s:
                continue
            s = str(s).strip()
            
            # Try to split by common separators
            for sep in ['/', '-', '.']:
                if sep in s:
                    parts = s.split(sep)
                    if len(parts) >= 2:
                        try:
                            first_parts.append(int(parts[0]))
                            second_parts.append(int(parts[1]))
                        except ValueError:
                            pass
                    break
        
        if not first_parts or not second_parts:
            return None
        
        max_first = max(first_parts)
        max_second = max(second_parts)
        
        # If first part ever > 12, it's day first (D/M/Y)
        if max_first > 12:
            return "DMY"
        # If second part ever > 12, it's month first (M/D/Y)
        if max_second > 12:
            return "MDY"
        # Ambiguous - default to M/D/Y (US format) for common datasets
        return "MDY"
    
    detected_format = detect_date_format(date_strings)
    print(f"Detected date format: {detected_format}")
    
    # Define format lists based on detection
    if detected_format == "DMY":
        date_formats = [
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d.%m.%Y",
            "%Y-%m-%d",
        ]
    else:  # MDY or unknown
        date_formats = [
            "%m/%d/%Y",
            "%m-%d-%Y",
            "%Y-%m-%d",
            "%d/%m/%Y",  # Fallback
        ]
    
    # Try each format and pick the one with most successful parses
    best_df = None
    best_success_count = 0
    best_format = None
    
    for fmt in date_formats:
        try:
            test_df = df.with_columns(
                pl.col(date_col).cast(pl.Utf8).str.strip_chars().str.to_date(fmt, strict=False).cast(pl.Datetime)
            )
            success_count = original_len - test_df[date_col].null_count()
            
            print(f"Format {fmt}: {success_count}/{original_len} rows parsed")
            
            if success_count > best_success_count:
                best_success_count = success_count
                best_df = test_df
                best_format = fmt
        except Exception as e:
            print(f"Format {fmt} failed: {e}")
            continue
    
    # Also try automatic parsing
    try:
        test_df = df.with_columns(
            pl.col(date_col).str.to_datetime(strict=False)
        )
        success_count = original_len - test_df[date_col].null_count()
        print(f"Auto parsing: {success_count}/{original_len} rows parsed")
        
        if success_count > best_success_count:
            best_success_count = success_count
            best_df = test_df
            best_format = "auto"
    except Exception:
        pass
    
    if best_df is None or best_success_count == 0:
        raise ValueError(f"Could not parse date column '{date_col}'")
    
    print(f"Best format: {best_format} with {best_success_count}/{original_len} rows")
    return best_df


def build_features(
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    feature_config: FeatureConfig
) -> tuple[pl.DataFrame, List[str]]:
    """
    Build all features for model training.
    
    Returns:
        - DataFrame with all features added
        - List of feature column names (for model training)
    """
    feature_names = []
    
    # 1. Target lags
    for lag in feature_config.target_lags:
        col_name = f"target_lag_{lag}"
        df = df.with_columns(
            pl.col(target_col).shift(lag).alias(col_name)
        )
        feature_names.append(col_name)
    
    # 2. Temporal features (from date column)
    temporal = feature_config.temporal
    
    if temporal.month:
        # Cyclical encoding: sin and cos for month (1-12)
        df = df.with_columns([
            (2 * np.pi * pl.col(date_col).dt.month() / 12).sin().alias("month_sin"),
            (2 * np.pi * pl.col(date_col).dt.month() / 12).cos().alias("month_cos"),
        ])
        feature_names.extend(["month_sin", "month_cos"])
    
    if temporal.day_of_week:
        # Cyclical encoding: sin and cos for day of week (0-6)
        df = df.with_columns([
            (2 * np.pi * pl.col(date_col).dt.weekday() / 7).sin().alias("dow_sin"),
            (2 * np.pi * pl.col(date_col).dt.weekday() / 7).cos().alias("dow_cos"),
        ])
        feature_names.extend(["dow_sin", "dow_cos"])
    
    if temporal.day_of_month:
        df = df.with_columns(
            pl.col(date_col).dt.day().alias("day_of_month")
        )
        feature_names.append("day_of_month")
    
    if temporal.week_of_year:
        df = df.with_columns(
            pl.col(date_col).dt.week().alias("week_of_year")
        )
        feature_names.append("week_of_year")
    
    if temporal.year:
        df = df.with_columns(
            pl.col(date_col).dt.year().alias("year")
        )
        feature_names.append("year")
    
    if temporal.hour_of_day:
        # Cyclical encoding: sin and cos for hour of day (0-23)
        df = df.with_columns([
            (2 * np.pi * pl.col(date_col).dt.hour() / 24).sin().alias("hour_sin"),
            (2 * np.pi * pl.col(date_col).dt.hour() / 24).cos().alias("hour_cos"),
        ])
        feature_names.extend(["hour_sin", "hour_cos"])
    
    if temporal.minute_of_day:
        # Cyclical encoding: sin and cos for minute of day (0-1439)
        # minute_of_day = hour * 60 + minute
        df = df.with_columns([
            (2 * np.pi * (pl.col(date_col).dt.hour() * 60 + pl.col(date_col).dt.minute()) / 1440).sin().alias("minute_of_day_sin"),
            (2 * np.pi * (pl.col(date_col).dt.hour() * 60 + pl.col(date_col).dt.minute()) / 1440).cos().alias("minute_of_day_cos"),
        ])
        feature_names.extend(["minute_of_day_sin", "minute_of_day_cos"])
    
    # 3. Exogenous features
    for exog in feature_config.exogenous:
        col = exog.column
        
        # Skip if column doesn't exist
        if col not in df.columns:
            print(f"Warning: exogenous column '{col}' not found, skipping")
            continue
        
        # Lags of exogenous variable
        for lag in exog.lags:
            col_name = f"{col}_lag_{lag}"
            df = df.with_columns(
                pl.col(col).shift(lag).alias(col_name)
            )
            feature_names.append(col_name)
        
        # Actual value (for features known at prediction time, e.g., planned promotions)
        if exog.use_actual:
            col_name = f"{col}_actual"
            df = df.with_columns(
                pl.col(col).alias(col_name)
            )
            feature_names.append(col_name)
        
        # Delta (difference vs lag)
        if exog.delta_lag is not None:
            col_name = f"{col}_delta_{exog.delta_lag}"
            df = df.with_columns(
                (pl.col(col) - pl.col(col).shift(exog.delta_lag)).alias(col_name)
            )
            feature_names.append(col_name)
        
        # Percentage change vs lag
        if exog.pct_change_lag is not None:
            col_name = f"{col}_pct_{exog.pct_change_lag}"
            df = df.with_columns(
                ((pl.col(col) - pl.col(col).shift(exog.pct_change_lag)) / 
                 pl.col(col).shift(exog.pct_change_lag).abs().clip(lower_bound=1e-10)).alias(col_name)
            )
            feature_names.append(col_name)
    
    # 4. Derived features (operations between existing features)
    for derived in feature_config.derived:
        col_a = derived.feature_a
        col_b = derived.feature_b
        
        # Check if columns exist
        if col_a not in df.columns or col_b not in df.columns:
            print(f"Warning: derived feature columns '{col_a}' or '{col_b}' not found, skipping")
            continue
            
        alias = derived.alias or f"{col_a}_{derived.operation}_{col_b}"
        
        if derived.operation == "sum":
            df = df.with_columns((pl.col(col_a) + pl.col(col_b)).alias(alias))
        elif derived.operation == "difference":
            df = df.with_columns((pl.col(col_a) - pl.col(col_b)).alias(alias))
        elif derived.operation == "product":
            df = df.with_columns((pl.col(col_a) * pl.col(col_b)).alias(alias))
        elif derived.operation == "ratio":
            df = df.with_columns(
                (pl.col(col_a) / pl.col(col_b).abs().clip(lower_bound=1e-10)).alias(alias)
            )
        
        feature_names.append(alias)
    
    return df, feature_names


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

def train_lag(
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    training_ranges: List[DateRange],
    prediction_ranges: List[DateRange],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Baseline model: predict using a simple lag.
    
    Params:
    - lag: int - which lag to use for prediction (default: 1)
    
    Returns metrics and predictions.
    """
    start_time = time.time()
    
    # Get lag parameter
    lag = params.get("lag", 1)
    
    # Create lagged column
    df_lagged = df.with_columns(
        pl.col(target_col).shift(lag).alias(f"lag_{lag}")
    )
    
    # Build training data
    train_dfs = []
    for tr in training_ranges:
        range_df = filter_by_date_range(df_lagged, date_col, tr.start, tr.end)
        train_dfs.append(range_df)
    
    train_df = pl.concat(train_dfs) if train_dfs else df_lagged.head(0)
    train_df = train_df.drop_nulls(subset=[target_col, f"lag_{lag}"])
    
    if train_df.height == 0:
        raise ValueError(f"No training data available after applying lag {lag}")
    
    # Calculate metrics on training data
    y_true = train_df[target_col].to_numpy()
    y_pred = train_df[f"lag_{lag}"].to_numpy()
    
    metrics = calculate_metrics(y_true, y_pred)
    
    # Generate predictions for all prediction ranges
    all_forecasts = []
    for pr in prediction_ranges:
        pred_df = filter_by_date_range(df_lagged, date_col, pr.start, pr.end).drop_nulls(subset=[target_col, f"lag_{lag}"])
        
        for row in pred_df.iter_rows(named=True):
            date_value = row[date_col]
            all_forecasts.append({
                date_col: date_value.isoformat() if hasattr(date_value, 'isoformat') else str(date_value),
                "prediction": float(row[f"lag_{lag}"]),
                target_col: float(row[target_col])
            })
    
    execution_time = time.time() - start_time
    
    return {
        "metrics": {
            **metrics,
            "execution_time": execution_time
        },
        "forecast": all_forecasts,
        "feature_importance": None
    }


def train_linear_regression(
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    training_ranges: List[DateRange],
    prediction_ranges: List[DateRange],
    params: Dict[str, Any]
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
    # Support both legacy "lags" param and new "feature_config" structure
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
        # Ensure we have the residual lag feature
        if residual_col not in df_features.columns:
            df_features = df_features.with_columns(
                pl.col(target_col).shift(residual_lag).alias(residual_col)
            )
            if residual_col not in feature_names:
                feature_names.append(residual_col)
        
        # Create residual target: y_residual = y - y_lag
        df_features = df_features.with_columns(
            (pl.col(target_col) - pl.col(residual_col)).alias("target_residual")
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
    
    # Drop rows with NaN (from lag/feature creation)
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
        
        # Check required columns for prediction
        cols_to_check_pred = feature_names.copy()
        if target_mode == "residual":
            residual_col = f"target_lag_{residual_lag}"
            if residual_col not in cols_to_check_pred:
                cols_to_check_pred.append(residual_col)
        pred_df = pred_df.drop_nulls(subset=cols_to_check_pred)
        
        if pred_df.height == 0:
            continue
        
        X_pred = pred_df.select(feature_names).to_numpy()
        y_actual_original = pred_df.select(target_col).to_numpy().flatten()
        dates = pred_df.select(date_col).to_series().to_list()
        
        # Standardize prediction features if needed
        if standardize and feature_means is not None:
            X_pred = (X_pred - feature_means) / feature_stds
        
        y_pred_raw = model.predict(X_pred)
        
        # If residual mode, reconstruct original scale: y_pred = y_pred_residual + y_lag
        if target_mode == "residual":
            y_lag_values = pred_df.select(f"target_lag_{residual_lag}").to_numpy().flatten()
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
    for feat_name, coef in zip(feature_names, model.coef_):
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


def train_xgboost(
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    training_ranges: List[DateRange],
    prediction_ranges: List[DateRange],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train an XGBoost model with configurable features.
    Uses the same feature engineering as Linear Regression.
    Supports target_mode (raw/residual) like Linear Regression.
    """
    start_time = time.time()
    
    # Get XGBoost specific params
    n_estimators = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", 6)
    learning_rate = params.get("learning_rate", 0.1)
    
    # Get target mode params (same as linear regression)
    target_mode = params.get("target_mode", "raw")  # "raw" or "residual"
    residual_lag = params.get("residual_lag", 1)
    
    # Build feature config (same as linear regression)
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
    
    # For residual mode, create residual target and ensure lag feature exists
    effective_target = target_col
    if target_mode == "residual":
        residual_col = f"target_lag_{residual_lag}"
        if residual_col not in df_features.columns:
            df_features = df_features.with_columns(
                pl.col(target_col).shift(residual_lag).alias(residual_col)
            )
            if residual_col not in feature_names:
                feature_names.append(residual_col)
        
        df_features = df_features.with_columns(
            (pl.col(target_col) - pl.col(residual_col)).alias("_target_residual")
        )
        effective_target = "_target_residual"
    
    # Build training data
    train_dfs = []
    for tr in training_ranges:
        chunk = filter_by_date_range(df_features, date_col, tr.start, tr.end)
        train_dfs.append(chunk)
    
    if not train_dfs:
        raise ValueError("No training data found")
    
    # Drop nulls including residual target if needed
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
    
    # Predict
    all_predictions = []
    all_actuals = []
    forecast_output = []
    
    for pr in prediction_ranges:
        pred_df = filter_by_date_range(df_features, date_col, pr.start, pr.end)
        
        # Check required columns for prediction
        cols_to_check_pred = feature_names.copy()
        if target_mode == "residual":
            residual_col = f"target_lag_{residual_lag}"
            if residual_col not in cols_to_check_pred:
                cols_to_check_pred.append(residual_col)
        pred_df = pred_df.drop_nulls(subset=cols_to_check_pred)
        
        if pred_df.height == 0:
            continue
        
        X_pred = pred_df.select(feature_names).to_numpy()
        y_actual_original = pred_df.select(target_col).to_numpy().flatten()
        dates = pred_df.select(date_col).to_series().to_list()
        
        y_pred_raw = model.predict(X_pred)
        
        # If residual mode, reconstruct original scale
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
                target_col: float(actual_val)
            })
    
    # Metrics (on original scale)
    metrics = calculate_metrics(np.array(all_actuals), np.array(all_predictions)) if all_actuals else {"rmse": 0, "mae": 0, "mape": 0, "r2": 0}
    metrics["execution_time"] = time.time() - start_time
    
    # Feature importance from XGBoost
    feature_importance = [
        {"feature": name, "importance": float(imp)}
        for name, imp in zip(feature_names, model.feature_importances_)
    ]
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    
    return {
        "metrics": metrics,
        "forecast": forecast_output,
        "feature_importance": feature_importance
    }


def train_arima(
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    training_ranges: List[DateRange],
    prediction_ranges: List[DateRange],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train an ARIMA model.
    ARIMA is a univariate model - it only uses the target variable's history.
    """
    start_time = time.time()
    
    # Get ARIMA params (p, d, q)
    p = params.get("p", 1)  # AR order
    d = params.get("d", 1)  # Differencing order
    q = params.get("q", 1)  # MA order
    
    # Build training data
    train_dfs = []
    for tr in training_ranges:
        chunk = filter_by_date_range(df, date_col, tr.start, tr.end)
        train_dfs.append(chunk)
    
    if not train_dfs:
        raise ValueError("No training data found")
    
    train_df = pl.concat(train_dfs).sort(date_col)
    y_train = train_df.select(target_col).to_numpy().flatten()
    
    if len(y_train) < p + d + q + 5:
        raise ValueError(f"Not enough training data for ARIMA({p},{d},{q})")
    
    # Fit ARIMA
    try:
        model = ARIMA(y_train, order=(p, d, q))
        fitted = model.fit()
    except Exception as e:
        raise ValueError(f"ARIMA fitting failed: {e}")
    
    # Predict on prediction ranges
    all_predictions = []
    all_actuals = []
    forecast_output = []
    
    for pr in prediction_ranges:
        pred_df = filter_by_date_range(df, date_col, pr.start, pr.end).sort(date_col)
        
        if pred_df.height == 0:
            continue
        
        y_actual = pred_df.select(target_col).to_numpy().flatten()
        dates = pred_df.select(date_col).to_series().to_list()
        
        # For ARIMA, we need to forecast from the end of training
        # Use in-sample predictions for overlapping periods, out-of-sample for future
        n_forecast = len(y_actual)
        
        try:
            # Get forecasts
            forecast = fitted.forecast(steps=n_forecast)
            y_pred = np.array(forecast)
        except Exception:
            # Fallback: use last known value
            y_pred = np.full(n_forecast, y_train[-1])
        
        all_predictions.extend(y_pred)
        all_actuals.extend(y_actual)
        
        for date, pred_val, actual_val in zip(dates, y_pred, y_actual):
            forecast_output.append({
                date_col: date.isoformat() if hasattr(date, 'isoformat') else str(date),
                "prediction": float(pred_val),
                target_col: float(actual_val)
            })
    
    # Metrics
    metrics = calculate_metrics(np.array(all_actuals), np.array(all_predictions)) if all_actuals else {"rmse": 0, "mae": 0, "mape": 0, "r2": 0}
    metrics["execution_time"] = time.time() - start_time
    
    return {
        "metrics": metrics,
        "forecast": forecast_output,
        "feature_importance": None  # ARIMA doesn't have feature importance
    }


def train_prophet(
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    training_ranges: List[DateRange],
    prediction_ranges: List[DateRange],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train a Prophet model with optional lag regressors.
    Prophet decomposes series into trend + seasonality.
    Adding lag regressors helps capture short-term fluctuations.
    """
    start_time = time.time()
    
    # Get Prophet params
    daily_seasonality = params.get("daily_seasonality", False)
    weekly_seasonality = params.get("weekly_seasonality", True)
    yearly_seasonality = params.get("yearly_seasonality", True)
    seasonality_mode = params.get("seasonality_mode", "additive")
    
    # Get lag regressors (new feature)
    use_lag_regressors = params.get("use_lag_regressors", True)
    lag_regressors = params.get("lag_regressors", [1, 7])
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
        chunk = filter_by_date_range(df_with_lags, date_col, tr.start, tr.end)
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
        
        # Create future dataframe for Prophet (must include regressors)
        cols_for_future = [pl.col(date_col).alias("ds")]
        for reg_name in regressor_names:
            cols_for_future.append(pl.col(reg_name))
        
        future = pred_df.select(cols_for_future).to_pandas()
        
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
    metrics = calculate_metrics(np.array(all_actuals), np.array(all_predictions)) if all_actuals else {"rmse": 0, "mae": 0, "mape": 0, "r2": 0}
    metrics["execution_time"] = time.time() - start_time
    
    # Feature importance approximation from regressor coefficients
    feature_importance = None
    if regressor_names and hasattr(model, 'params'):
        try:
            # Prophet stores regressor coefficients in params
            feature_importance = []
            for reg_name in regressor_names:
                # Get coefficient from model (simplified)
                feature_importance.append({
                    "feature": reg_name,
                    "importance": 1.0 / len(regressor_names)  # Equal importance as placeholder
                })
        except Exception:
            pass
    
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

@app.post("/analyze", response_model=DatasetAnalysisResponse)
async def analyze_dataset(request: DatasetAnalysisRequest, _: None = Depends(verify_api_key)):
    """
    Analyze a dataset and return statistics.
    Auto-detects frequency, missing values, date range, etc.
    """
    try:
        # Convert to Polars DataFrame
        df = pl.DataFrame(request.data, infer_schema_length=None)
        date_col = request.date_column
        target_col = request.target_column
        
        # Parse dates
        df = parse_dates_flexible(df, date_col)
        
        # Filter out null dates
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
        
        # Sort by date
        df = df.sort(date_col)
        
        if df.height == 0:
            return DatasetAnalysisResponse(
                status="error",
                message="No valid data after parsing"
            )
        
        # Get date range
        date_min = df[date_col].min()
        date_max = df[date_col].max()
        
        # Detect frequency
        freq_code, freq_label, missing_dates = detect_frequency(df, date_col)
        
        # Calculate target stats
        target_series = df[target_col]
        missing_target = target_series.null_count()
        
        # Filter non-null for stats
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
        
        # Build normalized data for frontend (dates in ISO format, clean values)
        # Filter rows with valid target values
        clean_df = df.filter(pl.col(target_col).is_not_null())
        normalized_data = [
            NormalizedDataPoint(
                date=row[date_col].isoformat() if hasattr(row[date_col], 'isoformat') else str(row[date_col]),
                value=float(row[target_col])
            )
            for row in clean_df.iter_rows(named=True)
        ]
        
        # Build available columns info (excluding date and target)
        available_columns = []
        for col_name in df.columns:
            if col_name in [date_col, target_col]:
                continue
            
            col_series = df[col_name]
            dtype = col_series.dtype
            
            # Determine column type
            if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                dtype_str = "numeric"
            elif dtype == pl.Boolean:
                dtype_str = "boolean"
            elif dtype in [pl.Datetime, pl.Date]:
                dtype_str = "date"
            else:
                dtype_str = "string"
            
            # Get sample values (first 5 non-null)
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
        return DatasetAnalysisResponse(
            status="error",
            message=str(e)
        )


@app.post("/train", response_model=TrainingResponse)
async def train_models(request: TrainingRequest, _: None = Depends(verify_api_key)):
    """
    Main training endpoint.
    Receives data + config, trains requested models, returns predictions + metrics.
    """
    try:
        # 1. Convert data to Polars DataFrame with increased schema inference
        # Use infer_schema_length=None to scan all rows for type inference
        df = pl.DataFrame(request.data, infer_schema_length=None)
        
        date_col = request.data_config.date_column
        target_col = request.data_config.target_column
        
        # Parse dates using the intelligent format detection (same as /analyze)
        df = parse_dates_flexible(df, date_col)
        
        # Remove rows with null dates after parsing
        df = df.filter(pl.col(date_col).is_not_null())
        
        # Handle target column - convert to float, handling invalid values
        try:
            # First cast to string to handle any type, then clean, then to float
            df = df.with_columns(
                pl.col(target_col)
                  .cast(pl.Utf8)
                  .str.strip_chars()
                  .str.replace(r"^\?", "")  # Remove leading "?"
                  .str.replace(r"[^\d\.-]", "")  # Keep only digits, dots, and minus
                  .cast(pl.Float64)
            )
        except Exception as e:
            print(f"Warning: Could not convert {target_col} to float: {e}")
            # Try simpler conversion
            df = df.with_columns(
                pl.col(target_col).cast(pl.Float64, strict=False)
            )
        
        # Remove rows with null target values
        df = df.filter(pl.col(target_col).is_not_null())
        
        # Convert potential exogenous columns to numeric
        # (columns that are not date or target and might be used as features)
        for col_name in df.columns:
            if col_name in [date_col, target_col]:
                continue
            # Try to convert to float if not already numeric
            col_dtype = df[col_name].dtype
            if col_dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                try:
                    df = df.with_columns(
                        pl.col(col_name).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)
                    )
                except Exception:
                    pass  # Keep as is if conversion fails
        
        # Sort by date
        df = df.sort(date_col)
        
        if df.height == 0:
            raise ValueError("No valid data rows after cleaning")
        
        results = []
        
        # 2. Train each requested model
        for model_config in request.models:
            try:
                if model_config.type == "LAG":
                    result = train_lag(
                        df=df,
                        date_col=date_col,
                        target_col=target_col,
                        training_ranges=request.data_config.training_ranges,
                        prediction_ranges=request.data_config.prediction_ranges,
                        params=model_config.params
                    )
                elif model_config.type == "LINEAR_REGRESSION":
                    result = train_linear_regression(
                        df=df,
                        date_col=date_col,
                        target_col=target_col,
                        training_ranges=request.data_config.training_ranges,
                        prediction_ranges=request.data_config.prediction_ranges,
                        params=model_config.params
                    )
                elif model_config.type == "XGBOOST":
                    result = train_xgboost(
                        df=df,
                        date_col=date_col,
                        target_col=target_col,
                        training_ranges=request.data_config.training_ranges,
                        prediction_ranges=request.data_config.prediction_ranges,
                        params=model_config.params
                    )
                elif model_config.type == "ARIMA":
                    result = train_arima(
                        df=df,
                        date_col=date_col,
                        target_col=target_col,
                        training_ranges=request.data_config.training_ranges,
                        prediction_ranges=request.data_config.prediction_ranges,
                        params=model_config.params
                    )
                elif model_config.type == "PROPHET":
                    if not PROPHET_AVAILABLE:
                        raise ValueError("Prophet is not installed on this server. Please use Linear Regression, XGBoost, or ARIMA instead.")
                    result = train_prophet(
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
                error_msg = str(e)
                print(f"Error training {model_config.name}: {error_msg}")
                # Return result with error message
                results.append(ModelResult(
                    model_id=model_config.id,
                    model_name=model_config.name,
                    metrics=ModelMetrics(rmse=0, mae=0, mape=0, r2=0, execution_time=0),
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
