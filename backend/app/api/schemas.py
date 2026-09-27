"""Request and response schemas of the API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DateRange(BaseModel):
    start: str
    end: str


class ForecastStrategyConfig(BaseModel):
    """Configuration for multi-step forecasting."""
    horizon: int = Field(default=1, ge=1, le=1000)


class DataConfig(BaseModel):
    target_column: str
    date_column: str
    frequency: Optional[str] = None  # Informative: the backend detects it
    training_ranges: List[DateRange] = Field(min_length=1)
    prediction_ranges: List[DateRange] = Field(min_length=1)
    forecast_strategy: Optional[ForecastStrategyConfig] = None


class ModelConfig(BaseModel):
    id: str
    type: str
    name: str
    params: Dict[str, Any] = {}


class TrainingRequest(BaseModel):
    data: List[Dict[str, Any]]
    data_config: DataConfig
    models: List[ModelConfig] = Field(min_length=1)


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
    metrics: Optional[ModelMetrics] = None  # None when the model failed (see error)
    forecast: List[Dict[str, Any]] = []
    metrics_by_horizon: Optional[List[HorizonMetrics]] = None
    feature_importance: Optional[List[FeatureImportance]] = None
    shap_analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TrainingResponse(BaseModel):
    status: str
    results: List[ModelResult]
    message: Optional[str] = None


class ForecastRequest(BaseModel):
    """Retrain models on the whole history and forecast the next `steps` points."""
    data: List[Dict[str, Any]]
    date_column: str
    target_column: str
    steps: int = Field(ge=1, le=1000)
    models: List[ModelConfig] = Field(min_length=1)


class FuturePoint(BaseModel):
    date: str
    prediction: Optional[float] = None  # None when the model could not forecast this step
    step: int


class ForecastModelResult(BaseModel):
    model_id: str
    model_name: str
    forecast: List[FuturePoint] = []
    execution_time: Optional[float] = None
    warning: Optional[str] = None  # E.g. the forecast diverges
    error: Optional[str] = None


class ForecastResponse(BaseModel):
    status: str
    frequency: str
    last_date: str
    results: List[ForecastModelResult]


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


class Seasonality(BaseModel):
    period: int
    period_label: str
    strength: float
    suggested_feature: Optional[str] = None  # Calendar feature capturing the cycle


class LagSuggestion(BaseModel):
    suggested_lags: List[int]
    suggested_temporal: List[str] = []  # Calendar features recommended by the detected cycles
    acf: List[float]
    pacf: List[float]
    confidence_interval: float
    significant_lags: List[Dict[str, Any]]
    seasonality: Dict[str, Any]  # Strongest detected cycle
    seasonalities: List[Seasonality] = []
    n_observations: int


class DataAlert(BaseModel):
    type: str  # 'warning', 'info', 'error'
    category: str  # 'outliers', 'missing', 'trend', 'stationarity', 'seasonality'
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
