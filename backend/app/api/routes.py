"""
API endpoints.

They are plain `def` (not `async def`): FastAPI runs them in a thread pool, so
CPU-heavy training does not block the server event loop.
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from app.api.limits import RateLimiter, TrainingSlots, client_ip
from app.api.schemas import (
    DatasetAnalysisRequest,
    DatasetAnalysisResponse,
    TrainingRequest,
    TrainingResponse,
)
from app.config import settings
from app.forecasting.analysis import analyze_dataset
from app.forecasting.data import load_frame
from app.forecasting.models import ForecastContext
from app.forecasting.training import train_models

logger = logging.getLogger(__name__)
router = APIRouter()

analyze_limiter = RateLimiter(settings.analyze_rate_limit)
train_limiter = RateLimiter(settings.train_rate_limit)
training_slots = TrainingSlots(settings.max_concurrent_trainings, settings.train_queue_timeout_s)


def check_dataset_size(n_rows: int) -> None:
    if n_rows > settings.max_rows:
        raise HTTPException(status_code=413, detail=f"Dataset too large: {n_rows} rows (max {settings.max_rows})")


@router.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Time Series Forecaster API is running"}


@router.get("/health")
def health():
    """Health check for monitoring."""
    return {"status": "healthy"}


@router.post("/analyze", response_model=DatasetAnalysisResponse)
def analyze(request: DatasetAnalysisRequest, http_request: Request):
    """
    Analyze a dataset: statistics, frequency, missing values, ACF/PACF lag
    suggestions and data quality alerts.
    """
    analyze_limiter.check(client_ip(http_request))
    check_dataset_size(len(request.data))
    try:
        df = load_frame(request.data, request.date_column, request.target_column)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return DatasetAnalysisResponse(status="success", **analyze_dataset(df, request.date_column, request.target_column))


@router.post("/train", response_model=TrainingResponse)
def train(request: TrainingRequest, http_request: Request):
    """
    Train the requested models and evaluate them on the prediction ranges.
    A failing model does not fail the request: its result carries the error.
    """
    train_limiter.check(client_ip(http_request))
    check_dataset_size(len(request.data))
    if len(request.models) > settings.max_models:
        raise HTTPException(status_code=413, detail=f"Too many models: {len(request.models)} (max {settings.max_models})")

    config = request.data_config
    horizon = config.forecast_strategy.horizon if config.forecast_strategy else 1
    ctx = ForecastContext(date_col=config.date_column, target_col=config.target_column, horizon=horizon)

    with training_slots.acquire():
        deadline = time.monotonic() + settings.train_time_budget_s
        try:
            df = load_frame(request.data, ctx.date_col, ctx.target_col)
            results = train_models(df, ctx, config.training_ranges, config.prediction_ranges, request.models, deadline)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    return TrainingResponse(status="success", results=results, message=f"Trained {len(results)} model(s)")
