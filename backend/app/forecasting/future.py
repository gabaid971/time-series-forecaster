"""
Forecasting beyond the data: retrain a validated configuration on the whole history,
then forecast the next steps.

The future is one block of rows appended after the last observation, with an unknown
target: the models forecast it with the same code (and the same no-leakage rules) as
in the evaluation.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from app.forecasting.data import detect_frequency, prepare_for_training
from app.forecasting.features import FeatureConfig
from app.forecasting.models import Block, ForecastContext, create_forecaster

logger = logging.getLogger(__name__)


def future_dates(df: pl.DataFrame, date_col: str, frequency: str, steps: int) -> List[datetime]:
    """
    The next `steps` dates after the last one: calendar months for monthly data,
    otherwise the typical (median) interval between observations.
    """
    dates = df[date_col].sort()
    last = dates[-1]
    if frequency == "M":
        return [pl.select(pl.lit(last).dt.offset_by(f"{k}mo")).item() for k in range(1, steps + 1)]
    # Positive gaps only: duplicated dates must not give a zero interval
    seconds = [d.total_seconds() for d in dates.diff().drop_nulls().to_list() if d.total_seconds() > 0]
    if not seconds:
        raise ValueError("At least two distinct dates are needed to forecast the future")
    interval = timedelta(seconds=round(float(np.median(seconds))))
    return [last + k * interval for k in range(1, steps + 1)]


# A forecast this far outside the observed range means the model is unstable when it
# feeds on its own predictions (e.g. an autoregressive coefficient above 1)
DIVERGENCE_FACTOR = 10


def divergence_warning(predictions: np.ndarray, history: np.ndarray) -> Optional[str]:
    low, high = float(np.min(history)), float(np.max(history))
    margin = DIVERGENCE_FACTOR * max(high - low, abs(high), 1e-9)
    finite = predictions[np.isfinite(predictions)]
    if finite.size and (finite.min() < low - margin or finite.max() > high + margin):
        return (
            "The forecast diverges: values end up far outside the observed range. "
            "The model is unstable when it builds on its own forecasts this far ahead; "
            "forecast fewer steps or choose another recipe."
        )
    if finite.size < predictions.size:
        return f"{predictions.size - finite.size} step(s) could not be forecast."
    return None


def _known_in_advance_columns(model) -> List[str]:
    features = getattr(model, "features", None)
    if not isinstance(features, FeatureConfig):
        return []
    return [e.column for e in features.exogenous if e.known_in_advance]


def forecast_models(
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    models: List[Any],
    steps: int,
    deadline: float,
) -> Dict[str, Any]:
    """
    Retrain each model on all the history and forecast the next `steps` points.
    A failing model does not stop the others: its result carries the error.
    """
    history = prepare_for_training(df, date_col, target_col)
    frequency, _, _ = detect_frequency(history, date_col)
    dates = future_dates(history, date_col, frequency, steps)

    # Future rows: known dates, unknown target and exogenous values
    future = pl.DataFrame({date_col: dates}).with_columns(pl.col(date_col).cast(history[date_col].dtype))
    full = pl.concat([history, future], how="diagonal_relaxed")
    train_rows = np.arange(history.height)
    block = Block(history.height, full.height)
    ctx = ForecastContext(date_col=date_col, target_col=target_col, horizon=steps)

    results = []
    for model_config in models:
        result = {"model_id": model_config.id, "model_name": model_config.name}
        if time.monotonic() > deadline:
            results.append({**result, "error": "Skipped: time budget exceeded"})
            continue
        start = time.perf_counter()
        try:
            model = create_forecaster(model_config.type, model_config.params, ctx)
            known = _known_in_advance_columns(model)
            if known:
                raise ValueError(
                    f"Variables known in advance ({', '.join(known)}) need their future values, "
                    "which cannot be provided yet."
                )
            model.fit(full, train_rows)
            predictions = model.predict_blocks(full, [block])[0]
        except Exception as e:
            logger.warning("Forecast with %s (%s) failed: %s", model_config.name, model_config.type, e)
            results.append({**result, "error": str(e)})
            continue
        results.append({
            **result,
            "warning": divergence_warning(predictions, history[target_col].to_numpy()),
            "forecast": [
                {"date": d.isoformat(), "prediction": None if np.isnan(p) else float(p), "step": i + 1}
                for i, (d, p) in enumerate(zip(dates, predictions))
            ],
            "execution_time": time.perf_counter() - start,
        })

    return {
        "frequency": frequency,
        "last_date": history[date_col][-1].isoformat(),
        "results": results,
    }
