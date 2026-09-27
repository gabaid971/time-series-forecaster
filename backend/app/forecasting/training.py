"""Training service: fit and evaluate several models on the same data and ranges."""

import logging
import time
from typing import Any, Dict, List

import polars as pl

from app.forecasting.backtest import make_blocks, run_backtest, training_rows
from app.forecasting.data import prepare_for_training
from app.forecasting.models import ForecastContext, create_forecaster

logger = logging.getLogger(__name__)


def train_models(
    df: pl.DataFrame,
    ctx: ForecastContext,
    training_ranges: List[Any],
    prediction_ranges: List[Any],
    models: List[Any],
) -> List[Dict[str, Any]]:
    """
    Train and evaluate each model. A failing model does not stop the others: its
    result carries the error.

    Raises ValueError when the request itself is unusable (no training or prediction rows).
    """
    df = prepare_for_training(df, ctx.date_col, ctx.target_col)
    train_rows = training_rows(df, ctx.date_col, training_ranges)
    if len(train_rows) == 0:
        raise ValueError("No data in the training ranges")
    blocks = make_blocks(df, ctx.date_col, prediction_ranges, ctx.horizon)
    if not blocks:
        raise ValueError("No data in the prediction ranges")

    results = []
    for model_config in models:
        result = {"model_id": model_config.id, "model_name": model_config.name}
        start = time.perf_counter()
        try:
            model = create_forecaster(model_config.type, model_config.params, ctx)
            model.fit(df, train_rows)
            evaluation = run_backtest(model, df, blocks, ctx.horizon)
            explanation = model.explain()
        except Exception as e:
            logger.warning("Model %s (%s) failed: %s", model_config.name, model_config.type, e)
            results.append({**result, "error": str(e)})
            continue
        evaluation["metrics"]["execution_time"] = time.perf_counter() - start
        results.append({**result, **evaluation, **explanation})
    return results
