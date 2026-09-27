"""
Evaluation engine, common to all models.

Each prediction range is split into blocks of `horizon` consecutive rows. A block is
forecast from its origin (the row before it) knowing only the past: step 1 is one
step ahead, step h is h steps ahead. Between blocks, the model gets the actuals back.
"""

from typing import Any, Dict, List

import numpy as np
import polars as pl

from app.forecasting.data import row_indices_in_range
from app.forecasting.metrics import calculate_metrics, calculate_metrics_by_horizon
from app.forecasting.models.base import Block, Forecaster


def make_blocks(df: pl.DataFrame, date_col: str, prediction_ranges: List[Any], horizon: int) -> List[Block]:
    """Split each prediction range (inclusive dates) into blocks of `horizon` rows."""
    blocks = []
    for pr in prediction_ranges:
        rows = row_indices_in_range(df, date_col, pr.start, pr.end, inclusive_end=True)
        if len(rows) == 0:
            continue
        first, last = int(rows[0]), int(rows[-1]) + 1
        blocks.extend(Block(start, min(start + horizon, last)) for start in range(first, last, horizon))
    return blocks


def training_rows(df: pl.DataFrame, date_col: str, training_ranges: List[Any]) -> np.ndarray:
    """Rows of the training ranges (end date excluded), sorted and without duplicates."""
    rows = [row_indices_in_range(df, date_col, tr.start, tr.end, inclusive_end=False) for tr in training_ranges]
    return np.unique(np.concatenate(rows)) if rows else np.array([], dtype=int)


def run_backtest(model: Forecaster, df: pl.DataFrame, blocks: List[Block], horizon: int) -> Dict[str, Any]:
    """Forecast every block and compute metrics (overall and by horizon step)."""
    date_col, target_col = model.ctx.date_col, model.ctx.target_col
    predictions = model.predict_blocks(df, blocks)

    dates = df[date_col].to_list()
    actuals = df[target_col].to_numpy()
    forecast = []
    for block, block_predictions in zip(blocks, predictions):
        for step, prediction in enumerate(block_predictions, start=1):
            if np.isnan(prediction):
                continue  # Features not available (e.g. start of the data)
            row = block.start + step - 1
            forecast.append({
                date_col: dates[row].isoformat() if hasattr(dates[row], "isoformat") else str(dates[row]),
                "prediction": float(prediction),
                target_col: float(actuals[row]),
                "horizon_step": step,
            })

    if not forecast:
        raise ValueError("No prediction could be made on the prediction ranges")

    return {
        "forecast": forecast,
        "metrics": calculate_metrics(
            np.array([f[target_col] for f in forecast]),
            np.array([f["prediction"] for f in forecast])
        ),
        "metrics_by_horizon": calculate_metrics_by_horizon(forecast, target_col) if horizon > 1 else None,
    }
