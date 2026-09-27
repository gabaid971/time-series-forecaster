"""ARIMA model (univariate: only uses the history of the target)."""

import logging
import warnings
from typing import Annotated, List

import numpy as np
import polars as pl
from pydantic import Field

from app.forecasting.models.base import Block, Forecaster, ModelParams

logger = logging.getLogger(__name__)


class ArimaParams(ModelParams):
    p: Annotated[int, Field(ge=0, le=10)] = 1  # AR order
    d: Annotated[int, Field(ge=0, le=2)] = 1   # Differencing order
    q: Annotated[int, Field(ge=0, le=10)] = 1  # MA order


class ArimaForecaster(Forecaster):
    """
    Coefficients are estimated once on the training rows. Before each block, the
    model state is set from all actual observations preceding it (without re-estimating
    the coefficients), so a gap between training and prediction is filled with actuals,
    like lag-based models do.
    """

    Params = ArimaParams

    def fit(self, df: pl.DataFrame, train_rows: np.ndarray) -> None:
        from statsmodels.tsa.arima.model import ARIMA

        p, d, q = self.params.p, self.params.d, self.params.q
        y_train = df[self.ctx.target_col].to_numpy()[train_rows]
        if len(y_train) < p + d + q + 5:
            raise ValueError(f"Not enough training data for ARIMA({p},{d},{q})")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fitted = ARIMA(y_train, order=(p, d, q)).fit()
        except Exception as e:
            raise ValueError(f"ARIMA fitting failed: {e}")

    def predict_blocks(self, df: pl.DataFrame, blocks: List[Block]) -> List[np.ndarray]:
        y = df[self.ctx.target_col].to_numpy()
        if any(b.start == 0 for b in blocks):
            raise ValueError("No history before the prediction range")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if all(len(b) == 1 for b in blocks):
                return self._one_step_ahead(y, blocks)
            return self._multi_step(y, blocks)

    def _one_step_ahead(self, y: np.ndarray, blocks: List[Block]) -> List[np.ndarray]:
        """Horizon 1: one filtering pass per run of consecutive rows instead of one update per row."""
        predictions = []
        run_start = 0
        for i in range(1, len(blocks) + 1):
            if i < len(blocks) and blocks[i].start == blocks[i - 1].end:
                continue
            first, last = blocks[run_start].start, blocks[i - 1].start
            state = self.fitted.apply(y[:last + 1], refit=False)
            values = state.predict(start=first, end=last)
            predictions.extend(np.array([v]) for v in values)
            run_start = i
        return predictions

    def _multi_step(self, y: np.ndarray, blocks: List[Block]) -> List[np.ndarray]:
        predictions = []
        state, position = None, None
        for block in blocks:
            if state is None or position != block.start:
                state = self.fitted.apply(y[:block.start], refit=False)
            try:
                predictions.append(np.asarray(state.forecast(steps=len(block)), dtype=float))
            except Exception:
                logger.warning("ARIMA forecast failed, using last known value", exc_info=True)
                predictions.append(np.full(len(block), y[block.start - 1]))
            try:
                # Update the state with the actuals of the block (fast, no refit)
                state = state.append(y[block.start:block.end], refit=False)
                position = block.end
            except Exception:
                state = None
        return predictions
