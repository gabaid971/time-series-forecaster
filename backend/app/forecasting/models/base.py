"""Common interface of all forecasting models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class ForecastContext:
    """What every model needs to know about the task."""
    date_col: str
    target_col: str
    horizon: int


@dataclass(frozen=True)
class Block:
    """
    A block of consecutive rows forecast from the same origin.

    Rows are positions in the full frame (sorted by date); the origin is the row
    just before `start`: the block is forecast 1..(end - start) steps ahead.
    """
    start: int
    end: int  # exclusive

    def __len__(self) -> int:
        return self.end - self.start


class ModelParams(BaseModel):
    """Base class of model parameters (unknown keys sent by the UI are ignored)."""
    model_config = ConfigDict(extra="ignore")


class Forecaster(ABC):
    """
    A forecasting model.

    Contract of predict_blocks: the forecast of a block may only use target values
    of rows before block.start (plus exogenous values known in advance). This is
    what makes the evaluation honest: see backtest.run_backtest.
    """

    Params: ClassVar[type[ModelParams]]

    def __init__(self, params: ModelParams, ctx: ForecastContext):
        self.params = params
        self.ctx = ctx

    @abstractmethod
    def fit(self, df: pl.DataFrame, train_rows: np.ndarray) -> None:
        """Fit on the rows `train_rows` of `df` (the full frame, sorted by date)."""

    @abstractmethod
    def predict_blocks(self, df: pl.DataFrame, blocks: List[Block]) -> List[np.ndarray]:
        """One array of predictions per block (NaN where no prediction can be made)."""

    def explain(self) -> Dict[str, Any]:
        """Optional explanations: feature_importance, shap_analysis."""
        return {}
