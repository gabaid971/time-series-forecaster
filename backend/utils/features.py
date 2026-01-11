"""Feature engineering utilities."""

import polars as pl
import numpy as np
from typing import List, Optional
from pydantic import BaseModel


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


def build_features(
    df: pl.DataFrame,
    date_col: str,
    target_col: str,
    feature_config: FeatureConfig
) -> tuple[pl.DataFrame, List[str]]:
    """
    Build all features for model training.
    
    Args:
        df: Input DataFrame
        date_col: Name of date column
        target_col: Name of target column
        feature_config: Feature configuration
    
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
