"""Tests for feature engineering utilities."""

import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta

from utils.features import (
    build_features, 
    FeatureConfig, 
    TemporalFeatureConfig,
    ExogenousFeatureConfig,
    DerivedFeatureConfig,
    parse_feature_config,
    validate_no_future_leakage
)


class TestBuildFeatures:
    """Tests for build_features function."""
    
    def test_target_lags_created(self, sample_daily_df):
        """Should create target lag columns."""
        config = FeatureConfig(target_lags=[1, 7])
        
        result_df, feature_names = build_features(
            sample_daily_df, "date", "value", config
        )
        
        assert "target_lag_1" in result_df.columns
        assert "target_lag_7" in result_df.columns
        assert "target_lag_1" in feature_names
        assert "target_lag_7" in feature_names
    
    def test_lag_values_correct(self, sample_daily_df):
        """Lag values should match shifted target."""
        config = FeatureConfig(target_lags=[1])
        
        result_df, _ = build_features(
            sample_daily_df, "date", "value", config
        )
        
        # Second row's lag_1 should equal first row's value
        lag_value = result_df["target_lag_1"][1]
        original_value = sample_daily_df["value"][0]
        
        assert lag_value == pytest.approx(original_value)
    
    def test_temporal_month_cyclical(self, sample_daily_df):
        """Should create cyclical month encoding."""
        config = FeatureConfig(
            target_lags=[1],
            temporal=TemporalFeatureConfig(month=True)
        )
        
        result_df, feature_names = build_features(
            sample_daily_df, "date", "value", config
        )
        
        assert "month_sin" in result_df.columns
        assert "month_cos" in result_df.columns
        assert "month_sin" in feature_names
        assert "month_cos" in feature_names
    
    def test_temporal_day_of_week(self, sample_daily_df):
        """Should create day of week encoding."""
        config = FeatureConfig(
            target_lags=[1],
            temporal=TemporalFeatureConfig(day_of_week=True)
        )
        
        result_df, feature_names = build_features(
            sample_daily_df, "date", "value", config
        )
        
        assert "dow_sin" in feature_names
        assert "dow_cos" in feature_names
    
    def test_temporal_hour_of_day(self, sample_hourly_df):
        """Should create hour of day encoding."""
        config = FeatureConfig(
            target_lags=[1],
            temporal=TemporalFeatureConfig(hour_of_day=True)
        )
        
        result_df, feature_names = build_features(
            sample_hourly_df, "date", "value", config
        )
        
        assert "hour_sin" in feature_names
        assert "hour_cos" in feature_names
    
    def test_exogenous_lags(self, sample_df_with_exogenous):
        """Should create exogenous lag columns."""
        config = FeatureConfig(
            target_lags=[1],
            exogenous=[
                ExogenousFeatureConfig(column="temperature", lags=[1, 7])
            ]
        )
        
        result_df, feature_names = build_features(
            sample_df_with_exogenous, "date", "value", config
        )
        
        assert "temperature_lag_1" in feature_names
        assert "temperature_lag_7" in feature_names
    
    def test_exogenous_actual(self, sample_df_with_exogenous):
        """Should include actual exogenous value."""
        config = FeatureConfig(
            target_lags=[1],
            exogenous=[
                ExogenousFeatureConfig(column="temperature", use_actual=True)
            ]
        )
        
        result_df, feature_names = build_features(
            sample_df_with_exogenous, "date", "value", config
        )
        
        assert "temperature_actual" in feature_names
    
    def test_exogenous_delta(self, sample_df_with_exogenous):
        """Should create delta feature."""
        config = FeatureConfig(
            target_lags=[1],
            exogenous=[
                ExogenousFeatureConfig(column="temperature", delta_lag=1)
            ]
        )
        
        result_df, feature_names = build_features(
            sample_df_with_exogenous, "date", "value", config
        )
        
        assert "temperature_delta_1" in feature_names
    
    def test_derived_features(self, sample_df_with_exogenous):
        """Should create derived features from operations."""
        # First add lags to create features to operate on
        sample_df_with_exogenous = sample_df_with_exogenous.with_columns([
            pl.col("temperature").alias("temp"),
            pl.col("value").alias("val")
        ])
        
        config = FeatureConfig(
            target_lags=[1],
            derived=[
                DerivedFeatureConfig(
                    operation="sum",
                    feature_a="temp",
                    feature_b="val",
                    alias="temp_plus_val"
                )
            ]
        )
        
        result_df, feature_names = build_features(
            sample_df_with_exogenous, "date", "value", config
        )
        
        assert "temp_plus_val" in feature_names
    
    def test_missing_exogenous_column_skipped(self, sample_daily_df):
        """Should skip missing exogenous columns with warning."""
        config = FeatureConfig(
            target_lags=[1],
            exogenous=[
                ExogenousFeatureConfig(column="nonexistent", lags=[1])
            ]
        )
        
        # Should not raise, just skip
        result_df, feature_names = build_features(
            sample_daily_df, "date", "value", config
        )
        
        assert "nonexistent_lag_1" not in feature_names
    
    def test_cyclical_values_in_range(self, sample_daily_df):
        """Cyclical encoded values should be in [-1, 1]."""
        config = FeatureConfig(
            target_lags=[1],
            temporal=TemporalFeatureConfig(month=True, day_of_week=True)
        )
        
        result_df, _ = build_features(
            sample_daily_df, "date", "value", config
        )
        
        for col in ["month_sin", "month_cos", "dow_sin", "dow_cos"]:
            assert result_df[col].min() >= -1.0
            assert result_df[col].max() <= 1.0
    
    def test_empty_config_returns_no_features(self, sample_daily_df):
        """Empty config should return only target lags."""
        config = FeatureConfig(target_lags=[])
        
        result_df, feature_names = build_features(
            sample_daily_df, "date", "value", config
        )
        
        assert len(feature_names) == 0


class TestParseFeatureConfig:
    """Tests for parse_feature_config."""

    def test_lags_param_wins_over_stale_feature_config(self):
        """Lags edited in the UI (params.lags) must not be ignored once feature_config exists."""
        params = {"lags": [1, 2, 3], "feature_config": {"target_lags": [1, 7], "temporal": {"month": True}}}
        config = parse_feature_config(params, default_lags=[1, 7])
        assert config.target_lags == [1, 2, 3]
        assert config.temporal.month is True

    def test_defaults(self):
        assert parse_feature_config({}, default_lags=[1, 7]).target_lags == [1, 7]
        assert parse_feature_config({"lags": "1, 14"}, default_lags=[1]).target_lags == [1, 14]


class TestValidateNoFutureLeakage:
    """Features must be known at the forecast origin."""

    def test_unknown_exogenous_short_lag_rejected(self):
        config = FeatureConfig(exogenous=[ExogenousFeatureConfig(column="sensor", lags=[0, 1])])
        with pytest.raises(ValueError, match=r"'sensor' lag\(s\) \[0, 1\] < horizon 7"):
            validate_no_future_leakage(config, "value", horizon=7)

    def test_unknown_exogenous_lag_zero_rejected_at_horizon_one(self):
        config = FeatureConfig(exogenous=[ExogenousFeatureConfig(column="sensor", lags=[0])])
        with pytest.raises(ValueError):
            validate_no_future_leakage(config, "value", horizon=1)

    def test_unknown_exogenous_long_lag_accepted(self):
        config = FeatureConfig(exogenous=[ExogenousFeatureConfig(column="sensor", lags=[7, 14])])
        validate_no_future_leakage(config, "value", horizon=7)

    def test_known_in_advance_accepts_any_lag(self):
        config = FeatureConfig(exogenous=[
            ExogenousFeatureConfig(column="promo", lags=[0, 1], use_actual=True, known_in_advance=True)
        ])
        validate_no_future_leakage(config, "value", horizon=7)

    def test_unknown_exogenous_actual_value_rejected(self):
        config = FeatureConfig(exogenous=[ExogenousFeatureConfig(column="sensor", use_actual=True)])
        with pytest.raises(ValueError, match="known in advance"):
            validate_no_future_leakage(config, "value", horizon=1)

    def test_derived_on_short_target_lag_rejected(self):
        config = FeatureConfig(derived=[
            DerivedFeatureConfig(operation="product", feature_a="target_lag_1", feature_b="month_sin")
        ])
        validate_no_future_leakage(config, "value", horizon=1)
        with pytest.raises(ValueError, match="target_lag_1"):
            validate_no_future_leakage(config, "value", horizon=3)

    def test_derived_on_target_or_raw_unknown_column_rejected(self):
        for operand in ["value", "sensor"]:
            config = FeatureConfig(
                exogenous=[ExogenousFeatureConfig(column="sensor", lags=[7])],
                derived=[DerivedFeatureConfig(operation="sum", feature_a=operand, feature_b="month_sin")]
            )
            with pytest.raises(ValueError, match=operand):
                validate_no_future_leakage(config, "value", horizon=7)
