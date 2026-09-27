"""SHAP analysis of tree models, aggregated into temporal and exogenous effects."""

from typing import Any, Dict, List

import numpy as np
import polars as pl

from app.forecasting.features import FeatureConfig


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: List[str],
    df_aligned: pl.DataFrame,
    date_col: str,
    feature_config: FeatureConfig
) -> Dict[str, Any]:
    """
    Compute SHAP values for XGBoost and aggregate them into
    human-interpretable temporal and exogenous effects.

    XGBoost computes exact TreeSHAP values itself (pred_contribs): no need for the
    shap library, which costs ~80 MB of memory on a small instance.
    """
    import xgboost as xgb

    # One column per feature, plus the bias in the last column
    contributions = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1].astype(np.float64)
    column = {name: i for i, name in enumerate(feature_names)}
    assert df_aligned.height == X.shape[0], "SHAP / data misalignment"

    dates = df_aligned[date_col]
    hours = dates.dt.hour().cast(pl.Int32).to_numpy()
    output: Dict[str, Any] = {
        "temporal": {},
        "exogenous": {},
    }

    # Temporal features (cyclical aggregation): effect by value (by month, by weekday...)
    temporal_groups = {
        "hour_of_day": (feature_config.temporal.hour_of_day, ["hour_sin", "hour_cos"], hours, range(24)),
        # Monday = 0 (polars counts weekdays from 1)
        "day_of_week": (feature_config.temporal.day_of_week, ["dow_sin", "dow_cos"],
                        dates.dt.weekday().cast(pl.Int32).to_numpy() - 1, range(7)),
        "month": (feature_config.temporal.month, ["month_sin", "month_cos"],
                  dates.dt.month().cast(pl.Int32).to_numpy(), range(1, 13)),
        "minute_of_day": (feature_config.temporal.minute_of_day, ["minute_of_day_sin", "minute_of_day_cos"],
                          hours * 60 + dates.dt.minute().cast(pl.Int32).to_numpy(), range(1440)),
    }

    for name, (enabled, (f1, f2), values, value_range) in temporal_groups.items():
        if not enabled or f1 not in column or f2 not in column:
            continue

        shap_sum = contributions[:, column[f1]] + contributions[:, column[f2]]
        full_rows = []
        for v in value_range:
            mask = values == v
            count = int(mask.sum())
            full_rows.append({
                "value": int(v),
                "shap": float(shap_sum[mask].mean()) if count else 0.0,
                "count": count,
            })

        # Normalization
        max_abs = max(abs(r["shap"]) for r in full_rows) or 1.0
        for r in full_rows:
            r["shap_norm"] = r["shap"] / max_abs

        output["temporal"][name] = full_rows

    # Exogenous features
    for exog in feature_config.exogenous:
        base_col = exog.column
        related_features = [
            f for f in feature_names
            if f.startswith(base_col + "_") or f == f"{base_col}_actual"
        ]

        if not related_features:
            continue

        related = contributions[:, [column[f] for f in related_features]]
        mean_abs_shap = float(np.abs(related).mean())
        mean_shap = float(related.mean())

        output["exogenous"][base_col] = {
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": mean_shap,
            "direction": "positive" if mean_shap > 0 else "negative" if mean_shap < 0 else "neutral",
            "features": related_features,
        }

    return output
