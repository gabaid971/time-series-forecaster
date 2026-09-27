"""SHAP analysis of tree models, aggregated into temporal and exogenous effects."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import shap

from app.forecasting.features import FeatureConfig


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: List[str],
    df_aligned,
    date_col: str,
    feature_config: FeatureConfig
) -> Dict[str, Any]:
    """
    Compute SHAP values for XGBoost and aggregate them into
    human-interpretable temporal and exogenous effects.
    """
    # Compute raw SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # Convert df_aligned to pandas if needed
    if not isinstance(df_aligned, pd.DataFrame):
        df_aligned = df_aligned.to_pandas()

    assert len(df_aligned) == X.shape[0], "SHAP / data misalignment"

    output: Dict[str, Any] = {
        "temporal": {},
        "exogenous": {},
    }

    # Temporal features (cyclical aggregation)
    temporal_groups = {
        "hour_of_day": {
            "enabled": feature_config.temporal.hour_of_day,
            "features": ["hour_sin", "hour_cos"],
            "values": df_aligned[date_col].dt.hour,
            "range": range(24),
        },
        "day_of_week": {
            "enabled": feature_config.temporal.day_of_week,
            "features": ["dow_sin", "dow_cos"],
            "values": df_aligned[date_col].dt.weekday,
            "range": range(7),
        },
        "month": {
            "enabled": feature_config.temporal.month,
            "features": ["month_sin", "month_cos"],
            "values": df_aligned[date_col].dt.month,
            "range": range(1, 13),
        },
        "minute_of_day": {
            "enabled": feature_config.temporal.minute_of_day,
            "features": ["minute_of_day_sin", "minute_of_day_cos"],
            "values": (
                df_aligned[date_col].dt.hour * 60
                + df_aligned[date_col].dt.minute
            ),
            "range": range(1440),
        },
    }

    for name, cfg in temporal_groups.items():
        if not cfg["enabled"]:
            continue

        f1, f2 = cfg["features"]
        if f1 not in shap_df.columns or f2 not in shap_df.columns:
            continue

        shap_sum = shap_df[f1] + shap_df[f2]
        temp_df = pd.DataFrame({
            "value": cfg["values"].values,
            "shap": shap_sum.values,
        })

        grouped = temp_df.groupby("value", as_index=False).agg(
            shap=("shap", "mean"),
            count=("shap", "size"),
        )

        # Fill missing values
        existing = set(grouped["value"].tolist())
        full_rows = []
        for v in cfg["range"]:
            if v in existing:
                row = grouped[grouped["value"] == v].iloc[0]
                full_rows.append({
                    "value": int(v),
                    "shap": float(row["shap"]),
                    "count": int(row["count"]),
                })
            else:
                full_rows.append({
                    "value": int(v),
                    "shap": 0.0,
                    "count": 0,
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

        mean_abs_shap = float(np.abs(shap_df[related_features].values).mean())
        mean_shap = float(shap_df[related_features].values.mean())

        output["exogenous"][base_col] = {
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": mean_shap,
            "direction": "positive" if mean_shap > 0 else "negative" if mean_shap < 0 else "neutral",
            "features": related_features,
        }

    return output
