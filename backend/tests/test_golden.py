"""
Golden (snapshot) tests: end-to-end API results on reference scenarios.

They freeze metrics and every prediction, so that refactorings can prove they
do not change results. When a change is intended, regenerate the snapshots:

    UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py

and review the diff of tests/golden/*.json.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"
REL_TOL = 1e-6
ABS_TOL = 1e-8


def load_csv(relative_path):
    with open(REPO_ROOT / relative_path) as f:
        return list(csv.DictReader(f))


TEMPERATURES = ("daily-minimum-temperatures.csv", "Date", "Daily minimum temperatures")
SALES = ("demo_data/ventes_capteur.csv", "date", "ventes")
MINUTES = ("demo_data/minutes_cycle_horaire.csv", "timestamp", "charge")

LAGS = [1, 2, 3, 7]
MONTH = {"target_lags": LAGS, "temporal": {"month": True}, "exogenous": [], "derived": []}


def model(model_id, model_type, params):
    return {"id": model_id, "type": model_type, "name": model_id, "params": params}


def exog_config(lags, exogenous, day_of_week=False):
    return {"target_lags": lags, "temporal": {"day_of_week": day_of_week}, "exogenous": exogenous, "derived": []}


TEMPERATURE_MODELS = [
    model("lag1", "LAG", {"lag": 1}),
    model("lr", "LINEAR_REGRESSION", {"lags": LAGS}),
    model("lr_month_std", "LINEAR_REGRESSION", {"lags": LAGS, "standardize": True, "feature_config": MONTH}),
    model("lr_residual", "LINEAR_REGRESSION", {"lags": LAGS, "target_mode": "residual", "feature_config": MONTH}),
    model("xgb_month", "XGBOOST", {"lags": LAGS, "n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "feature_config": MONTH}),
    model("arima111", "ARIMA", {"p": 1, "d": 1, "q": 1}),
    model("prophet", "PROPHET", {"daily_seasonality": False, "weekly_seasonality": False, "yearly_seasonality": True}),
]

SALES_MODELS = [
    model("lr_sensor_14", "LINEAR_REGRESSION", {"lags": [1, 7], "feature_config": exog_config([1, 7], [{"column": "capteur", "lags": [14]}])}),
    model("lr_sensor_known", "LINEAR_REGRESSION", {"lags": [1, 7], "feature_config": exog_config(
        [1, 7], [{"column": "capteur", "lags": [0, 1], "known_in_advance": True}], day_of_week=True)}),
    model("xgb_residual_sensor", "XGBOOST", {"lags": [1, 7], "target_mode": "residual", "n_estimators": 50, "max_depth": 3,
                                             "learning_rate": 0.1, "feature_config": exog_config([1, 7], [{"column": "capteur", "lags": [14, 21]}], day_of_week=True)}),
    model("lag7", "LAG", {"lag": 7}),
    model("arima_fails", "ARIMA", {"p": 5, "d": 2, "q": 5}),
    model("lr_leak_rejected", "LINEAR_REGRESSION", {"lags": [1, 7], "feature_config": exog_config([1, 7], [{"column": "capteur", "lags": [0, 1]}])}),
]

TRAIN_SCENARIOS = {
    "temperatures_h1": (TEMPERATURES, [("1981-01-01", "1989-01-01")], [("1989-01-01", "1990-12-31")], 1, TEMPERATURE_MODELS),
    "temperatures_h7": (TEMPERATURES, [("1981-01-01", "1989-01-01")], [("1989-01-01", "1990-12-31")], 7, TEMPERATURE_MODELS),
    "temperatures_h30": (TEMPERATURES, [("1981-01-01", "1989-01-01")], [("1989-01-01", "1990-12-31")], 30, TEMPERATURE_MODELS),
    "temperatures_gap": (TEMPERATURES, [("1981-01-01", "1985-01-01")], [("1989-06-01", "1990-12-31")], 30, TEMPERATURE_MODELS),
    "temperatures_multi_ranges": (TEMPERATURES, [("1981-01-01", "1983-01-01"), ("1985-01-01", "1987-01-01")],
                                  [("1984-01-01", "1984-06-30"), ("1989-01-01", "1989-06-30")], 7, TEMPERATURE_MODELS),
    "sales_h14": (SALES, [("2023-01-01", "2024-07-01")], [("2024-07-01", "2024-12-30")], 14, SALES_MODELS),
    "sales_short_training_h1": (SALES, [("2023-01-01", "2023-01-11")], [("2023-01-11", "2023-03-01")], 1,
                                [model("arima_fails", "ARIMA", {"p": 5, "d": 2, "q": 5}), model("lr", "LINEAR_REGRESSION", {"lags": [1, 7]})]),
}

ANALYZE_SCENARIOS = {"temperatures": TEMPERATURES, "sales": SALES, "minutes": MINUTES}


def train_payload(dataset, training, prediction, horizon, models):
    path, date_col, target_col = dataset
    return {
        "data": load_csv(path),
        "data_config": {
            "target_column": target_col,
            "date_column": date_col,
            "frequency": "D",
            "training_ranges": [{"start": s, "end": e} for s, e in training],
            "prediction_ranges": [{"start": s, "end": e} for s, e in prediction],
            "forecast_strategy": {"horizon": horizon},
        },
        "models": models,
    }


def fingerprint(values):
    """Exact, compact fingerprint of a list (dates, actuals, horizon steps)."""
    return hashlib.sha1(json.dumps(values).encode()).hexdigest()


def summarize_train(response_json, date_col, target_col):
    """Keep everything that defines the results; drop timings and error wording."""
    summary = {}
    for r in response_json["results"]:
        if r.get("error"):
            summary[r["model_id"]] = {"error": True}
            continue
        forecast = r["forecast"]
        summary[r["model_id"]] = {
            "error": False,
            "metrics": {k: v for k, v in r["metrics"].items() if k != "execution_time"},
            "metrics_by_horizon": r.get("metrics_by_horizon"),
            "forecast": {
                "dates": fingerprint([f[date_col] for f in forecast]),
                "actuals": fingerprint([f[target_col] for f in forecast]),
                "horizon_steps": fingerprint([f.get("horizon_step") for f in forecast]),
                "predictions": [round(f["prediction"], 8) for f in forecast],
            },
            "feature_importance": r.get("feature_importance"),
            "shap_temporal": (r.get("shap_analysis") or {}).get("temporal"),
        }
    return summary


def summarize_analyze(response_json):
    lag = response_json["lag_analysis"]
    return {
        "stats": response_json["stats"],
        "suggested_lags": lag["suggested_lags"],
        "acf": lag["acf"],
        "pacf": lag["pacf"],
        "seasonality": lag["seasonality"],
        "alerts": [a["message"] for a in response_json["alerts"] or []],
        "n_normalized": len(response_json["normalized_data"]),
    }


def assert_close(actual, expected, path="root"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {actual!r}"
        assert actual.keys() == expected.keys(), f"{path}: keys {sorted(actual)} != {sorted(expected)}"
        for k in expected:
            assert_close(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), \
            f"{path}: length {len(actual) if isinstance(actual, list) else actual!r} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, float) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)), f"{path}: expected number, got {actual!r}"
        assert math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL), f"{path}: {actual} != {expected}"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def check_golden(name, summary):
    path = GOLDEN_DIR / f"{name}.json"
    if UPDATE or not path.exists():
        path.write_text(json.dumps(summary, sort_keys=True) + "\n")
        if not UPDATE:
            pytest.skip(f"Golden file {path.name} created")
        return
    assert_close(summary, json.loads(path.read_text()), name)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.mark.parametrize("name", TRAIN_SCENARIOS)
def test_train_golden(client, name):
    dataset, training, prediction, horizon, models = TRAIN_SCENARIOS[name]
    response = client.post("/train", json=train_payload(dataset, training, prediction, horizon, models))
    assert response.status_code == 200, response.text
    check_golden(f"train_{name}", summarize_train(response.json(), dataset[1], dataset[2]))


@pytest.mark.parametrize("name", ANALYZE_SCENARIOS)
def test_analyze_golden(client, name):
    path, date_col, target_col = ANALYZE_SCENARIOS[name]
    response = client.post("/analyze", json={"data": load_csv(path), "date_column": date_col, "target_column": target_col})
    assert response.status_code == 200, response.text
    check_golden(f"analyze_{name}", summarize_analyze(response.json()))
