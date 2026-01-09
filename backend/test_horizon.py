"""
Tests simples pour vérifier le comportement de l'horizon de prévision.

Questions à valider:
1. L'entraînement (coefficients) est-il identique avec horizon=1 et horizon=5 ?
2. La validation block-wise fonctionne-t-elle correctement ?
3. Le modèle Lag gère-t-il l'horizon ?
"""

import numpy as np
import polars as pl
from datetime import date, timedelta
from pprint import pprint

# Import du backend
from main import (
    train_linear_regression,
    train_xgboost,
    train_lag,
    DateRange,
    ForecastStrategyConfig,
    FeatureConfig,
    block_recursive_forecast,
    calculate_metrics_by_horizon
)
from sklearn.linear_model import LinearRegression


def create_simple_dataset(n=100, seed=42):
    """
    Crée un dataset simple avec une tendance + bruit.
    y[t] = 0.8 * y[t-1] + 0.1 * y[t-7] + noise
    """
    np.random.seed(seed)
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(n)]
    
    # Generate AR-like series
    y = np.zeros(n)
    y[0:7] = 100 + np.random.randn(7) * 2
    
    for i in range(7, n):
        y[i] = 0.8 * y[i-1] + 0.1 * y[i-7] + 5 + np.random.randn() * 1
    
    return pl.DataFrame({
        "date": dates,
        "value": y
    })


def test_training_unchanged_by_horizon():
    """
    Test 1: Vérifie que les coefficients du modèle sont IDENTIQUES
    avec horizon=1 et horizon=5 (seule la validation change).
    """
    print("\n" + "="*60)
    print("TEST 1: L'entraînement est-il identique avec différents horizons ?")
    print("="*60)
    
    df = create_simple_dataset(n=100)
    
    # Training: jour 1-60, Validation: jour 61-90
    training_ranges = [DateRange(start="2023-01-08", end="2023-02-28")]
    prediction_ranges = [DateRange(start="2023-03-01", end="2023-03-31")]
    
    params = {
        "lags": [1, 7],
        "target_mode": "raw",
        "standardize": False
    }
    
    # Entraîner avec horizon=1
    result_h1 = train_linear_regression(
        df=df,
        date_col="date",
        target_col="value",
        training_ranges=training_ranges,
        prediction_ranges=prediction_ranges,
        params=params,
        forecast_strategy=ForecastStrategyConfig(horizon=1)
    )
    
    # Entraîner avec horizon=5
    result_h5 = train_linear_regression(
        df=df,
        date_col="date",
        target_col="value",
        training_ranges=training_ranges,
        prediction_ranges=prediction_ranges,
        params=params,
        forecast_strategy=ForecastStrategyConfig(horizon=5)
    )
    
    # Comparer les coefficients
    coefs_h1 = result_h1.get("coefficients", {})
    coefs_h5 = result_h5.get("coefficients", {})
    
    print(f"\nCoefficients horizon=1: {coefs_h1}")
    print(f"Coefficients horizon=5: {coefs_h5}")
    
    # Vérifier si les coefficients sont identiques
    if coefs_h1 == coefs_h5:
        print("\n✅ SUCCÈS: Les coefficients sont IDENTIQUES (l'entraînement ne change pas)")
    else:
        print("\n❌ ÉCHEC: Les coefficients sont DIFFÉRENTS !")
        # Comparer détail
        for key in set(list(coefs_h1.keys()) + list(coefs_h5.keys())):
            v1 = coefs_h1.get(key)
            v2 = coefs_h5.get(key)
            if v1 != v2:
                print(f"   {key}: h1={v1} vs h5={v2}")
    
    # Vérifier les métriques (elles DEVRAIENT différer car la validation change)
    metrics_h1 = result_h1["metrics"]
    metrics_h5 = result_h5["metrics"]
    
    print(f"\nMétriques horizon=1: RMSE={metrics_h1['rmse']:.4f}, MAE={metrics_h1['mae']:.4f}")
    print(f"Métriques horizon=5: RMSE={metrics_h5['rmse']:.4f}, MAE={metrics_h5['mae']:.4f}")
    
    if metrics_h1["rmse"] != metrics_h5["rmse"]:
        print("✅ SUCCÈS: Les métriques de validation DIFFÈRENT (normal car récursif)")
    else:
        print("⚠️ Les métriques sont identiques (étrange si horizon > minLag)")
    
    return coefs_h1 == coefs_h5


def test_block_recursive_behavior():
    """
    Test 2: Vérifie le comportement block-wise recursive.
    - Dans un bloc: on utilise les prédictions précédentes
    - Entre blocs: on remet les vraies valeurs
    """
    print("\n" + "="*60)
    print("TEST 2: Comportement block-wise recursive")
    print("="*60)
    
    df = create_simple_dataset(n=100)
    
    # Entraîner avec horizon=3 pour voir les blocs
    training_ranges = [DateRange(start="2023-01-08", end="2023-02-28")]
    prediction_ranges = [DateRange(start="2023-03-01", end="2023-03-15")]  # 15 jours = 5 blocs
    
    params = {
        "lags": [1, 7],
        "target_mode": "raw",
        "standardize": False
    }
    
    result = train_linear_regression(
        df=df,
        date_col="date",
        target_col="value",
        training_ranges=training_ranges,
        prediction_ranges=prediction_ranges,
        params=params,
        forecast_strategy=ForecastStrategyConfig(horizon=3)
    )
    
    forecasts = result["forecast"]
    
    print(f"\nNombre de prévisions: {len(forecasts)}")
    print("\nDétail par bloc et step:")
    
    current_block = 0
    for f in forecasts[:12]:  # Premiers 12 = 4 blocs
        block = f.get("block_num", "?")
        step = f.get("step_in_block", "?")
        date_val = f.get("date", "?")
        pred = f.get("prediction", 0)
        actual = f.get("value", 0)
        
        if block != current_block:
            print(f"\n--- Bloc {block} ---")
            current_block = block
        
        print(f"  Step {step}: {date_val} | pred={pred:.2f} | actual={actual:.2f}")
    
    # Vérifier que step_in_block va de 1 à horizon puis reset
    blocks = {}
    for f in forecasts:
        b = f.get("block_num", 0)
        s = f.get("step_in_block", 0)
        if b not in blocks:
            blocks[b] = []
        blocks[b].append(s)
    
    print(f"\nBlocs trouvés: {len(blocks)}")
    all_ok = True
    for b, steps in blocks.items():
        expected = list(range(1, len(steps) + 1))
        if steps == expected:
            print(f"  Bloc {b}: steps={steps} ✅")
        else:
            print(f"  Bloc {b}: steps={steps} (attendu: {expected}) ❌")
            all_ok = False
    
    return all_ok


def test_metrics_by_horizon():
    """
    Test 3: Vérifie que les métriques par horizon step sont calculées.
    """
    print("\n" + "="*60)
    print("TEST 3: Métriques par horizon step")
    print("="*60)
    
    df = create_simple_dataset(n=100)
    
    training_ranges = [DateRange(start="2023-01-08", end="2023-02-28")]
    prediction_ranges = [DateRange(start="2023-03-01", end="2023-03-31")]
    
    params = {
        "lags": [1, 7],
        "target_mode": "raw"
    }
    
    result = train_linear_regression(
        df=df,
        date_col="date",
        target_col="value",
        training_ranges=training_ranges,
        prediction_ranges=prediction_ranges,
        params=params,
        forecast_strategy=ForecastStrategyConfig(horizon=5)
    )
    
    horizon_metrics = result.get("metrics_by_horizon", [])
    
    if not horizon_metrics:
        print("❌ Pas de metrics_by_horizon retournées!")
        return False
    
    print(f"\nMétriques par horizon step (horizon=5):")
    print("-" * 50)
    
    prev_rmse = 0
    rmse_increasing = True
    
    for m in horizon_metrics:
        step = m["horizon_step"]
        rmse = m["rmse"]
        mae = m["mae"]
        count = m["count"]
        
        print(f"  Step {step}: RMSE={rmse:.4f}, MAE={mae:.4f}, count={count}")
        
        # On s'attend à ce que RMSE augmente avec l'horizon (erreur se propage)
        if step > 1 and rmse < prev_rmse * 0.9:  # Tolérance 10%
            rmse_increasing = False
        prev_rmse = rmse
    
    if rmse_increasing:
        print("\n✅ RMSE tend à augmenter avec l'horizon (erreurs se propagent)")
    else:
        print("\n⚠️ RMSE ne croît pas uniformément (peut être OK selon les données)")
    
    return len(horizon_metrics) > 0


def test_lag_model_with_horizon():
    """
    Test 4: Le modèle Lag avec horizon utilise block-wise recursive.
    Block 1: pred(t+1) = y(t₀), pred(t+2) = pred(t+1), pred(t+3) = pred(t+2)
    Block 2: reset, then recursive again
    """
    print("\n" + "="*60)
    print("TEST 4: Modèle Lag avec horizon (block-wise recursive)")
    print("="*60)
    
    df = create_simple_dataset(n=100)
    
    training_ranges = [DateRange(start="2023-01-08", end="2023-02-28")]
    prediction_ranges = [DateRange(start="2023-03-01", end="2023-03-15")]
    
    # Test avec horizon=1 (comportement normal)
    result_h1 = train_lag(
        df=df,
        date_col="date",
        target_col="value",
        training_ranges=training_ranges,
        prediction_ranges=prediction_ranges,
        params={"lag": 1},
        forecast_strategy=ForecastStrategyConfig(horizon=1)
    )
    
    # Test avec horizon=3 (block-wise recursive)
    result_h3 = train_lag(
        df=df,
        date_col="date",
        target_col="value",
        training_ranges=training_ranges,
        prediction_ranges=prediction_ranges,
        params={"lag": 1},
        forecast_strategy=ForecastStrategyConfig(horizon=3)
    )
    
    print(f"\nModèle Lag(1) - Horizon=1 (naïf):")
    forecasts_h1 = result_h1["forecast"][:6]
    for f in forecasts_h1:
        print(f"  {f.get('date')}: pred={f['prediction']:.2f}, actual={f['value']:.2f}")
    
    print(f"\nModèle Lag(1) - Horizon=3 (block-wise recursive):")
    forecasts_h3 = result_h3["forecast"][:9]  # 3 blocs
    current_block = 0
    for f in forecasts_h3:
        block = f.get("block_num", "?")
        step = f.get("step_in_block", "?")
        if block != current_block:
            print(f"\n  --- Bloc {block} ---")
            current_block = block
        print(f"  Step {step}: {f.get('date')} | pred={f['prediction']:.2f} | actual={f['value']:.2f}")
    
    # Vérifier que dans un bloc, les prédictions aux steps 2+ sont identiques à step 1
    # (car lag=1, donc pred(t+2) = pred(t+1) = pred(t+1) = ...)
    blocks = {}
    for f in result_h3["forecast"]:
        b = f.get("block_num", 0)
        if b not in blocks:
            blocks[b] = []
        blocks[b].append(f["prediction"])
    
    print(f"\nVérification: toutes les prédictions d'un bloc doivent être identiques (lag=1):")
    all_ok = True
    for b, preds in list(blocks.items())[:3]:
        all_same = all(abs(p - preds[0]) < 0.001 for p in preds)
        status = "✅" if all_same else "❌"
        print(f"  Bloc {b}: {[round(p, 2) for p in preds]} {status}")
        if not all_same:
            all_ok = False
    
    # Vérifier les métriques par horizon
    horizon_metrics = result_h3.get("metrics_by_horizon", [])
    if horizon_metrics:
        print(f"\nMétriques par horizon:")
        for m in horizon_metrics:
            print(f"  Step {m['horizon_step']}: RMSE={m['rmse']:.4f}")
    
    return all_ok


def test_xgboost_horizon():
    """
    Test 5: XGBoost avec horizon - vérifie que ça fonctionne.
    """
    print("\n" + "="*60)
    print("TEST 5: XGBoost avec horizon")
    print("="*60)
    
    df = create_simple_dataset(n=100)
    
    training_ranges = [DateRange(start="2023-01-08", end="2023-02-28")]
    prediction_ranges = [DateRange(start="2023-03-01", end="2023-03-15")]
    
    params = {
        "lags": [1, 7],
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.1
    }
    
    try:
        result = train_xgboost(
            df=df,
            date_col="date",
            target_col="value",
            training_ranges=training_ranges,
            prediction_ranges=prediction_ranges,
            params=params,
            forecast_strategy=ForecastStrategyConfig(horizon=5)
        )
        
        print(f"\n✅ XGBoost avec horizon=5 fonctionne!")
        print(f"  Métriques: RMSE={result['metrics']['rmse']:.4f}")
        print(f"  Nombre de prévisions: {len(result['forecast'])}")
        
        horizon_metrics = result.get("metrics_by_horizon", [])
        if horizon_metrics:
            print(f"  Métriques par horizon: {len(horizon_metrics)} steps")
            for m in horizon_metrics:
                print(f"    Step {m['horizon_step']}: RMSE={m['rmse']:.4f}")
        
        return True
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Exécute tous les tests."""
    print("\n" + "#"*60)
    print("# TESTS DE L'HORIZON DE PRÉVISION")
    print("#"*60)
    
    results = []
    
    results.append(("Entraînement identique", test_training_unchanged_by_horizon()))
    results.append(("Block recursive", test_block_recursive_behavior()))
    results.append(("Métriques par horizon", test_metrics_by_horizon()))
    results.append(("Modèle Lag", test_lag_model_with_horizon()))
    results.append(("XGBoost horizon", test_xgboost_horizon()))
    
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("✅ TOUS LES TESTS PASSENT" if all_passed else "❌ CERTAINS TESTS ÉCHOUENT"))
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
