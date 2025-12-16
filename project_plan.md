# Plan de Développement : Interface Web de Prédiction de Time Series

## 1. Concept
Interface web flexible pour la prédiction de séries temporelles, supportant une approche hybride (Statistiques + Machine Learning + Deep Learning).

## 2. Modèles Supportés
### Classiques
- ARIMA
- SARIMAX
- ETS
- Theta

### Prophet++
- Prophet
- NeuralProphet

### ML Tabulaire (Nécessite Feature Engineering)
- Linear Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

### Deep Learning Moderne (SOTA)
- N-BEATS
- N-HiTS
- TFT (Temporal Fusion Transformer)

## 3. Architecture Fonctionnelle

### Étape 1 : Upload & Ingestion
- Upload fichier (CSV/Excel).
- Détection auto : Date, Fréquence (D, W, M).
- Gestion des variables exogènes (merge sur date).

### Étape 2 : Feature Engineering (Crucial pour ML)
- **Lags** : Sélection manuelle ou auto (ex: lag_1, lag_7).
- **Rolling Stats** : Moyenne mobile, écart-type, min/max.
- **Transformations** : Différenciation ($\Delta$), Pourcentage ($\Delta\%$), Log returns.
- **Interactions** : Ratios, combinaisons.
- **Target** : Raw, Diff, ou Résidus (pour boosting sur résidus ARIMA).

### Étape 3 : Sélection des Modèles
- Multi-select.
- Configuration des hyperparamètres par modèle (menus collapsibles).

### Étape 4 : Entraînement & Validation
- Split Train/Test (Slider temporel ou "Last N").
- Options : Rolling forecast, Walk-forward validation.

### Étape 5 : Résultats & Export
- Métriques : RMSE, MAE, MAPE, R².
- Visualisation (Plotly) : Réel vs Pred vs Test.
- Analyse des résidus.
- Export : CSV prédictions, JSON config modèle, Notebook auto-généré.

## 4. Stack Technique Recommandée
- **Frontend** : Streamlit (MVP rapide) ou Next.js (Production).
- **Backend** : Python + FastAPI.
- **Bibliothèque Core** : **Darts** (Unifie l'API pour ARIMA, Prophet, Torch, Sklearn).
- **Data** : Polars (Performance) ou Pandas.
