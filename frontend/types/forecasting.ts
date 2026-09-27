// Codes returned by the backend frequency detection
export type Frequency = 's' | 'min' | 'H' | 'D' | 'W' | 'M';

// 1. Définition des données brutes
export interface TimeSeriesData {
  filename: string;
  columns: string[];
  targetColumn: string;
  dateColumn: string;
  frequency: Frequency;
  exogenousFeatures: string[]; // Colonnes disponibles pour aider la prédiction
}

// 2. Types de Modèles
export type ModelType = 
  | 'LAG'
  | 'LINEAR_REGRESSION'
  | 'ARIMA' 
  | 'PROPHET' 
  | 'XGBOOST' 
  | 'NBEATS';

// 3. Configuration spécifique par modèle
export interface ModelConfig {
  id: string; // unique id (ex: "xgb-run-1")
  type: ModelType;
  name: string; // Nom affiché (ex: "XGBoost avec Lags")
  colorIndex?: number; // Fixed palette slot (see lib/modelColors.ts)

  // Config spécifique (Union type)
  params: LagParams | ArimaParams | ProphetParams | XGBoostParams | LinearRegressionParams;
}

export interface LagParams {
  lag: number;
}

export interface ArimaParams {
  p: number;
  d: number;
  q: number;
  seasonal_order?: [number, number, number, number];
}

export interface ProphetParams {
  daily_seasonality: boolean;
  weekly_seasonality: boolean;
  yearly_seasonality: boolean;
  seasonality_mode?: 'additive' | 'multiplicative';
  country_holidays?: string;
}

// Pour XGBoost
export interface XGBoostParams {
  lags: number[];
  n_estimators?: number;
  max_depth?: number;
  learning_rate?: number;
  feature_config?: FeatureConfig;
}

// Feature configuration (new unified structure)
export interface ExogenousFeatureConfig {
  column: string;
  lags: number[];           // Lag values to create
  use_actual: boolean;      // Use actual value at prediction time
  delta_lag?: number;       // Compute delta vs this lag
  pct_change_lag?: number;  // Compute % change vs this lag
  // Future values known at forecast time (calendar, planned promotions...).
  // If false, the backend only accepts lags >= forecast horizon.
  known_in_advance?: boolean;
}

export interface TemporalFeatureConfig {
  month: boolean;           // Cyclical encoded month
  day_of_week: boolean;     // Cyclical encoded day of week
  day_of_month: boolean;    // Day of month (1-31)
  week_of_year: boolean;    // Week of year (1-52)
  year: boolean;            // Year as numeric
  hour_of_day?: boolean;    // Hour of day (0-23)
  minute_of_day?: boolean;  // Minute of day (0-1439)
}

export interface DerivedFeatureConfig {
  operation: 'sum' | 'product' | 'ratio' | 'difference';
  feature_a: string;
  feature_b: string;
  alias?: string;
}

export interface FeatureConfig {
  target_lags: number[];
  temporal: TemporalFeatureConfig;
  exogenous: ExogenousFeatureConfig[];
  derived?: DerivedFeatureConfig[];
}

// Pour Linear Regression
export interface LinearRegressionParams {
  lags: number[];  // Legacy support
  target_mode?: 'raw' | 'residual';
  residual_lag?: number;
  standardize?: boolean;
  feature_config?: FeatureConfig;  // New unified feature config
}

// Column info from /analyze response
export interface ColumnInfo {
  name: string;
  dtype: 'numeric' | 'string' | 'date' | 'boolean';
  missing_count: number;
  sample_values: any[];
}

// 4. L'objet complet envoyé au Backend pour l'entraînement
export interface DateRange {
  start: string;
  end: string;
}

// Forecast strategy configuration
export interface ForecastStrategyConfig {
  horizon: number;              // Number of steps to forecast ahead (block size)
}

export interface TrainingRequest {
  data: any[]; // Raw CSV data
  data_config: {
    target_column: string;
    date_column: string;
    frequency: Frequency;
    training_ranges: DateRange[];
    prediction_ranges: DateRange[];
    forecast_strategy?: ForecastStrategyConfig; // Optional multi-step config
  };
  models: ModelConfig[]; // Liste des modèles à entraîner
}

// 5. Résultats (Backend -> Frontend)
export interface ForecastPoint {
  date: string;
  value: number;
  lower_bound?: number;
  upper_bound?: number;
}

export interface ModelMetrics {
  rmse: number;
  mae: number;
  mape: number;
  r2: number;
  msle: number;
  execution_time: number;
}

// Metrics per forecast horizon step
export interface HorizonMetrics {
  horizon_step: number;
  rmse: number;
  mae: number;
  mape: number;
  msle: number;
  count: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

// SHAP value for a specific temporal value (e.g., Monday = 0, January = 1)
export interface ShapValue {
  value: number;        // The temporal value (0-6 for day_of_week, 1-12 for month, etc.)
  shap: number;         // Raw SHAP value
  count: number;        // Number of samples with this value
  shap_norm: number;    // Normalized SHAP value (-1 to 1)
}

// SHAP analysis for exogenous features
export interface ExogenousShap {
  mean_abs_shap: number;
  mean_shap: number;
  direction: 'positive' | 'negative' | 'neutral';
  features: string[];
}

// Complete SHAP analysis structure from XGBoost
export interface ShapAnalysis {
  temporal: {
    hour_of_day?: ShapValue[];
    day_of_week?: ShapValue[];
    month?: ShapValue[];
    minute_of_day?: ShapValue[];
  };
  exogenous: {
    [key: string]: ExogenousShap;
  };
}

export interface ModelResult {
  model_id: string;
  model_name: string;
  metrics: ModelMetrics | null;  // null when the model failed (see error)
  forecast: ForecastPoint[];
  metrics_by_horizon?: HorizonMetrics[];  // Per-step metrics for multi-horizon
  feature_importance?: FeatureImportance[];
  shap_analysis?: ShapAnalysis;  // SHAP analysis for XGBoost models
  error?: string;  // Error message if model training failed
}

export interface TrainingResponse {
  status: 'success' | 'error';
  results: ModelResult[];
}

// 6. Dataset analysis (Backend -> Frontend, /analyze)
export interface DatasetStats {
  date_min: string;
  date_max: string;
  total_rows: number;
  frequency: string;
  frequency_label: string;
  missing_dates: number;
  missing_values_target: number;
  value_min: number;
  value_max: number;
  value_mean: number;
}

export interface Seasonality {
  period: number;               // In rows (e.g. 365 for a yearly cycle of daily data)
  period_label: string;         // "Yearly", "Weekly"...
  strength: number;             // Autocorrelation at the period
  suggested_feature?: string | null;  // Calendar feature capturing the cycle
}

export interface LagAnalysis {
  suggested_lags: number[];
  suggested_temporal: string[];  // Calendar features recommended by the detected cycles
  seasonalities: Seasonality[];
  acf: number[];
  pacf: number[];
  confidence_interval: number;
  significant_lags: { lag: number; pacf: number; significant: boolean }[];
  seasonality: {
    detected: boolean;
    period?: number;
    period_label?: string;
    strength?: number;
  };
  n_observations: number;
}

export interface DataAlert {
  type: 'warning' | 'info' | 'error';
  category: string;
  message: string;
  details?: Record<string, unknown>;
}

// 7. Forecast space: library of validated recipes and forecasts of future dates
export interface Recipe {
  id: string;
  name: string;
  createdAt: string;
  colorIndex: number;            // Fixed palette slot of the recipe in the Forecast space
  model: ModelConfig;            // Type and parameters to retrain
  dataset: { filename: string; dateColumn: string; targetColumn: string; frequency: string };
  validation: {
    horizon: number;
    trainingRanges: DateRange[];
    predictionRanges: DateRange[];
    points: number;
    rmse: number;
    mae: number;
    r2: number;
    gain: number;                // vs naive forecast (0.2 = 20% less error)
  };
  // Quantiles of |forecast - actual| on the validation period, per step ahead (conformal intervals)
  intervals: { step: number; q80: number; q95: number }[];
  source: { modelId: string; rmse: number };  // To know if a result is already saved
}

export interface FuturePoint {
  date: string;
  prediction: number | null;
  step: number;
}

export interface ForecastOutput {
  frequency: string;
  lastDate: string;
  steps: number;
  results: { recipeId: string; name: string; forecast: FuturePoint[]; warning?: string; error?: string }[];
}
