export type Frequency = 'D' | 'W' | 'M' | 'H';

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
  
  // Config spécifique (Union type)
  params: LagParams | ArimaParams | ProphetParams | XGBoostParams | MLParams | LinearRegressionParams;
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

// Pour les modèles ML génériques (RF, LightGBM)
export interface MLParams {
  lags: number[]; // ex: [1, 7, 30]
  rolling_features: {
    window_size: number;
    operation: 'mean' | 'std' | 'min' | 'max';
  }[];
  use_exogenous: boolean;
}

// Feature configuration (new unified structure)
export interface ExogenousFeatureConfig {
  column: string;
  lags: number[];           // Lag values to create
  use_actual: boolean;      // Use actual value at prediction time
  delta_lag?: number;       // Compute delta vs this lag
  pct_change_lag?: number;  // Compute % change vs this lag
}

export interface TemporalFeatureConfig {
  month: boolean;           // Cyclical encoded month
  day_of_week: boolean;     // Cyclical encoded day of week
  day_of_month: boolean;    // Day of month (1-31)
  week_of_year: boolean;    // Week of year (1-52)
  year: boolean;            // Year as numeric
}

export interface FeatureConfig {
  target_lags: number[];
  temporal: TemporalFeatureConfig;
  exogenous: ExogenousFeatureConfig[];
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

export interface TrainingRequest {
  data: any[]; // Raw CSV data
  data_config: {
    target_column: string;
    date_column: string;
    frequency: Frequency;
    training_ranges: DateRange[];
    prediction_ranges: DateRange[];
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
  execution_time: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ModelResult {
  model_id: string;
  model_name: string;
  metrics: ModelMetrics;
  forecast: ForecastPoint[];
  feature_importance?: FeatureImportance[];
  error?: string;  // Error message if model training failed
}

export interface TrainingResponse {
  status: 'success' | 'error';
  results: ModelResult[];
}
