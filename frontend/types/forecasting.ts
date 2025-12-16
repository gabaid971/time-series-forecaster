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
  params: ArimaParams | ProphetParams | MLParams | LinearRegressionParams;
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
  country_holidays?: string;
}

// Pour les modèles ML (XGBoost, RF, LightGBM)
export interface MLParams {
  lags: number[]; // ex: [1, 7, 30]
  rolling_features: {
    window_size: number;
    operation: 'mean' | 'std' | 'min' | 'max';
  }[];
  use_exogenous: boolean;
}

// Pour Linear Regression
export interface LinearRegressionParams {
  lags: number[];
  exogenous_features?: string[];
  target_mode?: 'raw' | 'residual';
  residual_lag?: number;
  standardize?: boolean;
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
}

export interface TrainingResponse {
  status: 'success' | 'error';
  results: ModelResult[];
}
