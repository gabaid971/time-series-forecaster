/**
 * API Configuration and utilities for the Time Series Forecaster frontend.
 */

// Mode: 'direct' = appel direct au backend (contourne limite Vercel, clé exposée)
//       'proxy'  = passe par /api routes Vercel (clé cachée, limite 4.5MB)
export const API_MODE = (process.env.NEXT_PUBLIC_API_MODE || 'direct') as 'direct' | 'proxy';

// URLs et clé API
export const BACKEND_URL = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/$/, '');
export const API_KEY = process.env.NEXT_PUBLIC_API_KEY || '';

/**
 * Get the API URL for a given endpoint based on the configured mode.
 */
export const getApiUrl = (endpoint: string): string => {
  if (API_MODE === 'proxy') {
    return `/api/${endpoint}`;
  }
  return `${BACKEND_URL}/${endpoint}`;
};

/**
 * Get headers for API requests based on the configured mode.
 */
export const getApiHeaders = (): HeadersInit => {
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (API_MODE === 'direct' && API_KEY) {
    headers['X-API-Key'] = API_KEY;
  }
  return headers;
};

/**
 * Analyze a dataset via the backend /analyze endpoint.
 */
export async function analyzeDataset(
  data: any[],
  dateColumn: string,
  targetColumn: string
): Promise<{
  status: string;
  stats?: {
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
  };
  normalized_data?: { date: string; value: number }[];
  available_columns?: Array<{
    name: string;
    dtype: 'numeric' | 'string' | 'date' | 'boolean';
    missing_count: number;
    sample_values: any[];
  }>;
  message?: string;
}> {
  const response = await fetch(getApiUrl('analyze'), {
    method: 'POST',
    headers: getApiHeaders(),
    body: JSON.stringify({
      data,
      date_column: dateColumn,
      target_column: targetColumn,
    }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Train models via the backend /train endpoint.
 */
export async function trainModels(request: {
  data: any[];
  data_config: {
    target_column: string;
    date_column: string;
    frequency: string;
    training_ranges: { start: string; end: string }[];
    prediction_ranges: { start: string; end: string }[];
    forecast_strategy?: {
      horizon: number;
      mode: string;
    };
  };
  models: Array<{
    id: string;
    type: string;
    name: string;
    params: any;
  }>;
}): Promise<{
  status: string;
  results: Array<{
    model_id: string;
    model_name: string;
    metrics: {
      rmse: number;
      mae: number;
      mape: number;
      r2: number;
      msle: number;
      execution_time: number;
    };
    forecast: any[];
    metrics_by_horizon?: Array<{
      horizon_step: number;
      rmse: number;
      mae: number;
      mape: number;
      msle: number;
      count: number;
    }>;
    feature_importance?: Array<{ feature: string; importance: number }>;
    shap_analysis?: any;
    error?: string;
  }>;
  message?: string;
}> {
  const response = await fetch(getApiUrl('train'), {
    method: 'POST',
    headers: getApiHeaders(),
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

// Debug logging (client-side only)
if (typeof window !== 'undefined') {
  console.log('🔗 API Mode:', API_MODE);
  console.log('🔗 Backend URL:', BACKEND_URL);
}
