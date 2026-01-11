/**
 * Feature name formatter for cleaner display in UI.
 */
export function formatFeatureName(feature: string): string {
  // Handle lag features
  if (feature.match(/^target_lag_\d+$/)) {
    return feature.replace('target_lag_', 'Lag ');
  }
  if (feature.match(/_lag_\d+$/)) {
    return feature.replace(/_lag_(\d+)$/, ' (t-$1)');
  }
  
  // Handle temporal features
  const temporalMap: Record<string, string> = {
    'dow_sin': 'Day of week (sin)',
    'dow_cos': 'Day of week (cos)',
    'month_sin': 'Month (sin)',
    'month_cos': 'Month (cos)',
    'hour_sin': 'Hour (sin)',
    'hour_cos': 'Hour (cos)',
    'minute_of_day_sin': 'Minute (sin)',
    'minute_of_day_cos': 'Minute (cos)',
    'day_of_month': 'Day of month',
    'week_of_year': 'Week',
    'year': 'Year',
  };
  
  return temporalMap[feature] || feature;
}

/**
 * Check if a feature has associated SHAP values and return the key.
 */
export function getShapKeyForFeature(feature: string): string | null {
  if (feature.includes('dow_') || feature === 'day_of_week') return 'day_of_week';
  if (feature.includes('month_') && !feature.includes('day_of_month')) return 'month';
  if (feature.includes('hour_')) return 'hour_of_day';
  if (feature.includes('minute_of_day')) return 'minute_of_day';
  return null;
}

/**
 * Format a metric value for display.
 */
export function formatMetric(value: number | undefined | null, decimals: number = 4): string {
  if (value === undefined || value === null || isNaN(value)) return 'N/A';
  return value.toFixed(decimals);
}

/**
 * Format execution time for display.
 */
export function formatTime(seconds: number): string {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${(seconds / 60).toFixed(1)}m`;
}

/**
 * Format percentage for display.
 */
export function formatPercent(value: number, decimals: number = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Get color class based on R² value.
 */
export function getR2ColorClass(r2: number): string {
  if (r2 >= 0.9) return 'text-emerald-400';
  if (r2 >= 0.7) return 'text-amber-400';
  return 'text-red-400';
}

/**
 * Get color class based on MAPE value.
 */
export function getMapeColorClass(mape: number): string {
  if (mape <= 0.1) return 'text-emerald-400';
  if (mape <= 0.25) return 'text-amber-400';
  return 'text-red-400';
}
