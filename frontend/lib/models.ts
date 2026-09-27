/**
 * Model catalog, smart defaults and helpers to read/update model parameters.
 */

import type {
  ExogenousFeatureConfig,
  FeatureConfig,
  LagAnalysis,
  ModelConfig,
  ModelType,
  TemporalFeatureConfig,
} from '../types/forecasting';

export type Row = Record<string, unknown>;

export interface CatalogEntry {
  type: ModelType;
  name: string;
  description: string;
  whenToUse: string;
  tabular: boolean;  // Uses lags / calendar / exogenous features
}

export const MODEL_CATALOG: CatalogEntry[] = [
  {
    type: 'LAG', name: 'Naive baseline', tabular: false,
    description: 'Repeats the value observed a few steps earlier.',
    whenToUse: 'Always: the reference every model must beat.',
  },
  {
    type: 'LINEAR_REGRESSION', name: 'Linear regression', tabular: true,
    description: 'Weighted sum of past values and calendar features.',
    whenToUse: 'Fast, robust, explainable. Extrapolates trends.',
  },
  {
    type: 'XGBOOST', name: 'XGBoost', tabular: true,
    description: 'Gradient boosted trees on the same features.',
    whenToUse: 'Non-linear effects and interactions. Cannot extrapolate beyond seen values.',
  },
  {
    type: 'ARIMA', name: 'ARIMA', tabular: false,
    description: 'Classic statistical model of the series dynamics.',
    whenToUse: 'Short-term forecasts of a single series, trends via differencing.',
  },
  {
    type: 'PROPHET', name: 'Prophet', tabular: false,
    description: 'Trend plus yearly / weekly / daily seasonality.',
    whenToUse: 'Strong calendar seasonality over long periods.',
  },
];

export const catalogEntry = (type: ModelType) => MODEL_CATALOG.find(m => m.type === type)!;

export const TEMPORAL_FEATURES: { key: keyof TemporalFeatureConfig; label: string; frequencies: string[] }[] = [
  { key: 'minute_of_day', label: 'Minute of day', frequencies: ['s', 'min'] },
  { key: 'hour_of_day', label: 'Hour of day', frequencies: ['s', 'min', 'H'] },
  { key: 'day_of_week', label: 'Day of week', frequencies: ['min', 'H', 'D'] },
  { key: 'day_of_month', label: 'Day of month', frequencies: ['D'] },
  { key: 'week_of_year', label: 'Week of year', frequencies: ['D', 'W'] },
  { key: 'month', label: 'Month', frequencies: ['D', 'W', 'M'] },
  { key: 'year', label: 'Year', frequencies: ['D', 'W', 'M'] },
];

/** Calendar features that make sense for a frequency. */
export const temporalFeaturesFor = (frequency: string) =>
  TEMPORAL_FEATURES.filter(f => f.frequencies.includes(frequency));

const EMPTY_TEMPORAL: TemporalFeatureConfig = {
  month: false, day_of_week: false, day_of_month: false, week_of_year: false, year: false, hour_of_day: false, minute_of_day: false,
};

/** Feature configuration of a tabular model (lags live in params.lags, not here). */
export function featureConfigOf(params: Row): FeatureConfig {
  const fc = params.feature_config as Partial<FeatureConfig> | undefined;
  return {
    target_lags: (params.lags as number[]) || fc?.target_lags || [1],
    temporal: { ...EMPTY_TEMPORAL, ...(fc?.temporal || {}) },
    exogenous: fc?.exogenous || [],
    derived: fc?.derived || [],
  };
}

/** Parameters of a new model, from what the analysis found. */
export function defaultParams(type: ModelType, defaultLags: number[], analysis: LagAnalysis | null): Row {
  const temporal = { ...EMPTY_TEMPORAL };
  (analysis?.suggested_temporal || []).forEach(key => {
    if (key in temporal) temporal[key as keyof TemporalFeatureConfig] = true;
  });
  const features = { temporal, exogenous: [], derived: [] };
  const cycles = new Set((analysis?.seasonalities || []).map(s => s.period_label));

  switch (type) {
    case 'LAG':
      return { lag: 1 };
    case 'LINEAR_REGRESSION':
      return { lags: [...defaultLags], feature_config: features };
    case 'XGBOOST':
      return { lags: [...defaultLags], n_estimators: 100, max_depth: 3, learning_rate: 0.1, feature_config: features };
    case 'ARIMA':
      return { p: 1, d: 1, q: 1 };
    case 'PROPHET':
      // Seasonalities of the detected cycles (Prophet's own defaults when nothing was detected)
      return cycles.size > 0
        ? { yearly_seasonality: cycles.has('Yearly'), weekly_seasonality: cycles.has('Weekly'), daily_seasonality: cycles.has('Daily'), seasonality_mode: 'additive' }
        : { yearly_seasonality: true, weekly_seasonality: true, daily_seasonality: false, seasonality_mode: 'additive' };
    default:
      return {};
  }
}

/** Models added by "Add recommended models". */
export function recommendedTypes(frequency: string): ModelType[] {
  const types: ModelType[] = ['LAG', 'LINEAR_REGRESSION', 'XGBOOST', 'ARIMA'];
  // Prophet is built for calendar seasonality of daily-ish data
  if (['H', 'D', 'W', 'M'].includes(frequency)) types.push('PROPHET');
  return types;
}

/** Exogenous lags that would leak future values with this horizon. */
export const shortExogenousLags = (exog: ExogenousFeatureConfig, horizon: number) =>
  exog.known_in_advance ? [] : exog.lags.filter(lag => lag < horizon);

/** Problems that prevent a meaningful training, shown before launching it. */
export function configurationIssues(models: ModelConfig[], horizon: number, validationPoints: number): string[] {
  const issues: string[] = [];
  if (models.length === 0) issues.push('Add at least one model.');
  if (validationPoints === 0) issues.push('The validation period contains no data.');
  else if (horizon > validationPoints) issues.push(`The horizon (${horizon}) is longer than the validation period (${validationPoints} points).`);
  models.forEach(model => {
    if (!catalogEntry(model.type)?.tabular) return;
    featureConfigOf(model.params as unknown as Row).exogenous.forEach(exog => {
      const short = shortExogenousLags(exog, horizon);
      if (short.length > 0) {
        issues.push(`${model.name}: "${exog.column}" lag ${short.join(', ')} < horizon ${horizon} would use unknown future values.`);
      }
    });
  });
  return issues;
}

/** Key settings of a model in plain words ("Lags 1, 7", "Month", "Order (1, 1, 1)"). */
export function modelSummary(model: ModelConfig, frequency: string): string[] {
  const params = model.params as unknown as Row;
  const features = featureConfigOf(params);
  const items: string[] = [];
  const lag = (params.lag as number) ?? 1;

  if (model.type === 'LAG') items.push(`Value ${lag} step${lag === 1 ? '' : 's'} before`);
  if (catalogEntry(model.type).tabular) {
    items.push(`Lags ${((params.lags as number[]) || []).join(', ') || 'none'}`);
    temporalFeaturesFor(frequency).filter(f => features.temporal[f.key]).forEach(f => items.push(f.label));
    features.exogenous.forEach(e => items.push(`${e.column} (lag ${e.lags.join(', ')})`));
    if (params.target_mode === 'residual') items.push('Residual target');
  }
  if (model.type === 'XGBOOST') items.push(`${params.n_estimators ?? 100} trees, depth ${params.max_depth ?? 3}`);
  if (model.type === 'ARIMA') items.push(`Order (${params.p ?? 1}, ${params.d ?? 1}, ${params.q ?? 1})`);
  if (model.type === 'PROPHET') {
    const seasons = (['yearly', 'weekly', 'daily'] as const).filter(s => params[`${s}_seasonality`]);
    items.push(seasons.length ? `${seasons.map(s => s[0].toUpperCase() + s.slice(1)).join(' + ')} seasonality` : 'No seasonality');
  }
  return items;
}
