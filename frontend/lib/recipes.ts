/**
 * Recipes: validated model configurations saved from the Results page, and the
 * confidence intervals derived from their validation errors.
 */

import { featureConfigOf, Row } from './models';
import type { ForecastPoint, ModelEvaluation } from './results';
import type { DateRange, ModelConfig, Recipe } from '../types/forecasting';

/**
 * Split-conformal quantile: the smallest error bound that covers at least `level`
 * of new errors, assuming they behave like the validation errors.
 */
export function conformalQuantile(absErrors: number[], level: number): number {
  const sorted = absErrors.filter(e => !Number.isNaN(e)).sort((a, b) => a - b);
  if (sorted.length === 0) return NaN;
  const k = Math.ceil((sorted.length + 1) * level);
  return sorted[Math.min(k, sorted.length) - 1];
}

/** Error quantiles per step ahead, from the validation forecasts. */
export function intervalsByStep(points: ForecastPoint[]): Recipe['intervals'] {
  const steps = Array.from(new Set(points.map(p => p.step))).sort((a, b) => a - b);
  return steps.map(step => {
    const errors = points.filter(p => p.step === step).map(p => Math.abs(p.prediction - p.actual));
    return { step, q80: conformalQuantile(errors, 0.8), q95: conformalQuantile(errors, 0.95) };
  });
}

export interface RecipeSource {
  model: ModelConfig;
  name: string;
  points: ForecastPoint[];
  evaluation: ModelEvaluation;
  metrics: { rmse: number; mae: number; r2: number };
  dataset: Recipe['dataset'];
  run: { horizon: number; trainingRanges: DateRange[]; predictionRanges: DateRange[] };
  colorIndex: number;
}

export function buildRecipe(source: RecipeSource): Recipe {
  return {
    id: `recipe-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    name: source.name,
    createdAt: new Date().toISOString(),
    colorIndex: source.colorIndex,
    model: source.model,
    dataset: source.dataset,
    validation: {
      horizon: source.run.horizon,
      trainingRanges: source.run.trainingRanges,
      predictionRanges: source.run.predictionRanges,
      points: source.points.length,
      rmse: source.metrics.rmse,
      mae: source.metrics.mae,
      r2: source.metrics.r2,
      gain: source.evaluation.gain,
    },
    intervals: intervalsByStep(source.points),
    source: { modelId: source.model.id, rmse: source.metrics.rmse },
  };
}

/** Half-width of the interval at a step, or null beyond the validated horizon. */
export function intervalHalfWidth(recipe: Recipe, step: number, level: 0.8 | 0.95): number | null {
  const entry = recipe.intervals.find(i => i.step === step);
  if (!entry) return null;
  const value = level === 0.8 ? entry.q80 : entry.q95;
  return Number.isNaN(value) ? null : value;
}

/** Why a recipe cannot be applied to the loaded dataset (null if it can). */
export function incompatibility(
  recipe: Recipe,
  data: { columns: string[]; targetColumn: string } | null,
  frequency: string | undefined,
): string | null {
  if (!data) return 'Load a dataset in the Experiment space first.';
  const { dateColumn, targetColumn, filename } = recipe.dataset;
  const exogenous = featureConfigOf(recipe.model.params as unknown as Row).exogenous.map(e => e.column);
  const missing = [dateColumn, targetColumn, ...exogenous].filter(c => !data.columns.includes(c));
  if (missing.length > 0) {
    return `Needs column${missing.length > 1 ? 's' : ''} ${missing.map(c => `"${c}"`).join(', ')} (made on ${filename}).`;
  }
  if (frequency && recipe.dataset.frequency && frequency !== recipe.dataset.frequency) {
    return `Validated on ${FREQUENCY_LABELS[recipe.dataset.frequency] ?? recipe.dataset.frequency} data, the loaded data is ${FREQUENCY_LABELS[frequency] ?? frequency}.`;
  }
  return null;
}

const FREQUENCY_LABELS: Record<string, string> = {
  s: 'second', min: 'minute', H: 'hourly', D: 'daily', W: 'weekly', M: 'monthly',
};
