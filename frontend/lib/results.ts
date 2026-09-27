/**
 * Result analysis done in the browser: naive reference, gain, bias, errors by step.
 *
 * The naive reference repeats the last value known at the forecast origin: for a
 * point forecast k steps ahead, it predicts the actual value k rows earlier. It is
 * computed on exactly the points each model forecast, so the comparison is fair.
 */

import { isoKey } from './ranges';
import type { ModelResult } from '../types/forecasting';

export interface ForecastPoint {
  date: string;
  actual: number;
  prediction: number;
  step: number;
}

/** Forecast rows of a result in a typed shape (the API keys them by column name). */
export function forecastPoints(result: ModelResult, dateColumn: string, targetColumn: string): ForecastPoint[] {
  return (result.forecast as unknown as Record<string, unknown>[]).map(row => ({
    date: String(row[dateColumn]),
    actual: Number(row[targetColumn]),
    prediction: Number(row.prediction),
    step: Number(row.horizon_step ?? 1),
  }));
}

export const rmse = (errors: number[]) =>
  errors.length ? Math.sqrt(errors.reduce((sum, e) => sum + e * e, 0) / errors.length) : NaN;

export const mean = (values: number[]) =>
  values.length ? values.reduce((sum, v) => sum + v, 0) / values.length : NaN;

/** Index of each date of the series (normalized ISO keys). */
export function dateIndex(dates: string[]): Map<string, number> {
  return new Map(dates.map((d, i) => [isoKey(d), i]));
}

/** Naive predictions (last known value) for the given points, NaN when unknown. */
export function naivePredictions(points: ForecastPoint[], index: Map<string, number>, values: number[]): number[] {
  return points.map(p => {
    const i = index.get(isoKey(p.date));
    return i !== undefined && i - p.step >= 0 ? values[i - p.step] : NaN;
  });
}

export interface ModelEvaluation {
  rmse: number;
  naiveRmse: number;
  gain: number;          // 1 - rmse / naiveRmse: 0.12 = 12% less error than naive
  bias: number;          // Mean of prediction - actual
  errors: number[];      // prediction - actual, per point
  byStep: { step: number; rmse: number; naiveRmse: number; count: number }[];
}

/** Everything the results page shows about one model, compared with the naive reference. */
export function evaluate(points: ForecastPoint[], index: Map<string, number>, values: number[]): ModelEvaluation {
  const naive = naivePredictions(points, index, values);
  const valid = points.map((_, i) => !Number.isNaN(naive[i]));
  const errors = points.map(p => p.prediction - p.actual);
  const naiveErrors = points.map((p, i) => naive[i] - p.actual);
  const pick = (list: number[]) => list.filter((_, i) => valid[i]);

  const steps = Array.from(new Set(points.map(p => p.step))).sort((a, b) => a - b);
  const byStep = steps.map(step => {
    const idx = points.map((p, i) => (p.step === step && valid[i] ? i : -1)).filter(i => i >= 0);
    return {
      step,
      rmse: rmse(idx.map(i => errors[i])),
      naiveRmse: rmse(idx.map(i => naiveErrors[i])),
      count: idx.length,
    };
  });

  const modelRmse = rmse(errors);
  const naiveRmse = rmse(pick(naiveErrors));
  return {
    rmse: modelRmse,
    naiveRmse,
    gain: naiveRmse > 0 ? 1 - rmse(pick(errors)) / naiveRmse : NaN,
    bias: mean(errors),
    errors,
    byStep,
  };
}

/** Percentage with sign, e.g. "−12.3%" / "+4.0%". */
export const signedPercent = (value: number, decimals = 1) =>
  `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value * 100).toFixed(decimals)}%`;
