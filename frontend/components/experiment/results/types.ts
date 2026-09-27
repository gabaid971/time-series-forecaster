import type { ModelEvaluation, ForecastPoint } from '../../../lib/results';
import type { ModelConfig, ModelResult } from '../../../types/forecasting';

/** One model of the results page: API result + its configuration + browser-side evaluation. */
export interface ResultRow {
  id: string;
  name: string;
  colorIndex: number;
  result: ModelResult;
  model?: ModelConfig;
  points: ForecastPoint[];
  evaluation: ModelEvaluation | null;  // null when the model failed
}

/** Tone of a gain vs the naive forecast: better, worse, or no real difference (< 0.5%). */
export const gainTone = (gain: number): 'positive' | 'negative' | 'neutral' =>
  Number.isNaN(gain) || Math.abs(gain) < 0.005 ? 'neutral' : gain > 0 ? 'positive' : 'negative';

const TONE_TEXT = { positive: 'text-positive', negative: 'text-negative', neutral: 'text-fg-muted' };
const TONE_BG = { positive: 'bg-positive', negative: 'bg-negative', neutral: 'bg-fg-subtle' };
export const gainText = (gain: number) => TONE_TEXT[gainTone(gain)];
export const gainBg = (gain: number) => TONE_BG[gainTone(gain)];

/** Compact display of a metric: 4 significant digits, thousands separators. */
export function formatMetric(value: number | undefined | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  const abs = Math.abs(value);
  const decimals = abs >= 1000 ? 0 : abs >= 100 ? 1 : abs >= 10 ? 2 : 3;
  return value.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}
