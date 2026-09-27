'use client';

import { Trophy } from 'lucide-react';

import { seriesColor } from '../../../lib/modelColors';
import { signedPercent } from '../../../lib/results';
import { formatMetric, gainText, gainTone, ResultRow } from './types';

/** The headline: which model forecasts best, and how much better than a naive forecast. */
export function BestModelBanner({ best, horizon }: { best: ResultRow; horizon: number }) {
  const evaluation = best.evaluation!;
  const tone = gainTone(evaluation.gain);

  return (
    <section className="rounded-xl border border-accent/30 bg-gradient-to-r from-accent/15 via-accent/5 to-transparent p-5 sm:p-6">
      <div className="flex flex-wrap items-center gap-x-10 gap-y-4">
        <div className="flex items-center gap-4 min-w-0">
          <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-accent text-on-accent shadow-lg shadow-accent/20">
            <Trophy size={22} />
          </span>
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-wider text-accent-text">Best model</p>
            <h2 className="flex items-center gap-2 text-2xl font-bold text-fg">
              <span className="h-3 w-3 shrink-0 rounded-full" style={{ background: seriesColor(best.colorIndex) }} aria-hidden />
              <span className="truncate">{best.name}</span>
            </h2>
          </div>
        </div>

        <div>
          <p className="text-xs text-fg-muted">Error vs naive forecast</p>
          <p className={`text-3xl font-bold ${gainText(evaluation.gain)}`}>
            {Number.isNaN(evaluation.gain) ? '—' : signedPercent(-evaluation.gain)}
          </p>
          <p className="text-xs text-fg-subtle">
            {tone === 'positive' ? 'less error than repeating the last value'
              : tone === 'negative' ? 'more error than repeating the last value'
              : 'about the same error as repeating the last value'}
          </p>
        </div>

        <div className="flex gap-8">
          <div>
            <p className="text-xs text-fg-muted">RMSE</p>
            <p className="text-xl font-semibold text-fg">{formatMetric(best.result.metrics?.rmse)}</p>
          </div>
          <div>
            <p className="text-xs text-fg-muted">R²</p>
            <p className="text-xl font-semibold text-fg">{formatMetric(best.result.metrics?.r2)}</p>
          </div>
          <div>
            <p className="text-xs text-fg-muted">Horizon</p>
            <p className="text-xl font-semibold text-fg">{horizon} step{horizon > 1 ? 's' : ''}</p>
          </div>
        </div>
      </div>
    </section>
  );
}
