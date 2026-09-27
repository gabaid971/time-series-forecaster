'use client';

import { Activity } from 'lucide-react';

import { useAppStore } from '../../../lib/store';
import { CorrelogramChart } from '../../charts/CorrelogramChart';
import { Card } from '../../ui/Card';
import { TEMPORAL_LABELS } from './DataInsights';

/** ACF / PACF with the suggested lags and calendar features (advanced mode). */
export function AutocorrelationCard() {
  const { lagAnalysis, defaultLags } = useAppStore();
  if (!lagAnalysis) return null;

  const markers = (lagAnalysis.seasonalities ?? [])
    .filter(s => s.period < lagAnalysis.acf.length)
    .map(s => ({ lag: s.period, label: s.period_label }));

  return (
    <Card
      title="Autocorrelation"
      subtitle="How much the series resembles itself k steps earlier. Bars outside the shaded band are significant."
      icon={<Activity size={18} />}
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <p className="text-sm font-medium text-fg">ACF</p>
          <p className="text-xs text-fg-muted mb-2">Total correlation: reveals cycles (peaks at their period)</p>
          <CorrelogramChart values={lagAnalysis.acf} confidence={lagAnalysis.confidence_interval} markers={markers} />
        </div>
        <div>
          <p className="text-sm font-medium text-fg">PACF</p>
          <p className="text-xs text-fg-muted mb-2">Direct correlation: suggests which lags to use as features</p>
          <CorrelogramChart values={lagAnalysis.pacf} confidence={lagAnalysis.confidence_interval} />
        </div>
      </div>

      <div className="mt-5 pt-4 border-t border-ink/10 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-fg-muted">Suggested lags</span>
          {defaultLags.map(lag => (
            <span key={lag} className="px-2 py-0.5 rounded-md bg-accent/15 text-accent-text font-medium tabular-nums">{lag}</span>
          ))}
        </div>
        {lagAnalysis.suggested_temporal?.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-fg-muted">Calendar features</span>
            {lagAnalysis.suggested_temporal.map(feature => (
              <span key={feature} className="px-2 py-0.5 rounded-md bg-ink/5 text-fg">{TEMPORAL_LABELS[feature] ?? feature}</span>
            ))}
          </div>
        )}
        <span className="text-xs text-fg-subtle">Used as defaults for new models.</span>
      </div>
    </Card>
  );
}
