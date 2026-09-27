'use client';

import { AlertTriangle, CalendarRange, Sparkles } from 'lucide-react';

import { formatDate } from '../../lib/formatters';
import { useAppStore } from '../../lib/store';
import { Button } from '../ui/Button';
import { Card } from '../ui/Card';
import { Field, Segmented } from '../ui/fields';

const UNITS: Record<string, string> = { s: 'seconds', min: 'minutes', H: 'hours', D: 'days', W: 'weeks', M: 'months' };
const PRESETS: Record<string, number[]> = { s: [60, 600], min: [60, 240, 1440], H: [24, 48, 168], D: [7, 30, 90, 365], W: [4, 13, 52], M: [3, 6, 12] };
const MS: Record<string, number> = { s: 1e3, min: 6e4, H: 3.6e6, D: 8.64e7, W: 6.048e8 };
const MAX_STEPS = 1000;

/** Number of steps between the last date and a target date, for the data frequency. */
function stepsUntil(lastIso: string, targetIso: string, frequency: string): number {
  const last = new Date(lastIso);
  const target = new Date(targetIso);
  if (frequency === 'M') return (target.getFullYear() - last.getFullYear()) * 12 + target.getMonth() - last.getMonth();
  return Math.round((target.getTime() - last.getTime()) / (MS[frequency] ?? MS.D));
}

/** How far to forecast, the confidence level, and the button. */
export function ForecastSettings({ validatedHorizon, selectedCount }: { validatedHorizon: number | null; selectedCount: number }) {
  const { datasetStats, forecastSteps, intervalLevel, isForecasting, setForecastSteps, setIntervalLevel, runForecast } = useAppStore();
  const frequency = datasetStats?.frequency || 'D';
  const lastDate = datasetStats?.date_max || '';
  const unit = UNITS[frequency] || 'steps';
  const presets = Array.from(new Set([...(validatedHorizon ? [validatedHorizon] : []), ...(PRESETS[frequency] || [7, 30])])).sort((a, b) => a - b);
  const beyond = validatedHorizon !== null && forecastSteps > validatedHorizon;
  const intraday = ['s', 'min', 'H'].includes(frequency);

  return (
    <Card title="Forecast" subtitle={lastDate ? `Starts right after the last date of your data: ${formatDate(lastDate, frequency)}.` : undefined} icon={<CalendarRange size={18} />}>
      <div className="grid gap-5 lg:grid-cols-[1fr_auto] lg:items-end">
        <div className="flex flex-wrap items-end gap-x-6 gap-y-4">
          <Field label={`How many ${unit}`}>
            <div className="flex flex-wrap items-center gap-3">
              <Segmented
                ariaLabel="Forecast length presets"
                options={presets.map(p => ({ value: p, label: p === validatedHorizon ? `${p} ✓` : String(p) }))}
                value={presets.includes(forecastSteps) ? forecastSteps : -1}
                onChange={setForecastSteps}
              />
              <input
                type="number"
                min={1}
                max={MAX_STEPS}
                value={forecastSteps}
                onChange={(e) => setForecastSteps(Math.max(1, Math.min(MAX_STEPS, parseInt(e.target.value, 10) || 1)))}
                className="w-24 rounded-lg border border-ink/15 bg-panel px-2 py-1 text-fg tabular-nums focus:border-accent focus:outline-none"
                aria-label={`Number of ${unit}`}
              />
            </div>
          </Field>
          {lastDate && (
            <Field label="Or until">
              <input
                type={intraday ? 'datetime-local' : 'date'}
                onChange={(e) => {
                  const steps = stepsUntil(lastDate, e.target.value, frequency);
                  if (steps >= 1) setForecastSteps(Math.min(MAX_STEPS, steps));
                }}
                className="rounded-lg border border-ink/15 bg-panel px-2 py-1 text-sm text-fg focus:border-accent focus:outline-none"
                aria-label="Forecast until"
              />
            </Field>
          )}
          <Field label="Confidence interval" help="Range that should contain the actual value with this probability, based on the errors made on the validation period.">
            <Segmented
              ariaLabel="Confidence level"
              options={[{ value: '0.8', label: '80%' }, { value: '0.95', label: '95%' }]}
              value={String(intervalLevel)}
              onChange={(v) => setIntervalLevel(v === '0.95' ? 0.95 : 0.8)}
            />
          </Field>
        </div>
        <Button variant="primary" size="lg" onClick={runForecast} disabled={selectedCount === 0 || isForecasting}>
          <Sparkles size={16} />
          {isForecasting ? 'Forecasting…'
            : selectedCount === 0 ? 'Select a compatible recipe'
            : `Forecast with ${selectedCount} recipe${selectedCount > 1 ? 's' : ''}`}
        </Button>
      </div>
      {beyond && (
        <p className="mt-4 flex items-start gap-2 text-sm text-fg">
          <AlertTriangle size={16} className="mt-0.5 shrink-0 text-warning" />
          <span>
            The selected recipes were validated up to {validatedHorizon} {unit} ahead (✓). Beyond that, the forecast is shown without an interval: its reliability was not measured.
          </span>
        </p>
      )}
    </Card>
  );
}
