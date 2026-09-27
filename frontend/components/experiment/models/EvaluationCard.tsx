'use client';

import { Plus, Scissors, Trash2 } from 'lucide-react';

import { formatDate, formatDuration } from '../../../lib/formatters';
import { countInRanges, indexAtOrAfter, inputValue } from '../../../lib/ranges';
import { useAppStore } from '../../../lib/store';
import type { DateRange } from '../../../types/forecasting';
import { SplitChart } from '../../charts/SplitChart';
import { Card } from '../../ui/Card';
import { Field, Segmented } from '../../ui/fields';

const INTRADAY = new Set(['s', 'min', 'H']);

const UNITS: Record<string, [string, string]> = {
  s: ['second', 'seconds'], min: ['minute', 'minutes'], H: ['hour', 'hours'],
  D: ['day', 'days'], W: ['week', 'weeks'], M: ['month', 'months'],
};

const HORIZON_PRESETS: Record<string, number[]> = {
  s: [1, 10, 60], min: [1, 15, 60], H: [1, 6, 24, 48], D: [1, 7, 14, 30], W: [1, 4, 13], M: [1, 3, 6, 12],
};

const SPLIT_PRESETS = [0.1, 0.2, 0.3];

const plural = (n: number, frequency: string) => {
  const [one, many] = UNITS[frequency] || ['step', 'steps'];
  return `${n} ${n === 1 ? one : many}`;
};

/** How models are evaluated: training / validation split and forecast horizon. */
export function EvaluationCard({ validationPoints }: { validationPoints: number }) {
  const {
    data, fullData, datasetStats, trainingRanges, predictionRanges, forecastHorizon, mode,
    setSplit, setTrainingRanges, setPredictionRanges, setForecastHorizon,
  } = useAppStore();
  if (!data || fullData.length === 0) return null;

  const frequency = datasetStats?.frequency || 'D';
  const intraday = INTRADAY.has(frequency);
  const dates = fullData.map(row => String(row[data.dateColumn]));
  const values = fullData.map(row => Number(row[data.targetColumn]));
  const trainingPoints = countInRanges(dates, trainingRanges, false);
  const splitIndex = predictionRanges[0] ? indexAtOrAfter(dates, predictionRanges[0].start) : Math.floor(dates.length * 0.8);
  const validationShare = 1 - splitIndex / dates.length;
  const simpleSplit = trainingRanges.length === 1 && predictionRanges.length === 1;
  const maxHorizon = Math.max(1, validationPoints);

  const presets = (HORIZON_PRESETS[frequency] || [1, 7, 30]).filter(h => h <= maxHorizon);
  const unit = UNITS[frequency]?.[1] || 'steps';

  return (
    <Card
      title="Evaluation"
      subtitle="Models learn on the training period, then forecast the validation period they have never seen."
      icon={<Scissors size={18} />}
    >
      <SplitChart dates={dates} values={values} training={trainingRanges} validation={predictionRanges} />

      <div className="mt-4 grid gap-6 lg:grid-cols-2">
        {/* Split */}
        <div className="space-y-3">
          <Field
            label="Validation period"
            help="The last part of the series is kept aside to measure how well each model forecasts data it did not learn from."
          >
            {simpleSplit ? (
              <>
                <div className="flex flex-wrap items-center gap-2 mb-3">
                  {SPLIT_PRESETS.map(share => (
                    <button
                      key={share}
                      type="button"
                      onClick={() => setSplit(dates[Math.floor(dates.length * (1 - share))])}
                      className={`rounded-full border px-3 py-1 text-sm transition-colors ${
                        Math.abs(validationShare - share) < 0.005
                          ? 'border-accent/60 bg-accent/15 text-accent-text font-medium'
                          : 'border-ink/15 text-fg-muted hover:text-fg'
                      }`}
                    >
                      Last {share * 100}%
                    </button>
                  ))}
                </div>
                <input
                  type="range"
                  min={Math.floor(dates.length * 0.3)}
                  max={dates.length - 2}
                  value={splitIndex}
                  onChange={(e) => setSplit(dates[parseInt(e.target.value, 10)])}
                  className="w-full cursor-pointer accent-accent"
                  aria-label="Start of the validation period"
                />
              </>
            ) : (
              <p className="text-sm text-fg-muted">Custom periods (edit them below).</p>
            )}
          </Field>
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-sm">
            <dt className="flex items-center gap-2 text-fg-muted"><span className="h-2.5 w-2.5 rounded-sm bg-accent" />Training</dt>
            <dd className="text-fg">
              {trainingPoints.toLocaleString('en-US')} points
              {trainingRanges[0] && <span className="text-fg-muted"> · {formatDate(trainingRanges[0].start, frequency)} → {formatDate(trainingRanges[trainingRanges.length - 1].end, frequency)}</span>}
            </dd>
            <dt className="flex items-center gap-2 text-fg-muted"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: 'rgb(var(--series-1))' }} />Validation</dt>
            <dd className="text-fg">
              {validationPoints.toLocaleString('en-US')} points
              {predictionRanges[0] && (
                <span className="text-fg-muted"> · {formatDuration(predictionRanges[0].start, predictionRanges[predictionRanges.length - 1].end)}</span>
              )}
            </dd>
          </dl>
        </div>

        {/* Horizon */}
        <div className="space-y-3">
          <Field
            label="Forecast horizon"
            help="How far ahead each forecast looks. The validation period is cut into blocks of this length; each block is forecast from the data just before it, then the model gets the actual values back."
          >
            <div className="flex flex-wrap items-center gap-3">
              {presets.length > 0 && (
                <Segmented
                  ariaLabel="Horizon presets"
                  options={presets.map(h => ({ value: h, label: String(h) }))}
                  value={presets.includes(forecastHorizon) ? forecastHorizon : -1}
                  onChange={setForecastHorizon}
                />
              )}
              <label className="flex items-center gap-2 text-sm text-fg-muted">
                <input
                  type="number"
                  min={1}
                  max={maxHorizon}
                  value={forecastHorizon}
                  onChange={(e) => setForecastHorizon(Math.max(1, Math.min(maxHorizon, parseInt(e.target.value, 10) || 1)))}
                  className="w-20 rounded-lg border border-ink/15 bg-panel px-2 py-1 text-fg tabular-nums focus:border-accent focus:outline-none"
                />
                {unit}
              </label>
            </div>
          </Field>
          <p className="text-sm text-fg-muted">
            {forecastHorizon === 1
              ? `Each forecast predicts the next ${UNITS[frequency]?.[0] || 'step'}, knowing all values up to the one before.`
              : `Each forecast predicts the next ${plural(forecastHorizon, frequency)} at once. Errors are also reported for each step ahead (1 to ${forecastHorizon}).`}
          </p>
        </div>
      </div>

      {/* Exact periods (advanced) */}
      {mode === 'advanced' && (
        <div className="mt-5 pt-4 border-t border-ink/10 grid gap-4 lg:grid-cols-2">
          <RangeEditor
            title="Training periods"
            ranges={trainingRanges}
            onChange={setTrainingRanges}
            intraday={intraday}
            note="End excluded"
          />
          <RangeEditor
            title="Validation periods"
            ranges={predictionRanges}
            onChange={setPredictionRanges}
            intraday={intraday}
            note="End included"
          />
        </div>
      )}
    </Card>
  );
}

function RangeEditor({ title, ranges, onChange, intraday, note }: {
  title: string; ranges: DateRange[]; onChange: (ranges: DateRange[]) => void; intraday: boolean; note: string;
}) {
  const update = (index: number, key: keyof DateRange, value: string) =>
    onChange(ranges.map((r, i) => (i === index ? { ...r, [key]: value } : r)));
  const INPUT = 'rounded-lg border border-ink/15 bg-panel px-2 py-1 text-sm text-fg focus:border-accent focus:outline-none';

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-medium text-fg-muted">{title} <span className="text-fg-subtle font-normal">({note})</span></p>
        <button
          type="button"
          onClick={() => onChange([...ranges, { start: '', end: '' }])}
          className="inline-flex items-center gap-1 text-xs text-accent-text hover:underline"
        >
          <Plus size={12} /> Add period
        </button>
      </div>
      <div className="space-y-2">
        {ranges.map((range, i) => (
          <div key={i} className="flex flex-wrap items-center gap-2">
            <input type={intraday ? 'datetime-local' : 'date'} value={inputValue(range.start, intraday)} onChange={(e) => update(i, 'start', e.target.value)} className={INPUT} aria-label={`${title} ${i + 1} start`} />
            <span className="text-fg-subtle">→</span>
            <input type={intraday ? 'datetime-local' : 'date'} value={inputValue(range.end, intraday)} onChange={(e) => update(i, 'end', e.target.value)} className={INPUT} aria-label={`${title} ${i + 1} end`} />
            {ranges.length > 1 && (
              <button type="button" onClick={() => onChange(ranges.filter((_, j) => j !== i))} className="text-fg-subtle hover:text-negative" aria-label="Remove period">
                <Trash2 size={14} />
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
