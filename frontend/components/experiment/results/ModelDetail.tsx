'use client';

import { useState } from 'react';
import { BookmarkCheck, BookmarkPlus, Microscope } from 'lucide-react';

import { formatFeatureName } from '../../../lib/formatters';
import { seriesColor } from '../../../lib/modelColors';
import { modelSummary } from '../../../lib/models';
import { buildRecipe } from '../../../lib/recipes';
import { signedPercent } from '../../../lib/results';
import { useAppStore } from '../../../lib/store';
import type { ShapValue } from '../../../types/forecasting';
import { ErrorHistogram, HorizonErrorChart, ShapEffectChart } from '../../charts/DetailCharts';
import { Button } from '../../ui/Button';
import { Card } from '../../ui/Card';
import { Segmented } from '../../ui/fields';
import { formatMetric, ResultRow } from './types';

const SHAP_LABELS: Record<string, string> = {
  month: 'Month', day_of_week: 'Day of week', hour_of_day: 'Hour of day', minute_of_day: 'Minute of day',
};

/** Everything about one model: how its error grows with the horizon, its errors, what drives it. */
export function ModelDetail({ row, horizon, frequency, advanced }: { row: ResultRow; horizon: number; frequency: string; advanced: boolean }) {
  const evaluation = row.evaluation!;
  const { data, datasetStats, lastRun, library, addRecipe, setSpace } = useAppStore();
  const saved = library.some(r => r.source.modelId === row.id && r.source.rmse === row.result.metrics?.rmse);
  const save = () => {
    if (!data || !row.model || !row.result.metrics || !lastRun) return;
    addRecipe(buildRecipe({
      model: row.model,
      name: row.name,
      points: row.points,
      evaluation,
      metrics: row.result.metrics,
      dataset: { filename: data.filename, dateColumn: data.dateColumn, targetColumn: data.targetColumn, frequency: datasetStats?.frequency || 'D' },
      run: lastRun,
      colorIndex: row.colorIndex,
    }));
  };
  const importance = (row.result.feature_importance || []).slice(0, 10);
  const maxImportance = Math.max(...importance.map(f => f.importance), 1e-9);
  const temporal = Object.entries(row.result.shap_analysis?.temporal || {}).filter(([, values]) => values && values.length > 0) as [string, ShapValue[]][];
  const [shapFeature, setShapFeature] = useState<string | null>(null);
  const activeShap = temporal.find(([key]) => key === shapFeature) ?? temporal[0];
  const scale = Math.max(Math.abs(evaluation.rmse), 1e-9);
  const biasNote = Math.abs(evaluation.bias) < 0.05 * scale
    ? 'No systematic bias.'
    : `Tends to ${evaluation.bias > 0 ? 'over' : 'under'}-forecast by ${formatMetric(Math.abs(evaluation.bias))} on average.`;

  return (
    <Card
      title={
        <span className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full" style={{ background: seriesColor(row.colorIndex) }} aria-hidden />
          {row.name}
        </span>
      }
      subtitle={row.model ? modelSummary(row.model, frequency).join(' · ') : undefined}
      icon={<Microscope size={18} />}
      actions={saved ? (
        <Button variant="ghost" size="sm" onClick={() => setSpace('forecast')} title="Open the Forecast space">
          <BookmarkCheck size={16} className="text-positive" /> Saved · Forecast
        </Button>
      ) : (
        <Button
          variant="secondary"
          size="sm"
          onClick={save}
          disabled={!row.model || !lastRun}
          title="Keep this configuration and its validation results to forecast future dates"
        >
          <BookmarkPlus size={16} /> Save to library
        </Button>
      )}
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <p className="text-sm font-medium text-fg">Error by steps ahead</p>
          {horizon > 1 ? (
            <>
              <p className="text-xs text-fg-muted mb-1">
                How the error grows the further ahead the model forecasts, compared with the naive forecast.
              </p>
              <HorizonErrorChart byStep={evaluation.byStep} colorIndex={row.colorIndex} name={row.name} />
            </>
          ) : (
            <p className="mt-2 text-sm text-fg-muted">
              Horizon 1: every forecast is one step ahead. Choose a longer horizon on the Models page to see how the error grows further ahead.
            </p>
          )}
        </div>
        <div>
          <p className="text-sm font-medium text-fg">Distribution of errors</p>
          <p className="text-xs text-fg-muted mb-1">{biasNote} Error vs naive: {signedPercent(-evaluation.gain)}.</p>
          <ErrorHistogram errors={evaluation.errors} bias={evaluation.bias} colorIndex={row.colorIndex} />
        </div>
      </div>

      {/* Table view of the error by step (advanced) */}
      {advanced && horizon > 1 && (
        <details className="mt-4 rounded-lg border border-ink/10">
          <summary className="cursor-pointer px-4 py-2 text-sm text-fg-muted hover:text-fg">Error by step (table)</summary>
          <div className="overflow-x-auto custom-scrollbar">
            <table className="w-full text-sm tabular-nums">
              <thead>
                <tr className="border-y border-ink/10 text-left text-fg-muted">
                  <th className="px-4 py-2 font-medium">Step</th>
                  <th className="px-4 py-2 font-medium text-right">RMSE</th>
                  <th className="px-4 py-2 font-medium text-right">Naive RMSE</th>
                  <th className="px-4 py-2 font-medium text-right">Points</th>
                </tr>
              </thead>
              <tbody>
                {evaluation.byStep.map(s => (
                  <tr key={s.step} className="border-b border-ink/5 last:border-0 text-fg">
                    <td className="px-4 py-1.5">{s.step}</td>
                    <td className="px-4 py-1.5 text-right">{formatMetric(s.rmse)}</td>
                    <td className="px-4 py-1.5 text-right text-fg-muted">{formatMetric(s.naiveRmse)}</td>
                    <td className="px-4 py-1.5 text-right text-fg-muted">{s.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}

      {(importance.length > 0 || temporal.length > 0) && (
        <div className="mt-6 pt-5 border-t border-ink/10 grid gap-6 lg:grid-cols-2">
          {importance.length > 0 && (
            <div>
              <p className="text-sm font-medium text-fg">What the model relies on</p>
              <p className="text-xs text-fg-muted mb-3">Relative importance of each input (top {importance.length}).</p>
              <ul className="space-y-1.5">
                {importance.map(f => (
                  <li key={f.feature} className="grid grid-cols-[9rem_1fr_3.5rem] items-center gap-3 text-sm">
                    <span className="truncate text-fg-muted" title={f.feature}>{formatFeatureName(f.feature)}</span>
                    <span className="h-2.5 rounded-full bg-ink/5 overflow-hidden">
                      <span
                        className="block h-full rounded-full"
                        style={{ width: `${(f.importance / maxImportance) * 100}%`, background: seriesColor(row.colorIndex) }}
                      />
                    </span>
                    <span className="text-right text-fg tabular-nums">{(f.importance * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {activeShap && (
            <div>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-sm font-medium text-fg">Calendar effects</p>
                {temporal.length > 1 && (
                  <Segmented
                    ariaLabel="Calendar feature"
                    options={temporal.map(([key]) => ({ value: key, label: SHAP_LABELS[key] ?? key }))}
                    value={activeShap[0]}
                    onChange={setShapFeature}
                  />
                )}
              </div>
              <p className="text-xs text-fg-muted mb-1">
                Average effect of the {(SHAP_LABELS[activeShap[0]] ?? activeShap[0]).toLowerCase()} on the forecast (SHAP values): above zero raises it, below lowers it.
              </p>
              <ShapEffectChart feature={activeShap[0]} values={activeShap[1]} />
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
