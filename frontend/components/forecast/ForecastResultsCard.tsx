'use client';

import { useState } from 'react';
import { AlertTriangle, Download, LineChart } from 'lucide-react';

import { formatDate } from '../../lib/formatters';
import { seriesColor } from '../../lib/modelColors';
import { intervalHalfWidth } from '../../lib/recipes';
import { useAppStore } from '../../lib/store';
import type { Recipe } from '../../types/forecasting';
import { FutureChart, FutureSeries } from '../charts/FutureChart';
import { Button } from '../ui/Button';
import { Card } from '../ui/Card';
import { formatMetric } from '../experiment/results/types';

/** Chart and table of the last forecast, with confidence intervals where validated. */
export function ForecastResultsCard() {
  const { forecastOutput: output, library, intervalLevel, data, fullData } = useAppStore();
  const [focus, setFocus] = useState<string | null>(null);
  if (!output || !data) return null;

  const recipeOf = (id: string) => library.find(r => r.id === id);
  const successful = output.results.filter(r => !r.error && r.forecast.length > 0 && recipeOf(r.recipeId));
  const failed = output.results.filter(r => r.error);
  const focusId = successful.some(r => r.recipeId === focus) ? focus : successful[0]?.recipeId ?? null;
  const frequency = output.frequency;

  // Interval of a recipe at a step: prediction ± validation error quantile (null beyond validation)
  const bounds = (recipe: Recipe, step: number, prediction: number | null) => {
    const half = prediction === null ? null : intervalHalfWidth(recipe, step, intervalLevel);
    return half === null || prediction === null ? null : [prediction - half, prediction + half] as const;
  };

  const series: FutureSeries[] = successful.map(r => {
    const recipe = recipeOf(r.recipeId)!;
    const withBand = r.forecast.map(p => ({ p, b: bounds(recipe, p.step, p.prediction) })).filter(x => x.b);
    return {
      id: r.recipeId,
      name: r.name,
      colorIndex: recipe.colorIndex,
      // Start the line at the last observation so it continues the history
      dates: [output.lastDate, ...r.forecast.map(p => p.date)],
      predictions: [Number(fullData[fullData.length - 1]?.[data.targetColumn]), ...r.forecast.map(p => p.prediction)],
      band: withBand.length ? { dates: withBand.map(x => x.p.date), lower: withBand.map(x => x.b![0]), upper: withBand.map(x => x.b![1]) } : undefined,
    };
  });

  // History shown before the forecast: enough context, not the whole series
  const tail = Math.min(fullData.length, Math.max(60, output.steps * 3));
  const historyRows = fullData.slice(-tail);
  const history = {
    dates: historyRows.map(row => String(row[data.dateColumn])),
    values: historyRows.map(row => Number(row[data.targetColumn])),
  };

  const exportCsv = () => {
    const header = ['date', 'step', ...successful.flatMap(r => [r.name, `${r.name} low`, `${r.name} high`])];
    const lines = successful[0]?.forecast.map((p, i) => [
      p.date, p.step,
      ...successful.flatMap(r => {
        const point = r.forecast[i];
        const b = bounds(recipeOf(r.recipeId)!, point.step, point.prediction);
        return [point.prediction ?? '', b?.[0] ?? '', b?.[1] ?? ''];
      }),
    ]) || [];
    const csv = [header, ...lines].map(line => line.map(v => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v))).join(',')).join('\n');
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    link.download = `forecast_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  return (
    <Card
      title={`Next ${output.steps} steps`}
      subtitle={`From ${formatDate(output.lastDate, frequency)}. Shaded: ${Math.round(intervalLevel * 100)}% interval of the highlighted recipe, where it was validated.`}
      icon={<LineChart size={18} />}
      actions={<Button variant="secondary" size="sm" onClick={exportCsv} disabled={successful.length === 0}><Download size={14} /> CSV</Button>}
      bodyClassName="px-2 sm:px-3 pb-3"
    >
      {failed.map(r => (
        <p key={r.recipeId} className="mx-2 mt-3 flex items-start gap-2 text-sm text-negative">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" /> <span><span className="font-medium">{r.name}</span>: {r.error}</span>
        </p>
      ))}

      {successful.filter(r => r.warning).map(r => (
        <p key={r.recipeId} className="mx-2 mt-3 flex items-start gap-2 text-sm text-fg">
          <AlertTriangle size={16} className="mt-0.5 shrink-0 text-warning" /> <span><span className="font-medium">{r.name}</span>: {r.warning}</span>
        </p>
      ))}

      {successful.length > 0 && (
        <>
          {/* Legend: click to highlight a recipe (and see its interval) */}
          <div className="flex flex-wrap items-center gap-2 px-2 pt-3 pb-1">
            <span className="inline-flex items-center gap-2 rounded-full border border-ink/15 px-3 py-1 text-sm text-fg">
              <span className="h-0.5 w-4 rounded bg-fg" aria-hidden /> History
            </span>
            {successful.map(r => {
              const recipe = recipeOf(r.recipeId)!;
              return (
                <button
                  key={r.recipeId}
                  type="button"
                  onClick={() => setFocus(r.recipeId)}
                  aria-pressed={r.recipeId === focusId}
                  className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm transition-colors ${
                    r.recipeId === focusId ? 'border-ink/30 bg-ink/5 text-fg font-medium' : 'border-ink/10 text-fg-muted hover:text-fg'
                  }`}
                >
                  <span className="h-0.5 w-4 rounded" style={{ background: seriesColor(recipe.colorIndex) }} aria-hidden />
                  {r.name}
                </button>
              );
            })}
          </div>
          <FutureChart history={history} series={series} focusId={focusId} lastDate={output.lastDate} />

          <div className="mt-2 max-h-80 overflow-auto custom-scrollbar rounded-lg border border-ink/10 mx-2">
            <table className="w-full text-sm tabular-nums">
              <thead className="sticky top-0 bg-panel">
                <tr className="border-b border-ink/10 text-left text-fg-muted">
                  <th className="px-3 py-2 font-medium">Date</th>
                  {successful.map(r => (
                    <th key={r.recipeId} className="px-3 py-2 font-medium text-right whitespace-nowrap">{r.name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {successful[0].forecast.map((p, i) => (
                  <tr key={p.date} className="border-b border-ink/5 last:border-0">
                    <td className="px-3 py-1.5 text-fg-muted whitespace-nowrap">{formatDate(p.date, frequency)}</td>
                    {successful.map(r => {
                      const point = r.forecast[i];
                      const b = bounds(recipeOf(r.recipeId)!, point.step, point.prediction);
                      return (
                        <td key={r.recipeId} className="px-3 py-1.5 text-right whitespace-nowrap">
                          <span className="text-fg">{formatMetric(point.prediction ?? NaN)}</span>
                          {b && <span className="ml-2 text-xs text-fg-subtle">[{formatMetric(b[0])} – {formatMetric(b[1])}]</span>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  );
}
