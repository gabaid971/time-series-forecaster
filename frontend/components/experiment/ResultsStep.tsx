'use client';

import { useMemo, useState } from 'react';
import { AlertTriangle, ArrowLeft, Loader2, RotateCcw } from 'lucide-react';

import { seriesColor } from '../../lib/modelColors';
import { dateIndex, evaluate, forecastPoints } from '../../lib/results';
import { useAppStore } from '../../lib/store';
import { Button } from '../ui/Button';
import { BestModelBanner } from './results/BestModelBanner';
import { ForecastCard } from './results/ForecastCard';
import { Leaderboard } from './results/Leaderboard';
import { ModelDetail } from './results/ModelDetail';
import { ResultRow } from './results/types';

/** Step 3: which model forecasts best, by how much, and why. */
export default function ResultsStep() {
  const { data, fullData, datasetStats, results, lastRun, selectedModels, forecastHorizon, isTraining, trainError, mode, setStep } = useAppStore();
  const run = lastRun ?? { models: selectedModels, horizon: forecastHorizon };

  const { rows, failed } = useMemo(() => {
    if (!data) return { rows: [] as ResultRow[], failed: [] as ResultRow[] };
    const dates = fullData.map(row => String(row[data.dateColumn]));
    const values = fullData.map(row => Number(row[data.targetColumn]));
    const index = dateIndex(dates);
    const all: ResultRow[] = results.map((result, i) => {
      const model = run.models.find(m => m.id === result.model_id);
      const ok = !result.error && !!result.metrics;
      const points = ok ? forecastPoints(result, data.dateColumn, data.targetColumn) : [];
      return {
        id: result.model_id,
        name: result.model_name,
        colorIndex: model?.colorIndex ?? i,
        result,
        model,
        points,
        evaluation: ok ? evaluate(points, index, values) : null,
      };
    });
    return {
      rows: all.filter(r => r.evaluation).sort((a, b) => a.result.metrics!.rmse - b.result.metrics!.rmse),
      failed: all.filter(r => !r.evaluation),
    };
  }, [data, fullData, results, run.models]);

  const best = rows[0];
  const [selection, setSelection] = useState<string | null>(null);
  const [visibleIds, setVisibleIds] = useState<Set<string> | null>(null);
  const [view, setView] = useState<'forecast' | 'errors'>('forecast');
  const selectedId = rows.some(r => r.id === selection) ? selection : best?.id ?? null;
  const visible = visibleIds ?? new Set(selectedId ? [selectedId] : []);
  const selected = rows.find(r => r.id === selectedId);

  const select = (id: string) => {
    setSelection(id);
    setVisibleIds(new Set(Array.from(visible).concat(id)));
  };
  const toggle = (id: string) => {
    const next = new Set(visible);
    if (next.has(id)) next.delete(id); else next.add(id);
    setVisibleIds(next);
  };

  // A near-zero series makes percentage errors explode
  const minAbs = Math.min(...rows.flatMap(r => r.points.map(p => Math.abs(p.actual))));
  const mapeUnreliable = (datasetStats?.value_min ?? 1) <= 0 || minAbs < 1e-6;

  if (isTraining) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <Loader2 size={40} className="animate-spin text-accent-text mb-5" />
        <h2 className="text-xl font-semibold text-fg mb-2">Training {run.models.length} model{run.models.length > 1 ? 's' : ''}…</h2>
        <p className="text-fg-muted mb-6">Each model learns on the training period, then forecasts the validation period.</p>
        <ul className="flex flex-wrap justify-center gap-2">
          {run.models.map(m => (
            <li key={m.id} className="inline-flex items-center gap-2 rounded-full border border-ink/10 px-3 py-1 text-sm text-fg-muted">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: seriesColor(m.colorIndex) }} aria-hidden /> {m.name}
            </li>
          ))}
        </ul>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {trainError && (
        <div role="alert" className="flex items-start gap-3 rounded-xl border border-negative/30 bg-negative/10 px-4 py-3 text-sm">
          <AlertTriangle size={18} className="text-negative shrink-0 mt-0.5" />
          <p className="text-fg">{trainError}</p>
        </div>
      )}

      {!trainError && rows.length === 0 && (
        <div className="rounded-xl border border-ink/10 bg-panel px-5 py-10 text-center text-fg-muted">
          {failed.length > 0 ? 'All models failed: see the errors below.' : 'No results yet.'}
        </div>
      )}

      {best && <BestModelBanner best={best} horizon={run.horizon} />}

      {(rows.length > 0 || failed.length > 0) && (
        <Leaderboard
          rows={rows}
          failed={failed}
          selectedId={selectedId}
          onSelect={select}
          advanced={mode === 'advanced'}
          mapeUnreliable={mapeUnreliable}
        />
      )}

      {data && rows.length > 0 && (
        <ForecastCard
          rows={rows}
          visible={visible}
          onToggle={toggle}
          selectedId={selectedId}
          view={view}
          onViewChange={setView}
          dateColumn={data.dateColumn}
        />
      )}

      {selected && (
        <ModelDetail row={selected} horizon={run.horizon} frequency={datasetStats?.frequency || 'D'} advanced={mode === 'advanced'} />
      )}

      <div className="flex flex-wrap justify-between gap-3 pt-2">
        <Button variant="secondary" onClick={() => setStep(2)}>
          <ArrowLeft size={16} /> Edit configuration
        </Button>
        <Button variant="ghost" onClick={() => setStep(1)}>
          <RotateCcw size={16} /> Change data
        </Button>
      </div>
    </div>
  );
}
