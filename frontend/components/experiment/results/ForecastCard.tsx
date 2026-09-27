'use client';

import { Download, LineChart } from 'lucide-react';

import { seriesColor } from '../../../lib/modelColors';
import { ForecastChart } from '../../charts/ForecastChart';
import { Button } from '../../ui/Button';
import { Card } from '../../ui/Card';
import { Segmented } from '../../ui/fields';
import { ResultRow } from './types';

interface ForecastCardProps {
  rows: ResultRow[];
  visible: Set<string>;
  onToggle: (id: string) => void;
  selectedId: string | null;
  view: 'forecast' | 'errors';
  onViewChange: (view: 'forecast' | 'errors') => void;
  dateColumn: string;
}

/** Forecasts of the validation period. Model chips (the legend) show or hide each model. */
export function ForecastCard({ rows, visible, onToggle, selectedId, view, onViewChange, dateColumn }: ForecastCardProps) {
  const shown = rows.filter(r => visible.has(r.id));

  const exportCsv = () => {
    const byDate = new Map<string, Record<string, number | string>>();
    rows.forEach(row => row.points.forEach(p => {
      const line = byDate.get(p.date) ?? { [dateColumn]: p.date, actual: p.actual };
      line[row.name] = p.prediction;
      byDate.set(p.date, line);
    }));
    const header = [dateColumn, 'actual', ...rows.map(r => r.name)];
    const escape = (v: unknown) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v ?? ''));
    const csv = [header, ...Array.from(byDate.values()).sort((a, b) => String(a[dateColumn]).localeCompare(String(b[dateColumn])))
      .map(line => header.map(col => line[col] ?? ''))].map(line => line.map(escape).join(',')).join('\n');
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    link.download = `forecasts_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  return (
    <Card
      title={view === 'forecast' ? 'Forecasts on the validation period' : 'Forecast errors (forecast − actual)'}
      icon={<LineChart size={18} />}
      actions={
        <>
          <Segmented
            ariaLabel="Chart view"
            options={[{ value: 'forecast', label: 'Forecast' }, { value: 'errors', label: 'Errors' }]}
            value={view}
            onChange={onViewChange}
          />
          <Button variant="secondary" size="sm" onClick={exportCsv}>
            <Download size={14} /> CSV
          </Button>
        </>
      }
      bodyClassName="px-2 sm:px-3 pb-3"
    >
      {/* Legend and filter: one row above the chart */}
      <div className="flex flex-wrap items-center gap-2 px-2 pt-3 pb-1">
        {view === 'forecast' && (
          <span className="inline-flex items-center gap-2 rounded-full border border-ink/15 px-3 py-1 text-sm text-fg">
            <span className="h-0.5 w-4 rounded bg-fg" aria-hidden /> Actual
          </span>
        )}
        {rows.map(row => {
          const on = visible.has(row.id);
          return (
            <button
              key={row.id}
              type="button"
              onClick={() => onToggle(row.id)}
              aria-pressed={on}
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm transition-colors ${
                on ? 'border-ink/30 bg-ink/5 text-fg' : 'border-ink/10 text-fg-subtle hover:text-fg-muted'
              } ${row.id === selectedId ? 'font-medium' : ''}`}
            >
              <span
                className="h-0.5 w-4 rounded"
                style={{ background: on ? seriesColor(row.colorIndex) : 'rgb(var(--fg-subtle))' }}
                aria-hidden
              />
              {row.name}
            </button>
          );
        })}
      </div>
      {shown.length > 0 ? (
        <ForecastChart
          view={view}
          series={shown.map(r => ({ id: r.id, name: r.name, colorIndex: r.colorIndex, points: r.points, emphasized: r.id === selectedId }))}
        />
      ) : (
        <p className="h-[380px] flex items-center justify-center text-sm text-fg-muted">Select a model above to display its forecasts.</p>
      )}
    </Card>
  );
}
