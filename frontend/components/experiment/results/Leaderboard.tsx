'use client';

import { AlertTriangle, ListOrdered, Trophy } from 'lucide-react';

import { formatTime } from '../../../lib/formatters';
import { seriesColor } from '../../../lib/modelColors';
import { signedPercent } from '../../../lib/results';
import { Card } from '../../ui/Card';
import { HelpTip } from '../../ui/fields';
import { formatMetric, gainBg, gainText, ResultRow } from './types';

interface LeaderboardProps {
  rows: ResultRow[];            // Successful models, best first
  failed: ResultRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  advanced: boolean;
  mapeUnreliable: boolean;
}

const TH = 'px-3 py-2 font-medium text-fg-muted whitespace-nowrap';
const TD = 'px-3 py-2.5 whitespace-nowrap tabular-nums';

/** Ranking by error, gain vs the naive forecast, click a row to see the model's details. */
export function Leaderboard({ rows, failed, selectedId, onSelect, advanced, mapeUnreliable }: LeaderboardProps) {
  const maxGain = Math.max(0.05, ...rows.map(r => Math.abs(r.evaluation?.gain || 0)));

  return (
    <Card
      title="Leaderboard"
      subtitle="Ranked by error on the validation period (lower is better). Select a model to see its details."
      icon={<ListOrdered size={18} />}
      bodyClassName="pt-3 pb-2"
    >
      <div className="overflow-x-auto custom-scrollbar">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-ink/10 text-left">
              <th className={`${TH} pl-4 sm:pl-5`}>Model</th>
              <th className={TH}>
                <span className="inline-flex items-center gap-1">vs naive <HelpTip>Error compared with a naive forecast that repeats the last known value. −20% means 20% less error.</HelpTip></span>
              </th>
              <th className={`${TH} text-right`}>
                <span className="inline-flex items-center gap-1">RMSE <HelpTip>Root mean squared error, in the unit of the series. Penalizes large errors.</HelpTip></span>
              </th>
              <th className={`${TH} text-right`}>
                <span className="inline-flex items-center gap-1">MAE <HelpTip>Mean absolute error, in the unit of the series: the typical size of an error.</HelpTip></span>
              </th>
              <th className={`${TH} text-right`}>
                <span className="inline-flex items-center gap-1">R² <HelpTip>Share of the variations explained: 1 is perfect, 0 is no better than the average.</HelpTip></span>
              </th>
              {advanced && (
                <>
                  <th className={`${TH} text-right`}>
                    <span className="inline-flex items-center gap-1">
                      MAPE
                      <HelpTip>
                        Mean absolute percentage error.
                        {mapeUnreliable && ' Unreliable here: the series has values at or near zero, where percentages explode.'}
                      </HelpTip>
                      {mapeUnreliable && <AlertTriangle size={13} className="text-warning" aria-label="unreliable" />}
                    </span>
                  </th>
                  <th className={`${TH} text-right`}>MSLE</th>
                  <th className={`${TH} text-right`}>
                    <span className="inline-flex items-center gap-1">Bias <HelpTip>Mean of forecast − actual: positive when the model tends to over-forecast.</HelpTip></span>
                  </th>
                </>
              )}
              <th className={`${TH} text-right pr-4 sm:pr-5`}>Time</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rank) => {
              const metrics = row.result.metrics!;
              const gain = row.evaluation?.gain ?? NaN;
              const selected = row.id === selectedId;
              return (
                <tr
                  key={row.id}
                  onClick={() => onSelect(row.id)}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(row.id); } }}
                  tabIndex={0}
                  aria-selected={selected}
                  className={`cursor-pointer border-b border-ink/5 last:border-0 outline-none transition-colors focus-visible:bg-ink/5 ${
                    selected ? 'bg-accent/10' : 'hover:bg-ink/5'
                  }`}
                >
                  <td className={`${TD} pl-4 sm:pl-5`}>
                    <span className="flex items-center gap-2.5">
                      <span className="w-4 text-xs text-fg-subtle">{rank + 1}</span>
                      <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: seriesColor(row.colorIndex) }} aria-hidden />
                      <span className={`font-medium ${selected ? 'text-fg' : 'text-fg'}`}>{row.name}</span>
                      {rank === 0 && <Trophy size={14} className="text-accent-text" aria-label="best" />}
                    </span>
                  </td>
                  <td className={TD}>
                    <span className="flex items-center gap-2">
                      <span className="relative h-2 w-20 rounded-full bg-ink/10 overflow-hidden" aria-hidden>
                        {!Number.isNaN(gain) && (
                          <span
                            className={`absolute inset-y-0 left-0 rounded-full ${gainBg(gain)}`}
                            style={{ width: `${Math.min(100, (Math.abs(gain) / maxGain) * 100)}%` }}
                          />
                        )}
                      </span>
                      <span className={gainText(gain)}>
                        {Number.isNaN(gain) ? '—' : signedPercent(-gain)}
                      </span>
                    </span>
                  </td>
                  <td className={`${TD} text-right text-fg font-medium`}>{formatMetric(metrics.rmse)}</td>
                  <td className={`${TD} text-right text-fg`}>{formatMetric(metrics.mae)}</td>
                  <td className={`${TD} text-right text-fg`}>{formatMetric(metrics.r2)}</td>
                  {advanced && (
                    <>
                      <td className={`${TD} text-right ${mapeUnreliable ? 'text-fg-subtle' : 'text-fg'}`}>{(metrics.mape * 100).toFixed(1)}%</td>
                      <td className={`${TD} text-right text-fg`}>{formatMetric(metrics.msle)}</td>
                      <td className={`${TD} text-right text-fg`}>{formatMetric(row.evaluation?.bias)}</td>
                    </>
                  )}
                  <td className={`${TD} text-right text-fg-muted pr-4 sm:pr-5`}>{formatTime(metrics.execution_time)}</td>
                </tr>
              );
            })}
            {failed.map(row => (
              <tr key={row.id} className="border-b border-ink/5 last:border-0">
                <td className={`${TD} pl-4 sm:pl-5`}>
                  <span className="flex items-center gap-2.5">
                    <span className="w-4" />
                    <AlertTriangle size={14} className="text-negative shrink-0" aria-label="failed" />
                    <span className="font-medium text-fg">{row.name}</span>
                  </span>
                </td>
                <td colSpan={advanced ? 8 : 5} className="px-3 py-2.5 text-sm text-negative whitespace-normal">
                  {row.result.error}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
