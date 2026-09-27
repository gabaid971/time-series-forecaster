'use client';

import { Table } from 'lucide-react';

import { useAppStore } from '../../../lib/store';
import { Card } from '../../ui/Card';

const PREVIEW_ROWS = 10;

/** First rows of the file, as loaded (advanced mode). */
export function DataPreview() {
  const { data, rawData } = useAppStore();
  if (!data) return null;

  // Numbers are right-aligned (header included) so digits line up
  const numeric = new Set(data.columns.filter(col => rawData.slice(0, PREVIEW_ROWS).some(row => typeof row[col] === 'number')));
  const role = (col: string) =>
    col === data.dateColumn ? 'date' : col === data.targetColumn ? 'target' : null;

  return (
    <Card
      title="Data preview"
      subtitle={`First ${Math.min(PREVIEW_ROWS, rawData.length)} of ${rawData.length.toLocaleString('en-US')} rows, as read from the file`}
      icon={<Table size={18} />}
      bodyClassName="pt-3"
    >
      <div className="overflow-x-auto custom-scrollbar">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-ink/10 text-left">
              {data.columns.map(col => (
                <th key={col} className={`px-4 sm:px-5 py-2 font-medium text-fg-muted whitespace-nowrap ${numeric.has(col) ? 'text-right' : ''}`}>
                  {col}
                  {role(col) && (
                    <span className={`ml-2 text-xs px-1.5 py-0.5 rounded ${
                      role(col) === 'target' ? 'bg-accent/15 text-accent-text' : 'bg-info/15 text-info'
                    }`}>
                      {role(col)}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rawData.slice(0, PREVIEW_ROWS).map((row, i) => (
              <tr key={i} className="border-b border-ink/5 last:border-0">
                {data.columns.map(col => (
                  <td
                    key={col}
                    className={`px-4 sm:px-5 py-2 whitespace-nowrap tabular-nums ${
                      numeric.has(col) ? 'text-right text-fg' : 'text-fg-muted'
                    }`}
                  >
                    {String(row[col] ?? '')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
