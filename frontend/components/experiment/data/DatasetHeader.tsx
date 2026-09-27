'use client';

import { FileText, RefreshCw } from 'lucide-react';

import { useAppStore } from '../../../lib/store';
import { Button } from '../../ui/Button';

const SELECT = 'w-full rounded-lg border border-ink/15 bg-panel px-3 py-2 text-sm text-fg focus:border-accent focus:outline-none [&>option]:bg-panel';

/** Loaded file, date and target column selection, change file. */
export function DatasetHeader() {
  const { data, rawData, datasetStats, setColumns, clearData } = useAppStore();
  if (!data) return null;

  return (
    <div className="flex flex-col lg:flex-row lg:items-end gap-4">
      <div className="flex items-center gap-3 min-w-0 flex-1">
        <div className="w-11 h-11 rounded-lg bg-accent/15 text-accent-text flex items-center justify-center shrink-0">
          <FileText size={20} />
        </div>
        <div className="min-w-0">
          <h2 className="font-semibold text-fg truncate">{data.filename}</h2>
          <p className="text-sm text-fg-muted">
            {rawData.length.toLocaleString('en-US')} rows · {data.columns.length} columns
            {datasetStats && ` · ${datasetStats.frequency_label}`}
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={clearData} className="ml-auto lg:ml-2 shrink-0">
          <RefreshCw size={14} /> Change file
        </Button>
      </div>

      <div className="grid grid-cols-[2fr_3fr] gap-3 lg:w-[34rem]">
        <label className="text-xs text-fg-muted">
          Date column
          <select value={data.dateColumn} onChange={(e) => setColumns({ dateColumn: e.target.value })} className={`${SELECT} mt-1`}>
            {data.columns.map(col => <option key={col} value={col}>{col}</option>)}
          </select>
        </label>
        <label className="text-xs text-fg-muted">
          Value to forecast
          <select value={data.targetColumn} onChange={(e) => setColumns({ targetColumn: e.target.value })} className={`${SELECT} mt-1`}>
            {data.columns.filter(col => col !== data.dateColumn).map(col => <option key={col} value={col}>{col}</option>)}
          </select>
        </label>
      </div>
    </div>
  );
}
