'use client';

import { AlertTriangle, ArrowRight, LineChart } from 'lucide-react';

import { formatDate, formatDuration, formatNumber } from '../../lib/formatters';
import { useAppStore } from '../../lib/store';
import { SeriesChart } from '../charts/SeriesChart';
import { Button } from '../ui/Button';
import { Card } from '../ui/Card';
import { StatTile } from '../ui/StatTile';
import { AutocorrelationCard } from './data/AutocorrelationCard';
import { DataInsights } from './data/DataInsights';
import { DataPreview } from './data/DataPreview';
import { DatasetHeader } from './data/DatasetHeader';
import { UploadPanel } from './data/UploadPanel';

/** Step 1: load a series, check what it looks like and what the analysis found. */
export default function DataStep() {
  const { data, fullData, datasetStats: stats, analyzeError, isAnalyzing, mode, setMode, setStep } = useAppStore();

  if (!data) return <UploadPanel />;

  const dates = fullData.map(row => String(row[data.dateColumn]));
  const values = fullData.map(row => Number(row[data.targetColumn]));
  const missing = stats ? stats.missing_dates + stats.missing_values_target : 0;

  return (
    <div className="space-y-5">
      <DatasetHeader />

      {analyzeError && (
        <div role="alert" className="flex items-start gap-3 rounded-xl border border-warning/30 bg-warning/10 px-4 py-3 text-sm">
          <AlertTriangle size={18} className="text-warning shrink-0 mt-0.5" />
          <p className="text-fg">
            {analyzeError}
            <span className="text-fg-muted"> Only basic statistics computed in the browser are shown.</span>
          </p>
        </div>
      )}

      <Card
        title={data.targetColumn}
        subtitle={stats ? `${formatDate(stats.date_min, stats.frequency)} → ${formatDate(stats.date_max, stats.frequency)}` : undefined}
        icon={<LineChart size={18} />}
        bodyClassName="px-2 sm:px-3 pb-3"
      >
        {fullData.length > 0 ? (
          <SeriesChart dates={dates} values={values} name={data.targetColumn} />
        ) : (
          <div className="h-[340px] flex items-center justify-center text-sm text-fg-muted animate-pulse">
            {isAnalyzing ? 'Analyzing…' : 'No data to display'}
          </div>
        )}
      </Card>

      {stats && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <StatTile
            label="Period"
            value={formatDuration(stats.date_min, stats.date_max)}
            detail={`${formatDate(stats.date_min, stats.frequency)} → ${formatDate(stats.date_max, stats.frequency)}`}
          />
          <StatTile label="Observations" value={stats.total_rows.toLocaleString('en-US')} detail={`${stats.frequency_label} frequency`} />
          <StatTile
            label="Missing"
            value={missing.toLocaleString('en-US')}
            detail={`${stats.missing_dates} dates · ${stats.missing_values_target} values`}
            tone={missing / stats.total_rows > 0.01 ? 'warning' : 'default'}
          />
          <StatTile
            label="Range"
            value={`${formatNumber(stats.value_min, 1)} – ${formatNumber(stats.value_max, 1)}`}
            detail={`mean ${formatNumber(stats.value_mean)}`}
          />
        </div>
      )}

      <DataInsights />

      {mode === 'advanced' ? (
        <>
          <AutocorrelationCard />
          <DataPreview />
        </>
      ) : (
        <p className="text-sm text-fg-muted">
          Autocorrelation charts and a preview of the file are available in{' '}
          <button onClick={() => setMode('advanced')} className="text-accent-text underline underline-offset-2 hover:no-underline">
            advanced mode
          </button>.
        </p>
      )}

      <div className="flex justify-end pt-2">
        <Button variant="primary" size="lg" onClick={() => setStep(2)} disabled={isAnalyzing}>
          Continue to models <ArrowRight size={18} />
        </Button>
      </div>
    </div>
  );
}
