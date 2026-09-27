'use client';

import type { ReactNode } from 'react';
import { AlertTriangle, CheckCircle2, CircleSlash, Repeat, TrendingUp, Waves } from 'lucide-react';

import { useAppStore } from '../../../lib/store';
import { Card } from '../../ui/Card';

export const TEMPORAL_LABELS: Record<string, string> = {
  month: 'month',
  day_of_week: 'day of week',
  day_of_month: 'day of month',
  week_of_year: 'week of year',
  hour_of_day: 'hour of day',
  minute_of_day: 'minute of day',
  year: 'year',
};

const CATEGORY_ICONS: Record<string, typeof Repeat> = {
  trend: TrendingUp,
  outliers: AlertTriangle,
  missing: CircleSlash,
  stationarity: Waves,
};

const capitalize = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

interface Insight {
  key: string;
  Icon: typeof Repeat;
  tone: 'info' | 'warning';
  title: ReactNode;
  detail?: ReactNode;
}

/** What the analysis found, in plain language, with what it means for the models. */
export function DataInsights() {
  const { lagAnalysis, dataAlerts, isAnalyzing } = useAppStore();

  const insights: Insight[] = [
    ...(lagAnalysis?.seasonalities || []).map(s => ({
      key: `season-${s.period}`,
      Icon: Repeat,
      tone: 'info' as const,
      title: `${s.period_label} cycle (every ${s.period} points, strength ${s.strength.toFixed(2)})`,
      detail: capitalize([
        s.suggested_feature && `${TEMPORAL_LABELS[s.suggested_feature] ?? s.suggested_feature} features recommended`,
        s.period <= 60 && `lag ${s.period} added to the suggested lags`,
      ].filter(Boolean).join(' · ')),
    })),
    ...(dataAlerts || [])
      .filter(alert => alert.category !== 'seasonality')
      .map((alert, i) => ({
        key: `${alert.category}-${i}`,
        Icon: CATEGORY_ICONS[alert.category] ?? AlertTriangle,
        tone: alert.type === 'warning' ? 'warning' as const : 'info' as const,
        title: alert.message,
      })),
  ];

  return (
    <Card title="What we found" subtitle="Patterns and data quality checks on the value to forecast">
      <ul className={`space-y-3 transition-opacity ${isAnalyzing ? 'opacity-50' : ''}`}>
        {insights.length === 0 && (
          <li className="flex items-center gap-3 text-sm text-fg-muted">
            <CheckCircle2 size={18} className="text-positive shrink-0" />
            {isAnalyzing ? 'Analyzing…' : 'No cycle or data issue detected.'}
          </li>
        )}
        {insights.map(({ key, Icon, tone, title, detail }) => (
          <li key={key} className="flex items-start gap-3">
            <span className={`mt-0.5 shrink-0 ${tone === 'warning' ? 'text-warning' : 'text-info'}`}>
              <Icon size={18} aria-label={tone} />
            </span>
            <div className="text-sm">
              <p className="text-fg">{title}</p>
              {detail && <p className="text-fg-muted mt-0.5">{detail}</p>}
            </div>
          </li>
        ))}
      </ul>
    </Card>
  );
}
