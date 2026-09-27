'use client';

import dynamic from 'next/dynamic';
import type { Data, Layout } from 'plotly.js';

import { basePlotLayout } from '../../lib/plotTheme';
import type { ForecastPoint } from '../../lib/results';
import { useThemeColors } from '../../lib/useThemeColors';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export interface ForecastSeries {
  id: string;
  name: string;
  colorIndex: number;
  points: ForecastPoint[];
  emphasized: boolean;  // The selected model: full weight; others are drawn lighter
}

interface ForecastChartProps {
  series: ForecastSeries[];
  view: 'forecast' | 'errors';
  height?: number;
}

/**
 * Validation period: actual values (neutral) and the visible models' forecasts,
 * or their errors (forecast − actual) around zero. Range slider to zoom.
 */
export function ForecastChart({ series, view, height = 380 }: ForecastChartProps) {
  const colors = useThemeColors();
  const base = basePlotLayout(colors);
  const reference = series[0]?.points ?? [];

  const models: Data[] = series.map(s => ({
    x: s.points.map(p => p.date),
    y: s.points.map(p => (view === 'errors' ? p.prediction - p.actual : p.prediction)),
    type: 'scatter',
    mode: 'lines',
    name: s.name,
    line: { color: colors.series[s.colorIndex % colors.series.length], width: s.emphasized ? 2 : 1.5 },
    opacity: s.emphasized ? 1 : 0.6,
    hovertemplate: `${s.name}: %{y:,.2f}<extra></extra>`,
  }));

  const actual: Data[] = view === 'forecast' && reference.length > 0 ? [{
    x: reference.map(p => p.date),
    y: reference.map(p => p.actual),
    type: 'scatter',
    mode: 'lines',
    name: 'Actual',
    line: { color: colors.fg, width: 2 },
    hovertemplate: 'Actual: %{y:,.2f}<extra></extra>',
  }] : [];

  const layout: Partial<Layout> = {
    ...base,
    height,
    yaxis: { ...base.yaxis, zeroline: view === 'errors', zerolinecolor: colors.fgMuted, zerolinewidth: 1 },
    xaxis: {
      ...base.xaxis,
      rangeslider: { visible: true, thickness: 0.08, bgcolor: 'rgba(0,0,0,0)', bordercolor: colors.grid, borderwidth: 1 },
    },
  };

  return (
    <Plot data={[...actual, ...models]} layout={layout} useResizeHandler style={{ width: '100%', height }} config={{ displayModeBar: false, responsive: true }} />
  );
}
