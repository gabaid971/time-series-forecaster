'use client';

import dynamic from 'next/dynamic';
import type { Data, Layout } from 'plotly.js';

import { basePlotLayout } from '../../lib/plotTheme';
import { useThemeColors } from '../../lib/useThemeColors';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export interface FutureSeries {
  id: string;
  name: string;
  colorIndex: number;
  dates: string[];
  predictions: (number | null)[];
  band?: { dates: string[]; lower: number[]; upper: number[] };  // Only where the recipe was validated
}

interface FutureChartProps {
  history: { dates: string[]; values: number[] };
  series: FutureSeries[];
  focusId: string | null;
  lastDate: string;
  height?: number;
}

const withAlpha = (rgba: string, alpha: number) => rgba.replace(/,\s*[\d.]+\)$/, `, ${alpha})`);

/** Recent history, then each recipe's forecast; the focused recipe shows its confidence band. */
export function FutureChart({ history, series, focusId, lastDate, height = 380 }: FutureChartProps) {
  const colors = useThemeColors();
  const base = basePlotLayout(colors);
  const color = (s: FutureSeries) => colors.series[s.colorIndex % colors.series.length];

  const bands: Data[] = series
    .filter(s => s.id === focusId && s.band && s.band.dates.length > 0)
    .flatMap(s => [
      {
        x: s.band!.dates, y: s.band!.upper, type: 'scatter', mode: 'lines', line: { width: 0 },
        hoverinfo: 'skip', showlegend: false,
      },
      {
        x: s.band!.dates, y: s.band!.lower, type: 'scatter', mode: 'lines', line: { width: 0 },
        fill: 'tonexty', fillcolor: withAlpha(color(s), 0.18), hoverinfo: 'skip', showlegend: false,
      },
    ] as Data[]);

  const lines: Data[] = series.map(s => ({
    x: s.dates, y: s.predictions, type: 'scatter', mode: 'lines', name: s.name,
    line: { color: color(s), width: s.id === focusId ? 2.5 : 1.5 },
    opacity: s.id === focusId || !focusId ? 1 : 0.55,
    hovertemplate: `${s.name}: %{y:,.2f}<extra></extra>`,
  }));

  const actual: Data = {
    x: history.dates, y: history.values, type: 'scatter', mode: 'lines', name: 'History',
    line: { color: colors.fg, width: 2 }, hovertemplate: 'History: %{y:,.2f}<extra></extra>',
  };

  const layout: Partial<Layout> = {
    ...base,
    height,
    shapes: [{ type: 'line', x0: lastDate, x1: lastDate, yref: 'paper', y0: 0, y1: 1, line: { color: colors.fgMuted, width: 1, dash: 'dot' } }],
    annotations: [{
      x: lastDate, yref: 'paper', y: 1, xanchor: 'left', yanchor: 'top', xshift: 6, showarrow: false,
      text: 'Forecast →', font: { size: 12, color: colors.fgMuted },
    }],
  };

  return <Plot data={[actual, ...bands, ...lines]} layout={layout} useResizeHandler style={{ width: '100%', height }} config={{ displayModeBar: false, responsive: true }} />;
}
