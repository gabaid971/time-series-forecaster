'use client';

import dynamic from 'next/dynamic';
import type { Data, Layout } from 'plotly.js';

import { basePlotLayout } from '../../lib/plotTheme';
import type { ModelEvaluation } from '../../lib/results';
import { useThemeColors } from '../../lib/useThemeColors';
import type { ShapValue } from '../../types/forecasting';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });
const CONFIG = { displayModeBar: false, responsive: true };

/** RMSE at each step ahead: the model vs the naive reference. */
export function HorizonErrorChart({ byStep, colorIndex, name, height = 240 }: {
  byStep: ModelEvaluation['byStep']; colorIndex: number; name: string; height?: number;
}) {
  const colors = useThemeColors();
  const base = basePlotLayout(colors);
  const x = byStep.map(s => s.step);
  const data: Data[] = [
    {
      x, y: byStep.map(s => s.naiveRmse), type: 'scatter', mode: 'lines+markers', name: 'Naive',
      line: { color: colors.fgSubtle, width: 2, dash: 'dot' }, marker: { size: 8, color: colors.fgSubtle },
      hovertemplate: 'Naive: %{y:.3f}<extra></extra>',
    },
    {
      x, y: byStep.map(s => s.rmse), type: 'scatter', mode: 'lines+markers', name,
      line: { color: colors.series[colorIndex % colors.series.length], width: 2 },
      marker: { size: 8, color: colors.series[colorIndex % colors.series.length], line: { color: colors.panel, width: 2 } },
      hovertemplate: `${name}: %{y:.3f}<extra></extra>`,
    },
  ];
  const layout: Partial<Layout> = {
    ...base,
    height,
    showlegend: true,
    legend: { orientation: 'h', x: 0, y: 1.15, font: { color: colors.fgMuted, size: 12 } },
    margin: { ...base.margin, t: 24 },
    xaxis: { ...base.xaxis, title: { text: 'Steps ahead', font: { size: 12 } }, dtick: byStep.length > 15 ? undefined : 1 },
    yaxis: { ...base.yaxis, title: { text: 'RMSE', font: { size: 12 } }, rangemode: 'tozero' },
  };
  return <Plot data={data} layout={layout} useResizeHandler style={{ width: '100%', height }} config={CONFIG} />;
}

/** Distribution of the errors (forecast − actual), with zero and the mean marked. */
export function ErrorHistogram({ errors, bias, colorIndex, height = 240 }: {
  errors: number[]; bias: number; colorIndex: number; height?: number;
}) {
  const colors = useThemeColors();
  const base = basePlotLayout(colors);
  const data: Data[] = [{
    x: errors, type: 'histogram',
    marker: { color: colors.series[colorIndex % colors.series.length], line: { color: colors.panel, width: 1 } },
    opacity: 0.85,
    hovertemplate: '%{x}: %{y} points<extra></extra>',
  }];
  const layout: Partial<Layout> = {
    ...base,
    height,
    hovermode: 'closest',
    bargap: 0.05,
    margin: { ...base.margin, t: 24 },
    xaxis: { ...base.xaxis, title: { text: 'Forecast − actual', font: { size: 12 } }, zeroline: false },
    yaxis: { ...base.yaxis, title: { text: 'Points', font: { size: 12 } } },
    shapes: [
      { type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1, line: { color: colors.fgMuted, width: 1 } },
      { type: 'line', x0: bias, x1: bias, yref: 'paper', y0: 0, y1: 1, line: { color: colors.fg, width: 1, dash: 'dot' } },
    ],
    annotations: [{
      x: bias, yref: 'paper', y: 1, yanchor: 'bottom', showarrow: false,
      text: `mean ${bias >= 0 ? '+' : ''}${bias.toFixed(2)}`, font: { size: 11, color: colors.fgMuted },
    }],
  };
  return <Plot data={data} layout={layout} useResizeHandler style={{ width: '100%', height }} config={CONFIG} />;
}

const LABELS: Record<string, (v: number) => string> = {
  day_of_week: v => ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][v] ?? String(v),
  month: v => ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][v - 1] ?? String(v),
  hour_of_day: v => `${v}h`,
  minute_of_day: v => `${Math.floor(v / 60)}:${String(v % 60).padStart(2, '0')}`,
};

/**
 * Average SHAP effect of a calendar feature on the forecast, per value (e.g. per month).
 * Diverging: raises the forecast (accent) / lowers it (blue), zero as the neutral baseline.
 */
export function ShapEffectChart({ feature, values, height = 240 }: { feature: string; values: ShapValue[]; height?: number }) {
  const colors = useThemeColors();
  const base = basePlotLayout(colors);
  const label = LABELS[feature] ?? String;
  const shown = values.filter(v => v.count > 0);
  const data: Data[] = [{
    x: shown.map(v => label(v.value)),
    y: shown.map(v => v.shap),
    type: 'bar',
    marker: { color: shown.map(v => (v.shap >= 0 ? colors.accent : colors.series[0])) },
    hovertemplate: '%{x}: %{y:+.3f}<extra></extra>',
  }];
  const layout: Partial<Layout> = {
    ...base,
    height,
    hovermode: 'closest',
    bargap: 0.35,
    margin: { ...base.margin, t: 8 },
    xaxis: { ...base.xaxis, type: 'category' },
    yaxis: { ...base.yaxis, title: { text: 'Effect on forecast', font: { size: 12 } }, zeroline: true, zerolinecolor: colors.fgMuted },
  };
  return <Plot data={data} layout={layout} useResizeHandler style={{ width: '100%', height }} config={CONFIG} />;
}
