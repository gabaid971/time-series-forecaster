'use client';

import dynamic from 'next/dynamic';
import type { Annotations, Data, Layout, Shape } from 'plotly.js';

import { basePlotLayout } from '../../lib/plotTheme';
import { useThemeColors } from '../../lib/useThemeColors';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface CorrelogramChartProps {
  values: number[];              // Index = lag (index 0 = lag 0, not drawn)
  confidence: number;            // Half-width of the 95% band around 0
  markers?: { lag: number; label: string }[];  // Detected cycles to point out
  height?: number;
}

/**
 * ACF or PACF by lag: bars from 0, the non-significance band shaded,
 * significant lags in the accent color, detected cycles annotated.
 */
export function CorrelogramChart({ values, confidence, markers = [], height = 220 }: CorrelogramChartProps) {
  const colors = useThemeColors();
  const base = basePlotLayout(colors);
  const lags = values.map((_, i) => i).slice(1);
  const bars = values.slice(1);

  const data: Data[] = [{
    x: lags,
    y: bars,
    type: 'bar',
    marker: { color: bars.map(v => (Math.abs(v) > confidence ? colors.accent : colors.fgSubtle)) },
    hovertemplate: 'Lag %{x}: %{y:.3f}<extra></extra>',
  }];

  const band: Partial<Shape> = {
    type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: -confidence, y1: confidence,
    fillcolor: colors.grid, line: { width: 0 }, layer: 'below',
  };
  const markerLines: Partial<Shape>[] = markers.map(m => ({
    type: 'line', x0: m.lag, x1: m.lag, yref: 'paper', y0: 0, y1: 1,
    line: { color: colors.fgMuted, width: 1, dash: 'dot' },
  }));
  const annotations: Partial<Annotations>[] = markers.map(m => ({
    x: m.lag, yref: 'paper', y: 1, text: m.label, showarrow: false, yanchor: 'bottom',
    font: { color: colors.fgMuted, size: 11 },
  }));

  const layout: Partial<Layout> = {
    ...base,
    height,
    bargap: 0.3,
    hovermode: 'closest',
    margin: { ...base.margin, t: markers.length ? 20 : 8 },
    yaxis: { ...base.yaxis, range: [Math.min(-0.2, ...bars) - 0.05, 1.05], zeroline: true },
    xaxis: { ...base.xaxis, title: { text: 'Lag', font: { size: 12, color: colors.fgMuted } } },
    shapes: [band, ...markerLines],
    annotations,
  };

  return (
    <Plot
      data={data}
      layout={layout}
      useResizeHandler
      style={{ width: '100%', height }}
      config={{ displayModeBar: false, responsive: true }}
    />
  );
}
