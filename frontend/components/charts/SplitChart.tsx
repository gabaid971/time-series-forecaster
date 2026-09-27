'use client';

import dynamic from 'next/dynamic';
import type { Annotations, Data, Layout, Shape } from 'plotly.js';

import { basePlotLayout } from '../../lib/plotTheme';
import { useThemeColors } from '../../lib/useThemeColors';
import type { DateRange } from '../../types/forecasting';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SplitChartProps {
  dates: string[];
  values: number[];
  training: DateRange[];
  validation: DateRange[];
  height?: number;
}

/** The series with its training (accent) and validation (info) periods shaded and labelled. */
export function SplitChart({ dates, values, training, validation, height = 260 }: SplitChartProps) {
  const colors = useThemeColors();
  const base = basePlotLayout(colors);
  const info = colors.series[0];

  const zone = (range: DateRange, fill: string): Partial<Shape> => ({
    type: 'rect', xref: 'x', yref: 'paper', x0: range.start, x1: range.end, y0: 0, y1: 1,
    fillcolor: fill, opacity: 0.14, line: { width: 0 }, layer: 'below',
  });
  const label = (range: DateRange, text: string): Partial<Annotations> => ({
    x: range.start, xref: 'x', xanchor: 'left', yref: 'paper', y: 1, yanchor: 'top',
    text, showarrow: false, font: { size: 12, color: colors.fg }, xshift: 6, yshift: -4,
  });

  const data: Data[] = [{
    x: dates, y: values, type: 'scatter', mode: 'lines',
    line: { color: colors.fgMuted, width: 1.5 },
    hovertemplate: '%{y:,.2f}<extra></extra>',
  }];

  const layout: Partial<Layout> = {
    ...base,
    height,
    shapes: [...training.map(r => zone(r, colors.accent)), ...validation.map(r => zone(r, info))],
    annotations: [
      ...training.slice(0, 1).map(r => label(r, '<b>Training</b>')),
      ...validation.slice(0, 1).map(r => label(r, '<b>Validation</b>')),
    ],
  };

  return (
    <Plot data={data} layout={layout} useResizeHandler style={{ width: '100%', height }} config={{ displayModeBar: false, responsive: true }} />
  );
}
