'use client';

import dynamic from 'next/dynamic';
import type { Data, Layout } from 'plotly.js';

import { basePlotLayout } from '../../lib/plotTheme';
import { useThemeColors } from '../../lib/useThemeColors';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SeriesChartProps {
  dates: string[];
  values: number[];
  name: string;
  height?: number;
}

/** A single time series: 2px line, hover readout, range slider to zoom on a period. */
export function SeriesChart({ dates, values, name, height = 340 }: SeriesChartProps) {
  const colors = useThemeColors();
  const base = basePlotLayout(colors);

  const data: Data[] = [{
    x: dates,
    y: values,
    type: 'scatter',  // SVG: WebGL traces are not drawn in the range slider
    mode: 'lines',
    name,
    line: { color: colors.accent, width: 2 },
    hovertemplate: '%{y:,.2f}<extra></extra>',
  }];

  const layout: Partial<Layout> = {
    ...base,
    height,
    xaxis: {
      ...base.xaxis,
      rangeslider: { visible: true, thickness: 0.08, bgcolor: 'rgba(0,0,0,0)', bordercolor: colors.grid, borderwidth: 1 },
    },
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
