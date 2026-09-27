import type { Layout } from 'plotly.js';

import type { ThemeColors } from './useThemeColors';

/**
 * Common Plotly layout: transparent background, recessive hairline grid,
 * theme text colors, crosshair-like unified hover.
 */
export function basePlotLayout(colors: ThemeColors): Partial<Layout> {
  const axis = {
    gridcolor: colors.grid,
    zerolinecolor: colors.grid,
    linecolor: colors.grid,
    tickfont: { color: colors.fgMuted, size: 12 },
    showgrid: true,
  };
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, system-ui, sans-serif', color: colors.fgMuted, size: 12 },
    xaxis: { ...axis, showgrid: false },
    yaxis: axis,
    margin: { t: 8, r: 8, l: 48, b: 32 },
    autosize: true,
    hovermode: 'x unified',
    hoverlabel: { bgcolor: colors.panel, bordercolor: colors.grid, font: { color: colors.fg, size: 12 } },
    showlegend: false,
  };
}
