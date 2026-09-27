'use client';

import { useEffect, useState } from 'react';

import { SERIES_COUNT } from './modelColors';

export interface ThemeColors {
  fg: string;
  fgMuted: string;
  fgSubtle: string;
  grid: string;
  panel: string;
  accent: string;
  series: string[];
}

function readColors(): ThemeColors {
  const style = getComputedStyle(document.documentElement);
  const rgb = (name: string, alpha = 1) => {
    const [r, g, b] = style.getPropertyValue(`--${name}`).trim().split(/\s+/);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };
  return {
    fg: rgb('fg'),
    fgMuted: rgb('fg-muted'),
    fgSubtle: rgb('fg-subtle'),
    grid: rgb('ink', 0.1),
    panel: rgb('panel'),
    accent: rgb('accent'),
    series: Array.from({ length: SERIES_COUNT }, (_, i) => rgb(`series-${i + 1}`)),
  };
}

const FALLBACK: ThemeColors = {
  fg: '#f4f4f5', fgMuted: '#a1a1aa', fgSubtle: '#71717a', grid: 'rgba(255,255,255,0.1)', panel: '#18181b', accent: '#f59e0b',
  series: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
};

/**
 * Concrete colors of the current theme, for libraries that cannot use CSS variables
 * (Plotly). Updates when the theme changes.
 */
export function useThemeColors(): ThemeColors {
  const [colors, setColors] = useState<ThemeColors>(FALLBACK);

  useEffect(() => {
    setColors(readColors());
    const observer = new MutationObserver(() => setColors(readColors()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);

  return colors;
}
