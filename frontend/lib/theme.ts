'use client';

import { useCallback, useEffect, useState } from 'react';

import { THEME_STORAGE_KEY as STORAGE_KEY } from './themeScript';

export type Theme = 'dark' | 'light';

/** Current theme and a setter that persists it (localStorage, read synchronously at load). */
export function useTheme(): [Theme, (theme: Theme) => void] {
  const [theme, setThemeState] = useState<Theme>('dark');

  useEffect(() => {
    setThemeState(document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');
  }, []);

  const setTheme = useCallback((next: Theme) => {
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Private browsing or blocked storage: the theme simply won't persist
    }
    setThemeState(next);
  }, []);

  return [theme, setTheme];
}
