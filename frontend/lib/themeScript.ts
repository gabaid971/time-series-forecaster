/** Theme bootstrap, usable from server components (no 'use client'). */

export const THEME_STORAGE_KEY = 'tss-theme';

/**
 * Inline script for <head>: applies the saved theme before the first paint
 * (no flash of the wrong theme). Dark is the default identity of the app.
 */
export const themeInitScript = `
try {
  var t = localStorage.getItem('${THEME_STORAGE_KEY}');
  document.documentElement.dataset.theme = t === 'light' ? 'light' : 'dark';
} catch (e) {
  document.documentElement.dataset.theme = 'dark';
}`;
