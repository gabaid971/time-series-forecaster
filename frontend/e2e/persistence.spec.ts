import { expect, test } from '@playwright/test';

/**
 * State saved in the browser by an older version of the app must not break the new one:
 * the persisted state is versioned and outdated parts are recomputed.
 */
test('state saved by an older version is migrated', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(String(e)));

  await page.goto('/');
  await expect(page.getByText('Load a time series')).toBeVisible();

  // Version 1 state: an analysis without the fields added later (seasonalities...)
  const csv = 'date,value\n' + Array.from({ length: 60 }, (_, i) =>
    `2024-01-${String((i % 28) + 1).padStart(2, '0')},${i}`).join('\n');
  const rows = csv.split('\n').slice(1).map((line, i) => ({ date: `2024-${String(Math.floor(i / 28) + 1).padStart(2, '0')}-${line.split(',')[0].slice(8)}`, value: i }));
  const data = { filename: 'old.csv', columns: ['date', 'value'], dateColumn: 'date', targetColumn: 'value', frequency: 'D', exogenousFeatures: [] };
  const oldState = {
    version: 1,
    state: {
      mode: 'advanced', space: 'experiment', step: 1, data, rawData: rows, fullData: [],
      analyzedKey: `old.csv|${rows.length}|date|value`,
      lagAnalysis: { suggested_lags: [1], acf: [1, 0.5], pacf: [1, 0.5], confidence_interval: 0.1, significant_lags: [], seasonality: { detected: false }, n_observations: 60 },
      dataAlerts: null, availableColumns: [], datasetStats: null, analyzeError: null,
      trainingRanges: [], predictionRanges: [], forecastHorizon: 1, defaultLags: [1], selectedModels: [], results: [], trainError: null,
    },
  };
  await page.evaluate(async (value) => {
    await new Promise<void>((resolve, reject) => {
      const open = indexedDB.open('keyval-store');
      open.onupgradeneeded = () => open.result.createObjectStore('keyval');
      open.onsuccess = () => {
        const tx = open.result.transaction('keyval', 'readwrite');
        tx.objectStore('keyval').put(value, 'tss-app-state');
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      };
    });
  }, JSON.stringify(oldState));

  const analyzed = page.waitForResponse(r => r.url().endsWith('/analyze'));
  await page.reload();
  await analyzed;  // The outdated analysis is recomputed
  await expect(page.getByText('Autocorrelation')).toBeVisible();
  expect(errors).toEqual([]);
});
