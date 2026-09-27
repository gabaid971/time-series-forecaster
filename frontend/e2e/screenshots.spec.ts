import path from 'path';
import { expect, test } from '@playwright/test';

const DATASET = path.resolve(__dirname, '../../daily-minimum-temperatures.csv');

for (const theme of ['dark', 'light'] as const) {
  test(`walkthrough (${theme} theme)`, async ({ page }, testInfo) => {
    const errors: string[] = [];
    page.on('pageerror', e => errors.push(String(e)));
    page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });

    // Every captured page is also checked: no horizontal overflow (phones), no console error
    const shot = async (name: string) => {
      await page.screenshot({ path: `screenshots/${testInfo.project.name}/${theme}/${name}.png`, fullPage: true });
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      expect(overflow, `horizontal overflow on ${name}`).toBeLessThanOrEqual(0);
      expect(errors, `console errors on ${name}`).toEqual([]);
    };

    // The theme is read from localStorage before the first paint
    await page.addInitScript(t => localStorage.setItem('tss-theme', t), theme);
    await page.goto('/');
    await expect(page.getByText('Load a time series')).toBeVisible();
    await shot('1-data-upload');

    await page.getByRole('navigation', { name: 'Spaces' }).getByRole('button', { name: 'Forecast' }).click();
    await expect(page.getByText('Your library is empty')).toBeVisible();
    await shot('0-forecast-empty');
    await page.getByRole('navigation', { name: 'Spaces' }).getByRole('button', { name: 'Experiment' }).click();

    const analyzed = page.waitForResponse(r => r.url().endsWith('/analyze'));
    await page.setInputFiles('input[type=file]', DATASET);
    await analyzed;
    await expect(page.getByText('What we found')).toBeVisible();
    await page.waitForTimeout(1000); // Let charts render
    await shot('2-data-loaded');

    await page.getByRole('button', { name: 'Advanced', exact: true }).click();
    await expect(page.getByText('Autocorrelation')).toBeVisible();
    await page.waitForTimeout(1000);
    await shot('2b-data-advanced');
    await page.getByRole('button', { name: 'Simple', exact: true }).click();

    await page.getByRole('button', { name: /Continue to models/ }).click();
    await page.getByRole('button', { name: /Add recommended models/ }).click();
    await page.getByRole('group', { name: 'Horizon presets' }).getByRole('button', { name: '7', exact: true }).click();
    await page.waitForTimeout(800);
    await shot('3-models');

    await page.getByRole('button', { name: 'Advanced', exact: true }).click();
    await page.waitForTimeout(500);
    await shot('3b-models-advanced');
    await page.getByRole('button', { name: 'Simple', exact: true }).click();

    const trained = page.waitForResponse(r => r.url().endsWith('/train'));
    await page.getByRole('button', { name: /^Train \d+ model/ }).click();
    await trained;
    await expect(page.getByText('Best model')).toBeVisible();
    await page.waitForTimeout(1500);
    await shot('4-results');
    await page.getByRole('button', { name: /Save to library/ }).click();

    await page.getByRole('button', { name: 'Advanced', exact: true }).click();
    await page.getByRole('row').filter({ hasText: 'XGBoost' }).click();
    await page.waitForTimeout(1500);
    await shot('4b-results-advanced');
    await page.getByRole('button', { name: /Save to library/ }).click();
    await page.getByRole('button', { name: 'Simple', exact: true }).click();

    await page.getByRole('navigation', { name: 'Spaces' }).getByRole('button', { name: 'Forecast' }).click();
    await expect(page.getByText('Library', { exact: true })).toBeVisible();
    await shot('5-forecast-library');

    const forecasted = page.waitForResponse(r => r.url().endsWith('/forecast'));
    await page.getByRole('button', { name: /Forecast with 2 recipes/ }).click();
    await forecasted;
    await expect(page.getByText('Next 30 steps')).toBeVisible();
    await page.waitForTimeout(1500);
    await shot('6-forecast-results');
  });
}

