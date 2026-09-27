import { expect, Page, test } from '@playwright/test';

/**
 * Edge cases of the Forecast space: every model at the shortest and longest
 * horizon, and what happens to models, results and recipes when the data changes.
 * Behavior tests: run on the desktop project only (see playwright.config.ts).
 */

test.describe.configure({ timeout: 240_000 });

const spaces = (page: Page) => page.getByRole('navigation', { name: 'Spaces' });

async function loadExample(page: Page, title: string) {
  const analyzed = page.waitForResponse(r => r.url().endsWith('/analyze'));
  await page.getByRole('button', { name: new RegExp(title) }).click();
  await analyzed;
  await expect(page.getByText('What we found')).toBeVisible();
}

async function trainRecommended(page: Page, horizon: string) {
  await page.getByRole('button', { name: /Continue to models/ }).click();
  await page.getByRole('button', { name: /Add recommended models/ }).click();
  await page.getByRole('group', { name: 'Horizon presets' }).getByRole('button', { name: horizon, exact: true }).click();
  const trained = page.waitForResponse(r => r.url().endsWith('/train'));
  await page.getByRole('button', { name: /^Train \d+ model/ }).click();
  await trained;
  await expect(page.getByText('Best model')).toBeVisible();
}

/** Save every successful model of the leaderboard to the library. */
async function saveAll(page: Page) {
  const rows = page.getByRole('row').filter({ has: page.locator('td') });
  const count = await rows.count();
  for (let i = 0; i < count; i++) {
    await rows.nth(i).click();
    const save = page.getByRole('button', { name: /Save to library/ });
    if (await save.isVisible()) await save.click();
  }
}

async function forecast(page: Page, steps: string) {
  await page.getByRole('spinbutton', { name: /Number of/ }).fill(steps);
  const done = page.waitForResponse(r => r.url().endsWith('/forecast'));
  await page.getByRole('button', { name: /Forecast with \d+ recipe/ }).click();
  await done;
  await expect(page.getByText(`Next ${steps} steps`)).toBeVisible();
}

test('every model forecasts at the shortest and the longest horizon', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto('/');
  await loadExample(page, 'Daily minimum temperatures');
  await trainRecommended(page, '7');
  await saveAll(page);

  await spaces(page).getByRole('button', { name: 'Forecast' }).click();
  await expect(page.getByRole('button', { name: /Forecast with 5 recipes/ })).toBeEnabled();

  for (const steps of ['1', '365']) {
    await forecast(page, steps);
    // Every recipe produced a forecast: no error nor warning line, 5 series in the legend
    await expect(page.locator('p.text-negative')).toHaveCount(0);
    await expect(page.getByText(/diverges|could not be forecast/)).toHaveCount(0);
    for (const name of ['Naive baseline', 'Linear regression', 'XGBoost', 'ARIMA', 'Prophet']) {
      await expect(page.getByRole('button', { name, exact: true })).toBeVisible();
    }
  }
  expect(errors).toEqual([]);
});

test('loading another dataset resets models and results but keeps the library', async ({ page }) => {
  await page.goto('/');
  await loadExample(page, 'Daily minimum temperatures');
  await trainRecommended(page, '7');
  await page.getByRole('button', { name: /Save to library/ }).click();
  await spaces(page).getByRole('button', { name: 'Forecast' }).click();
  await forecast(page, '7');

  // Load another file
  await spaces(page).getByRole('button', { name: 'Experiment' }).click();
  await page.getByRole('button', { name: 'Data', exact: true }).click();
  await page.getByRole('button', { name: /Change file/ }).click();
  await loadExample(page, 'Sales with a sensor');

  // Experiment: no model configured, no result
  await expect(page.getByRole('button', { name: 'Results', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: /Continue to models/ }).click();
  await expect(page.getByText('Add at least one model.')).toBeVisible();

  // Forecast: the previous forecast is gone, the recipe is kept but flagged
  await spaces(page).getByRole('button', { name: 'Forecast' }).click();
  await expect(page.getByText(/Next \d+ steps/)).toHaveCount(0);
  await expect(page.getByText(/Needs columns "Date", "Daily minimum temperatures"/)).toBeVisible();
  await expect(page.getByRole('button', { name: /Select a compatible recipe/ })).toBeDisabled();
});

test('a recipe is refused on data of another frequency', async ({ page }) => {
  await page.goto('/');
  await loadExample(page, 'Daily minimum temperatures');
  await trainRecommended(page, '1');
  await page.getByRole('button', { name: /Save to library/ }).click();

  // Same column names, monthly data
  await spaces(page).getByRole('button', { name: 'Experiment' }).click();
  await page.getByRole('button', { name: 'Data', exact: true }).click();
  await page.getByRole('button', { name: /Change file/ }).click();
  const rows = Array.from({ length: 60 }, (_, i) => `${2000 + Math.floor(i / 12)}-${String(i % 12 + 1).padStart(2, '0')}-01,${10 + (i % 12)}`);
  const analyzed = page.waitForResponse(r => r.url().endsWith('/analyze'));
  await page.setInputFiles('input[type=file]', {
    name: 'monthly.csv', mimeType: 'text/csv', buffer: Buffer.from(['Date,Daily minimum temperatures', ...rows].join('\n')),
  });
  await analyzed;

  await spaces(page).getByRole('button', { name: 'Forecast' }).click();
  await expect(page.getByText('Validated on daily data, the loaded data is monthly.')).toBeVisible();
  await expect(page.getByRole('button', { name: /Select a compatible recipe/ })).toBeDisabled();
});

test('changing the target column clears the results', async ({ page }) => {
  await page.goto('/');
  await loadExample(page, 'Sales with a sensor');
  await trainRecommended(page, '1');
  await page.getByRole('button', { name: 'Data', exact: true }).click();

  const analyzed = page.waitForResponse(r => r.url().endsWith('/analyze'));
  await page.getByLabel('Value to forecast').selectOption('sensor');
  await analyzed;
  await expect(page.getByRole('button', { name: 'Results', exact: true })).toBeDisabled();
});
