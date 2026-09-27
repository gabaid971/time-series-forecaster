import { expect, test } from '@playwright/test';

/** The free backend sleeps when idle: the app says so while it wakes up. */

test('a slow server shows the waking-up banner, then hides it', async ({ page }) => {
  await page.route('**/health', async route => {
    await new Promise(resolve => setTimeout(resolve, 4000));  // Cold start
    await route.fulfill({ status: 200, contentType: 'application/json', body: '{"status":"healthy"}' });
  });
  await page.goto('/');
  await expect(page.getByText('Waking up the server…')).toBeVisible({ timeout: 4000 });
  await expect(page.getByText('Waking up the server…')).toBeHidden({ timeout: 10000 });
});

test('an unreachable server is reported', async ({ page }) => {
  await page.route('**/health', route => route.abort());
  await page.goto('/');
  await expect(page.getByText('The server cannot be reached.')).toBeVisible();
});

test('a running server shows no banner', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('Load a time series')).toBeVisible();
  await page.waitForTimeout(3000);
  await expect(page.getByText(/Waking up the server|cannot be reached/)).toHaveCount(0);
});
