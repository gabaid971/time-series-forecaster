import { defineConfig, devices } from '@playwright/test';

/**
 * Visual walkthrough of the app (npm run screenshots): captures every page in both
 * themes, desktop and mobile, into screenshots/. Requires the frontend (npm run dev)
 * and the backend (uv run python -m app) to be running.
 */
export default defineConfig({
  testDir: './e2e',
  outputDir: './test-results',
  timeout: 120_000,
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
  },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1440, height: 900 } } },
    // Layout checks on mobile; behavior tests (forecast.spec) once is enough
    { name: 'mobile', use: { ...devices['iPhone 13'], browserName: 'chromium' }, testIgnore: /(forecast|server)\.spec/ },
  ],
});
