// This lane intentionally has no Julia web server. Generate the required exports
// with the native acceptance owners, then stop Julia before reopening them.
import { defineConfig } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const directory = path.dirname(fileURLToPath(import.meta.url));
export default defineConfig({
  testDir: directory,
  testMatch: ['portable-session.spec.mjs', 'composition-offline.spec.mjs', 'static-export.spec.mjs', 'source-preview-offline.spec.mjs'],
  fullyParallel: false,
  forbidOnly: true,
  workers: 1,
  retries: 0,
  timeout: 120_000,
  expect: { timeout: 15_000 },
  outputDir: process.env.TAMEROP_BROWSER_OUTPUT || path.join(directory, 'test-results', 'portable-session'),
  reporter: [['list'], [path.join(directory, 'acceptance-reporter.mjs')]],
  use: {
    browserName: 'chromium',
    headless: true,
    viewport: { width: 1440, height: 1050 },
    launchOptions: { chromiumSandbox: true },
    acceptDownloads: true,
    trace: { mode: 'retain-on-failure', screenshots: false, snapshots: true },
    screenshot: 'only-on-failure',
  },
});
