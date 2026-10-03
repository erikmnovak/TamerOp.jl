import { defineConfig } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const directory = path.dirname(fileURLToPath(import.meta.url));
const intervalPort = Number(process.env.TAMEROP_BROWSER_INTERVAL_PORT || 8848);
const quote = value => `'${String(value).replaceAll("'", "'\\''")}'`;
const julia = process.env.TAMEROP_JULIA || 'julia';
const project = process.env.TAMEROP_JULIA_PROJECT || directory;

export default defineConfig({
  testDir: directory,
  testMatch: '*.spec.mjs',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 600_000,
  expect: { timeout: 30_000 },
  outputDir: process.env.TAMEROP_BROWSER_OUTPUT || path.join(directory, 'test-results'),
  reporter: [['list'], ['html', { open: 'never' }]],
  globalTeardown: path.join(directory, 'stop-server.mjs'),
  use: {
    browserName: 'chromium',
    headless: true,
    viewport: { width: 1440, height: 1050 },
    launchOptions: { chromiumSandbox: true },
    // Keep action/DOM traces without continuously recording large WebGL frames.
    // Explicit checkpoints and failures still capture full-page screenshots.
    trace: { mode: 'retain-on-failure', screenshots: false, snapshots: true },
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: `${quote(julia)} --startup-file=no --compiled-modules=existing --threads=1 --heap-size-hint=2400M --project=${quote(project)} ${quote(path.join(directory, 'serve.jl'))}`,
    url: `http://127.0.0.1:${intervalPort}/health`,
    reuseExistingServer: process.env.TAMEROP_BROWSER_REUSE_SERVER === '1',
    timeout: 600_000,
    stdout: 'pipe',
    stderr: 'pipe',
    env: {
      JULIA_NUM_THREADS: '1',
      JULIA_NUM_PRECOMPILE_TASKS: '1',
      OPENBLAS_NUM_THREADS: '1',
    },
    gracefulShutdown: { signal: 'SIGINT', timeout: 15_000 },
  },
});
