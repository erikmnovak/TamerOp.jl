import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

// Check both launch paths before Julia compilation or fixture construction.
const browser = await chromium.launch({ headless: true, chromiumSandbox: true });
try {
  const page = await browser.newPage();
  await page.setContent('<p>Sandboxed Chromium</p>');
  assert.equal(await page.locator('p').textContent(), 'Sandboxed Chromium');
  console.log(`Sandboxed headless Chromium starts: ${browser.version()}`);
} finally {
  await browser.close();
}

const directory = path.dirname(fileURLToPath(import.meta.url));
const extension = path.join(directory, 'zoom-extension');
const context = await chromium.launchPersistentContext('', {
  channel: 'chromium', headless: true, chromiumSandbox: true,
  args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`],
});
try {
  const worker = context.serviceWorkers()[0]
    ?? await context.waitForEvent('serviceworker', { timeout: 30_000 });
  assert.equal(await worker.evaluate(() => chrome.runtime.getManifest().name),
    'TamerOp local browser zoom checks');
  console.log('Sandboxed full Chromium starts and loads the local zoom fixture.');
} finally {
  await context.close();
}
