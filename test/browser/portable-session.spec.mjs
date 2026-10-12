import { test, expect } from '@playwright/test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { checkOfflineChartScroll } from './helpers.mjs';

const fixture = process.env.TAMEROP_PORTABLE_SESSION;
if (!fixture) throw new Error('Generate TAMEROP_PORTABLE_SESSION before running the offline lane.');

async function ready(page, id) {
  await expect(page.locator('html')).toHaveAttribute('data-ready', 'true');
  if (id !== undefined) await expect(page.locator('html')).toHaveAttribute('data-selected-state', id);
  await expect.poll(() => page.locator('#saved-figure').evaluate(img =>
    img.complete && img.naturalWidth > 0 && img.naturalHeight > 0)).toBe(true);
}

async function downloaded(page, button) {
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: button, exact: true }).click();
  const download = await downloadPromise;
  return fs.readFile(await download.path(), 'utf8');
}

test('A100 moved session reopens offline with exact records, keyboard selections and no Julia', async ({ browser }, info) => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'tamerop-portable-'));
  const copied = path.join(directory, 'moved-session.html');
  await fs.copyFile(path.resolve(fixture), copied);
  const network = [], errors = [];
  let context;
  try {
    context = await browser.newContext({ offline: true, acceptDownloads: true, viewport: { width: 1440, height: 1050 } });
    context.on('request', request => {
      if (/^https?:/i.test(request.url())) network.push(request.url());
    });
    let page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    page.on('websocket', socket => network.push(socket.url()));
    await page.goto(pathToFileURL(copied).href);
    await ready(page);
    const bundle = await page.locator('#portable-visual-data').evaluate(node => JSON.parse(node.textContent));
    expect(bundle.states.length).toBeGreaterThanOrEqual(2);
    expect(await page.locator('#tamerop-visual-session').evaluate(node => JSON.parse(node.textContent)))
      .toEqual(JSON.parse(bundle.session_json));
    const initial = bundle.states.find(state => state.id === bundle.initial);
    await expect(page.locator('#saved-selection')).toHaveText(initial.selection);
    await expect(page.locator('#saved-context')).toHaveText(initial.context);
    expect(await downloaded(page, 'Download session JSON')).toBe(bundle.session_json);
    expect(await downloaded(page, 'Download selected values')).toBe(initial.plotted_json);
    expect(() => JSON.parse(bundle.session_json)).not.toThrow();
    expect(() => JSON.parse(initial.plotted_json)).not.toThrow();
    await page.screenshot({ path: info.outputPath('portable-initial.png'), fullPage: true });

    const chooser = page.getByLabel('Saved view', { exact: true });
    const last = bundle.states.at(-1);
    await chooser.focus();
    await chooser.press('End');
    await chooser.press('Enter');
    await ready(page, last.id);
    await expect(page.locator('#saved-caption')).toHaveText(last.label);
    await expect(page.locator('#saved-selection')).toHaveText(last.selection);
    await expect(page.locator('#saved-context')).toHaveText(last.context);
    expect(await downloaded(page, 'Download selected values')).toBe(last.plotted_json);
    expect(await page.locator('#saved-figure').getAttribute('src')).toBe(last.image);
    expect(last.image).not.toBe(initial.image);
    const figurePromise = page.waitForEvent('download');
    await page.getByRole('link', { name: 'Download selected SVG', exact: true }).click();
    const figureDownload = await figurePromise;
    expect(await fs.readFile(await figureDownload.path(), 'utf8'))
      .toBe(Buffer.from(last.image.split(',')[1], 'base64').toString('utf8'));
    await expect(page.getByRole('button', { name: 'Next', exact: true })).toBeDisabled();
    const link = await page.locator('#state-link').evaluate(node => node.href);
    expect(link).toContain('#state=');
    await page.screenshot({ path: info.outputPath('portable-selected.png'), fullPage: true });
    await page.getByRole('button', { name: 'Previous', exact: true }).click();
    await ready(page, bundle.states.at(-2).id);
    await page.getByRole('button', { name: 'Next', exact: true }).click();
    await ready(page, last.id);

    await context.close();
    context = await browser.newContext({ offline: true, acceptDownloads: true, viewport: { width: 480, height: 900 } });
    context.on('request', request => {
      if (/^https?:/i.test(request.url())) network.push(request.url());
    });
    page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    page.on('websocket', socket => network.push(socket.url()));
    await page.goto(link);
    await ready(page, last.id);
    await expect(page.locator('#saved-selection')).toHaveText(last.selection);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
    const image=page.locator('#saved-figure');
    const size=await image.evaluate(el=>({width:el.getBoundingClientRect().width,height:el.getBoundingClientRect().height,
      naturalWidth:el.naturalWidth,naturalHeight:el.naturalHeight}));
    expect(size.width).toBeCloseTo(size.naturalWidth,1);
    expect(size.width/size.height).toBeCloseTo(size.naturalWidth/size.naturalHeight,3);
    await checkOfflineChartScroll(page,page.getByRole('region',{name:'Saved figure',exact:true}),image);
    await expect(page.locator('html')).toHaveAttribute('data-selected-state',last.id);
    expect(await image.getAttribute('src')).toBe(last.image);
    await page.screenshot({ path: info.outputPath('portable-narrow-reopened.png'), fullPage: true });

    // The fixture includes literal markup in its title, caption or state labels.
    // It must remain text; CSP also forbids network and active injected scripts.
    expect(await page.evaluate(() => globalThis.a100Injected)).toBeUndefined();
    expect(await page.locator('script:not([type="application/json"])').count()).toBe(1);
    expect(await page.locator('iframe,object,embed').count()).toBe(0);
    await page.goto(pathToFileURL(copied).href + '#state=missing-state');
    await ready(page, bundle.initial);
    await expect(page.locator('#view-status')).toContainText('Unknown saved state');
    expect(network).toEqual([]);
    expect(errors).toEqual([]);
  } finally {
    if (context) await context.close();
    await fs.rm(directory, { recursive: true, force: true });
  }
});
