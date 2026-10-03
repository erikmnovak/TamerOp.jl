import { test as base, expect, chromium } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { state, control, openInspector, pick, capture, checkNarrowLayout, scrollableCharts, scrollChart, fullyClosed } from './helpers.mjs';

const directory = path.dirname(fileURLToPath(import.meta.url));
const url = `http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT || 8849}/squares-grayscale`;
const test = base.extend({
  context: async ({ headless }, use) => {
    const extension = path.join(directory, 'zoom-extension');
    // A fresh disposable profile gives access to Chrome's real zoom API. This
    // test extension has loopback-only host permission and no content scripts.
    const context = await chromium.launchPersistentContext('', {
      channel: 'chromium', headless, chromiumSandbox: true,
      viewport: { width: 1440, height: 1050 },
      args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`],
    });
    try { await use(context); } finally { await context.close(); }
  },
});

async function exactPair(page, source, target) {
  for (const [suffix, value] of [['source-x', source[0]], ['source-y', source[1]],
    ['target-x', target[0]], ['target-y', target[1]]]) {
    await control(page, suffix).fill(value);
    await control(page, suffix).press('Tab');
  }
  const revision = (await state(page)).selection.revision;
  await control(page, 'select-points').click();
  await expect.poll(async () => (await state(page)).selection.revision).toBeGreaterThan(revision);
  await expect(control(page, 'error')).toBeEmpty();
}

test('previous manual A35 checks: large grayscale, chart scrolling and real 150/200 percent browser zoom', async ({ page, context }, testInfo) => {
  const errors = [];
  const worker = context.serviceWorkers()[0] || await context.waitForEvent('serviceworker');
  await openInspector(page, url, errors);
  expect((await state(page)).style).toEqual({ palette: 'grayscale', fontsize: 24 });
  await exactPair(page, ['1/2', '1/2'], ['5/2', '5/2']);
  let current = await state(page);
  expect(current.inspection.defined).toBe(true);
  expect(current.inspection.rank).toBe(0);
  expect(current.inspection.matrix_size).toEqual([1, 1]);
  expect(current.main_ui.markers.region.source).toHaveLength(1);
  expect(current.main_ui.markers.region.target).toHaveLength(1);
  await expect(page.locator('body')).toContainText('circle = source');
  await expect(page.locator('body')).toContainText('square = target');
  await capture(page, testInfo, 'grayscale-zero-composite');

  await exactPair(page, ['5/4', '7/4'], ['7/4', '5/4']);
  current = await state(page);
  expect(current.selection.pair[0]).toBe(current.selection.pair[1]);
  expect(current.inspection.defined).toBe(false);
  expect(current.inspection.parameter_relation).toBe('incomparable');
  const fixedQuery = current.selection.query_points;
  await checkNarrowLayout(page, testInfo, 'grayscale-narrow');
  expect((await state(page)).selection.query_points).toEqual(fixedQuery);

  const baselineDPR = await page.evaluate(() => devicePixelRatio);
  for (const factor of [1.5, 2]) {
    const actual = await worker.evaluate(async ({ url, factor }) => {
      const tabs = await chrome.tabs.query({ url });
      if (tabs.length !== 1) throw new Error('Expected exactly one fixture tab');
      await chrome.tabs.setZoomSettings(tabs[0].id, { mode: 'automatic', scope: 'per-tab' });
      await chrome.tabs.setZoom(tabs[0].id, factor);
      return chrome.tabs.getZoom(tabs[0].id);
    }, { url: page.url(), factor });
    expect(actual).toBeCloseTo(factor, 5);
    await expect.poll(() => page.evaluate(() => devicePixelRatio)).toBeCloseTo(baselineDPR * factor, 5);
    // Browser zoom changes layout and pixel density; pinch zoom does not.
    expect(await page.evaluate(() => visualViewport.scale)).toBe(1);
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
    await control(page, 'point-x').focus();
    await page.keyboard.press('Tab');
    await expect(control(page, 'point-y')).toBeFocused();
    const focus = await control(page, 'point-y').evaluate(element => {
      const style = getComputedStyle(element);
      return { style: style.outlineStyle, width: parseFloat(style.outlineWidth) };
    });
    expect(focus.style).not.toBe('none');
    expect(focus.width).toBeGreaterThan(0);
    // Full-page captures use CSS bounds that can crop a genuinely zoomed page.
    // Record the actual viewport at the controls and both scrolled chart ends.
    await capture(page, testInfo, `grayscale-zoom-${factor * 100}-controls`, { fullPage: false });
    const panels = await scrollableCharts(page);
    expect(panels.length).toBeGreaterThanOrEqual(2);
    for (const panel of panels) await scrollChart(page, panel, 'right');
    await pick(page, 1, 'diagram', { scroll: false });
    await capture(page, testInfo, `grayscale-zoom-${factor * 100}-right`, { fullPage: false });
    for (const panel of panels) await scrollChart(page, panel, 'left');
    await pick(page, 2, 'barcode', { scroll: false });
    await capture(page, testInfo, `grayscale-zoom-${factor * 100}-left`, { fullPage: false });
    current = await state(page);
    expect(current.selection.query_points).toEqual(fixedQuery);
    expect(current.inspection.defined).toBe(false);
    expect(current.records.map(record => [record.birth, record.death])).toEqual([['0', '2'], ['1', '3']]);
  }
  await worker.evaluate(async url => {
    const [tab] = await chrome.tabs.query({ url });
    await chrome.tabs.setZoom(tab.id, 1);
  }, page.url());
  await expect.poll(() => page.evaluate(() => devicePixelRatio)).toBe(baselineDPR);
  await control(page, 'close').click();
  await fullyClosed(page);
  expect(errors).toEqual([]);
});
