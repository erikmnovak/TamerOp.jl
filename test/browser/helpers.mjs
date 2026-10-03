import { expect } from '@playwright/test';
import { writeFile } from 'node:fs/promises';

export const state = async page => JSON.parse(await page.getByTestId('julia-session-state').textContent());
export const control = (page, suffix) => page.locator(`${['member', 'vertex'].includes(suffix) ? 'input' : ''}[id$="-${suffix}"]`);

export async function openInspector(page, url, errors) {
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 300_000 });
  await page.getByTestId('julia-session-state').waitFor({ state: 'attached', timeout: 300_000 });
  await page.waitForFunction(() => {
    const canvases = [...document.querySelectorAll('canvas')];
    // Three resets draw-call counts for each scene, including empty scenes.
    // The frame counter records that rendering has actually started.
    return canvases.length > 0 && canvases.every(canvas => canvas.wglmakie_screen?.renderer?.info.render.frame > 0);
  }, null, { timeout: 300_000 });
  await expect.poll(async () => (await state(page)).summary.listener_errors).toEqual([]);
}

export async function selected(page, id) {
  await expect.poll(async () => (await state(page)).selection.interval).toBe(id);
  await expect.poll(async () => (await state(page)).ui.selected_point.length).toBe(id === null ? 0 : 1);
}

export async function stableCanvas(canvas) {
  let previous;
  let unchanged = 0;
  await expect.poll(async () => {
    const current = await canvas.screenshot();
    unchanged = previous && Buffer.compare(previous, current) === 0 ? unchanged + 1 : 0;
    previous = current;
    return unchanged;
  }).toBeGreaterThanOrEqual(2);
  return previous;
}

export async function fullyClosed(page) {
  await expect.poll(async () => {
    const current = await state(page);
    return [current.summary.closed, current.ui.closed, current.viewer_count, current.summary.listener_errors];
  }).toEqual([true, true, 0, []]);
}

// Read-only fixture telemetry tells us where a plotted mark is drawn. The
// selection itself must travel through browser mouse events, WebSocket, Julia,
// and the rendered update. No test invokes a Julia selection function directly.
export async function pick(page, id, panel, { scroll = true } = {}) {
  const canvas = page.locator('canvas').last();
  if (scroll) await canvas.scrollIntoViewIfNeeded();
  let current = await state(page);
  if (current.selection.interval === id) {
    await control(page, current.fixture === 'ordinary' ? 'interval' : 'slice-interval').selectOption({ index: 0 });
    await selected(page, null);
    // Selecting through the dropdown may scroll the page to that control.
    if (scroll) await canvas.scrollIntoViewIfNeeded();
    else await canvas.locator('xpath=ancestor::div[@tabindex="0"][1]').scrollIntoViewIfNeeded();
    current = await state(page);
  }
  const target = current.ui.pick_targets.find(item => item.id === id && item.panel === panel);
  expect(target).toBeTruthy();
  const box = await canvas.boundingBox();
  const viewport = await page.evaluate(() => ({ width: innerWidth, height: innerHeight }));
  expect(box.x + box.width * target.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width * target.x).toBeLessThan(viewport.width);
  expect(box.y + box.height * target.y).toBeGreaterThanOrEqual(0);
  expect(box.y + box.height * target.y).toBeLessThan(viewport.height);
  const point = { x: box.x + box.width * target.x, y: box.y + box.height * target.y };
  expect(await canvas.evaluate((element, point) => document.elementFromPoint(point.x, point.y) === element, point)).toBe(true);
  await page.mouse.move(box.x + box.width * target.x, box.y + box.height * target.y);
  // WGL throttles pointer delivery. Wait for Julia to see the new coordinates
  // before pressing the button, instead of relying on a fixed timing delay.
  await expect.poll(async () => {
    const updated = await state(page);
    const [x, y] = updated.ui.mouseposition;
    return Math.hypot(x - target.x * current.ui.figure_size[0],
      y - (1 - target.y) * current.ui.figure_size[1]);
  }).toBeLessThan(3);
  await page.mouse.down();
  await page.mouse.up();
  await selected(page, id);
}

export async function capture(page, testInfo, name, { fullPage = true } = {}) {
  const filename = testInfo.outputPath(`${name}.png`);
  if (fullPage) {
    await page.screenshot({ path: filename, fullPage: true });
  } else {
    // Playwright's document-coordinate clip can miss the visible surface at
    // real page zoom. Ask Chromium for that surface without a computed clip.
    const client = await page.context().newCDPSession(page);
    try {
      const { data } = await client.send('Page.captureScreenshot', {
        format: 'png', fromSurface: true, captureBeyondViewport: false,
      });
      await writeFile(filename, Buffer.from(data, 'base64'));
    } finally {
      await client.detach();
    }
  }
  await testInfo.attach(name, { path: filename, contentType: 'image/png' });
  await testInfo.attach(`${name}-state`, {
    body: JSON.stringify(await state(page), null, 2), contentType: 'application/json',
  });
}

export async function checkNarrowLayout(page, testInfo, name) {
  const previous = (await state(page)).selection.interval;
  await page.setViewportSize({ width: 640, height: 1000 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
  await expect(control(page, 'close')).toBeVisible();
  const panels = await scrollableCharts(page);
  expect(panels.length).toBeGreaterThan(0);
  for (const panel of panels) await scrollChart(page, panel, 'left');
  await capture(page, testInfo, name);
  for (const panel of panels) await scrollChart(page, panel, 'right');
  const current = await state(page);
  // The right-hand diagram must remain reachable and pickable after scrolling.
  if (current.ui.pick_targets.length) {
    const id = current.ui.pick_targets.find(target => target.panel === 'diagram').id;
    await pick(page, id, 'diagram', { scroll: false });
    await capture(page, testInfo, `${name}-right`);
    await scrollChart(page, panels.at(-1), 'left');
    await pick(page, id, 'barcode', { scroll: false });
    const suffix = current.fixture === 'ordinary' ? 'interval' : 'slice-interval';
    const index = previous === null ? 0 : current.records.findIndex(record => record.id === previous) + 1;
    await control(page, suffix).selectOption({ index });
    await selected(page, previous);
  } else {
    await capture(page, testInfo, `${name}-right`);
  }
  for (const panel of panels) await scrollChart(page, panel, 'left');
  await page.setViewportSize({ width: 1440, height: 1050 });
}

export async function scrollableCharts(page) {
  const charts = page.locator('div[tabindex="0"][aria-label]').filter({ has: page.locator('canvas') });
  const result = [];
  for (const panel of await charts.all()) {
    if (await panel.isVisible() && await panel.evaluate(element => element.scrollWidth - element.clientWidth > 1)) result.push(panel);
  }
  return result;
}

// Focus the real scroll container and use browser keyboard scrolling. Setting
// scrollLeft in JavaScript would miss keyboard-access and event-routing bugs.
export async function scrollChart(page, panel, side) {
  await panel.scrollIntoViewIfNeeded();
  await panel.focus();
  await expect(panel).toBeFocused();
  const extent = await panel.evaluate(element => element.scrollWidth - element.clientWidth);
  expect(extent).toBeGreaterThan(0);
  const remaining = () => panel.evaluate((element, side) => side === 'right'
    ? element.scrollWidth - element.clientWidth - element.scrollLeft : element.scrollLeft, side);
  for (let i = 0; i < Math.ceil(extent / 20) + 2; i++) {
    if (await remaining() <= 1) break;
    await page.keyboard.press(side === 'right' ? 'ArrowRight' : 'ArrowLeft');
  }
  await expect.poll(remaining).toBeLessThanOrEqual(1);
  const canvas = await panel.locator('canvas').boundingBox();
  const bounds = await panel.boundingBox();
  if (side === 'right') expect(canvas.x + canvas.width).toBeLessThanOrEqual(bounds.x + bounds.width + 1);
  else expect(canvas.x).toBeGreaterThanOrEqual(bounds.x - 1);
}
