import { test, expect } from '@playwright/test';
import { state, control, openInspector, capture, fullyClosed, scrollableCharts, scrollChart } from './helpers.mjs';

const baseURL = `http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT || 8849}`;
async function edit(page, suffix, text) {
  const input = page.locator(`input[id$="-${suffix}"]`);
  await input.fill(text);
  await input.press('Tab');
}
async function pair(page, source, target, timeout = 30_000) {
  for (const [key, value] of [['source-x', source[0]], ['source-y', source[1]],
    ['target-x', target[0]], ['target-y', target[1]]]) await edit(page, key, value);
  const before = (await state(page)).selection.revision;
  const previousError = await control(page, 'error').textContent();
  await control(page, 'select-points').click();
  // Surface a UI failure immediately instead of waiting for a revision that
  // a rejected transaction will never commit.
  await expect.poll(async () => [(await state(page)).selection.revision,
    await control(page, 'error').textContent()], { timeout }).not.toEqual([before, previousError]);
  await expect(control(page, 'error')).toBeEmpty();
  await expect.poll(async () => (await state(page)).selection.revision, { timeout }).toBeGreaterThan(before);
}
async function view(page, label, symbol) {
  await control(page, 'view').selectOption({ label });
  await expect.poll(async () => (await state(page)).selection.view).toBe(symbol);
  await expect.poll(async () => (await state(page)).summary.listener_errors).toEqual([]);
}
async function rankCanvas(page) {
  const canvas = page.locator('[aria-label="Anchored rank section"] canvas');
  await expect(canvas).toHaveCount(1, { timeout: 120_000 });
  await canvas.scrollIntoViewIfNeeded();
  await expect.poll(async () => canvas.evaluate(c => c.wglmakie_screen?.renderer?.info.render.frame || 0)).toBeGreaterThan(0);
}

test('A84 source and target sections preserve ranks, masks, exact selection and lifecycle', async ({ page }, testInfo) => {
  const errors = [];
  await openInspector(page, `${baseURL}/rank-sections`, errors);
  await view(page, 'Rank from source', 'rank_from');
  await expect(page.locator('[aria-label="Anchored rank section"] canvas')).toHaveCount(0);
  // A standalone cold run compiles the first rank canvas here; subsequent
  // interactions retain the normal 30-second deadline.
  await pair(page, ['1/2', '1/2'], ['3/2', '3/2'], 180_000);
  await rankCanvas(page);
  let current = await state(page);
  expect(current.inspection.rank).toBe(1);
  expect(current.inspection.source_dimension).toBe(1);
  expect(current.inspection.target_dimension).toBe(2);
  expect(current.rank_sections[0].all_pairs_table).toBe(false);
  await capture(page, testInfo, 'rank-from-source');

  // Adjacent maps have rank one but their composite is zero.
  await pair(page, ['3/2', '3/2'], ['5/2', '5/2']);
  expect((await state(page)).inspection.rank).toBe(1);
  await pair(page, ['1/2', '1/2'], ['5/2', '5/2']);
  current = await state(page);
  expect(current.inspection.rank).toBe(0);
  expect(current.inspection.defined).toBe(true);
  expect(current.inspection.source_dimension).toBe(1);
  expect(current.inspection.target_dimension).toBe(1);
  await expect(control(page, 'readout')).toContainText(/rank/i);
  await capture(page, testInfo, 'rank-zero-composite');

  await view(page, 'Rank to target', 'rank_to');
  await rankCanvas(page);
  expect((await state(page)).rank_sections[0].from).toBe(false);
  expect((await state(page)).inspection.rank).toBe(0);
  await capture(page, testInfo, 'rank-to-target');

  // Incomparable points in one fiber do not define its identity map.
  await pair(page, ['5/4', '7/4'], ['7/4', '5/4']);
  current = await state(page);
  expect(current.inspection.source).toBe(current.inspection.target);
  expect(current.inspection.defined).toBe(false);
  expect(current.inspection.rank).toBeNull();
  await expect(control(page, 'readout')).toContainText('incomparable');
  await rankCanvas(page);
  await capture(page, testInfo, 'rank-incomparable-same-fiber');

  // A failed text query must preserve both section and selection.
  const before = (await state(page)).selection;
  await edit(page, 'target-x', 'not a coordinate');
  await control(page, 'select-points').click();
  await expect(control(page, 'error')).not.toBeEmpty();
  expect((await state(page)).selection).toEqual(before);
  await pair(page, ['1/2', '1/2'], ['3/2', '3/2']);

  // The rank plane itself picks the varying endpoint and preserves the anchor.
  await view(page, 'Rank from source', 'rank_from');
  await rankCanvas(page);
  current = await state(page);
  const fixedAnchor = current.rank_anchors;
  const rankPick = current.rank_ui.pick_targets.find(t => t.id === 'second');
  const sectionCanvas = page.locator('[aria-label="Anchored rank section"] canvas');
  const sectionBox = await sectionCanvas.boundingBox();
  await page.mouse.move(sectionBox.x + sectionBox.width * rankPick.x, sectionBox.y + sectionBox.height * rankPick.y);
  await expect.poll(async () => {
    const [x, y] = (await state(page)).rank_ui.mouseposition;
    return Math.hypot(x - rankPick.x * current.rank_ui.figure_size[0],
      y - (1 - rankPick.y) * current.rank_ui.figure_size[1]);
  }).toBeLessThan(3);
  expect((await state(page)).summary.snapshot_builds).toBe(current.summary.snapshot_builds);
  await page.mouse.down();
  await page.mouse.up();
  await expect.poll(async () => (await state(page)).selection.revision).toBeGreaterThan(current.selection.revision);
  expect((await state(page)).rank_anchors).toEqual(fixedAnchor);
  expect((await state(page)).inspection.rank).toBe(0);
  await rankCanvas(page);

  // Hover over the navigation canvas reads dimensions without new rank work.
  const navigation = page.locator('[aria-label="Parameter regions and finite poset"] canvas');
  await navigation.scrollIntoViewIfNeeded();
  current = await state(page);
  const pick = current.main_ui.pick_targets.find(t => t.panel === 'region');
  const rect = await navigation.boundingBox();
  await page.mouse.move(rect.x + rect.width * pick.x, rect.y + rect.height * pick.y);
  await expect.poll(async () => {
    const [x, y] = (await state(page)).main_ui.mouseposition;
    return Math.hypot(x - pick.x * current.main_ui.figure_size[0],
      y - (1 - pick.y) * current.main_ui.figure_size[1]);
  }).toBeLessThan(3);
  await expect(control(page, 'hover')).toContainText(`Vertex ${pick.vertex}; dimension ${pick.dimension}.`);
  expect((await state(page)).summary.snapshot_builds).toBe(current.summary.snapshot_builds);
  await page.mouse.click(rect.x + rect.width * pick.x, rect.y + rect.height * pick.y);
  await expect.poll(async () => (await state(page)).selection.revision).toBeGreaterThan(current.selection.revision);
  await expect(control(page, 'error')).toBeEmpty();
  await rankCanvas(page);
  await expect.poll(async () => (await state(page)).summary.listener_errors).toEqual([]);

  await page.setViewportSize({ width: 390, height: 900 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
  const charts = await scrollableCharts(page);
  expect(charts.length).toBeGreaterThan(0);
  const mobileSelection = (await state(page)).selection;
  for (const chart of charts) {
    await scrollChart(page, chart, 'right');
    await scrollChart(page, chart, 'left');
  }
  expect((await state(page)).selection).toEqual(mobileSelection);
  await capture(page, testInfo, 'rank-mobile');
  await view(page, 'Module coordinates', 'module');
  await expect(page.locator('[aria-label="Anchored rank section"] canvas')).toHaveCount(0);
  await view(page, 'Rank from source', 'rank_from');
  await control(page, 'reset').click();
  await expect.poll(async () => (await state(page)).rank_sections).toEqual([]);
  await expect(page.locator('[aria-label="Anchored rank section"] canvas')).toHaveCount(0);
  await control(page, 'close').click();
  await fullyClosed(page);
  expect(errors).toEqual([]);
});
