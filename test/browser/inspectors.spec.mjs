import { test, expect } from '@playwright/test';

const ordinaryURL = `http://127.0.0.1:${process.env.TAMEROP_BROWSER_INTERVAL_PORT || 8848}`;
const sliceURL = `http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT || 8849}`;
import { state, control, openInspector, selected, stableCanvas, fullyClosed, pick, capture, checkNarrowLayout } from './helpers.mjs';

test('ordinary intervals: real picking, duplicate members, retained chains and lifecycle', async ({ page, context }, testInfo) => {
  const errors = [];
  await openInspector(page, ordinaryURL, errors);
  await control(page, 'reset').click();
  await selected(page, null);
  expect((await state(page)).records.map(r => [r.birth, r.death, r.multiplicity])).toEqual([[0, 1, 2], [0, 'Inf', 1]]);
  await expect(control(page, 'member')).toBeDisabled();
  const canvas = page.locator('canvas').last();
  const before = await stableCanvas(canvas);
  await pick(page, 1, 'barcode');
  await expect(control(page, 'interval').locator('option:checked')).toHaveText('#1 [0, 1) x2');
  await expect.poll(async () => Buffer.compare(before, await canvas.screenshot())).not.toBe(0);

  await control(page, 'representative').click();
  await expect(control(page, 'error')).not.toBeEmpty();
  await expect(control(page, 'representative')).not.toBeChecked();
  expect((await state(page)).selection.interval).toBe(1);
  expect((await state(page)).selection.member).toBeNull();

  await control(page, 'member').fill('1');
  await control(page, 'apply-member').click();
  await expect.poll(async () => (await state(page)).selection.member).toBe(1);
  await control(page, 'representative').click();
  await expect.poll(async () => (await state(page)).representative?.available).toBe(true);
  const first = (await state(page)).representative;
  expect(first.cycle.cell_ids.length).toBeGreaterThan(0);
  expect(first.bounding_chain.cell_ids.length).toBeGreaterThan(0);
  await expect(control(page, 'readout')).toContainText('Bounding chain at death');
  await capture(page, testInfo, 'ordinary-finite-representative');

  await control(page, 'member').fill('2');
  await control(page, 'apply-member').click();
  await expect.poll(async () => (await state(page)).selection.member).toBe(2);
  await expect(control(page, 'representative')).not.toBeChecked();
  await control(page, 'representative').click();
  await expect.poll(async () => (await state(page)).representative?.available).toBe(true);
  expect((await state(page)).representative.cycle.cell_ids).not.toEqual(first.cycle.cell_ids);

  const valid = (await state(page)).selection;
  await control(page, 'member').fill('999');
  await control(page, 'apply-member').click();
  await expect(control(page, 'error')).not.toBeEmpty();
  expect((await state(page)).selection).toEqual(valid);
  await pick(page, 2, 'diagram');
  await control(page, 'member').fill('1');
  await control(page, 'apply-member').click();
  await expect.poll(async () => (await state(page)).selection.member).toBe(1);
  await control(page, 'representative').click();
  await expect.poll(async () => (await state(page)).representative?.available).toBe(true);
  expect((await state(page)).representative.bounding_chain).toBeNull();
  await capture(page, testInfo, 'ordinary-essential-representative');

  // Actual keyboard navigation changes the same interval group.
  await control(page, 'interval').focus();
  await page.keyboard.press('Home');
  await selected(page, null);
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Tab');
  await selected(page, 1);
  await checkNarrowLayout(page, testInfo, 'ordinary-narrow');

  const linked = await context.newPage();
  await openInspector(linked, ordinaryURL, errors);
  await selected(linked, 1);
  await control(linked, 'interval').selectOption({ label: '#2 [0, Inf) x1' });
  await selected(page, 2);
  await selected(linked, 2);
  await linked.close();
  // Bonito keeps disconnected clients for a 30-second reconnect grace period.
  await expect.poll(async () => (await state(page)).viewer_count, { timeout: 45_000 }).toBe(1);
  await openInspector(page, ordinaryURL, errors);
  await selected(page, 2);
  await control(page, 'reset').click();
  await selected(page, null);
  const closingPeer = await context.newPage();
  await openInspector(closingPeer, ordinaryURL, errors);
  await control(page, 'close').click();
  await fullyClosed(page);
  await fullyClosed(closingPeer);
  await expect(control(closingPeer, 'close')).toBeDisabled();
  await capture(page, testInfo, 'ordinary-closed');
  expect(errors).toEqual([]);
});

test('slices: certified infinity, window censoring, drafts, canvas selection and linked views', async ({ page, context }, testInfo) => {
  const errors = [];
  await openInspector(page, sliceURL, errors);
  const scope = control(page, 'slice-scope');
  const apply = control(page, 'slice-apply');
  const interval = control(page, 'slice-interval');
  const initial = await state(page);
  const globalEndpoints = [['-Inf', '1'], ['-Inf', 'Inf'], ['0', 'Inf']];
  expect(initial.records.map(r => [r.birth, r.death])).toEqual(globalEndpoints);
  expect(initial.slice.essential_status).toBe('certified');
  const sliceCanvas = page.locator('canvas').last();
  const unselected = await stableCanvas(sliceCanvas);
  for (const id of [1, 2, 3]) {
    await pick(page, id, id === 2 ? 'diagram' : 'barcode');
    if (id === 1) await expect.poll(async () => Buffer.compare(unselected, await sliceCanvas.screenshot())).not.toBe(0);
    const selectedState = await state(page);
    expect(selectedState.ui.selected_region.length).toBeGreaterThan(0);
    expect(selectedState.ui.selected_region.flat().every(Number.isFinite)).toBe(true);
  }
  await capture(page, testInfo, 'slices-certified');
  // Keep a full-resolution crop of the actual charts for finite/infinity tick
  // spacing review. Native renderer tests measure the individual glyph bounds.
  const chartName = `slices-certified-charts-font-${initial.style.fontsize}`;
  const chartPath = testInfo.outputPath(`${chartName}.png`);
  await sliceCanvas.screenshot({ path: chartPath });
  await testInfo.attach(chartName, { path: chartPath, contentType: 'image/png' });
  expect((await state(page)).records.map(r => [r.birth, r.death])).toEqual(globalEndpoints);

  await scope.selectOption({ label: 'Viewing-window restriction' });
  await expect(control(page, 'slice-draft')).toContainText('unapplied changes');
  expect((await state(page)).selection.slice_scope).toBe('global');
  await apply.click();
  await expect.poll(async () => (await state(page)).selection.slice_scope).toBe('window');
  await selected(page, null);
  const windowed = await state(page);
  expect(windowed.records.every(r => !['Inf', '-Inf'].includes(r.birth) && !['Inf', '-Inf'].includes(r.death))).toBe(true);
  expect(windowed.records.some(r => r.left_clipped || r.right_clipped)).toBe(true);
  await expect(control(page, 'slice-status')).toContainText('window');
  await interval.selectOption({ index: 1 });
  await selected(page, 1);
  await capture(page, testInfo, 'slices-window-cuts');

  await scope.selectOption({ label: 'Whole line (certified endpoints)' });
  await apply.click();
  await expect.poll(async () => (await state(page)).selection.slice_scope).toBe('global');
  expect((await state(page)).records.map(r => [r.birth, r.death])).toEqual(globalEndpoints);
  const beforeInvalid = (await state(page)).selection;
  await control(page, 'slice-direction-x').fill('-1');
  await apply.click();
  await expect(control(page, 'error')).not.toBeEmpty();
  expect((await state(page)).selection).toEqual(beforeInvalid);
  await control(page, 'slice-direction-x').fill('1');
  await apply.click();
  await expect(control(page, 'error')).toBeEmpty();

  const oldLine = (await state(page)).selection.slice;
  await control(page, 'slice-angle').focus();
  await page.keyboard.press('ArrowRight');
  await expect(control(page, 'slice-draft')).toContainText('Draft');
  expect((await state(page)).selection.slice).toEqual(oldLine);
  await apply.click();
  await expect.poll(async () => (await state(page)).selection.slice).not.toEqual(oldLine);
  // Restore the exact reference line before checking shared interval identities.
  for (const [suffix, value] of [['base-x', '0'], ['base-y', '0'], ['direction-x', '1'], ['direction-y', '1']]) {
    await control(page, `slice-${suffix}`).fill(value);
    await control(page, `slice-${suffix}`).press('Tab');
  }
  await apply.click();
  await expect.poll(async () => (await state(page)).records.map(r => [r.birth, r.death])).toEqual(globalEndpoints);
  await checkNarrowLayout(page, testInfo, 'slices-narrow');

  const linked = await context.newPage();
  await openInspector(linked, sliceURL, errors);
  await control(linked, 'slice-interval').selectOption({ label: '#2 (-Inf, Inf) x1' });
  await selected(page, 2);
  await selected(linked, 2);
  await linked.close();
  await expect.poll(async () => (await state(page)).viewer_count, { timeout: 45_000 }).toBe(1);
  await pick(page, 3, 'diagram');
  const closingPeer = await context.newPage();
  await openInspector(closingPeer, sliceURL, errors);
  await control(page, 'close').click();
  await fullyClosed(page);
  await fullyClosed(closingPeer);
  await expect(control(closingPeer, 'close')).toBeDisabled();
  await capture(page, testInfo, 'slices-closed');
  expect(errors).toEqual([]);
});
