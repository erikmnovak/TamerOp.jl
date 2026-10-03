import { test, expect } from '@playwright/test';
import { state, control, openInspector, selected, fullyClosed, pick, capture, checkNarrowLayout } from './helpers.mjs';

const baseURL = `http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT || 8849}`;

// These examples reproduce the two-square checks manually accepted on
// 30 September. The mathematical oracles come from the direct sum of the
// closed square modules [0,2]^2 and [1,3]^2, not from a second call to the API.
const query = selection => ({
  vertex: selection.vertex, pair: selection.pair,
  query_points: selection.query_points, parameter_relation: selection.parameter_relation,
});
const work = summary => [summary.revision, summary.snapshot_builds,
  summary.cache_hits, summary.cache_misses, summary.cache_entries,
  summary.slice_cache_hits, summary.slice_cache_misses, summary.slice_cache_entries];

async function edit(page, suffix, value) {
  // "vertex" is also a suffix of the three navigation buttons.
  const input = page.locator(`input[id$="-${suffix}"]`);
  await input.fill(value);
  await input.press('Tab');
}

async function clickAndCommit(page, suffix) {
  const revision = (await state(page)).selection.revision;
  await control(page, suffix).click();
  await expect.poll(async () => (await state(page)).selection.revision).toBeGreaterThan(revision);
  await expect(control(page, 'error')).toBeEmpty();
}

async function exactPoint(page, [x, y]) {
  await edit(page, 'point-x', x);
  await edit(page, 'point-y', y);
  await clickAndCommit(page, 'select-point');
}

async function exactPair(page, [sx, sy], [tx, ty]) {
  for (const [suffix, value] of [['source-x', sx], ['source-y', sy], ['target-x', tx], ['target-y', ty]]) {
    await edit(page, suffix, value);
  }
  await clickAndCommit(page, 'select-points');
}

async function line(page, basepoint, direction = ['1', '1']) {
  for (const [suffix, value] of [['base-x', basepoint[0]], ['base-y', basepoint[1]],
    ['direction-x', direction[0]], ['direction-y', direction[1]]]) {
    await edit(page, `slice-${suffix}`, value);
  }
  await clickAndCommit(page, 'slice-apply');
}

async function endpoints(page, expected) {
  await expect.poll(async () => (await state(page)).records.map(record =>
    [record.birth, record.death, record.left_closed, record.right_closed, record.multiplicity]))
    .toEqual(expected.map(([birth, death]) => [birth, death, true, true, 1]));
}

async function mainPointer(page, id, panel, click = false) {
  const canvas = page.locator('[aria-label="Parameter regions and finite poset"] canvas');
  await canvas.scrollIntoViewIfNeeded();
  const current = await state(page);
  const target = current.main_ui.pick_targets.find(item => item.id === id && item.panel === panel);
  expect(target).toBeTruthy();
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width * target.x, box.y + box.height * target.y);
  await expect.poll(async () => {
    const [x, y] = (await state(page)).main_ui.mouseposition;
    return Math.hypot(x - target.x * current.main_ui.figure_size[0],
      y - (1 - target.y) * current.main_ui.figure_size[1]);
  }).toBeLessThan(3);
  await expect(control(page, 'hover')).toContainText(`Vertex ${target.vertex}; dimension ${target.dimension}.`);
  if (click) {
    const revision = (await state(page)).selection.revision;
    await page.mouse.down();
    await page.mouse.up();
    await expect.poll(async () => (await state(page)).selection.revision).toBeGreaterThan(revision);
  }
  return target;
}

function matrixProduct(a, b) {
  expect(a.size[1]).toBe(b.size[0]);
  // All entries of this particular fixture's presentation are exact 0 or 1.
  const numeric = entry => {
    expect(['0', '1']).toContain(String(entry));
    return Number(entry);
  };
  return a.rows.map(row => Array.from({ length: b.size[1] }, (_, col) =>
    row.reduce((total, entry, inner) => total + numeric(entry) * numeric(b.rows[inner][col]), 0)));
}

test('previous manual A37 checks: exact spaces, maps, presentation bases, hover and linked lifecycle', async ({ page, context }, testInfo) => {
  const errors = [];
  const url = `${baseURL}/squares`;
  await openInspector(page, url, errors);
  await expect(page.locator('body')).not.toContainText('Bonito.Dropdown(');
  for (const [point, dimension] of [[['1/2', '1/2'], 1], [['3/2', '3/2'], 2],
    [['5/2', '5/2'], 1], [['2', '2'], 2],
    [['20000000000000000000001/10000000000000000000000', '2'], 1]]) {
    await exactPoint(page, point);
    expect((await state(page)).inspection.dimension).toBe(dimension);
    expect((await state(page)).selection.input).toBe('provided');
  }

  const first = ['1/2', '1/2'], overlap = ['3/2', '3/2'], second = ['5/2', '5/2'];
  for (const [source, target, rank, shape] of [[first, overlap, 1, [2, 1]],
    [overlap, second, 1, [1, 2]], [first, second, 0, [1, 1]]]) {
    await exactPair(page, source, target);
    const map = (await state(page)).inspection;
    expect(map.defined).toBe(true);
    expect(map.rank).toBe(rank);
    expect(map.matrix_size).toEqual(shape);
    expect(map.kernel_dimension).toBe(shape[1] - rank);
    expect(map.image_dimension).toBe(rank);
  }
  await expect(control(page, 'readout')).toContainText('rank 0');
  await capture(page, testInfo, 'manual-module-zero-composite');

  const zeroQuery = query((await state(page)).selection);
  await control(page, 'view').selectOption({ label: 'Presentation image coordinates' });
  await expect.poll(async () => (await state(page)).selection.view).toBe('presentation');
  expect(query((await state(page)).selection)).toEqual(zeroQuery);
  const presentation = (await state(page)).presentation;
  expect(presentation.defined).toBe(true);
  expect(presentation.map).toEqual({ size: [1, 1], rows: [['0']] });
  expect(presentation.stalks.map(stalk => stalk.dimension)).toEqual([1, 1]);
  expect(matrixProduct(presentation.stalks[1].basis, presentation.map))
    .toEqual(matrixProduct(presentation.ambient_projection, presentation.stalks[0].basis));

  await exactPoint(page, ['1/2', '5/2']);
  await control(page, 'upset').selectOption({ label: 'U1' });
  await control(page, 'downset').selectOption({ label: 'D2' });
  await expect.poll(async () => (await state(page)).selection.downset).toBe(2);
  let active = (await state(page)).presentation.stalks[0];
  expect(active.dimension).toBe(0);
  expect(active.active_rows).toEqual([2]);
  expect(active.active_columns).toEqual([1]);
  expect(active.matrix).toEqual({ size: [1, 1], rows: [['0']] });
  expect(active.basis).toBeNull();
  await control(page, 'basis').check();
  await expect.poll(async () => (await state(page)).selection.basis).toBe(true);
  active = (await state(page)).presentation.stalks[0];
  expect(active.basis).toEqual({ size: [1, 0], rows: [[]] });
  await expect(control(page, 'readout')).toContainText('Empty matrix (1 x 0)');
  await capture(page, testInfo, 'manual-active-zero-image-basis');

  const activeQuery = query((await state(page)).selection);
  await control(page, 'upset').selectOption({ label: 'U2' });
  await control(page, 'downset').selectOption({ label: 'D1' });
  await expect.poll(async () => [(await state(page)).selection.upset, (await state(page)).selection.downset]).toEqual([2, 1]);
  expect(query((await state(page)).selection)).toEqual(activeQuery);
  expect((await state(page)).presentation.stalks[0].matrix.rows).toEqual([['0']]);
  await control(page, 'view').selectOption({ label: 'Module coordinates' });
  await expect.poll(async () => (await state(page)).selection.view).toBe('module');
  expect(query((await state(page)).selection)).toEqual(activeQuery);
  expect((await state(page)).inspection.dimension).toBe(0);

  await exactPair(page, ['5/4', '7/4'], ['7/4', '5/4']);
  const incomparable = await state(page);
  expect(incomparable.selection.pair[0]).toBe(incomparable.selection.pair[1]);
  expect(incomparable.inspection.parameter_relation).toBe('incomparable');
  expect(incomparable.inspection.defined).toBe(false);
  expect(incomparable.inspection.matrix).toBeNull();
  expect(incomparable.inspection.rank).toBeNull();
  await expect(control(page, 'readout')).toContainText('not a zero matrix');
  await capture(page, testInfo, 'manual-incomparable-equal-labels');

  // Hover must remain a cheap label/dimension query, with no new algebra,
  // snapshot, basis request, or committed selection.
  const beforeHover = await state(page);
  await mainPointer(page, 'first', 'region');
  const afterHover = await state(page);
  expect(afterHover.selection).toEqual(beforeHover.selection);
  expect(work(afterHover.summary)).toEqual(work(beforeHover.summary));
  await control(page, 'endpoint').selectOption({ label: 'Stalk' });
  const picked = await mainPointer(page, 'overlap', 'region', true);
  expect((await state(page)).selection.vertex).toBe(picked.vertex);
  expect((await state(page)).inspection.dimension).toBe(2);
  expect((await state(page)).selection.input).toBe('pointer');
  // Pick that actual finite-poset vertex as well as the parameter region.
  await mainPointer(page, picked.vertex, 'hasse', true);
  expect((await state(page)).selection.vertex).toBe(picked.vertex);
  expect((await state(page)).selection.query_points).toEqual([]);

  await control(page, 'endpoint').selectOption({ label: 'Source' });
  await mainPointer(page, 'first', 'region', true);
  await control(page, 'endpoint').selectOption({ label: 'Target' });
  await mainPointer(page, 'second', 'region', true);
  expect((await state(page)).inspection.rank).toBe(0);
  expect((await state(page)).main_ui.markers.region.source).toHaveLength(1);
  expect((await state(page)).main_ui.markers.region.target).toHaveLength(1);
  await control(page, 'endpoint').selectOption({ label: 'Stalk' });

  await exactPoint(page, overlap);
  const valid = await state(page);
  await edit(page, 'point-x', '1/0');
  await control(page, 'select-point').click();
  await expect(control(page, 'error')).not.toBeEmpty();
  expect((await state(page)).selection).toEqual(valid.selection);
  expect(work((await state(page)).summary)).toEqual(work(valid.summary));
  await exactPoint(page, first);

  await edit(page, 'vertex', '1');
  await control(page, 'select-vertex').focus();
  await page.keyboard.press('Enter');
  await expect.poll(async () => (await state(page)).selection.vertex).toBe(1);
  await control(page, 'next-vertex').focus();
  await page.keyboard.press('Enter');
  await expect.poll(async () => (await state(page)).selection.vertex).toBe(2);
  await control(page, 'previous-vertex').click();
  await expect.poll(async () => (await state(page)).selection.vertex).toBe(1);

  await clickAndCommit(page, 'reset');
  expect(query((await state(page)).selection)).toMatchObject({ vertex: null, pair: null, query_points: [] });
  await exactPair(page, first, second);
  const linked = await context.newPage();
  await openInspector(linked, url, errors);
  expect((await state(linked)).inspection.rank).toBe(0);
  await exactPoint(linked, overlap);
  await expect.poll(async () => (await state(page)).inspection.dimension).toBe(2);
  await linked.close();
  await expect.poll(async () => (await state(page)).viewer_count, { timeout: 45_000 }).toBe(1);
  await openInspector(page, url, errors);
  expect((await state(page)).inspection.dimension).toBe(2);
  const closingPeer = await context.newPage();
  await openInspector(closingPeer, url, errors);
  await control(page, 'close').click();
  await fullyClosed(page);
  await fullyClosed(closingPeer);
  await expect(control(closingPeer, 'close')).toBeDisabled();
  await capture(page, testInfo, 'manual-module-closed');
  expect(errors).toEqual([]);
});

test('previous manual A40/A41 checks: closed-square slices, singleton and empty restrictions, drafts and integration', async ({ page, context }, testInfo) => {
  const errors = [];
  const url = `${baseURL}/squares-slices`;
  await openInspector(page, url, errors);
  await endpoints(page, [['0', '2'], ['1', '3']]);
  expect((await state(page)).selection.slice_scope).toBe('window');
  for (const id of [1, 2]) {
    await pick(page, id, 'barcode');
    await pick(page, id, 'diagram');
    expect((await state(page)).ui.selected_region.length).toBe(2);
  }
  await control(page, 'slice-interval').selectOption({ label: '#1 [0, 2] x1' });
  await selected(page, 1);
  await capture(page, testInfo, 'manual-slices-diagonal');

  const initial = await state(page);
  for (const suffix of ['slice-angle', 'slice-offset']) {
    const previousValue = await control(page, suffix).inputValue();
    const previousDraft = await control(page, 'slice-draft').textContent();
    await control(page, suffix).focus();
    await page.keyboard.press('ArrowRight');
    await expect(control(page, suffix)).not.toHaveValue(previousValue);
    await expect(control(page, 'slice-draft')).not.toHaveText(previousDraft);
    await expect(control(page, 'slice-draft')).toContainText('Draft');
  }
  const drafted = await state(page);
  expect(drafted.selection).toEqual(initial.selection);
  expect(work(drafted.summary)).toEqual(work(initial.summary));
  expect(drafted.ui.rebuild_count).toBe(initial.ui.rebuild_count);
  await clickAndCommit(page, 'slice-apply');
  expect((await state(page)).ui.rebuild_count).toBe(initial.ui.rebuild_count + 1);
  await selected(page, null);

  await line(page, ['0', '1']);
  await endpoints(page, [['0', '1'], ['1', '2']]);
  await line(page, ['0', '2']);
  await endpoints(page, [['0', '0'], ['1', '1']]);
  expect((await state(page)).records.every(record => record.singleton)).toBe(true);
  await pick(page, 1, 'barcode');
  await pick(page, 2, 'diagram');
  const singleton = await state(page);
  expect(singleton.ui.selected_region[0]).toEqual(singleton.ui.selected_region[1]);
  expect(singleton.ui.selected_region[0]).toEqual([1, 3]);
  expect(singleton.ui.selected_region_point).toEqual([[1, 3]]);
  await capture(page, testInfo, 'manual-slices-singletons');

  await line(page, ['0', '3']);
  await endpoints(page, []);
  await selected(page, null);
  expect((await state(page)).ui.selected_region).toEqual([]);
  await expect(control(page, 'slice-interval')).toBeDisabled();
  await capture(page, testInfo, 'manual-slices-empty');
  await line(page, ['10', '0'], ['0', '1']);
  expect((await state(page)).slice.window).toBeNull();
  expect((await state(page)).records).toEqual([]);

  for (const [x, y, error] of [
    ['not-a-number', '1', 'Enter an integer, decimal, p/q, or p//q.'],
    ['0', '0', 'slice direction must be nonnegative and nonzero'],
    ['-1', '1', 'slice direction must be nonnegative and nonzero'],
  ]) {
    // Each rejection must follow a successful submission that clears the last
    // error; an ignored Apply cannot pass by reusing an earlier error message.
    await line(page, ['0', '0']);
    await endpoints(page, [['0', '2'], ['1', '3']]);
    const valid = await state(page);
    await edit(page, 'slice-direction-x', x);
    await edit(page, 'slice-direction-y', y);
    await control(page, 'slice-apply').click();
    await expect(control(page, 'error')).toContainText(error);
    expect((await state(page)).selection).toEqual(valid.selection);
    expect((await state(page)).records).toEqual(valid.records);
    expect((await state(page)).ui.rebuild_count).toBe(valid.ui.rebuild_count);
    expect(work((await state(page)).summary)).toEqual(work(valid.summary));
  }
  await line(page, ['0', '1']);
  await endpoints(page, [['0', '1'], ['1', '2']]);
  await exactPair(page, ['1/2', '1/2'], ['5/2', '5/2']);
  const beforeHide = query((await state(page)).selection);
  await clickAndCommit(page, 'slice-disable');
  expect((await state(page)).selection.slice).toBeNull();
  expect((await state(page)).records).toEqual([]);
  expect(query((await state(page)).selection)).toEqual(beforeHide);
  expect((await state(page)).inspection.rank).toBe(0);
  await expect(page.locator('[aria-label="Linked slice barcode and persistence diagram"] canvas')).toHaveCount(0);

  await line(page, ['0', '0']);
  await control(page, 'slice-interval').selectOption({ label: '#2 [1, 3] x1' });
  await selected(page, 2);
  const beforeView = await state(page);
  await control(page, 'view').selectOption({ label: 'Presentation image coordinates' });
  await expect.poll(async () => (await state(page)).selection.view).toBe('presentation');
  expect((await state(page)).selection.slice).toEqual(beforeView.selection.slice);
  expect((await state(page)).selection.interval).toBe(2);
  expect((await state(page)).ui.rebuild_count).toBe(beforeView.ui.rebuild_count);
  expect((await state(page)).presentation.map.rows).toEqual([['0']]);
  await exactPoint(page, ['3/2', '3/2']);
  expect((await state(page)).selection.slice).toEqual(beforeView.selection.slice);
  await control(page, 'view').selectOption({ label: 'Module coordinates' });
  await expect.poll(async () => (await state(page)).selection.view).toBe('module');
  await clickAndCommit(page, 'reset');
  await selected(page, null);
  const reset = await state(page);
  expect(query(reset.selection)).toMatchObject({ vertex: null, pair: null, query_points: [] });
  expect(reset.selection.slice).toEqual(beforeView.selection.slice);
  await endpoints(page, [['0', '2'], ['1', '3']]);
  await checkNarrowLayout(page, testInfo, 'manual-slices-narrow');

  const linked = await context.newPage();
  await openInspector(linked, url, errors);
  await line(linked, ['0', '1']);
  await endpoints(page, [['0', '1'], ['1', '2']]);
  await control(page, 'slice-interval').selectOption({ label: '#1 [0, 1] x1' });
  await selected(linked, 1);
  await openInspector(page, url, errors);
  await selected(page, 1);
  await endpoints(page, [['0', '1'], ['1', '2']]);
  await control(page, 'close').click();
  await fullyClosed(page);
  await fullyClosed(linked);
  await expect(control(linked, 'close')).toBeDisabled();
  await capture(page, testInfo, 'manual-slices-closed');
  expect(errors).toEqual([]);
});
