import { test, expect } from '@playwright/test';
import { checkKeyboardControls, keyboardActivate, scrollChart } from './helpers.mjs';
const url = `http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT || 8849}/volume`;
const state = async page => JSON.parse(await page.getByTestId('volume-state').textContent());
const slider = (page, axis) => page.getByRole('slider', { name: `Slice index on axis ${axis}` });
const expected = (j,l) => [1,2,3,4].map(k => [1000 + 100*j + 10*k + l, 2000 + 100*j + 10*k + l]);
async function open(page, errors) {
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 300_000 });
  await page.getByTestId('volume-state').waitFor({ state: 'attached', timeout: 300_000 });
  await page.waitForFunction(() => document.querySelector('canvas')?.wglmakie_screen?.renderer?.info.render.frame > 0, null, { timeout: 300_000 });
  await checkKeyboardControls(page);
}
test('A36/A37 array slices: labelled keyboard controls, pointer, unequal axes and teardown', async ({ page }, info) => {
  const errors = [];
  await open(page, errors);
  expect((await state(page)).rows).toEqual(expected(1,2));
  expect([(await state(page)).xlabel, (await state(page)).ylabel]).toEqual(['axis 3','axis 1']);
  await slider(page,2).focus();
  const outline = await slider(page,2).evaluate(el => ({ style: getComputedStyle(el).outlineStyle, width: parseFloat(getComputedStyle(el).outlineWidth) }));
  expect(outline.style).not.toBe('none'); expect(outline.width).toBeGreaterThan(0);
  await slider(page,2).press('End');
  await expect.poll(async () => (await state(page)).rows).toEqual(expected(3,2));
  await slider(page,4).press('Home');
  await expect.poll(async () => (await state(page)).rows).toEqual(expected(3,1));
  await slider(page,2).press('ArrowLeft');
  await expect.poll(async () => (await state(page)).rows).toEqual(expected(2,1));
  // A real pointer action reaches the same native control and Julia matrix.
  const box = await slider(page,4).boundingBox();
  await page.mouse.click(box.x + box.width - 5, box.y + box.height / 2);
  await expect.poll(async () => (await state(page)).rows).toEqual(expected(2,2));
  const second = await page.context().newPage(); await open(second,errors);
  await expect.poll(async () => (await state(page)).viewer_count).toBe(2);
  await slider(second,2).press('End');
  await expect.poll(async () => (await state(second)).rows).toEqual(expected(3,2));
  expect((await state(page)).rows).toEqual(expected(2,2));
  await second.close();
  await expect.poll(async () => (await state(page)).viewer_count, { timeout: 45_000 }).toBe(1);
  await page.setViewportSize({ width: 480, height: 900 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await scrollChart(page, page.getByLabel('Volume slice chart'), 'right');
  await page.screenshot({ path: info.outputPath('volume-narrow.png'), fullPage:true });
  await keyboardActivate(page.getByRole('button', { name:'Close viewer' }));
  await expect.poll(async () => { const s=await state(page); return [s.closed,s.observers,s.retained_source,s.viewer_count]; }).toEqual([true,0,false,0]);
  expect(errors).toEqual([]);
});
