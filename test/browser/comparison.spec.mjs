import { test, expect } from '@playwright/test';
import { state, keyboardActivate, control, openInspector, capture, fullyClosed, scrollChart } from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page) {
  await expect.poll(async()=>(await state(page)).summary.updating,{timeout:120_000}).toBe(false);
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].some(c=>c.wglmakie_screen?.renderer?.info.render.frame>0),null,{timeout:120_000});
  expect((await state(page)).summary.listener_errors).toEqual([]);
}
async function changed(page,suffix) {
  await settled(page);const revision=(await state(page)).summary.revision;
  await keyboardActivate(control(page,suffix));
  await expect.poll(async()=>(await state(page)).summary.revision,{timeout:120_000}).toBeGreaterThan(revision);
  await expect(control(page,'error')).toBeEmpty();await settled(page);
}
async function edit(page,suffix,value) {
  const el=page.locator(`input[id$="-${suffix}"]`);await el.fill(value);await el.press('Tab');
}
async function pick(page,probe) {
  await settled(page);const t=(await state(page)).ui.pick_targets.find(t=>t.axis_index===1&&t.probe===probe);
  expect(t).toBeTruthy();const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  let box=await canvas.boundingBox();await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+t.y*box.height);box=await canvas.boundingBox();
  const builds=(await state(page)).summary.snapshot_builds;
  await page.mouse.move(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect(control(page,'hover')).toHaveText(t.label);expect((await state(page)).summary.snapshot_builds).toBe(builds);
  await page.mouse.click(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect.poll(async()=>(await state(page)).selection.probe).toBe(probe);await settled(page);
}
test('A94 fields, rank differences, masks, linked selections and cleanup',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/comparisons`,errors);await settled(page);
  expect(await control(page,'apply').evaluate(el=>parseFloat(getComputedStyle(el).fontSize))).toBeGreaterThanOrEqual(16);
  expect((await state(page)).values).toEqual({f2:[2,2],f3:[2,2]});
  await changed(page,'ranks');expect((await state(page)).differences.f3).toEqual([0,1]);
  const rows=(await state(page)).summary.row_builds;
  await pick(page,2);expect((await state(page)).summary.row_builds).toBe(rows);
  await capture(page,info,'field-dependent-rank');
  await changed(page,'matrices');await expect(control(page,'readout')).toContainText('f2');
  await capture(page,info,'selected-matrices');
  await edit(page,'baseline','f3');await changed(page,'apply');expect((await state(page)).differences.f2).toEqual([0,-1]);
  const rev=(await state(page)).summary.revision;
  await edit(page,'probe','999');await control(page,'apply').click();
  await expect(control(page,'error')).not.toBeEmpty();expect((await state(page)).summary.revision).toBe(rev);
  await edit(page,'probe','1');await edit(page,'source','2');await changed(page,'apply');
  expect((await state(page)).values.f2).toEqual([null,2]);await expect(control(page,'readout')).toContainText('unordered');
  const linked=await page.context().newPage();await openInspector(linked,`${base}/comparisons`,errors);
  await changed(linked,'dimensions');await expect.poll(async()=>(await state(page)).selection.view).toBe('hilbert');
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  await changed(page,'reset');await page.setViewportSize({width:480,height:900});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await scrollChart(page,page.locator('[aria-label="Comparison charts"]'),'right');await capture(page,info,'narrow-differences');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
test('A94 ordinary bars, endpoint dimensions and synchronized query marker',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/comparisons-ordinary`,errors);await settled(page);
  expect((await state(page)).values).toEqual({f2:[1,1,1],f3:[1,0,0]});
  await edit(page,'probe','2');await changed(page,'apply');await capture(page,info,'characteristic-barcode');
  await expect(control(page,'readout')).toContainText('difference -1');
  await changed(page,'ranks');expect((await state(page)).values.f3).toEqual([1,0,0]);
  await changed(page,'dimensions');await changed(page,'barcodes');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
