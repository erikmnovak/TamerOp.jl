import { test, expect } from '@playwright/test';
import { state, keyboardActivate, openInspector, capture, fullyClosed, scrollChart } from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
const control=(page,suffix)=>page.locator(`[id^="tamerop-collection-inspector-"][id$="-${suffix}"]`);
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
async function edit(page,suffix,value) {const el=control(page,suffix);await el.fill(value);await el.press('Tab');}
async function pick(page,kind,sample,feature) {
  await settled(page);const t=(await state(page)).ui.pick_targets.find(t=>t.kind===kind&&t.sample===sample&&(feature===undefined||t.feature===feature));
  expect(t).toBeTruthy();const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  let box=await canvas.boundingBox();await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+t.y*box.height);box=await canvas.boundingBox();
  const builds=(await state(page)).summary.snapshot_builds;
  await page.mouse.move(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect(control(page,'hover')).toHaveText(t.label);expect((await state(page)).summary.snapshot_builds).toBe(builds);
  await page.mouse.click(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect.poll(async()=>(await state(page)).selection.sample).toBe(sample);await settled(page);
}
test('A98 tables, embedding, sample views, supplied matching, linked views and cleanup',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/collections`,errors);await settled(page);
  expect(await control(page,'apply').evaluate(el=>parseFloat(getComputedStyle(el).fontSize))).toBeGreaterThanOrEqual(16);
  await pick(page,'collection_matrix',3,2);
  expect((await state(page)).selected.value).toBe(0);
  await expect(control(page,'readout')).toContainText('Dimension at parameter 1');
  await capture(page,info,'feature-and-source');
  await changed(page,'distances');
  await edit(page,'sample','2');await edit(page,'target','3');await changed(page,'apply');
  expect((await state(page)).selected.value).toBe(1.5);
  await pick(page,'collection_embedding',1);
  expect((await state(page)).selected.sample).toBe('A');
  const revision=(await state(page)).summary.revision;
  await edit(page,'sample','999');await control(page,'apply').click();
  await expect(control(page,'error')).not.toBeEmpty();expect((await state(page)).summary.revision).toBe(revision);
  await changed(page,'reset');await changed(page,'distances');
  await control(page,'matching').click();
  await expect.poll(async()=>(await state(page)).pair_open,{timeout:120_000}).toBe(true);
  await expect(page.locator('[id^="tamerop-matching-inspector-"]').first()).toBeVisible({timeout:120_000});
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].every(c=>c.wglmakie_screen?.renderer?.info.render.frame>0),null,{timeout:120_000});
  await capture(page,info,'collection-pair-witness');
  await changed(page,'kernel');expect((await state(page)).pair_open).toBe(false);
  expect((await state(page)).selected.value).toBe(-1);
  const linked=await page.context().newPage();await openInspector(linked,`${base}/collections`,errors);
  await changed(linked,'features');await expect.poll(async()=>(await state(page)).selection.view).toBe('features');
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  await page.setViewportSize({width:480,height:900});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await scrollChart(page,page.locator('[aria-label="Collection charts"]'),'right');await capture(page,info,'narrow-collection');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
