import { test, expect } from '@playwright/test';
import { state, keyboardActivate, openInspector, capture, fullyClosed, scrollChart } from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
const control=(page,suffix)=>page.locator(`[id^="tamerop-ensemble-inspector-"][id$="-${suffix}"]`);
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
test('A95 probe picking, bands, replicate sources, linked viewers and cleanup',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/ensembles`,errors);await settled(page);
  expect((await state(page)).selected.mean).toBe(2);
  const t=(await state(page)).ui.pick_targets.find(t=>t.probe===2);
  const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  let box=await canvas.boundingBox();await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+t.y*box.height);box=await canvas.boundingBox();
  const builds=(await state(page)).summary.snapshot_builds;
  await page.mouse.move(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect(control(page,'hover')).toHaveText(t.label);expect((await state(page)).summary.snapshot_builds).toBe(builds);
  await page.mouse.click(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect.poll(async()=>(await state(page)).selected.probe).toBe(2);await settled(page);
  await control(page,'sample').fill('2');await control(page,'sample').press('Tab');await changed(page,'apply');
  expect((await state(page)).selected.value).toBe(3);
  await expect(control(page,'readout')).toContainText('descriptive');
  await expect(control(page,'readout')).toContainText('supplied probes only');
  await capture(page,info,'band-and-selected-source');
  for(const view of ['frequency','counts','std','lower','upper','mean']) await changed(page,view);
  const revision=(await state(page)).summary.revision;
  await control(page,'probe').fill('999');await control(page,'probe').press('Tab');await control(page,'apply').click();
  await expect(control(page,'error')).not.toBeEmpty();expect((await state(page)).summary.revision).toBe(revision);
  await changed(page,'reset');expect((await state(page)).selected.sample).toBe('A');
  const linked=await page.context().newPage();await openInspector(linked,`${base}/ensembles`,errors);await settled(linked);
  await changed(linked,'frequency');await expect.poll(async()=>(await state(page)).selection.view).toBe('frequency');
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  await page.setViewportSize({width:480,height:900});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await scrollChart(page,page.locator('[aria-label="Ensemble charts"]'),'right');await capture(page,info,'narrow-ensemble');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});

test('A95 nonuniform physical field probes and selected-replicate hover',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/ensemble-fields`,errors);await settled(page);
  const targets=(await state(page)).ui.pick_targets;
  const t=targets.filter(t=>t.kind==='ensemble_map'&&t.probe===5).at(-1);
  expect(t.label).toBe('Probe 5: (2.0, 5.0); 24.0');
  const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  let box=await canvas.boundingBox();await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+t.y*box.height);box=await canvas.boundingBox();
  const builds=(await state(page)).summary.snapshot_builds;
  await page.mouse.move(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect(control(page,'hover')).toHaveText(t.label);expect((await state(page)).summary.snapshot_builds).toBe(builds);
  await page.mouse.click(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect.poll(async()=>(await state(page)).selected.probe).toBe(5);
  expect((await state(page)).selected.coordinates).toEqual([2,5]);
  expect((await state(page)).selected.value).toBe(24);
  await changed(page,'frequency');expect((await state(page)).selected.frequency).toBe(1);
  await capture(page,info,'nonuniform-field-frequency');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
