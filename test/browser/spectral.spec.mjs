import { test, expect } from '@playwright/test';
import { state, keyboardActivate, control, openInspector, capture, fullyClosed, scrollChart } from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){
  await expect.poll(async()=>(await state(page)).summary.updating,{timeout:120_000}).toBe(false);
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].some(c=>c.wglmakie_screen?.renderer?.info.render.frame>0),null,{timeout:120_000});
  await expect.poll(async()=>(await state(page)).summary.listener_errors).toEqual([]);
}
async function changed(page,suffix){
  await settled(page);
  const r=(await state(page)).selection.revision;
  await keyboardActivate(control(page,suffix));
  await expect.poll(async()=>(await state(page)).selection.revision,{timeout:120_000}).toBeGreaterThan(r);
  await expect(control(page,'error')).toBeEmpty();await settled(page);
}
async function edit(page,suffix,value){
  const input=page.locator(`input[id$="-${suffix}"]`);
  await input.fill(value);await input.press('Tab');
}
test('A42d actual page navigation, unknown terms, differential and linked lifecycle',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/spectral`,errors);
  expect((await state(page)).selection.page).toBe(2);
  await changed(page,'compute');
  await changed(page,'next');
  expect((await state(page)).selection.page).toBe(3);
  expect((await state(page)).summary.page_status).toBe('uncomputed');
  await capture(page,info,'uncomputed-page');
  await changed(page,'compute');
  expect((await state(page)).summary.page_status).toBe('computed');
  expect((await state(page)).summary.limiting_dimensions).toBe(true);
  await changed(page,'previous');
  // Real pointer selection, with only geometry supplied by read-only telemetry.
  const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  const current=await state(page),t=current.ui.pick_targets.find(t=>t.bidegree[0]===0&&t.bidegree[1]===1);
  let box=await canvas.boundingBox();
  await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+t.y*box.height);
  box=await canvas.boundingBox();
  await page.mouse.move(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect.poll(async()=>{
    const p=(await state(page)).ui.mouseposition;
    return Math.hypot(p[0]-t.x*current.ui.figure_size[0],p[1]-(1-t.y)*current.ui.figure_size[1]);
  }).toBeLessThan(3);
  const builds=(await state(page)).summary.snapshot_builds;
  await expect(control(page,'hover')).toContainText('dimension 2');
  expect((await state(page)).summary.snapshot_builds).toBe(builds);
  await page.mouse.down();await page.mouse.up();
  await expect.poll(async()=>(await state(page)).selection.source,{timeout:120_000}).toEqual([0,1]);
  await settled(page);
  const selected=(await state(page)).selected;
  expect([selected.rank,selected.kernel,selected.next,selected.target_next]).toEqual([1,1,1,1]);
  expect(selected.target).toEqual([2,0]);
  await capture(page,info,'selected-differential');
  const linked=await page.context().newPage();await openInspector(linked,`${base}/spectral`,errors);
  expect((await state(linked)).selection.source).toEqual([0,1]);
  await changed(linked,'next');
  await expect.poll(async()=>(await state(page)).selection.page).toBe(3);
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  await edit(page,'page','0');await control(page,'apply-page').click();
  await expect(control(page,'error')).not.toBeEmpty();
  expect((await state(page)).selection.page).toBe(3);
  await changed(page,'limit');expect((await state(page)).selection.page).toBe('inf');
  await capture(page,info,'associated-graded');
  await changed(page,'reset');
  await edit(page,'p','0');await edit(page,'q','1');await changed(page,'select');
  await page.setViewportSize({width:480,height:900});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  const scroll=page.locator('[aria-label="Spectral page charts"]');
  await scrollChart(page,scroll,'right');await capture(page,info,'narrow-right');
  await scrollChart(page,scroll,'left');await capture(page,info,'narrow-left');
  await control(page,'close').click();await fullyClosed(page);
  expect(errors).toEqual([]);
});
