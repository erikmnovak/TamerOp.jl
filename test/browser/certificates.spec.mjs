import { test, expect } from '@playwright/test';
import { state, keyboardActivate, control, openInspector, capture, fullyClosed, scrollChart, checkKeyboardControls } from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){
  await expect.poll(async()=>(await state(page)).summary.updating,{timeout:120_000}).toBe(false);
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].some(c=>c.wglmakie_screen?.renderer?.info.render.frame>0),null,{timeout:120_000});
  await expect.poll(async()=>(await state(page)).summary.listener_errors).toEqual([]);
}
async function changed(page,suffix){
  await settled(page);const r=(await state(page)).selection.revision;
  await keyboardActivate(control(page,suffix));
  await expect.poll(async()=>(await state(page)).selection.revision,{timeout:120_000}).toBeGreaterThan(r);
  await expect(control(page,'error')).toBeEmpty();await settled(page);
}
test('A101 certificate equations, parameter domains, pruning maps and linked browser lifecycle',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/certificates`,errors);await settled(page);
  const initial=(await state(page)).selection.vertex;
  expect((await state(page)).equations).toEqual({forward:true,reverse:true});
  await capture(page,info,'composition');
  await changed(page,'next');await changed(page,'previous');
  expect((await state(page)).selection.vertex).toBe(initial);
  await control(page,'side').selectOption({index:1});
  await expect.poll(async()=>(await state(page)).selection.side).toBe('reverse');
  await settled(page);
  await control(page,'view').selectOption({index:1});
  await expect.poll(async()=>(await state(page)).selection.view).toBe('naturality');await settled(page);
  await capture(page,info,'naturality');
  await control(page,'view').selectOption({index:2});
  await expect.poll(async()=>(await state(page)).selection.view).toBe('domains');await settled(page);
  await capture(page,info,'translated-domains');
  await control(page,'view').selectOption({index:3});
  await expect.poll(async()=>(await state(page)).selection.view).toBe('pruning');await settled(page);
  await capture(page,info,'subquotient');
  const vertex=page.locator('input[id$="-vertex"]');
  await vertex.fill('0');await vertex.press('Tab');await control(page,'apply').click();
  await expect(control(page,'error')).not.toBeEmpty();expect((await state(page)).selection.vertex).toBe(initial);
  const point=page.locator('input[id$="-point"]');
  await point.fill('5/2, 7/2');await point.press('Tab');await changed(page,'apply-point');
  await expect(control(page,'readout')).toContainText('5//2');
  await changed(page,'reset');
  const linked=await page.context().newPage();await openInspector(linked,`${base}/certificates`,errors);await settled(linked);
  await changed(linked,'next');
  await expect.poll(async()=>(await state(page)).selection.vertex).toBe(initial+1);
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  // Canvas selection travels through genuine browser pointer events.
  const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  const current=await state(page),target=current.ui.pick_targets.find(t=>t.vertex===initial);
  let box=await canvas.boundingBox();
  await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+target.y*box.height);box=await canvas.boundingBox();
  const builds=(await state(page)).summary.snapshot_builds;
  await page.mouse.move(box.x+target.x*box.width,box.y+target.y*box.height);
  await expect(control(page,'hover')).toContainText(`Location ${initial}:`);
  expect((await state(page)).summary.snapshot_builds).toBe(builds);
  await page.mouse.down();await page.mouse.up();
  await expect.poll(async()=>(await state(page)).selection.vertex).toBe(initial);await settled(page);
  await page.setViewportSize({width:480,height:900});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  const scroll=page.locator('[aria-label="Certificate charts"]');
  await scrollChart(page,scroll,'right');await capture(page,info,'narrow-right');
  await scrollChart(page,scroll,'left');await capture(page,info,'narrow-left');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
