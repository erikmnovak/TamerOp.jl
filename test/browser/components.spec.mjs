import { test, expect } from '@playwright/test';
import { state, keyboardActivate, control, openInspector, capture, fullyClosed, scrollChart } from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){
  await expect.poll(async()=>(await state(page)).summary.updating,{timeout:120_000}).toBe(false);
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].some(c=>c.wglmakie_screen?.renderer?.info.render.frame>0),null,{timeout:120_000});
  await expect.poll(async()=>(await state(page)).summary.listener_errors).toEqual([]);
}
async function changed(page,suffix){
  await settled(page);const r=(await state(page)).summary.revision;await keyboardActivate(control(page,suffix));
  await expect.poll(async()=>(await state(page)).summary.revision,{timeout:120_000}).toBeGreaterThan(r);
  await expect(control(page,'error')).toBeEmpty();await settled(page);
}
async function edit(page,suffix,value){const el=page.locator(`input[id$="-${suffix}"]`);await el.fill(value);await el.press('Tab');}
async function scale(page,value){await edit(page,'scale',value);await changed(page,'apply-scale');}
async function pick(page,axis,selection){
  await settled(page);const current=await state(page),t=current.ui.pick_targets.find(t=>t.axis_index===axis&&Object.entries(selection).every(([k,v])=>t.selection[k]===v));
  expect(t).toBeTruthy();const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  let box=await canvas.boundingBox();await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+t.y*box.height);box=await canvas.boundingBox();
  await page.mouse.move(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect.poll(async()=>{const p=(await state(page)).ui.mouseposition;return Math.hypot(p[0]-t.x*current.ui.figure_size[0],p[1]-(1-t.y)*current.ui.figure_size[1]);}).toBeLessThan(3);
  const builds=(await state(page)).summary.snapshot_builds;
  await expect(control(page,'hover')).toHaveText(t.label);expect((await state(page)).summary.snapshot_builds).toBe(builds);
  await page.mouse.down();await page.mouse.up();
  for(const [key,value] of Object.entries(selection)) await expect.poll(async()=>(await state(page)).selection[key],{timeout:120_000}).toEqual(value);
  await settled(page);
}
test('A91 source memberships, merge nodes, exact endpoints and linked lifecycle',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/components`,errors);await settled(page);
  expect((await state(page)).summary.components).toBe(3);
  await pick(page,1,{component:2});expect((await state(page)).selected.vertices).toEqual([2]);
  expect((await state(page)).selected.sources).toEqual([[0,12]]);
  await scale(page,'199999999999999999999/100000000000000000000');
  expect((await state(page)).summary.components).toBe(3);expect((await state(page)).selected.status).toBe('alive');
  await capture(page,info,'before-merge');await scale(page,'2');
  expect((await state(page)).selected.status).toBe('merged');expect((await state(page)).selected.vertices).toEqual([1,2]);
  await expect(control(page,'readout')).toContainText('Its bar has died');
  await pick(page,2,{component:1});expect((await state(page)).selected.status).toBe('alive');
  await pick(page,3,{event:6,component:3});expect((await state(page)).parameter).toEqual(['3']);
  expect((await state(page)).selected.vertices).toEqual([1,2,3]);
  await expect(control(page,'readout')).toContainText('source edge (1, 102)');await capture(page,info,'selected-source-merge');
  const r=(await state(page)).summary.revision;await edit(page,'scale','bad');await control(page,'apply-scale').click();
  await expect(control(page,'error')).not.toBeEmpty();expect((await state(page)).summary.revision).toBe(r);
  await edit(page,'component','999');await control(page,'select').click();await expect(control(page,'error')).not.toBeEmpty();
  expect((await state(page)).summary.revision).toBe(r);
  await changed(page,'previous');expect((await state(page)).summary.components).toBe(2);expect((await state(page)).selection.event).toBeNull();
  await changed(page,'next');expect((await state(page)).summary.components).toBe(1);
  const linked=await page.context().newPage();await openInspector(linked,`${base}/components`,errors);
  await scale(linked,'4');await expect.poll(async()=>(await state(page)).summary.components).toBe(2);
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  await changed(page,'reset');expect((await state(page)).selection.component).toBeNull();expect((await state(page)).summary.components).toBe(3);
  await page.setViewportSize({width:480,height:900});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  const scroll=page.locator('[aria-label="Component and merge-history charts"]');
  await scrollChart(page,scroll,'right');await capture(page,info,'narrow-right');
  await scrollChart(page,scroll,'left');await capture(page,info,'narrow-left');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
test('A91 multicritical slice keeps exact source parameter and event',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/components-slice`,errors);await settled(page);
  expect((await state(page)).summary.components).toBe(2);
  await pick(page,3,{event:3,component:2});expect((await state(page)).parameter).toEqual(['3//2','3']);
  expect((await state(page)).summary.components).toBe(1);expect((await state(page)).selected.vertices).toEqual([1,2]);
  await expect(control(page,'readout')).toContainText('not to every slice');await capture(page,info,'multicritical-merge');
  await scale(page,'7/5');expect((await state(page)).selected.status).toBe('alive');expect((await state(page)).summary.components).toBe(2);
  await changed(page,'next');expect((await state(page)).summary.components).toBe(1);
  await changed(page,'clear');expect((await state(page)).selected).toBeNull();
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
