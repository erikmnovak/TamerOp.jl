import { test, expect } from '@playwright/test';
import { state, keyboardActivate, control, openInspector, capture, fullyClosed, scrollChart } from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){
  await expect.poll(async()=>(await state(page)).summary.updating,{timeout:120_000}).toBe(false);
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].some(c=>c.wglmakie_screen?.renderer?.info.render.frame>0),null,{timeout:120_000});
  await expect.poll(async()=>(await state(page)).summary.listener_errors).toEqual([]);
}
async function changed(page,suffix){
  await settled(page);const r=(await state(page)).summary.revision;
  await keyboardActivate(control(page,suffix));
  await expect.poll(async()=>(await state(page)).summary.revision,{timeout:120_000}).toBeGreaterThan(r);
  await expect(control(page,'error')).toBeEmpty();await settled(page);
}
async function edit(page,suffix,value){const el=page.locator(`input[id$="-${suffix}"]`);await el.fill(value);await el.press('Tab');}
async function vector(page,coefficients){await edit(page,'vector',coefficients);await changed(page,'select');}
async function pick(page,selection){
  await settled(page);const current=await state(page),t=current.ui.pick_targets.find(t=>Object.entries(selection).every(([k,v])=>t.selection[k]===v));
  expect(t).toBeTruthy();const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  let box=await canvas.boundingBox();await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+t.y*box.height);box=await canvas.boundingBox();
  await page.mouse.move(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect.poll(async()=>{const p=(await state(page)).ui.mouseposition;return Math.hypot(p[0]-t.x*current.ui.figure_size[0],p[1]-(1-t.y)*current.ui.figure_size[1]);}).toBeLessThan(3);
  const builds=(await state(page)).summary.snapshot_builds;
  await expect(control(page,'hover')).toContainText('dimension');
  expect((await state(page)).summary.snapshot_builds).toBe(builds);
  await page.mouse.down();await page.mouse.up();
  for(const [key,value] of Object.entries(selection)) await expect.poll(async()=>(await state(page)).selection[key],{timeout:120_000}).toEqual(value);
  await settled(page);
}
test('A89 cycle, boundary, noncycle, quotient coordinates and linked lifecycle',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/chains`,errors);
  expect((await state(page)).summary.quotient_computed).toBe(false);
  await vector(page,'1, 1, 0');let s=(await state(page)).selected;
  expect([s.is_cycle,s.is_boundary,s.quotient_equation.valid]).toEqual([true,false,true]);
  await capture(page,info,'cycle-with-boundary-adjustment');
  await vector(page,'0, 0, 1');expect((await state(page)).selected.is_cycle).toBe(false);
  await vector(page,'1, 0, 0');expect((await state(page)).selected.is_boundary).toBe(true);
  const r=(await state(page)).summary.revision;await edit(page,'vector','1, 2');await control(page,'select').click();
  await expect(control(page,'error')).toContainText('Expected 3 coefficients');expect((await state(page)).summary.revision).toBe(r);
  await vector(page,'0, 1/2, 0');expect((await state(page)).selected.coordinates).toEqual(['1//2']);
  await changed(page,'class-coordinates');expect((await state(page)).selection.vector).toBeNull();
  await vector(page,'1');expect((await state(page)).selected.coordinates).toEqual(['1']);
  await changed(page,'compute');expect((await state(page)).selection.compute).toBe(true);
  await pick(page,{degree:0});expect((await state(page)).selection.vector).toBeNull();
  const linked=await page.context().newPage();await openInspector(linked,`${base}/chains`,errors);
  await edit(linked,'degree','1');await changed(linked,'apply-location');
  await expect.poll(async()=>(await state(page)).selection.degree).toBe(1);
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  await changed(page,'reset');await vector(page,'1,1,0');
  await page.setViewportSize({width:480,height:900});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  const scroll=page.locator('[aria-label="Chain and connecting-map charts"]');
  await scrollChart(page,scroll,'right');await capture(page,info,'narrow-right');
  await scrollChart(page,scroll,'left');await capture(page,info,'narrow-left');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
test('A89 mapping-cone connecting map uses retained bases and shift',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/chains-cone`,errors);
  expect((await state(page)).selection.map).toBe('delta');await vector(page,'1');
  expect((await state(page)).selected.image_coordinates).toEqual(['1']);expect((await state(page)).selected.equation.valid).toBe(true);
  await capture(page,info,'cone-connecting-map');
  await pick(page,{map:'f'});expect((await state(page)).selection.vector).toBeNull();await vector(page,'[]');
  expect((await state(page)).selected.image_coordinates).toEqual([]);
  await edit(page,'map','bogus');await control(page,'apply-location').click();await expect(control(page,'error')).not.toBeEmpty();
  expect((await state(page)).selection.map).toBe('f');await changed(page,'reset');
  await vector(page,'2');expect((await state(page)).selected.image_coordinates).toEqual(['2']);
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
test('A89 source-cell boundary and exact parameter selection',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/chains-source`,errors);
  await vector(page,'-1,1,1,-1');expect((await state(page)).selected.is_boundary).toBe(true);
  expect((await state(page)).selected.quotient_equation.valid).toBe(true);await capture(page,info,'square-filling');
  await edit(page,'parameter','-1');await changed(page,'apply-location');expect((await state(page)).selection.vector).toBeNull();
  await vector(page,'[]');expect((await state(page)).selected.is_boundary).toBe(true);
  await changed(page,'reset');expect((await state(page)).selection.parameter).toEqual([0]);
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
