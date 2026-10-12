import { test, expect } from '@playwright/test';
import { state, keyboardActivate, control, openInspector, capture, fullyClosed } from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){
  await expect.poll(async()=>(await state(page)).summary.updating,{timeout:120_000}).toBe(false);
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].some(c=>c.wglmakie_screen?.renderer?.info.render.frame>0),null,{timeout:120_000});
  await expect.poll(async()=>(await state(page)).summary.listener_errors).toEqual([]);
}
async function changed(page,suffix){
  const r=(await state(page)).selection.revision;await keyboardActivate(control(page,suffix));
  await expect.poll(async()=>(await state(page)).selection.revision,{timeout:120_000}).toBeGreaterThan(r);
  await expect(control(page,'error')).toBeEmpty();await settled(page);
}
async function edit(page,suffix,value){
  const input=page.locator(`input[id$="-${suffix}"]`);await input.fill(value);await input.press('Tab');
}
async function pickCell(page,index){
  const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  const current=await state(page),t=current.ui.pick_targets.find(t=>t.index===index);
  expect(t).toBeTruthy();let box=await canvas.boundingBox();
  await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+t.y*box.height);
  box=await canvas.boundingBox();await page.mouse.move(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect.poll(async()=>{
    const p=(await state(page)).ui.mouseposition;
    return Math.hypot(p[0]-t.x*current.ui.figure_size[0],p[1]-(1-t.y)*current.ui.figure_size[1]);
  }).toBeLessThan(3);
  const builds=(await state(page)).summary.snapshot_builds;
  await expect(control(page,'hover')).toContainText(`Cell ${index},`);
  expect((await state(page)).summary.snapshot_builds).toBe(builds);
  await page.mouse.down();await page.mouse.up();
  await expect.poll(async()=>(await state(page)).selection.cell,{timeout:120_000}).toBe(index);await settled(page);
}
test('A93 radius-depth controls link exact cap cells, coverage, homology and rollback',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/geometric`,errors);
  expect((await state(page)).geometry.betti).toEqual([2,0,0]);
  await capture(page,info,'separated');
  await edit(page,'parameter','1,1');await changed(page,'apply');
  expect((await state(page)).geometry.betti).toEqual([1,0,0]);
  expect((await state(page)).geometry.panel_parameter).toEqual(['1',1]);
  await pickCell(page,3);
  expect((await state(page)).geometry.selected_sites).toEqual([1,2]);
  expect((await state(page)).geometry.radius_squared).toBe('1');
  expect((await state(page)).selected.boundary).toHaveLength(2);
  await capture(page,info,'closed-cap');
  const linked=await page.context().newPage();await openInspector(linked,`${base}/geometric`,errors);
  await edit(linked,'parameter','1152921504606846975/1152921504606846976,1');await changed(linked,'apply');
  await expect.poll(async()=>(await state(page)).geometry.betti).toEqual([2,0,0]);
  expect((await state(page)).selected.active).toBe(false);
  await linked.close();
  const revision=(await state(page)).selection.revision;
  await edit(page,'parameter','1,2');await control(page,'apply').click();
  await expect(control(page,'error')).toContainText('closed construction window');
  expect((await state(page)).selection.revision).toBe(revision);
  expect((await state(page)).geometry.betti).toEqual([2,0,0]);
  await changed(page,'reset');
  await page.setViewportSize({width:480,height:900});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await capture(page,info,'narrow');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
