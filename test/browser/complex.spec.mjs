import { test, expect } from '@playwright/test';
import { state, keyboardActivate, control, openInspector, capture, fullyClosed, scrollChart } from './helpers.mjs';
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
  await expect.poll(async()=>(await state(page)).selection.cell).toBe(index);await settled(page);
}
test('A92 ring parameter, actual cell picking, exact boundary and linked lifecycle',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/complex`,errors);
  expect((await state(page)).summary.active_counts).toEqual([16,24,8]);await capture(page,info,'ring');
  await edit(page,'parameter','5');await changed(page,'apply');
  expect((await state(page)).summary.active_counts).toEqual([16,24,9]);
  await pickCell(page,45);
  const selected=(await state(page)).selected;
  expect(selected.dimension).toBe(2);expect(selected.source.origin).toEqual([2,2]);
  expect(selected.boundary.map(f=>f.coefficient).sort()).toEqual([-1,-1,1,1]);
  await capture(page,info,'filled-selected');
  const linked=await page.context().newPage();await openInspector(linked,`${base}/complex`,errors);
  expect((await state(linked)).selection.cell).toBe(45);
  await edit(linked,'parameter','5764607523034234879/1152921504606846976');await changed(linked,'apply');
  await expect.poll(async()=>(await state(page)).summary.active_counts).toEqual([16,24,8]);
  expect((await state(page)).selected.active).toBe(false);
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  await changed(page,'inactive');expect((await state(page)).summary.drawn_counts).toEqual([16,24,9]);
  await changed(page,'boundary');expect((await state(page)).selection.boundary).toBe(false);
  await edit(page,'dimensions','0,1');await changed(page,'apply-dimensions');
  expect((await state(page)).summary.drawn_counts).toEqual([16,24,0]);
  const revision=(await state(page)).selection.revision;
  await edit(page,'parameter','not-a-number');await control(page,'apply').click();
  await expect(control(page,'error')).not.toBeEmpty();expect((await state(page)).selection.revision).toBe(revision);
  await changed(page,'reset');await edit(page,'parameter','-1');await changed(page,'apply');
  expect((await state(page)).summary.active_counts).toEqual([0,0,0]);await capture(page,info,'empty');
  await changed(page,'reset');await page.setViewportSize({width:480,height:900});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  const scroll=page.locator('[aria-label="Filtered complex chart"]');
  await scrollChart(page,scroll,'right');await capture(page,info,'narrow-right');
  await scrollChart(page,scroll,'left');await capture(page,info,'narrow-left');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
test('A92 three-dimensional skeleton keeps the active volume explicit',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/complex-3d`,errors);
  expect((await state(page)).summary.active_counts).toEqual([8,12,6,1]);
  expect((await state(page)).summary.drawn_counts).toEqual([8,12,6,0]);
  await edit(page,'cell','27');await changed(page,'select');
  expect((await state(page)).selected.dimension).toBe(3);
  expect((await state(page)).selected.boundary).toHaveLength(6);
  await capture(page,info,'volume-boundary');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
