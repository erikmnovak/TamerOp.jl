import { test, expect } from '@playwright/test';
import { state, keyboardActivate, control, openInspector, capture, fullyClosed } from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){
  await expect.poll(async()=>(await state(page)).summary.updating,{timeout:120_000}).toBe(false);
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].some(c=>c.wglmakie_screen?.renderer?.info.render.frame>0),null,{timeout:120_000});
  await expect.poll(async()=>(await state(page)).summary.listener_errors).toEqual([]);
}
async function edit(page,suffix,value){const i=page.locator(`input[id$="-${suffix}"]`);await i.fill(value);await i.press('Tab');}
async function changed(page,suffix){
  await settled(page);const r=(await state(page)).selection.revision;
  await keyboardActivate(control(page,suffix));
  await expect.poll(async()=>(await state(page)).selection.revision,{timeout:120_000}).toBeGreaterThan(r);
  await expect(control(page,'error')).toBeEmpty();await settled(page);
}
test('A88 pointer preimages, exact square, linked lifecycle and failure rollback',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/transport`,errors);await settled(page);
  expect((await state(page)).summary.comparison_validation.isomorphism).toBe(true);
  const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  const current=await state(page),t=current.ui.pick_targets.find(t=>t.side==='right'&&t.label===1);
  let box=await canvas.boundingBox();await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+t.y*box.height);box=await canvas.boundingBox();
  await page.mouse.move(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect(control(page,'hover')).toContainText('right label 1');
  expect((await state(page)).summary.snapshot_builds).toBe(current.summary.snapshot_builds);
  await page.mouse.down();await page.mouse.up();
  await expect.poll(async()=>(await state(page)).summary.candidates,{timeout:120_000}).toEqual([1,2]);await settled(page);
  await capture(page,info,'all-preimages');
  await edit(page,'p','1');await edit(page,'q','4');await changed(page,'square');
  expect((await state(page)).square.valid).toBe(true);
  expect((await state(page)).square.left).toEqual((await state(page)).square.right);
  await capture(page,info,'commuting-square');
  const linked=await page.context().newPage();await openInspector(linked,`${base}/transport`,errors);await settled(linked);
  expect((await state(linked)).selection.pair).toEqual([1,4]);
  await changed(linked,'overview');await expect.poll(async()=>(await state(page)).selection.view).toBe('overview');
  expect((await state(page)).selection.pair).toEqual([1,4]);
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  const revision=(await state(page)).selection.revision;
  await edit(page,'vertex','100');await control(page,'select').click();await expect(control(page,'error')).not.toBeEmpty();
  expect((await state(page)).selection.revision).toBe(revision);
  await edit(page,'x','3/2');await edit(page,'y','1/2');await changed(page,'point');
  expect((await state(page)).summary.candidates).toEqual([2]);
  await page.setViewportSize({width:480,height:900});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await capture(page,info,'narrow');
  await changed(page,'reset');await control(page,'close').click();await fullyClosed(page);
  expect(errors).toEqual([]);
});
