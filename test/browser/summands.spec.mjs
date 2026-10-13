import { test, expect } from '@playwright/test';
import { state, keyboardActivate, openInspector, capture, fullyClosed, scrollChart } from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
const control=(page,suffix)=>page.locator(`[id^="tamerop-summand-inspector-"][id$="-${suffix}"]`);
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
async function edit(page,suffix,value) {const input=control(page,suffix);await input.fill(value);await input.press('Tab');}
test('A87 summand selection reconstruction geometry Hom and linked lifecycle',async({page},info)=>{
  // Scene creation, repeated gallery updates and two connected viewers share
  // this scenario. Each gesture and mathematical assertion keeps its deadline.
  test.setTimeout(900_000);
  const errors=[];await openInspector(page,`${base}/summands`,errors);await settled(page);
  expect((await state(page)).summary.selected_dimensions).toEqual([1,2,1,4]);
  expect((await state(page)).summary.certification).toBe('checked_splitting');
  expect((await state(page)).summary.geometry_status).toBe('exact_classifier_fibers');
  await edit(page,'selected','2,3');await edit(page,'focus','3');await changed(page,'apply');
  expect((await state(page)).summary.selected_dimensions).toEqual([0,1,0,3]);
  expect((await state(page)).summary.complement_dimensions).toEqual([1,1,1,1]);
  await changed(page,'previous');expect((await state(page)).selection.focus).toBe(2);
  await changed(page,'next');expect((await state(page)).selection.focus).toBe(3);
  await edit(page,'layout','overlay');await changed(page,'apply');
  await changed(page,'scale');expect((await state(page)).selection.scale).toBe('log');
  await changed(page,'shared');expect((await state(page)).selection.shared_scale).toBe(false);
  const before=(await state(page)).summary;
  const target=(await state(page)).ui.pick_targets.find(t=>t.vertex===2);
  const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  let box=await canvas.boundingBox();await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+target.y*box.height);box=await canvas.boundingBox();
  await page.mouse.move(box.x+target.x*box.width,box.y+target.y*box.height);
  await expect(control(page,'hover')).toHaveText(target.label);
  expect((await state(page)).summary.revision).toBe(before.revision);
  await page.mouse.click(box.x+target.x*box.width,box.y+target.y*box.height);
  await expect.poll(async()=>(await state(page)).selection.vertex).toBe(2);await settled(page);
  await changed(page,'hom');
  expect((await state(page)).summary.hom_spaces).toBe(4);
  await edit(page,'hom-pair','3,2');await edit(page,'vertex','4');await changed(page,'apply');
  expect((await state(page)).hom_component.size).toEqual([1,2]);
  await capture(page,info,'summands-hom');
  const revision=(await state(page)).summary.revision;
  await edit(page,'focus','99');await control(page,'apply').click();
  await expect(control(page,'error')).not.toBeEmpty();expect((await state(page)).summary.revision).toBe(revision);
  await changed(page,'reset');
  await edit(page,'ranking','multiplicity');await edit(page,'limit','1');await changed(page,'apply');
  expect((await state(page)).selection.ids).toEqual([3]);
  expect((await state(page)).selection.focus).toBe(3);
  await changed(page,'reset');
  const linked=await page.context().newPage();await openInspector(linked,`${base}/summands`,errors);await settled(linked);
  await edit(linked,'focus','2');await changed(linked,'apply');
  await expect.poll(async()=>(await state(page)).selection.focus).toBe(2);
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  await page.setViewportSize({width:480,height:900});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await scrollChart(page,page.locator('[aria-label="Summand charts"]'),'right');await capture(page,info,'summands-narrow');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
