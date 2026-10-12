import { test, expect } from '@playwright/test';
import { state, keyboardActivate, openInspector, capture, fullyClosed, scrollChart } from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
const control=(page,suffix)=>page.locator(`[id^="tamerop-region-inspector-"][id$="-${suffix}"]`);
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
async function clickTarget(page,t,hover=true) {
  const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  let box=await canvas.boundingBox();await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+t.y*box.height);box=await canvas.boundingBox();
  const before=(await state(page)).summary;
  await page.mouse.move(box.x+t.x*box.width,box.y+t.y*box.height);
  if(hover) await expect(control(page,'hover')).toHaveText(t.label);
  expect((await state(page)).summary.rank_queries).toBe(before.rank_queries);
  await page.mouse.click(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect.poll(async()=>(await state(page)).summary.revision,{timeout:120_000}).toBeGreaterThan(before.revision);
  await settled(page);
}
test('A54 finite domains boundary occurrence picking matrices rollback and linked cleanup',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/generalized-rank`,errors);await settled(page);
  expect((await state(page)).boundary.path.vertices).toEqual([2,5,4,8,5,6]);
  expect((await state(page)).summary.rank).toBe(1);
  const before=(await state(page)).summary;
  await clickTarget(page,(await state(page)).ui.pick_targets.find(t=>t.kind==='occurrence'&&t.occurrence===5));
  expect((await state(page)).selection.occurrence).toBe(5);
  expect((await state(page)).summary.boundary_queries).toBe(before.boundary_queries);
  await clickTarget(page,(await state(page)).ui.pick_targets.find(t=>t.kind==='bar'));
  expect((await state(page)).selection.bar).toBeGreaterThan(0);
  await edit(page,'occurrence','2');await edit(page,'bar','0');await changed(page,'select');
  expect((await state(page)).selection.occurrence).toBe(2);
  expect((await state(page)).selection.bar).toBe(0);
  expect((await state(page)).summary.boundary_queries).toBe(before.boundary_queries);
  await changed(page,'boundary');expect((await state(page)).boundary).toBeNull();
  await changed(page,'boundary');expect((await state(page)).summary.rank).toBe(1);
  await changed(page,'witnesses');await capture(page,info,'boundary-and-matrices');
  const revision=(await state(page)).summary.revision;
  await edit(page,'vertices','1,9');await control(page,'apply').click();
  await expect(control(page,'error')).not.toBeEmpty();expect((await state(page)).summary.revision).toBe(revision);
  await edit(page,'vertices','5');await changed(page,'apply');
  expect((await state(page)).summary.rank).toBe(3);
  expect((await state(page)).boundary.path.vertices).toEqual([5,5]);
  await clickTarget(page,(await state(page)).ui.pick_targets.find(t=>t.kind==='vertex'&&t.vertex===2));
  expect((await state(page)).selection.vertices).toEqual([2,5]);
  await changed(page,'reset');
  const linked=await page.context().newPage();await openInspector(linked,`${base}/generalized-rank`,errors);await settled(linked);
  await edit(linked,'vertices','5');await changed(linked,'apply');
  await expect.poll(async()=>(await state(page)).summary.rank).toBe(3);
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  await page.setViewportSize({width:480,height:900});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await scrollChart(page,page.locator('[aria-label="Generalized rank charts"]'),'right');await capture(page,info,'narrow-region');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
test('A54 continuous worms exact entry layer reuse pointer centers and width endpoints',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/gril`,errors);await settled(page);
  const before=(await state(page)).summary;
  await clickTarget(page,(await state(page)).ui.pick_targets.find(t=>t.kind==='level'&&t.level===2));
  expect((await state(page)).summary.rank_queries).toBe(before.rank_queries);
  expect((await state(page)).summary.landscape_queries).toBe(before.landscape_queries);
  await edit(page,'center-x','1/3');await edit(page,'center-y','2/3');await edit(page,'width','1/3');
  await edit(page,'level','1');await changed(page,'apply');
  expect((await state(page)).selection.center).toEqual(['1//3','2//3']);
  expect((await state(page)).summary.rank).toBe(1);
  const landscape=(await state(page)).summary.landscape_queries;
  await edit(page,'width','34/100');await changed(page,'apply');
  expect((await state(page)).summary.rank).toBe(0);
  expect((await state(page)).summary.landscape_queries).toBe(landscape);
  const revision=(await state(page)).summary.revision;
  await edit(page,'width','-1');await control(page,'apply').click();
  await expect(control(page,'error')).not.toBeEmpty();expect((await state(page)).summary.revision).toBe(revision);
  await changed(page,'reset');
  await clickTarget(page,(await state(page)).ui.domain[0],false);
  expect((await state(page)).summary.landscape_queries).toBeGreaterThan(landscape);
  const coordinates=(await state(page)).selection.center.map(value=>{
    const [numerator,denominator='1']=String(value).split('//');return Number(numerator)/Number(denominator);
  });
  expect(coordinates[0]).toBeCloseTo(1,1);expect(coordinates[1]).toBeCloseTo(2,1);
  await edit(page,'length','2');await changed(page,'apply');
  await capture(page,info,'continuous-worm-and-layers');
  await page.setViewportSize({width:480,height:900});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await scrollChart(page,page.locator('[aria-label="Generalized rank charts"]'),'right');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
