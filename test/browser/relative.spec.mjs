import {test,expect} from '@playwright/test';
import {state,control,openInspector,capture,fullyClosed,keyboardActivate,scrollableCharts,scrollChart} from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){
  await expect.poll(async()=>{const s=await state(page);return [s.summary.requests.pending,s.summary.requests.running,s.summary.updating];}).toEqual([false,false,false]);
  expect((await state(page)).summary.listener_errors).toEqual([]);
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].every(c=>c.wglmakie_screen?.renderer?.info.render.frame>0),null,{timeout:120_000});
}
async function set(page,name,value){await control(page,name).fill(String(value));await control(page,name).press('Tab');}
async function action(page,name){const old=(await state(page)).selection.revision;await keyboardActivate(control(page,name));await expect.poll(async()=>(await state(page)).selection.revision).toBeGreaterThan(old);await settled(page);}
test('A128 spatial brush, exact pair, quotient witnesses and linked lifecycle',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/relative`,errors);
  await action(page,'reset');
  expect((await state(page)).counts).toEqual({ambient:0,subcomplex:0,relative:1});
  expect((await state(page)).added).toEqual([1,2,3]);
  await action(page,'maps');expect((await state(page)).maps.connecting.rows).toEqual([['1']]);
  await action(page,'representative');expect((await state(page)).representative.index).toBe(1);
  await capture(page,info,'a128-disk-relative-boundary');
  const before=await state(page);await set(page,'witness',99);await control(page,'representative').click();
  await expect(control(page,'error')).toContainText('witness');await settled(page);
  expect((await state(page)).selection).toEqual(before.selection);
  await action(page,'reset');
  const linked=await page.context().newPage();await openInspector(linked,`${base}/relative`,errors);
  await expect.poll(async()=>(await state(page)).viewer_count).toBe(2);
  await set(page,'entry','3');await action(page,'delay');expect((await state(page)).counts.relative).toBe(0);
  await set(page,'scale','3');await action(page,'query');expect((await state(page)).counts.relative).toBe(1);
  await expect.poll(async()=>(await state(linked)).selection.scale).toBe('3');
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  await action(page,'induced');
  const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  let box=await canvas.boundingBox();let s=await state(page);let p=s.ui.pick_targets.find(t=>t.index===1);
  await page.mouse.move(box.x+p.x*box.width,box.y+p.y*box.height);await expect(control(page,'hover')).toContainText('Cell 1');
  expect((await state(page)).selection.revision).toBe(s.selection.revision);
  await page.mouse.click(box.x+p.x*box.width,box.y+p.y*box.height);
  await expect.poll(async()=>(await state(page)).selected).toEqual([1]);await settled(page);
  // A real canvas drag selects the lower two vertices and their induced edge.
  await canvas.scrollIntoViewIfNeeded();box=await canvas.boundingBox();s=await state(page);
  const [a,b]=s.ui.brush;
  await page.mouse.move(box.x+a.x*box.width,box.y+a.y*box.height);await page.mouse.down();
  await page.mouse.move(box.x+b.x*box.width,box.y+b.y*box.height,{steps:8});await page.mouse.up();
  await expect.poll(async()=>(await state(page)).selected).toEqual([1,2,4]);await settled(page);
  // Keyboard brush tests exact closed endpoints, independently of screen precision.
  await set(page,'lower','-1, 0');await set(page,'upper','5, 0');await action(page,'brush');
  expect((await state(page)).selected).toEqual([1,2,4]);
  await action(page,'closure');await set(page,'cells','7');await action(page,'apply');
  expect((await state(page)).added).toEqual([1,2,3,4,5,6]);expect((await state(page)).counts.relative).toBe(0);
  await action(page,'clear');expect((await state(page)).selected).toEqual([]);
  await set(page,'cells','99');await control(page,'apply').click();await expect(control(page,'error')).toContainText('indices');await settled(page);
  expect((await state(page)).selected).toEqual([]);
  await action(page,'full');expect((await state(page)).selected).toEqual([1,2,3,4,5,6,7]);
  await page.setViewportSize({width:640,height:1000});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  for(const chart of await scrollableCharts(page)){await scrollChart(page,chart,'right');await scrollChart(page,chart,'left');}
  await capture(page,info,'a128-narrow');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
test('A128 periodic geometric closure differs from field chain closure',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/relative-periodic`,errors);
  expect((await state(page)).selected).toEqual([1,2,3,4]);
  await action(page,'chain');await set(page,'cells','4');await action(page,'apply');
  expect((await state(page)).selected).toEqual([4]);expect((await state(page)).added).toEqual([]);
  await action(page,'closure');await set(page,'cells','4');await action(page,'apply');
  expect((await state(page)).selected).toEqual([1,2,3,4]);
  await capture(page,info,'a128-periodic-closure');await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
