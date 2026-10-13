import {test,expect} from '@playwright/test';
import {state,control,openInspector,capture,fullyClosed,keyboardActivate,scrollableCharts,scrollChart} from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){
  await expect.poll(async()=>{
    const s=await state(page);
    return [s.summary.requests.pending,s.summary.requests.running,s.summary.updating];
  }).toEqual([false,false,false]);
  expect((await state(page)).summary.listener_errors).toEqual([]);
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].every(
    canvas=>canvas.wglmakie_screen?.renderer?.info.render.frame>0),null,{timeout:120_000});
}
async function pair(page,source,target){
  await control(page,'source').fill(String(source));await control(page,'source').press('Tab');
  await control(page,'target').fill(String(target));await control(page,'target').press('Tab');
  const revision=(await state(page)).selection.revision;
  await keyboardActivate(control(page,'inspect'));
  await expect.poll(async()=>(await state(page)).selection.revision).toBeGreaterThan(revision);
  await settled(page);
}
test('A125 atlas entries, cycle/filling witnesses, retained canvas and lifecycle',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/atlas`,errors);
  const initial=await state(page);expect(initial.transition.kernel).toBe(1);
  await keyboardActivate(control(page,'kernel'));
  await expect.poll(async()=>(await state(page)).selection.event).toBe('kernel');await settled(page);
  await expect(page.locator('canvas')).toHaveCount(2);await settled(page);
  await capture(page,info,'a125-kernel');
  const linked=await page.context().newPage();await openInspector(linked,`${base}/atlas`,errors);
  await expect.poll(async()=>(await state(page)).viewer_count).toBe(2);
  await pair(page,1,2);expect((await state(page)).transition.cokernel).toBe(1);
  await expect.poll(async()=>(await state(linked)).selection.source).toBe(1);
  await expect(control(linked,'source')).toHaveValue('1');
  // Bonito permits reconnection for 30 seconds before notifying child Apps.
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  await keyboardActivate(control(page,'entries'));
  await expect.poll(async()=>(await state(page)).selection.event).toBe('entries');await settled(page);
  expect((await state(page)).transition.entries).toEqual([4,5,6]);
  await keyboardActivate(control(page,'cokernel'));
  await expect.poll(async()=>(await state(page)).selection.event).toBe('cokernel');await settled(page);
  const old=await state(page);
  await control(page,'witness').fill('99');await control(page,'witness').press('Tab');await control(page,'cokernel').click();
  await expect(control(page,'error')).toContainText('witness');await settled(page);
  expect((await state(page)).selection).toEqual(old.selection);
  // Hover reads retained information and leaves the query revision unchanged.
  const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();const box=await canvas.boundingBox();
  const target=old.ui.pick_targets[2];
  await page.mouse.move(box.x+target.x*box.width,box.y+target.y*box.height);
  await expect(control(page,'hover')).toContainText('State 3');
  expect((await state(page)).selection.revision).toBe(old.selection.revision);
  await page.mouse.click(box.x+target.x*box.width,box.y+target.y*box.height);
  await expect.poll(async()=>(await state(page)).selection.source).toBe(3);await settled(page);
  expect((await state(page)).transition.defined).toBe(false);
  expect((await state(page)).ui.figure_id).toBe(initial.ui.figure_id);
  await keyboardActivate(control(page,'endpoint'));
  await keyboardActivate(page.getByRole('button',{name:'State 3',exact:true}));
  await expect.poll(async()=>(await state(page)).selection.target).toBe(3);await settled(page);
  expect((await state(page)).transition.defined).toBe(true);
  await control(page,'reset').click();await expect.poll(async()=>(await state(page)).selection.source).toBe(2);await settled(page);
  await page.setViewportSize({width:640,height:1000});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  const charts=await scrollableCharts(page);expect(charts.length).toBeGreaterThan(0);
  for(const chart of charts){await scrollChart(page,chart,'right');await scrollChart(page,chart,'left');}
  await capture(page,info,'a125-narrow');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
test('A125 grid refuses incomparable maps and finite maps retain zero composites',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/atlas-grid`,errors);
  await pair(page,1,5);let s=await state(page);
  expect(s.transition.relation).toBe('incomparable');expect(s.transition.matrix).toBeNull();
  await capture(page,info,'a125-incomparable');
  await control(page,'close').click();await fullyClosed(page);
  await openInspector(page,`${base}/atlas-module`,errors);
  expect((await state(page)).transition.rank).toBe(0);
  await pair(page,1,2);expect((await state(page)).transition.rank).toBe(1);
  await pair(page,2,3);expect((await state(page)).transition.rank).toBe(1);
  await pair(page,1,3);expect((await state(page)).transition.matrix.rows).toEqual([['0']]);
  await capture(page,info,'a125-zero-composite');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
