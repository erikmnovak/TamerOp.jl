import {test,expect} from '@playwright/test';
import {state,control,openInspector,capture,fullyClosed,keyboardActivate,scrollableCharts,scrollChart} from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){
  await expect.poll(async()=>{const s=await state(page);return [s.summary.requests.pending,s.summary.requests.running,s.summary.updating];}).toEqual([false,false,false]);
  expect((await state(page)).summary.listener_errors).toEqual([]);
}
async function set(page,name,value){await control(page,name).fill(String(value));await control(page,name).press('Tab');}
async function action(page,name,timeout=30000){const old=(await state(page)).selection.revision;await keyboardActivate(control(page,name));await expect.poll(async()=>(await state(page)).selection.revision,{timeout}).toBeGreaterThan(old);await settled(page);}
test('A96 certified path pointer, exact crossings, playback, linked views and cleanup',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/paths`,errors);
  expect((await state(page)).mode).toBe('certified');
  // The first retained-figure update compiles on a fresh Julia fixture. This
  // is correctness acceptance; subsequent actions keep the usual timeout.
  await action(page,'next-event',120000);expect((await state(page)).events.length).toBeGreaterThan(0);
  const crossing=(await state(page)).selection.frame;
  await action(page,'next-event');await action(page,'previous-event');
  expect((await state(page)).selection.frame).toBe(crossing);
  await capture(page,info,'a96-crossing');
  await action(page,'reset');
  const old=await state(page);const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  const box=await canvas.boundingBox();const p=old.ui.pick_targets.find(t=>t.panel===3&&t.id===2);
  expect(p).toBeTruthy();
  await page.mouse.move(box.x+p.x*box.width,box.y+p.y*box.height);
  await expect(control(page,'hover')).toContainText('Track 2');
  expect((await state(page)).selection.revision).toBe(old.selection.revision);
  await page.mouse.click(box.x+p.x*box.width,box.y+p.y*box.height);
  await expect.poll(async()=>(await state(page)).selection.track).toBe(2);await settled(page);
  await set(page,'scale',2);await control(page,'representative').check();await action(page,'apply');
  expect((await state(page)).representative).toBe(true);
  expect((await state(page)).source_representative_visible).toBe(true);
  await capture(page,info,'a96-selected-witness');
  const before=await state(page);await set(page,'frame',999);await control(page,'apply').click();
  await expect(control(page,'error')).toContainText('frame');await settled(page);
  expect((await state(page)).selection).toEqual(before.selection);
  await action(page,'reset');
  const linked=await page.context().newPage();await openInspector(linked,`${base}/paths`,errors);
  await expect.poll(async()=>(await state(page)).viewer_count).toBe(2);
  await control(page,'slider').focus();await control(page,'slider').press('End');
  const n=(await state(page)).nframes;
  await expect.poll(async()=>(await state(page)).selection.frame).toBe(n);await settled(page);
  await expect.poll(async()=>(await state(linked)).selection.frame).toBe(n);
  await keyboardActivate(control(page,'reverse'));
  await expect.poll(async()=>(await state(page)).selection.frame).toBeLessThan(n);
  await keyboardActivate(control(page,'pause'));await settled(page);
  expect((await state(page)).ui.timer_active).toBe(false);
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45000}).toBe(1);
  await set(page,'frame',n-1);await action(page,'apply');await keyboardActivate(control(page,'play'));
  await expect.poll(async()=>(await state(page)).selection.frame).toBe(n);
  await expect.poll(async()=>(await state(page)).ui.playing).toBe(false);await settled(page);
  const final=await state(page);expect(final.ui.rebuild_count).toBe(1);expect(final.ui.update_count).toBeGreaterThan(3);
  await page.setViewportSize({width:640,height:1000});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  for(const chart of await scrollableCharts(page)){await scrollChart(page,chart,'right');await scrollChart(page,chart,'left');}
  await capture(page,info,'a96-narrow');
  await keyboardActivate(control(page,'reverse'));await control(page,'close').click();await fullyClosed(page);
  expect((await state(page)).ui.timer_active).toBe(false);expect(errors).toEqual([]);
});
test('A96 matched paths preserve duplicate members and identify unmatched tracks',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/paths-matched`,errors);
  expect((await state(page)).mode).toBe('matched');
  await expect(control(page,'readout')).toContainText('Consecutive bottleneck matching');
  await set(page,'track',3);await action(page,'apply');
  await action(page,'next');await expect(control(page,'readout')).toContainText('absent at this state');
  expect((await state(page)).selection.track).toBe(3);
  await expect(control(page,'readout')).toContainText('Identity is ambiguous');
  await action(page,'previous');expect((await state(page)).records).toHaveLength(4);
  await capture(page,info,'a96-matched-duplicate-members');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
