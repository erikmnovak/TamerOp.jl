import {test,expect} from '@playwright/test';
import {state,control,openInspector,capture,fullyClosed,keyboardActivate,scrollableCharts,scrollChart} from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){await expect.poll(async()=>{const s=await state(page);return [s.summary.requests.pending,s.summary.requests.running,s.summary.updating];}).toEqual([false,false,false]);expect((await state(page)).summary.listener_errors).toEqual([]);}
async function set(page,name,value){await control(page,name).fill(String(value));await control(page,name).press('Tab');}
async function action(page,name,timeout=30000){const old=(await state(page)).selection.revision;await keyboardActivate(control(page,name));await expect.poll(async()=>(await state(page)).selection.revision,{timeout}).toBeGreaterThan(old);await settled(page);}
test('A124 edit source with pointer and keyboard, explicit rebuild, common scales and linked history',async({page},info)=>{
 const errors=[];await openInspector(page,`${base}/source-laboratory`,errors);
 const original=await state(page);expect(original.bars[0][0].birth).toBe(2);
 const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();const box=await canvas.boundingBox();
 const p=original.ui.pick_targets.find(p=>p.id===3);expect(p).toBeTruthy();
 await page.mouse.move(box.x+p.x*box.width,box.y+p.y*box.height);
 await expect(control(page,'hover')).toContainText('Source point 3');expect((await state(page)).selection.revision).toBe(0);
 await page.mouse.click(box.x+p.x*box.width,box.y+p.y*box.height);
 expect((await state(page)).selection.revision).toBe(0);
 await page.mouse.down();await expect.poll(async()=>(await state(page)).ui.dragging).toBe(true);
 await page.mouse.move(box.x+p.x*box.width+35,box.y+p.y*box.height-20,{steps:5});await expect(control(page,'hover')).toContainText('Draft move preview');await page.mouse.up();
 await expect.poll(async()=>(await state(page)).summary.dirty,{timeout:120000}).toBe(true);await settled(page);
 expect((await state(page)).current).toEqual(original.current);expect((await state(page)).selection.generation).toBe(1);
 await capture(page,info,'a124-draft-drag');
 await set(page,'point',3);await set(page,'x',2.4);await set(page,'y',1.3);await action(page,'move');
 await action(page,'rebuild',120000);
 expect((await state(page)).selection.generation).toBe(2);expect((await state(page)).summary.dirty).toBe(false);
 const linked=await page.context().newPage();await openInspector(linked,`${base}/source-laboratory`,errors);
 await expect.poll(async()=>(await state(page)).viewer_count).toBe(2);
 await set(page,'x',0.4);await set(page,'y',0.3);await action(page,'add');const added=(await state(page)).draft.ids.at(-1);
 await expect.poll(async()=>(await state(linked)).draft.ids.length).toBe(5);
 await action(page,'undo');await action(page,'redo');expect((await state(page)).draft.ids.at(-1)).toBe(added);
 await action(page,'rebuild');await set(page,'point',added);await action(page,'remove');await action(page,'rebuild');
 await set(page,'mass',0.7);await set(page,'cap',1.5);await action(page,'settings');await action(page,'rebuild');
 expect((await state(page)).current.dtm_mass).toBe(0.7);expect((await state(page)).current.max_diameter).toBe(1.5);
 await page.getByText('Inspect the constructed filtration',{exact:true}).click();
 await set(page,'diameter',1.0);await set(page,'codensity',1.2);const generation=(await state(page)).selection.generation;await action(page,'select');
 expect((await state(page)).selection.generation).toBe(generation);
 await capture(page,info,'a124-committed-comparison');
 const before=await state(page);await set(page,'mass',1);await control(page,'settings').click();
 await expect(control(page,'error')).toContainText('strictly between');expect((await state(page)).current).toEqual(before.current);
 await action(page,'reset');await action(page,'rebuild');expect((await state(page)).current.points).toEqual(original.current.points);
 await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45000}).toBe(1);
 await page.setViewportSize({width:640,height:1000});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
 for(const chart of await scrollableCharts(page)){await scrollChart(page,chart,'right');await scrollChart(page,chart,'left');}
 await capture(page,info,'a124-narrow');await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
test('A124 failed rebuild retains the last valid result and undo recovers',async({page},info)=>{
 const errors=[];await openInspector(page,`${base}/source-laboratory-limited`,errors);const original=await state(page);
 await set(page,'cap',3);await action(page,'settings');await control(page,'rebuild').click();
 await expect(control(page,'error')).toContainText('Committed result unchanged');await settled(page);
 expect((await state(page)).current).toEqual(original.current);expect((await state(page)).summary.dirty).toBe(true);
 await capture(page,info,'a124-budget-recovery');await action(page,'undo');await action(page,'rebuild');
 expect((await state(page)).summary.dirty).toBe(false);
 await control(page,'cancel').click();await expect(control(page,'status')).toContainText('cancelled');
 await control(page,'rebuild').click();await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
test('A124 codensity navigation distinguishes inactive sites and censored H0 bars',async({page},info)=>{
 const errors=[];await openInspector(page,`${base}/source-laboratory-activity`,errors);
 expect((await state(page)).source_activity).toEqual([[1,2],[1,2]]);
 await capture(page,info,'a124-inactive-site');
 await page.getByText('Inspect the constructed filtration',{exact:true}).click();
 await set(page,'codensity',0);await action(page,'select');
 expect((await state(page)).source_activity).toEqual([[],[]]);
 expect((await state(page)).selection.generation).toBe(1);
 await set(page,'codensity',3);await action(page,'select');
 const after=await state(page);expect(after.source_activity).toEqual([[1,2,3],[1,2,3]]);
 expect(after.bars[1].map(r=>r.death)).toEqual([1,3,5]);
 expect(after.bars[1].at(-1).right_status).toBe('censored');
 await capture(page,info,'a124-h0-censored');
 await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
