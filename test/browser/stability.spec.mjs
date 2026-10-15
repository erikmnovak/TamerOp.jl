import {test,expect} from '@playwright/test';
import {state,control,openInspector,capture,keyboardActivate,scrollChart} from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){await expect.poll(async()=>{const s=await state(page);return [s.summary.requests.pending,s.summary.requests.running,s.summary.updating];}).toEqual([false,false,false]);expect((await state(page)).summary.listener_errors).toEqual([]);}
async function apply(page){const rev=(await state(page)).selection.revision;await keyboardActivate(control(page,'apply'));await expect.poll(async()=>(await state(page)).selection.revision,{timeout:120_000}).toBeGreaterThan(rev);await settled(page);}
test('A51 exact chamber walls, factor maps, linked clients and rollback',async({page,context},info)=>{
 const errors=[];await openInspector(page,`${base}/stability`,errors);
 const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();const box=await canvas.boundingBox();
 const before=await state(page);const target=before.ui.pick_targets.find(t=>t.label.startsWith('Chamber'));
 expect(target).toBeTruthy();await page.mouse.move(box.x+target.x*box.width,box.y+target.y*box.height);
 await expect(control(page,'hover')).toContainText('Chamber');expect((await state(page)).selection.revision).toBe(before.selection.revision);
 await page.mouse.click(box.x+target.x*box.width,box.y+target.y*box.height);
 await expect.poll(async()=>(await state(page)).selection.revision,{timeout:120_000}).toBeGreaterThan(before.selection.revision);await settled(page);
 const linked=await context.newPage();await openInspector(linked,`${base}/stability`,errors);
 await control(page,'x').fill('-3//2');await control(page,'y').fill('21//4');await apply(page);
 expect((await state(page)).boundary).toBe(true);await expect(control(page,'readout')).toContainText('8//15');
 await expect(control(linked,'y')).toHaveValue('21//4');
 const valid=(await state(page)).selection;await control(page,'factor').fill('99');await keyboardActivate(control(page,'apply'));
 await expect(control(page,'error')).toContainText('factor');expect((await state(page)).selection).toEqual(valid);
 await control(page,'factor').fill('1');await apply(page);await expect(control(page,'readout')).toContainText('Projection to factor 1');
 await keyboardActivate(control(page,'reset'));await expect(control(page,'y')).toHaveValue('5//1');await settled(page);
 await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
 await capture(page,info,'stability-wide');await page.setViewportSize({width:520,height:1000});
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
 const chart=page.locator('[aria-label="Stability charts"]');await scrollChart(page,chart,'right');await scrollChart(page,chart,'left');await capture(page,info,'stability-narrow');
 await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);
 expect((await state(page)).ui.observer_count).toBe(0);expect(errors).toEqual([]);
});
test('A51 landscape thresholds, layers, difference values and sample picking',async({page},info)=>{
 const errors=[];await openInspector(page,`${base}/stability-landscape`,errors);
 const canvas=await page.locator('canvas').first().elementHandle();
 await control(page,'threshold').fill('1//2');await control(page,'layer').fill('2');await apply(page);
 await control(page,'view').selectOption({label:'difference'});await expect.poll(async()=>(await state(page)).selection.view).toBe('difference');await settled(page);
 await expect(control(page,'readout')).toContainText('radius difference in');await expect(control(page,'readout')).toContainText('enclosure');
 const valid=(await state(page)).selection;await control(page,'threshold').fill('1//7');await keyboardActivate(control(page,'apply'));
 await expect(control(page,'error')).toContainText('retained thresholds');expect((await state(page)).selection).toEqual(valid);
 await keyboardActivate(control(page,'reset'));await settled(page);
 const target=(await state(page)).ui.pick_targets.find(t=>t.selection.vertex===4);expect(target).toBeTruthy();
 await page.locator('canvas').first().scrollIntoViewIfNeeded();const box=await page.locator('canvas').first().boundingBox();
 await page.mouse.click(box.x+target.x*box.width,box.y+target.y*box.height);
 await expect.poll(async()=>(await state(page)).selection.vertex).toBe(4);await settled(page);
 expect(await canvas.evaluate(el=>el.isConnected)).toBe(true);expect((await state(page)).ui.rebuild_count).toBe(1);
 await capture(page,info,'landscape-wide');await keyboardActivate(control(page,'close'));
 await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);expect(errors).toEqual([]);
});
