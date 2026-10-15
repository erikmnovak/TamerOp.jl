import {test,expect} from '@playwright/test';
import {state,control,openInspector,capture,keyboardActivate,checkKeyboardControls,scrollChart} from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){await expect.poll(async()=>{const s=await state(page);return [s.summary.requests.pending,s.summary.requests.running,s.summary.updating];}).toEqual([false,false,false]);expect((await state(page)).summary.listener_errors).toEqual([]);}
async function apply(page){const revision=(await state(page)).selection.revision;await keyboardActivate(control(page,'apply'));await expect.poll(async()=>(await state(page)).selection.revision).toBeGreaterThan(revision);await settled(page);}
test('A127 exact curve selection, Betti grades, linked views and recovery',async({page,context},info)=>{
 const errors=[];await openInspector(page,`${base}/end-curves`,errors);expect((await state(page)).count).toBe(3);
 await checkKeyboardControls(page);
 const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();const box=await canvas.boundingBox();
 const before=await state(page);const target=before.ui.pick_targets.find(t=>t.label.startsWith('D1 at'));
 expect(target).toBeTruthy();await page.mouse.move(box.x+target.x*box.width,box.y+target.y*box.height);
 await expect(control(page,'hover')).toContainText('D1');expect((await state(page)).selection.revision).toBe(before.selection.revision);
 await page.mouse.click(box.x+target.x*box.width,box.y+target.y*box.height);await expect.poll(async()=>(await state(page)).selection.kind).toBe('death');await settled(page);
 const linked=await context.newPage();await openInspector(linked,`${base}/end-curves`,errors);
 await control(page,'x').fill('1');await control(page,'y').fill('6');await apply(page);
 await expect.poll(async()=>(await state(linked)).selection.point).toEqual([1,6]);
 const degree=control(page,'degree');await degree.focus();await degree.press('End');await expect.poll(async()=>(await state(page)).selection.betti_degree).toBe(2);await settled(page);
 expect((await state(page)).point.betti[2]).toBeGreaterThan(0);await expect(control(page,'readout')).toContainText('zero halo');
 const valid=(await state(page)).selection;await control(page,'curve').fill('99');await keyboardActivate(control(page,'apply'));await expect(control(page,'error')).toContainText('curve');expect((await state(page)).selection).toEqual(valid);
 await keyboardActivate(control(page,'reset'));await expect.poll(async()=>(await state(page)).selection.point).toEqual([-2,4]);await settled(page);await expect(control(page,'hover')).toBeEmpty();
 await expect.poll(async()=>(await state(linked)).selection.kind).toBe('birth');await linked.close();
 await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
 await capture(page,info,'end-curves-wide');await page.setViewportSize({width:520,height:1000});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
 const chart=page.locator('[aria-label="End-curve charts"]');await scrollChart(page,chart,'right');await scrollChart(page,chart,'left');await capture(page,info,'end-curves-narrow');
 await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);expect((await state(page)).ui.observer_count).toBe(0);expect(errors).toEqual([]);
});
test('A127 zero module, kind changes and rapid lattice gestures retain the canvas',async({page},info)=>{
 const errors=[];await openInspector(page,`${base}/end-curves-empty`,errors);expect((await state(page)).count).toBe(0);const canvas=await page.locator('canvas').first().elementHandle();
 await control(page,'kind').selectOption({label:'death'});await expect.poll(async()=>(await state(page)).selection.kind).toBe('death');await settled(page);
 await control(page,'curve').fill('1');await keyboardActivate(control(page,'apply'));await expect(control(page,'error')).toContainText('curve');expect((await state(page)).selection.curve).toBe(0);
 await keyboardActivate(control(page,'reset'));await expect.poll(async()=>(await state(page)).selection.kind).toBe('birth');await settled(page);
 const x=control(page,'x-slider');await x.focus();for(let i=0;i<6;i++)await x.press(i%2?'Home':'End');await x.press('End');await expect.poll(async()=>(await state(page)).selection.point[0]).toBe(2);await settled(page);
 await control(page,'show-betti').uncheck();await expect.poll(async()=>(await state(page)).selection.show_betti).toBe(false);await settled(page);
 expect(await canvas.evaluate(el=>el.isConnected)).toBe(true);expect((await state(page)).ui.rebuild_count).toBe(1);expect((await state(page)).ui.update_count).toBeGreaterThan(0);
 await capture(page,info,'end-curves-empty');await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);expect(errors).toEqual([]);
});
