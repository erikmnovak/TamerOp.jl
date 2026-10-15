import {test,expect} from '@playwright/test';
import {state,control,openInspector,capture,keyboardActivate,checkKeyboardControls} from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){await expect.poll(async()=>{const s=await state(page);return [s.summary.requests.pending,s.summary.requests.running,s.summary.updating];}).toEqual([false,false,false]);expect((await state(page)).summary.listener_errors).toEqual([]);}
async function apply(page){const rev=(await state(page)).selection.revision;await keyboardActivate(control(page,'apply'));await expect.poll(async()=>(await state(page)).selection.revision).toBeGreaterThan(rev);await settled(page);}
test('A129 event-grid selection, reproduced labels, threshold recovery and linked clients',async({page,context},info)=>{
 const errors=[];await openInspector(page,`${base}/clustering`,errors);expect((await state(page)).labels).toEqual([1,1,1,2,2,2,0]);
 await checkKeyboardControls(page);
 const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();const box=await canvas.boundingBox();
 const initial=await state(page);const t=initial.ui.pick_targets.find(t=>t.selection.neighbors===0 && t.selection.diameter===0);
 expect(t).toBeTruthy();await page.mouse.move(box.x+t.x*box.width,box.y+t.y*box.height);
 await expect(control(page,'hover')).toContainText('7 components');expect((await state(page)).selection.revision).toBe(initial.selection.revision);
 await page.mouse.click(box.x+t.x*box.width,box.y+t.y*box.height);await expect.poll(async()=>(await state(page)).labels).toEqual([1,2,3,4,5,6,7]);await settled(page);
 await control(page,'diameter').fill('0.3');await control(page,'neighbors').fill('1');await apply(page);expect((await state(page)).labels).toEqual([1,1,1,2,2,2,0]);
 const linked=await context.newPage();await openInspector(linked,`${base}/clustering`,errors);
 await control(page,'min-size').fill('4');await apply(page);expect((await state(page)).labels).toEqual([0,0,0,0,0,0,0]);await expect.poll(async()=>(await state(linked)).labels).toEqual([0,0,0,0,0,0,0]);
 await control(page,'diameter').fill('-1');const previous=(await state(page)).selection;await keyboardActivate(control(page,'apply'));await expect(control(page,'error')).toContainText('diameter must lie');expect((await state(page)).selection).toEqual(previous);
 await keyboardActivate(control(page,'reset'));await expect.poll(async()=>(await state(page)).labels).toEqual([1,1,1,2,2,2,0]);await settled(page);
 await expect(control(page,'hover')).toBeEmpty();
 await capture(page,info,'clustering-wide');await linked.close();
 await page.setViewportSize({width:520,height:1000});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);await capture(page,info,'clustering-narrow');
 await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);expect((await state(page)).ui.observer_count).toBe(0);expect(errors).toEqual([]);
});
test('A129 tied coordinates, degree gestures, lifetime gaps and rapid requests',async({page},info)=>{
 const errors=[];await openInspector(page,`${base}/clustering-ties`,errors);expect((await state(page)).labels).toEqual([1,1,0,0]);
 const originalCanvas=await page.locator('canvas').first().elementHandle();
 const gapInput=page.locator('input[id$="-gap"]');
 const initial=(await state(page)).selection;await gapInput.fill('999');await keyboardActivate(control(page,'apply-gap'));
 await expect(control(page,'error')).toContainText('available gap index');expect((await state(page)).selection).toEqual(initial);await gapInput.fill('1');
 const slider=control(page,'degree-slider');await slider.focus();await slider.press('End');await expect.poll(async()=>(await state(page)).selection.neighbors).toBe(3);await settled(page);expect((await state(page)).labels).toEqual([0,0,0,0]);
 await control(page,'diameter-slider').focus();await control(page,'diameter-slider').press('End');await expect.poll(async()=>(await state(page)).selection.diameter).toBe(2);await settled(page);expect((await state(page)).labels).toEqual([1,1,1,1]);
 expect((await state(page)).gaps.length).toBeGreaterThan(0);const revision=(await state(page)).selection.revision;await keyboardActivate(control(page,'apply-gap'));
 await expect.poll(async()=>(await state(page)).selection.revision).toBeGreaterThan(revision);await settled(page);
 await slider.focus();for(let i=0;i<8;i++)await slider.press(i%2?'Home':'End');await slider.press('Home');await expect.poll(async()=>(await state(page)).selection.neighbors).toBe(0);await settled(page);
 expect(await originalCanvas.evaluate(el=>el.isConnected)).toBe(true);expect((await state(page)).ui.rebuild_count).toBe(1);expect((await state(page)).ui.update_count).toBeGreaterThan(0);
 await capture(page,info,'clustering-ties');await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);expect(errors).toEqual([]);
});
