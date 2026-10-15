import {test,expect} from '@playwright/test';
import {state,control,openInspector,capture,keyboardActivate,checkKeyboardControls,scrollChart} from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){
 await expect.poll(async()=>{const s=await state(page);return [s.summary.requests.pending,s.summary.requests.running,s.summary.updating];}).toEqual([false,false,false]);
 expect((await state(page)).summary.listener_errors).toEqual([]);
}
async function activate(page,name){
 const revision=(await state(page)).selection.revision;
 await keyboardActivate(control(page,name));
 await expect.poll(async()=>(await state(page)).selection.revision).toBeGreaterThan(revision);await settled(page);
}
async function pickTarget(page,predicate){
 const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
 const box=await canvas.boundingBox(),s=await state(page),t=s.ui.pick_targets.find(predicate);
 expect(t).toBeTruthy();await page.mouse.move(box.x+t.x*box.width,box.y+t.y*box.height);
 await expect(control(page,'hover')).toHaveText(t.label);
 expect((await state(page)).selection.revision).toBe(s.selection.revision);
 await page.mouse.click(box.x+t.x*box.width,box.y+t.y*box.height);
 await expect.poll(async()=>(await state(page)).selection.revision).toBeGreaterThan(s.selection.revision);await settled(page);
 return t;
}
test('A130 operations, matrix/source/pair selection, exact witnesses and linked lifecycle',async({page,context},info)=>{
 const errors=[];await openInspector(page,`${base}/reduction`,errors);
 expect((await state(page)).prefix).toBe(0);expect((await state(page)).pairs).toEqual([]);
 await activate(page,'next');let s=await state(page);
 expect(s.prefix).toBe(0);expect(s.active).toBe(1);
 await activate(page,'next');expect((await state(page)).prefix).toBe(1);
 await activate(page,'previous');expect((await state(page)).prefix).toBe(0);
 await activate(page,'finish');s=await state(page);const finalStep=s.selection.step;
 expect(s.prefix).toBe(7);expect(s.pairs.map(p=>[p.birth_cell,p.death_cell])).toEqual([[1,0],[2,4],[3,5],[6,7]]);
 await pickTarget(page,t=>t.axis_index===3 && t.selection.birth===6);
 expect((await state(page)).witness.pair.birth).toBe('4');expect((await state(page)).witness.pair.death).toBe('7');
 await control(page,'chain').selectOption({index:1});await expect.poll(async()=>(await state(page)).selection.chain).toBe('filling');await settled(page);
 await expect(control(page,'readout')).toContainText('cell 7: 1');
 await control(page,'matrix').selectOption({index:2});await expect.poll(async()=>(await state(page)).selection.matrix).toBe('V');await settled(page);
 await pickTarget(page,t=>t.axis_index===2 && t.label.startsWith('V[7,7]'));
 expect((await state(page)).selection.cell).toBe(7);
 await pickTarget(page,t=>t.axis_index===1 && t.selection.cell===1);expect((await state(page)).selection.cell).toBe(1);
 const linked=await context.newPage();await openInspector(linked,`${base}/reduction`,errors);
 await activate(page,'previous');await expect.poll(async()=>(await state(linked)).selection.step).toBe(finalStep-1);
 expect((await state(page)).active).toBe(7);expect((await state(page)).prefix).toBe(6);
 expect((await state(page)).pairs.find(p=>p.birth_cell===6).death_cell).toBe(0);
 await expect(control(page,'readout')).toContainText('No filling exists');
 const valid=(await state(page)).selection;
 await control(page,'step').fill('-1');await keyboardActivate(control(page,'apply'));
 await expect(control(page,'error')).toContainText('step');await settled(page);expect((await state(page)).selection).toEqual(valid);
 await activate(page,'reset');expect((await state(page)).prefix).toBe(0);
 await expect(control(page,'hover')).toBeEmpty();await expect.poll(async()=>(await state(linked)).selection.step).toBe(0);
 const slider=control(page,'step-slider');await slider.focus();await slider.press('End');
 for(let i=0;i<5;i++)await slider.press('ArrowLeft');
 await slider.press('Home');await expect.poll(async()=>(await state(page)).selection.step).toBe(0);await settled(page);
 await activate(page,'finish');expect((await state(page)).ui.rebuild_count).toBe(1);
 await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
 await capture(page,info,'reduction-wide');await page.setViewportSize({width:520,height:1000});
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
 await checkKeyboardControls(page);
 const chart=page.locator('[aria-label="Reduction charts"]');await scrollChart(page,chart,'right');await capture(page,info,'reduction-narrow-matrix');
 await scrollChart(page,chart,'left');await capture(page,info,'reduction-narrow-source');
 await keyboardActivate(control(page,'close'));
 await expect.poll(async()=>{const s=await state(page);return [s.summary.closed,s.ui.closed,s.ui.observer_count,s.viewer_count];}).toEqual([true,true,0,0]);
 expect(errors).toEqual([]);
});
test('A130 empty reduction keeps controls and lifecycle meaningful',async({page},info)=>{
 const errors=[];await openInspector(page,`${base}/reduction-empty`,errors);
 expect((await state(page)).pairs).toEqual([]);await activate(page,'next');await activate(page,'previous');await activate(page,'finish');
 expect((await state(page)).selection.step).toBe(0);
 await control(page,'birth').fill('1');await keyboardActivate(control(page,'apply'));await expect(control(page,'error')).toContainText('Birth');await settled(page);
 expect((await state(page)).selection.birth).toBe(0);await activate(page,'reset');await capture(page,info,'reduction-empty');
 await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);
 expect(errors).toEqual([]);
});
