import {test,expect} from '@playwright/test';
import {state,control,openInspector,capture,keyboardActivate,checkKeyboardControls,scrollChart} from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settled(page){await expect.poll(async()=>{const s=await state(page);return [s.summary.requests.pending,s.summary.requests.running,s.summary.updating];}).toEqual([false,false,false]);expect((await state(page)).summary.listener_errors).toEqual([]);}
async function apply(page){const rev=(await state(page)).selection.revision;await keyboardActivate(control(page,'apply'));await expect.poll(async()=>(await state(page)).selection.revision).toBeGreaterThan(rev);await settled(page);}
test('A126 graphcode edges, exact coefficients, linked clients and recovery',async({page,context},info)=>{
 const errors=[];await openInspector(page,`${base}/graphcode`,errors);expect((await state(page)).coefficient).toBe(1);
 await checkKeyboardControls(page);
 const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();const box=await canvas.boundingBox();
 const initial=await state(page);const t=initial.ui.pick_targets.find(t=>t.label.startsWith('Map 1') && t.selection.source_interval===1);
 expect(t).toBeTruthy();await page.mouse.move(box.x+t.x*box.width,box.y+t.y*box.height);
 await expect(control(page,'hover')).toContainText('coefficient 1');expect((await state(page)).selection.revision).toBe(initial.selection.revision);
 await page.mouse.click(box.x+t.x*box.width,box.y+t.y*box.height);await expect.poll(async()=>(await state(page)).selection.row).toBe(1);await settled(page);
 await control(page,'source').fill('1');await control(page,'target').fill('2');await apply(page);expect((await state(page)).coefficient).toBe(0);
 await expect(control(page,'readout')).toContainText('Selected coefficient: 0');
 const linked=await context.newPage();await openInspector(linked,`${base}/graphcode`,errors);
 await control(page,'target').fill('1');await apply(page);await expect.poll(async()=>(await state(linked)).coefficient).toBe(1);
 const previous=(await state(page)).selection;await control(page,'source').fill('99');await keyboardActivate(control(page,'apply'));await expect(control(page,'error')).toContainText('source interval');expect((await state(page)).selection).toEqual(previous);
 await keyboardActivate(control(page,'reset'));await expect.poll(async()=>(await state(page)).selection.row).toBe(2);await settled(page);await expect(control(page,'hover')).toBeEmpty();
 const bar=(await state(page)).ui.pick_targets.find(t=>t.label.startsWith('Row 4, interval 2:'));
 await canvas.scrollIntoViewIfNeeded();const currentBox=await canvas.boundingBox();const bx=0.75*bar.segment[0].x+0.25*bar.segment[1].x;
 await page.mouse.click(currentBox.x+bx*currentBox.width,currentBox.y+bar.segment[0].y*currentBox.height);
 await expect.poll(async()=>(await state(page)).selection.row).toBe(4);await settled(page);expect((await state(page)).selection.source_interval).toBe(2);
 await keyboardActivate(control(page,'reset'));await expect.poll(async()=>(await state(page)).selection.row).toBe(2);await settled(page);
 await capture(page,info,'graphcode-wide');await linked.close();
 await page.setViewportSize({width:520,height:1000});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
 const panel=page.locator('[aria-label="Graphcode charts"]');await scrollChart(page,panel,'right');await scrollChart(page,panel,'left');
 await capture(page,info,'graphcode-narrow');
 await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);expect((await state(page)).ui.observer_count).toBe(0);expect(errors).toEqual([]);
});
test('A126 compression, terminal rows and rapid keyboard selection retain the canvas',async({page},info)=>{
 const errors=[];await openInspector(page,`${base}/graphcode`,errors);const canvas=await page.locator('canvas').first().elementHandle();
 await control(page,'compressed').check();await expect.poll(async()=>(await state(page)).selection.compressed).toBe(true);await settled(page);
 await capture(page,info,'graphcode-compressed');
 const row=control(page,'row-slider');await row.focus();await row.press('End');await expect.poll(async()=>(await state(page)).selection.row).toBe(4);await settled(page);
 expect((await state(page)).active_target).toEqual([]);await expect(control(page,'readout')).toContainText('final row has no outgoing map');
 const col=control(page,'column-slider');await col.focus();for(let i=0;i<6;i++)await col.press(i%2?'Home':'End');await col.press('End');await expect.poll(async()=>(await state(page)).selection.column).toBe(4);await settled(page);
 expect((await state(page)).active_source).toEqual([1]);expect(await canvas.evaluate(el=>el.isConnected)).toBe(true);expect((await state(page)).ui.rebuild_count).toBe(1);expect((await state(page)).ui.update_count).toBeGreaterThan(0);
 await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);expect(errors).toEqual([]);
});

for(const [suffix,field,firstCoefficient,secondCoefficient] of [['qq','QQ','-2/3','-1'],['f5','F5','2','4']]) {
 test(`A126 ${field} coefficients survive selection, compression and linked clients`,async({page,context},info)=>{
  const errors=[];await openInspector(page,`${base}/graphcode-${suffix}`,errors);
  expect((await state(page)).field).toBe(field);expect((await state(page)).coefficient_text).toBe(firstCoefficient);
  const canvas=page.locator('canvas').first();const retained=await canvas.elementHandle();await canvas.scrollIntoViewIfNeeded();
  const box=await canvas.boundingBox(),initial=await state(page),edge=initial.ui.pick_targets.find(t=>t.label.startsWith('Map 2'));
  expect(edge.label).toContain(`coefficient ${secondCoefficient} in ${field}`);
  await page.mouse.move(box.x+edge.x*box.width,box.y+edge.y*box.height);
  await expect(control(page,'hover')).toContainText(`coefficient ${secondCoefficient}`);
  expect((await state(page)).selection.revision).toBe(initial.selection.revision);
  await page.mouse.click(box.x+edge.x*box.width,box.y+edge.y*box.height);
  await expect.poll(async()=>(await state(page)).coefficient_text).toBe(secondCoefficient);await settled(page);
  const linked=await context.newPage();await openInspector(linked,`${base}/graphcode-${suffix}`,errors);
  await control(page,'compressed').check();await expect.poll(async()=>(await state(linked)).selection.compressed).toBe(true);await settled(page);
  await expect(control(page,'readout')).toContainText(`Selected coefficient: ${secondCoefficient}`);
  await keyboardActivate(control(page,'reset'));await expect.poll(async()=>(await state(linked)).coefficient_text).toBe(firstCoefficient);await settled(page);
  expect(await retained.evaluate(el=>el.isConnected)).toBe(true);expect((await state(page)).ui.rebuild_count).toBe(1);
  await capture(page,info,`graphcode-${suffix}-wide`);await linked.close();
  await page.setViewportSize({width:520,height:1000});await scrollChart(page,page.locator('[aria-label="Graphcode charts"]'),'right');
  await capture(page,info,`graphcode-${suffix}-narrow`);
  await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);
  expect(errors).toEqual([]);
 });
}
