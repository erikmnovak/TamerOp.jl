import {test,expect} from '@playwright/test';
import {state,control,openInspector,capture,keyboardActivate,checkKeyboardControls,scrollChart,checkOfflineChartScroll as checkChartScroll} from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settle(page){
 await expect.poll(async()=>{const s=await state(page);return [s.summary.requests.pending,s.summary.requests.running,s.summary.updating];},{timeout:120000}).toEqual([false,false,false]);
 expect((await state(page)).summary.listener_errors).toEqual([]);
}
test('A55 source states, arrows, temporal barcode and landscape remain synchronized',async({page,context},info)=>{
 const errors=[];await openInspector(page,`${base}/temporal`,errors);
 expect((await state(page)).directions).toEqual(['forward','backward','forward','backward']);
 expect((await state(page)).source_cycle).not.toBeNull();
 const prepared=(await state(page)).summary.landscape_queries;
 await control(page,'occurrence').fill('4');await keyboardActivate(control(page,'apply'));
 await expect.poll(async()=>(await state(page)).selection.occurrence).toBe(4);await settle(page);
 const linked=await context.newPage();await openInspector(linked,`${base}/temporal`,errors);
 await keyboardActivate(control(page,'previous'));await expect.poll(async()=>(await state(linked)).selection.occurrence).toBe(3);await settle(page);
 await control(page,'layer').selectOption({index:1});await expect.poll(async()=>(await state(page)).selection.layer).toBe(2);await settle(page);
 const before=(await state(page)).selection;
 await control(page,'occurrence').fill('99');await keyboardActivate(control(page,'apply'));
 await expect(control(page,'error')).toContainText('occurrence');await settle(page);expect((await state(page)).selection).toEqual(before);
 await control(page,'occurrence').fill('2');await control(page,'scale').fill('1');await control(page,'interval').fill('0');await keyboardActivate(control(page,'apply'));
 await expect.poll(async()=>(await state(page)).selection.scale_index).toBe(1);await settle(page);
 // Click the actual landscape canvas at a known finite grid point.
 const target=(await state(page)).targets.find(t=>t.axis_index===4 && t.selection.occurrence===3 && t.selection.scale_index===3);
 expect(target).toBeTruthy();const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
 await page.bringToFront();const box=await canvas.boundingBox();
 const point={x:box.x+box.width*target.x,y:box.y+box.height*target.y};
 expect(await canvas.evaluate((el,p)=>document.elementFromPoint(p.x,p.y)===el,point)).toBe(true);
 const beforeHover=(await state(page)).selection;
 // A structural update may replace the canvas asynchronously. Exercise real
 // pointer movement until the current scene receives it, rather than accepting
 // a coordinate computed for a canvas that is still being replaced.
 await expect.poll(async()=>{
  await page.mouse.move(point.x-2,point.y);await page.mouse.move(point.x,point.y);
  const current=await state(page),a=current.pointer.axes.find(a=>a.index===4);
  return a.inside&&Math.hypot(a.data[0]-3,a.data[1]-3)<0.04&&current.ui.hover.includes('occurrence 3, scale index 3');
 }).toBe(true);
 await expect(control(page,'hover')).toContainText('occurrence 3, scale index 3');
 expect((await state(page)).selection).toEqual(beforeHover);
 await page.mouse.down();await page.mouse.up();
 await expect.poll(async()=>(await state(page)).selection.scale_index).toBe(3);await settle(page);
 // Hover must not replace the scale control with its pixel conversion vector.
 await control(page,'scale').fill('2');await keyboardActivate(control(page,'apply'));
 await expect.poll(async()=>(await state(page)).selection.scale_index).toBe(2);await settle(page);
 expect((await state(page)).summary.landscape_queries).toBe(prepared);
 const retained=(await state(page)).ui.rebuilds;
 // Slider input is bounded and the last request wins.
 await control(page,'time-slider').evaluate(node=>{for(const value of [1,2,4,5]){node.value=String(value);node.dispatchEvent(new Event('input',{bubbles:true}));}});
 await expect.poll(async()=>(await state(page)).selection.occurrence,{timeout:120000}).toBe(5);await settle(page);
 expect((await state(page)).ui.rebuilds).toBe(retained);
 await capture(page,info,'temporal-wide');await page.setViewportSize({width:520,height:1000});
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);await checkKeyboardControls(page);await capture(page,info,'temporal-narrow');
 const charts=page.locator('[aria-label="Temporal charts"]');await scrollChart(page,charts,'right');await capture(page,info,'temporal-narrow-right');
 await scrollChart(page,charts,'left');
 await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45000}).toBe(1);
 await keyboardActivate(control(page,'reset'));await expect.poll(async()=>(await state(page)).selection.occurrence).toBe(3);
 await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);
 expect(errors).toEqual([]);
});

test('A55 charts preserve horizontal wheel and keyboard navigation',async({page},info)=>{
 const errors=[];await openInspector(page,`${base}/temporal`,errors);
 const before=(await state(page)).selection;
 await page.setViewportSize({width:520,height:1000});await page.bringToFront();
 const charts=page.getByLabel('Temporal charts',{exact:true});
 await scrollChart(page,charts,'right');await capture(page,info,'temporal-narrow-wheel');
 await checkChartScroll(page,charts,charts.locator('canvas'));
 expect((await state(page)).selection).toEqual(before);
 await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);
 expect(errors).toEqual([]);
});
