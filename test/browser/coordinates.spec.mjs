import {test,expect} from '@playwright/test';
import {state,control,openInspector,capture,keyboardActivate,checkKeyboardControls,scrollChart,checkOfflineChartScroll} from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settle(page){
 await expect.poll(async()=>{const s=await state(page);return [s.summary.requests.pending,s.summary.requests.running,s.summary.updating];},{timeout:120000}).toEqual([false,false,false]);
 expect((await state(page)).summary.listener_errors).toEqual([]);
}
test('A97 linked source, circle and torus coordinates',async({page,context},info)=>{
 const errors=[];await openInspector(page,`${base}/coordinates`,errors);
 const lifts=(await state(page)).lifts;
 await control(page,'vertex').fill('7');await keyboardActivate(control(page,'apply'));
 await expect.poll(async()=>(await state(page)).selection.vertex).toBe(7);await settle(page);
 const linked=await context.newPage();await openInspector(linked,`${base}/coordinates`,errors);
 await control(page,'coordinate').selectOption({index:1});
 await expect.poll(async()=>(await state(linked)).selection.coordinate).toBe(2);await settle(page);
 const before=(await state(page)).selection;
 await control(page,'vertex').fill('999');await keyboardActivate(control(page,'apply'));
 await expect(control(page,'error')).toContainText('active source vertex');await settle(page);
 expect((await state(page)).selection).toEqual(before);
 await control(page,'vertex').fill('3');await control(page,'phase').fill('0.125');await keyboardActivate(control(page,'apply'));
 await expect.poll(async()=>(await state(page)).selection.phase).toBe(0.125);await settle(page);
 // Select a genuinely displayed source point through the WebGL canvas.
 const target=(await state(page)).targets.find(t=>t.axis_index===1&&t.selection.vertex===1);
 const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();await page.bringToFront();
 const box=await canvas.boundingBox(),point={x:box.x+box.width*target.x,y:box.y+box.height*target.y};
 const hoverBefore=(await state(page)).selection;
 await expect.poll(async()=>{await page.mouse.move(point.x-2,point.y);await page.mouse.move(point.x,point.y);return (await state(page)).ui.hover.includes('Vertex 1');}).toBe(true);
 expect((await state(page)).selection).toEqual(hoverBefore);
 await page.mouse.down();await page.mouse.up();
 await expect.poll(async()=>(await state(page)).selection.vertex).toBe(1);await settle(page);
 const builds=(await state(page)).ui.rebuilds;
 await control(page,'phase-slider').evaluate(node=>{for(const value of [15,30,65,75]){node.value=String(value);node.dispatchEvent(new Event('input',{bubbles:true}));}});
 await expect.poll(async()=>(await state(page)).selection.phase).toBe(0.75);await settle(page);
 expect((await state(page)).lifts).toEqual(lifts);
 expect((await state(page)).ui.rebuilds).toBe(builds);
 await capture(page,info,'coordinates-wide');await page.setViewportSize({width:520,height:1000});
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
 await checkKeyboardControls(page);await capture(page,info,'coordinates-narrow');
 // Bonito retains a disconnected client for its 30-second reconnect window.
 await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
 await keyboardActivate(control(page,'reset'));await expect.poll(async()=>(await state(page)).selection.phase).toBe(0);
 await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);
 expect(errors).toEqual([]);
});
test('A97 coordinate charts retain readable narrow scrolling',async({page},info)=>{
 const errors=[];await openInspector(page,`${base}/coordinates`,errors);await page.setViewportSize({width:520,height:1000});
 const charts=page.getByLabel('Coordinate charts',{exact:true});await scrollChart(page,charts,'right');
 await capture(page,info,'coordinates-narrow-right');await checkOfflineChartScroll(page,charts,charts.locator('canvas'));
 await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);
 expect(errors).toEqual([]);
});
