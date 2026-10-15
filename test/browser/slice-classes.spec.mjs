import {test,expect} from '@playwright/test';
import {state,control,openInspector,capture,keyboardActivate,checkKeyboardControls,scrollChart,checkOfflineChartScroll as checkChartScroll} from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function settle(page){
 await expect.poll(async()=>{const s=await state(page);return [s.summary.requests.pending,s.summary.requests.running,s.summary.updating];},{timeout:120000}).toEqual([false,false,false]);
 expect((await state(page)).summary.listener_errors).toEqual([]);
}
test('A40 source cycles, pinned exact correspondence, cache reuse and linked lifecycle',async({page,context},info)=>{
 // Three complete keyboard audits (including a linked client and narrow view)
 // traverse several software-rendered figures. Keep their aggregate budget
 // separate from the shorter mathematical-update and disconnect deadlines.
 test.setTimeout(900_000);
 const errors=[];await openInspector(page,`${base}/slice-classes`,errors);
 await control(page,'slice-interval').selectOption({index:1});
 await expect.poll(async()=>(await state(page)).class!==null,{timeout:120000}).toBe(true);
 await control(page,'slice-class-parameter').fill('3/2');
 await keyboardActivate(control(page,'slice-class-pin'));
 await expect.poll(async()=>(await state(page)).correspondence?.status,{timeout:120000}).toBe('unique_preimage');await settle(page);
 expect((await state(page)).class.cycle.coefficients).toEqual([1,1,1]);
 const linked=await context.newPage();await openInspector(linked,`${base}/slice-classes`,errors);
 await control(page,'slice-base-x').fill('1/4');await keyboardActivate(control(page,'slice-apply'));
 await expect.poll(async()=>(await state(page)).correspondence?.target_point).toEqual(['7//4','3//2']);await settle(page);
 await expect.poll(async()=>(await state(linked)).correspondence?.target_point).toEqual(['7//4','3//2']);
 const before=(await state(page)).selection;
 await control(page,'slice-class-parameter').fill('100');await keyboardActivate(control(page,'slice-class-apply'));
 await expect(control(page,'error')).toContainText('outside');await settle(page);expect((await state(page)).selection).toEqual(before);
 await control(page,'slice-class-parameter').fill('5/2');await keyboardActivate(control(page,'slice-class-apply'));
 await expect.poll(async()=>(await state(page)).correspondence?.status).toBe('zero_at_join');await settle(page);
 await keyboardActivate(control(page,'slice-class-clear'));await expect.poll(async()=>(await state(page)).correspondence).toBe(null);
 await control(page,'slice-class-parameter').fill('3/2');await keyboardActivate(control(page,'slice-class-apply'));await settle(page);
 await control(page,'slice-base-x').fill('0');await keyboardActivate(control(page,'slice-apply'));
 await expect.poll(async()=>(await state(page)).summary.slice_cache_hits).toBeGreaterThan(0);await settle(page);
 await control(page,'slice-interval').selectOption({index:1});await expect.poll(async()=>(await state(page)).class!==null).toBe(true);
 await capture(page,info,'slice-source-wide');await page.setViewportSize({width:520,height:1000});
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);await checkKeyboardControls(page);
 await capture(page,info,'slice-source-narrow');await linked.close();
 await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45000}).toBe(1);
 await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);
 expect(errors).toEqual([]);
});

test('A40 source panels preserve horizontal wheel and keyboard navigation',async({page},info)=>{
 const errors=[];await openInspector(page,`${base}/slice-classes`,errors);
 await control(page,'slice-interval').selectOption({index:1});
 await expect.poll(async()=>(await state(page)).class!==null,{timeout:120000}).toBe(true);await settle(page);
 const charts=page.getByLabel('Slice source cycles',{exact:true});
 await page.waitForFunction(()=>document.querySelector('[aria-label="Slice source cycles"] canvas')?.wglmakie_screen?.renderer?.info.render.frame>0);
 const before=(await state(page)).selection;
 await page.setViewportSize({width:520,height:1000});await page.bringToFront();
 await scrollChart(page,charts,'right');await capture(page,info,'slice-source-narrow-right');
 // This intrinsic-size check also applies to a live canvas: a real wheel
 // gesture must scroll its container without resizing the scientific figure.
 await checkChartScroll(page,charts,charts.locator('canvas'));
 expect((await state(page)).selection).toEqual(before);
 await keyboardActivate(control(page,'close'));await expect.poll(async()=>(await state(page)).summary.closed).toBe(true);
 expect(errors).toEqual([]);
});
