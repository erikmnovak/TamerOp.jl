import {test,expect} from '@playwright/test';
import {state,control,openInspector,capture,fullyClosed} from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function idle(page){await expect.poll(async()=>{const s=await state(page);return [s.summary.requests.pending,s.summary.requests.running,s.summary.updating];}).toEqual([false,false,false]);}
test('A124-F1 changing diameter recalculates degree eligibility with exact events',async({page},info)=>{
 const errors=[];await openInspector(page,`${base}/radius-degree`,errors);
 expect((await state(page)).summary.active_counts).toEqual([0,0]);
 await expect(page.getByText('Minimum neighbors at this diameter k',{exact:false}).first()).toBeVisible();
 await control(page,'grade-1').focus();await control(page,'grade-1').press('End');
 await expect.poll(async()=>(await state(page)).summary.active_counts).toEqual([4,3]);await idle(page);
 await control(page,'grade-2').focus();await control(page,'grade-2').press('End');
 await expect.poll(async()=>(await state(page)).summary.active_counts).toEqual([1,0]);await idle(page);
 await control(page,'grade-1').focus();await control(page,'grade-1').press('Home');
 await expect.poll(async()=>(await state(page)).summary.active_counts).toEqual([0,0]);await idle(page);
 await control(page,'grade-2').focus();await control(page,'grade-2').press('Home');
 await expect.poll(async()=>(await state(page)).summary.active_counts).toEqual([4,0]);await idle(page);
 await capture(page,info,'radius-degree');await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
