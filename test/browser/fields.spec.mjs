import {test,expect} from '@playwright/test';
import {checkKeyboardControls,keyboardActivate,scrollChart} from './helpers.mjs';
const url=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}/fields`;
const state=async page=>JSON.parse(await page.getByTestId('julia-session-state').textContent());
async function open(page,errors){
  page.on('pageerror',e=>errors.push(e.message));
  page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.goto(url,{waitUntil:'domcontentloaded',timeout:300_000});
  await page.getByTestId('julia-session-state').waitFor({state:'attached',timeout:300_000});
  await page.waitForFunction(()=>document.querySelector('canvas')?.wglmakie_screen?.renderer?.info.render.frame>0,null,{timeout:300_000});
  await checkKeyboardControls(page);
}
test('A44 linked planes: real sliders, chart selection, layer, errors and teardown',async({page},info)=>{
  const errors=[];await open(page,errors);
  expect((await state(page)).summary.scale.limits).toEqual([11108,22342]);
  expect((await state(page)).native[0].rows).toEqual([1,2,3,4].map(k=>[11202+10*k,12202+10*k]));
  const firstCanvas=await page.locator('canvas').first().elementHandle();
  const rebuilds=(await state(page)).ui.rebuilds;
  await page.getByRole('slider',{name:'Selected index on b',exact:true}).press('End');
  await expect.poll(async()=>(await state(page)).selection.indices).toEqual([1,3,3,2]);
  await page.getByLabel('Layer ID',{exact:true}).fill('H1');
  await keyboardActivate(page.getByRole('button',{name:'Apply view',exact:true}));
  await expect.poll(async()=>(await state(page)).selection.layer).toBe('H1');
  expect((await state(page)).ui.rebuilds).toBe(rebuilds);
  expect((await state(page)).ui.updates).toBeGreaterThanOrEqual(2);
  expect(await firstCanvas.evaluate(c=>c.isConnected)).toBe(true);
  expect((await state(page)).summary.scale.limits).toEqual([11108,22342]);
  await expect.poll(async()=>(await state(page)).summary.updating).toBe(false);
  await page.waitForFunction(()=>document.querySelector('canvas')?.wglmakie_screen?.renderer?.info.render.frame>0);
  const target=(await state(page)).target,box=await page.locator('canvas').first().boundingBox();
  // WGL throttles pointer position separately from button events. Wait for
  // Julia's actual hover readout before pressing the button on the same point.
  await page.mouse.move(box.x+box.width*target.x,box.y+box.height*target.y);
  await expect(page.locator('[id^="tamerop-field-inspector-"][id$="-hover"]')).toHaveText(
    'A: (20.0, 5.0); value=22322.0. Nearest stored sample.');
  await page.mouse.click(box.x+box.width*target.x,box.y+box.height*target.y);
  await expect.poll(async()=>(await state(page)).selection.indices).toEqual([2,3,2,2]);
  await expect(page.getByRole('slider',{name:'Selected index on a',exact:true})).toHaveValue('2');
  await expect(page.getByRole('slider',{name:'Selected index on c',exact:true})).toHaveValue('2');
  await page.getByLabel('Ordered planes (1,2; 1,3)',{exact:true}).fill('4,2; 1,3');
  await page.getByLabel('Baseline ID (blank for values)',{exact:true}).fill('A');
  await keyboardActivate(page.getByRole('button',{name:'Apply view',exact:true}));
  await expect.poll(async()=>(await state(page)).selection.planes).toEqual([[4,2],[1,3]]);
  expect((await state(page)).summary.scale.limits).toEqual([-3,3]);
  expect((await state(page)).native[2].rows).toEqual([[-3,-3,-3],[-3,-3,-3]]);
  const revision=(await state(page)).summary.revision;
  await page.getByLabel('Layer ID',{exact:true}).fill('unknown');
  await keyboardActivate(page.getByRole('button',{name:'Apply view',exact:true}));
  await expect(page.getByRole('alert')).toContainText('Unknown layer');
  expect((await state(page)).summary.revision).toBe(revision);
  const second=await page.context().newPage();await open(second,errors);
  await expect.poll(async()=>(await state(page)).viewer_count).toBe(2);
  await keyboardActivate(second.getByRole('button',{name:'Reset',exact:true}));
  await expect.poll(async()=>(await state(page)).selection.indices).toEqual([1,2,3,2]);
  for (const [axis,value] of [['a','1'],['b','2'],['c','3'],['d','2']]) {
    await expect(page.getByRole('slider',{name:`Selected index on ${axis}`,exact:true})).toHaveValue(value);
  }
  await expect(page.getByRole('alert')).toBeEmpty();
  await second.close();
  await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  await page.setViewportSize({width:480,height:900});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await scrollChart(page,page.getByLabel('Coordinate section charts',{exact:true}),'right');
  await page.screenshot({path:info.outputPath('fields-narrow.png'),fullPage:true});
  await keyboardActivate(page.getByRole('button',{name:'Close inspector',exact:true}));
  await expect.poll(async()=>{const s=await state(page);return[s.summary.closed,s.ui.observers,s.ui.pointer_observers,s.viewer_count];}).toEqual([true,0,0,0]);
  expect(errors).toEqual([]);
});
