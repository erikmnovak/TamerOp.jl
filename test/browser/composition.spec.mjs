import {test,expect} from '@playwright/test';
import {createRequire} from 'node:module';
import {keyboardActivate,checkCanvasKeyboardEscape} from './helpers.mjs';
const {PNG}=createRequire(import.meta.url)('playwright-core/lib/utilsBundle');
const port=process.env.TAMEROP_BROWSER_SLICE_PORT || '8849';
const state=async page=>JSON.parse(await page.getByTestId('composition-state').textContent());
async function attached(page){
  await expect.poll(async()=>{const s=await state(page);return s.actual.every((p,i)=>p.every((v,k)=>typeof v==='number'&&Math.abs(v-s.expected[i][k])<1.5));}).toBe(true);
}
async function keyboardNavigation(page){
  const before=await state(page);
  await checkCanvasKeyboardEscape(page);
  const after=await state(page);
  expect(after.limits).toEqual(before.limits);
  expect(after.actual).toEqual(before.actual);
  expect(after.expected).toEqual(before.expected);
  expect(after.meaning).toEqual(before.meaning);
}
async function visibleOverPanels(page){
  // Coordinate telemetry alone misses a curve painted behind opaque panels.
  await expect.poll(async()=>{
    const s=await state(page),png=PNG.sync.read(await page.locator('canvas').first().screenshot());
    return s.ink_probes.map(([x,y])=>{
      const px=Math.round(x/s.size[0]*png.width),py=Math.round((1-y/s.size[1])*png.height);
      for(let dy=-3;dy<=3;dy++) for(let dx=-3;dx<=3;dx++){
        const xx=px+dx,yy=py+dy;if(xx<0||xx>=png.width||yy<0||yy>=png.height)continue;
        const i=4*(yy*png.width+xx);
        if(png.data[i]<90&&png.data[i+1]<90&&png.data[i+2]<90&&png.data[i+3]>180)return true;
      }
      return false;
    });
  }).toEqual([true,true]);
}
test('A102 live connector follows actual axes and resized layout',async({page},info)=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  page.on('console',m=>{if(m.type()==='error') errors.push(m.text());});
  await page.goto(`http://127.0.0.1:${port}/composition`,{waitUntil:'domcontentloaded',timeout:300_000});
  await expect(page.getByTestId('composition-state')).toContainText('computed_correspondence',{timeout:180_000});
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].some(c=>c.wglmakie_screen?.renderer?.info.render.frame>0),null,{timeout:180_000});
  await attached(page);await visibleOverPanels(page);const initial=await state(page);
  await keyboardNavigation(page);
  await keyboardActivate(page.getByRole('button',{name:'Change barcode limits'}));
  await expect.poll(async()=>JSON.stringify((await state(page)).actual)).not.toBe(JSON.stringify(initial.actual));
  await attached(page);await page.screenshot({path:info.outputPath('changed-limits.png'),fullPage:true});
  await keyboardActivate(page.getByRole('button',{name:'Resize figure'}));
  await expect.poll(async()=>(await state(page)).size).toEqual([1450,820]);await attached(page);
  await keyboardNavigation(page);
  await keyboardActivate(page.getByRole('button',{name:'Reset limits'}));await attached(page);
  const before=await state(page);const canvas=page.locator('canvas').first();const box=await canvas.boundingBox();
  const x=box.x+before.barcode_center[0]*box.width,y=box.y+before.barcode_center[1]*box.height;
  await page.mouse.move(x,y);await page.mouse.down({button:'right'});await page.mouse.move(x+45,y+15,{steps:8});await page.mouse.up({button:'right'});
  await expect.poll(async()=>JSON.stringify((await state(page)).limits)).not.toBe(JSON.stringify(before.limits));await attached(page);
  await visibleOverPanels(page);
  await page.screenshot({path:info.outputPath('panned-connector.png'),fullPage:true});
  await page.setViewportSize({width:480,height:850});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await attached(page);await page.screenshot({path:info.outputPath('narrow.png'),fullPage:true});
  expect(errors).toEqual([]);
});
