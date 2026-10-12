import {test,expect} from '@playwright/test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
import {checkCanvasKeyboardEscape} from './helpers.mjs';
const {PNG}=createRequire(import.meta.url)('playwright-core/lib/utilsBundle');
const fixture=process.env.TAMEROP_STATIC_HTML;
if (!fixture) throw new Error('Generate TAMEROP_STATIC_HTML before running the offline lane.');
const batchedFixture=process.env.TAMEROP_BATCHED_HTML;
if (!batchedFixture) throw new Error('Generate TAMEROP_BATCHED_HTML with the A38 native owner before running the offline lane.');

test('A36 A37 exported WGL scene actually renders after Julia exits',async({page,context},info)=>{
  const errors=[], requests=[];
  page.on('pageerror', e=>errors.push(e.message));
  page.on('console',m=>{if(m.type()==='error') errors.push(m.text());});
  page.on('request',r=>{if(/^https?:/.test(r.url())) requests.push(r.url());});
  await context.setOffline(true);
  await page.goto(pathToFileURL(path.resolve(fixture)).href);
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].some(c=>
    c.wglmakie_screen?.renderer?.info.render.frame>0));
  const canvas=page.locator('canvas').first();
  await expect(canvas).toBeVisible();
  const png=PNG.sync.read(await canvas.screenshot());
  let ink=0;
  for(let i=0;i<png.data.length;i+=4)
    if(png.data[i+3]>100&&Math.min(png.data[i],png.data[i+1],png.data[i+2])<180) ink++;
  expect(ink).toBeGreaterThan(100);
  expect(await canvas.evaluate(c=>c.width>0&&c.height>0)).toBe(true);
  await checkCanvasKeyboardEscape(page);
  await page.screenshot({path:info.outputPath('offline-wgl.png'),fullPage:true});
  await page.reload();
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].some(c=>
    c.wglmakie_screen?.renderer?.info.render.frame>0));
  await checkCanvasKeyboardEscape(page);
  expect(requests).toEqual([]);
  expect(errors).toEqual([]);
});

test('A38 batched geometry, halo labels and exact tables render offline after reload and resize',async({page,context},info)=>{
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('console',m=>{if(m.type()==='error') errors.push(m.text());});
  await context.setOffline(true);
  await page.goto(pathToFileURL(path.resolve(batchedFixture)).href);
  for(const width of [1440,720]) {
    await page.setViewportSize({width,height:1000});
    await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].some(c=>
      c.wglmakie_screen?.renderer?.info.render.frame>0));
    const canvas=page.locator('canvas').first();
    const png=PNG.sync.read(await canvas.screenshot());
    const ink=[0,0,0,0];
    for(let y=0;y<png.height;y++) for(let x=0;x<png.width;x++) {
      const p=4*(y*png.width+x);
      if(png.data[p+3]>100&&Math.min(...png.data.subarray(p,p+3))<180)
        ink[(y>=png.height/2?2:0)+(x>=png.width/2?1:0)]++;
    }
    for(const count of ink) expect(count).toBeGreaterThan(100);
    await checkCanvasKeyboardEscape(page);
    await page.screenshot({path:info.outputPath(`batched-${width}.png`),fullPage:true});
    await page.reload();
    await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].some(c=>
      c.wglmakie_screen?.renderer?.info.render.frame>0));
  }
  expect(errors).toEqual([]);
});
