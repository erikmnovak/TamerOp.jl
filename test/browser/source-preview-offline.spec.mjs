import {test,expect} from '@playwright/test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {checkCanvasKeyboardEscape} from './helpers.mjs';
const fixture=process.env.TAMEROP_SOURCE_PREVIEW_HTML;
if (!fixture) throw new Error('Generate TAMEROP_SOURCE_PREVIEW_HTML with test_visualization_a39.jl first.');
test('A39 source preview remains readable offline after reload and resize',async({page,context},info)=>{
  const errors=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});
  await context.setOffline(true);
  await page.goto(pathToFileURL(path.resolve(fixture)).href);
  for (const width of [1440,720]) {
    await page.setViewportSize({width,height:900});
    await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].some(c=>c.wglmakie_screen?.renderer?.info.render.frame>0));
    await expect(page.locator('canvas').first()).toBeVisible();
    await checkCanvasKeyboardEscape(page);
    await page.screenshot({path:info.outputPath(`source-preview-${width}.png`),fullPage:true});
    await page.reload();
  }
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].some(c=>c.wglmakie_screen?.renderer?.info.render.frame>0));
  expect(errors).toEqual([]);
  expect(requests).toEqual([]);
});
