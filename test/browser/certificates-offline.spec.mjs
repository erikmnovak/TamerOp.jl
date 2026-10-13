import {test,expect} from '@playwright/test';
import {pathToFileURL,fileURLToPath} from 'node:url';
import path from 'node:path';
import {checkOfflineChartScroll} from './helpers.mjs';
test('A101 retained certificate views reopen without Julia or network access',async({page,context},info)=>{
  const dir=process.env.TAMEROP_A101_OUTPUT||path.join(path.dirname(fileURLToPath(import.meta.url)),'artifacts','certificates');
  const requests=[],errors=[];
  page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});
  page.on('websocket',s=>requests.push(s.url()));page.on('pageerror',e=>errors.push(e.message));
  await context.setOffline(true);await page.goto(pathToFileURL(path.join(dir,'certificates.html')).href);
  await expect(page.locator('html')).toHaveAttribute('data-ready','true');
  const data=await page.locator('#portable-visual-data').evaluate(el=>JSON.parse(el.textContent));
  expect(data.states.map(s=>s.id)).toEqual(['composition','naturality','domains','pruning']);
  for(const name of ['composition','naturality','domains','pruning']){
    await page.getByLabel('Saved view',{exact:true}).selectOption(name);
    await expect(page.locator('html')).toHaveAttribute('data-selected-state',name);
    expect(data.states.find(s=>s.id===name).plotted_json).toContain('exact_interleaving');
  }
  await page.screenshot({path:info.outputPath('certificate-offline-wide.png'),fullPage:true});
  await page.setViewportSize({width:480,height:900});
  await checkOfflineChartScroll(page,page.getByRole('region',{name:'Saved figure',exact:true}),page.locator('#saved-figure'));
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await page.reload();await expect(page.locator('html')).toHaveAttribute('data-selected-state','pruning');
  await page.screenshot({path:info.outputPath('certificate-offline-narrow.png'),fullPage:true});
  expect(requests).toEqual([]);expect(errors).toEqual([]);
});
