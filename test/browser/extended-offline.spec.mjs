import {test,expect} from '@playwright/test';
import {pathToFileURL,fileURLToPath} from 'node:url';
import path from 'node:path';
import {checkOfflineChartScroll} from './helpers.mjs';

test('A119 typed extended diagrams and barcodes reopen without Julia',async({page,context},info)=>{
  const dir=process.env.TAMEROP_A119_OUTPUT||path.join(path.dirname(fileURLToPath(import.meta.url)),'artifacts','extended');
  const requests=[],errors=[];
  page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});
  page.on('websocket',s=>requests.push(s.url()));page.on('pageerror',e=>errors.push(e.message));
  await context.setOffline(true);await page.goto(pathToFileURL(path.join(dir,'extended.html')).href);
  await expect(page.locator('html')).toHaveAttribute('data-ready','true');
  const data=await page.locator('#portable-visual-data').evaluate(el=>JSON.parse(el.textContent));
  const names=['extended_persistence_0','extended_barcode_0','extended_persistence_1','extended_barcode_1'];
  expect(data.states.map(s=>s.id)).toEqual(names);
  for(const name of names){
    await page.getByLabel('Saved view',{exact:true}).selectOption(name);
    await expect(page.locator('html')).toHaveAttribute('data-selected-state',name);
    const plotted=data.states.find(s=>s.id===name).plotted_json;
    expect(plotted).toContain('birth_phase');expect(plotted).toContain('death_included');
    expect(plotted).toContain(name.endsWith('_1')?'extended_negative':'extended_positive');
    await expect(page.locator('#saved-figure')).toBeVisible();
  }
  await page.getByLabel('Saved view',{exact:true}).selectOption('extended_persistence_1');
  await page.screenshot({path:info.outputPath('extended-wide.png'),fullPage:true});
  await page.setViewportSize({width:480,height:900});
  await checkOfflineChartScroll(page,page.getByRole('region',{name:'Saved figure',exact:true}),page.locator('#saved-figure'));
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await page.reload();await expect(page.locator('html')).toHaveAttribute('data-selected-state','extended_persistence_1');
  await page.screenshot({path:info.outputPath('extended-narrow.png'),fullPage:true});
  expect(requests).toEqual([]);expect(errors).toEqual([]);
});
