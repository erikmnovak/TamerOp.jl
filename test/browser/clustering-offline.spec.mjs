import {test,expect} from '@playwright/test';
import {pathToFileURL} from 'node:url';
import {checkOfflineChartScroll} from './helpers.mjs';
test('A129 saved clustering labels and decisions survive without Julia',async({page,context},info)=>{
 const file=process.env.TAMEROP_CLUSTERING_HTML;expect(file,'Generate TAMEROP_CLUSTERING_HTML with the A129 native owner').toBeTruthy();
 const requests=[],errors=[];page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});page.on('websocket',s=>requests.push(s.url()));page.on('pageerror',e=>errors.push(e.message));
 await context.setOffline(true);await page.goto(pathToFileURL(file).href);await expect(page.locator('html')).toHaveAttribute('data-ready','true');
 const data=await page.locator('#portable-visual-data').evaluate(el=>JSON.parse(el.textContent));expect(data.states.map(s=>s.id)).toEqual(['original','other']);
 await page.getByLabel('Saved view',{exact:true}).selectOption('other');await expect(page.locator('html')).toHaveAttribute('data-selected-state','other');
 const selected=data.states.find(s=>s.id==='other');expect(selected.plotted_json).toContain('persistent_component_cut');expect(selected.plotted_json).toContain('labels');
 const download=page.waitForEvent('download');await page.getByRole('button',{name:'Download selected values',exact:true}).click();
 const {readFile}=await import('node:fs/promises');expect(await readFile(await (await download).path(),'utf8')).toBe(selected.plotted_json);
 await page.screenshot({path:info.outputPath('a129-offline-wide.png'),fullPage:true});await page.setViewportSize({width:480,height:900});
 await checkOfflineChartScroll(page,page.getByRole('region',{name:'Saved figure',exact:true}),page.locator('#saved-figure'));
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
 await page.reload();await expect(page.locator('html')).toHaveAttribute('data-selected-state','other');
 await page.screenshot({path:info.outputPath('a129-offline-narrow.png'),fullPage:true});expect(requests).toEqual([]);expect(errors).toEqual([]);
});
