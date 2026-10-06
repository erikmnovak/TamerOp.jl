import { test, expect } from '@playwright/test';
import { state, control, openInspector, capture, fullyClosed, scrollChart } from './helpers.mjs';
const base = `http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT || 8849}`;
async function settled(page,timeout=120_000) {
  await expect.poll(async()=>(await state(page)).summary.updating,{timeout}).toBe(false);
  await page.waitForFunction(()=>{
    const canvases=[...document.querySelectorAll('canvas')];
    return canvases.length>0 && canvases.every(c=>c.wglmakie_screen?.renderer?.info.render.frame>0);
  },null,{timeout});
}
async function edit(page, suffix, value) {
  const input = page.locator(`input[id$="-${suffix}"]`);
  await input.fill(value); await input.press('Tab');
}
async function changed(page, suffix, timeout=120_000) {
  await settled(page,timeout);
  const before=(await state(page)).selection.revision;
  const error=await control(page,'error').textContent();
  await control(page,suffix).click();
  await expect.poll(async()=>[(await state(page)).selection.revision,await control(page,'error').textContent()],{timeout}).not.toEqual([before,error]);
  await expect(control(page,'error')).toBeEmpty();
  await settled(page,timeout);
  await expect.poll(async()=>(await state(page)).summary.listener_errors).toEqual([]);
}
async function clickTarget(page, panel, id) {
  await settled(page);
  const canvas=page.locator('canvas').first(); await canvas.scrollIntoViewIfNeeded();
  const current=await state(page);
  const t=current.ui.pick_targets.find(p=>p.panel===panel && p.id===id);
  expect(t).toBeTruthy();
  let box=await canvas.boundingBox();
  await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+t.y*box.height);
  box=await canvas.boundingBox();
  const point={x:box.x+t.x*box.width,y:box.y+t.y*box.height};
  expect(await canvas.evaluate((element,p)=>document.elementFromPoint(p.x,p.y)===element,point)).toBe(true);
  await page.mouse.move(box.x+t.x*box.width,box.y+t.y*box.height);
  await expect.poll(async()=>{
    const p=(await state(page)).ui.mouseposition;
    return Math.hypot(p[0]-t.x*current.ui.figure_size[0],p[1]-(1-t.y)*current.ui.figure_size[1]);
  }).toBeLessThan(3);
  await page.mouse.down();await page.mouse.up();
}

test('A85 matching members, diagonal costs, duplicate picks and lifecycle',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/matching`,errors);
  expect((await state(page)).summary.distance).toBe(1);
  await edit(page,'pair','3');await changed(page,'apply-pair');
  expect((await state(page)).selection.pair).toBe(3);
  expect((await state(page)).records[2].diagonal).toBe(true);
  await capture(page,info,'diagonal-pair');
  const linked=await page.context().newPage();
  await openInspector(linked,`${base}/matching`,errors);
  expect((await state(linked)).selection.pair).toBe(3);
  await changed(linked,'next-pair');
  await expect.poll(async()=>(await state(page)).selection.pair).toBe(4);
  await linked.close();
  // Bonito retains disconnected clients for its 30-second reconnect grace.
  await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  expect((await state(page)).summary.closed).toBe(false);
  const count=(await state(page)).summary.matching_queries;
  await clickTarget(page,'matching_diagram',1);
  await expect.poll(async()=>(await state(page)).selection.pair).toBe(1);
  await clickTarget(page,'matching_diagram',1);
  await expect.poll(async()=>(await state(page)).selection.pair).toBe(2);
  await clickTarget(page,'matching_barcodes',4);
  await expect.poll(async()=>(await state(page)).selection.pair).toBe(4);
  expect((await state(page)).summary.matching_queries).toBe(count);
  await edit(page,'pair','999');await control(page,'apply-pair').click();
  await expect(control(page,'error')).toContainText('pair');
  expect((await state(page)).selection.pair).toBe(4);
  await changed(page,'reset');
  await page.setViewportSize({width:480,height:900});
  await capture(page,info,'matching-narrow');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  const scroll=page.locator('[aria-label="Linked matching charts"]');
  await scrollChart(page,scroll,'right');
  await capture(page,info,'matching-narrow-right');
  await scrollChart(page,scroll,'left');
  await control(page,'close').click();await fullyClosed(page);
  expect(errors).toEqual([]);
});

test('A85 distinguishes chosen samples from the exact interior optimizer',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/matching-slices`,errors);
  expect((await state(page)).summary.weighted_distance).toBe(1);
  await page.getByText('Compare slices and search the window',{exact:true}).click();
  await clickTarget(page,'matching_sample_map',2);
  await expect.poll(async()=>(await state(page)).selection.sample).toBe(2);
  await changed(page,'best-sample');
  expect((await state(page)).summary.scope).toBe('sampled_slice');
  await changed(page,'optimum',240_000);
  let s=await state(page);
  expect(s.summary.scope).toBe('certified_finite_window');
  expect(s.summary.weighted_distance).toBe('5//4');
  expect(s.context.status).toBe('attained');
  await capture(page,info,'exact-switch-witness');
  const count=s.summary.matching_queries;
  await changed(page,'next-pair');
  expect((await state(page)).summary.matching_queries).toBe(count);
  for(const [suffix,value] of [['line-1','0'],['line-2','1/2'],['line-3','1'],['line-4','1']])await edit(page,suffix,value);
  await changed(page,'apply-slice');
  s=await state(page);expect(s.summary.scope).toBe('selected_slice');
  expect(s.summary.weighted_distance).toBe(1.25);
  await edit(page,'line-3','0');await control(page,'apply-slice').click();
  await expect(control(page,'error')).toContainText('positive');
  await changed(page,'reset');
  expect((await state(page)).summary.scope).toBe('selected_slice');
  await control(page,'close').click();await fullyClosed(page);
  expect(errors).toEqual([]);
});
