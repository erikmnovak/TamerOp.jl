import { test, expect } from '@playwright/test';
import { state, keyboardActivate, control, openInspector, capture, fullyClosed, scrollChart } from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT || 8849}`;
async function settled(page) {
  await expect.poll(async()=>(await state(page)).summary.updating,{timeout:120_000}).toBe(false);
  await page.waitForFunction(()=>[...document.querySelectorAll('canvas')].every(c=>c.wglmakie_screen?.renderer?.info.render.frame>0));
}
async function edit(page,suffix,value) {
  const input=page.locator(`input[id$="-${suffix}"]`);await input.fill(value);await input.press('Tab');
}
async function change(page,suffix) {
  await settled(page);const before=(await state(page)).selection.revision;
  await keyboardActivate(page.locator(`button[id$="-${suffix}"]`));
  await expect.poll(async()=>(await state(page)).selection.revision,{timeout:120_000}).toBeGreaterThan(before);
  await settled(page);await expect(control(page,'error')).toBeEmpty();
  await expect.poll(async()=>(await state(page)).summary.listener_errors).toEqual([]);
}
async function hoverTerm(page,id) {
  await settled(page);const canvas=page.locator('canvas').first();await canvas.scrollIntoViewIfNeeded();
  const snapshot=await state(page),target=snapshot.ui.pick_targets.find(t=>t.panel==='signed_arithmetic' && t.id===id);
  expect(target).toBeTruthy();let box=await canvas.boundingBox();
  await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+target.y*box.height);box=await canvas.boundingBox();
  await page.mouse.move(box.x+target.x*box.width,box.y+target.y*box.height);
  await expect.poll(async()=>{
    const p=(await state(page)).ui.mouseposition;
    return Math.hypot(p[0]-target.x*snapshot.ui.figure_size[0],p[1]-(1-target.y)*snapshot.ui.figure_size[1]);
  }).toBeLessThan(3);
  await expect(control(page,'hover')).toContainText(`Term #${id}:`);
}
async function pickTerm(page,id) {
  await hoverTerm(page,id);
  await page.mouse.down();await page.mouse.up();
  await expect.poll(async()=>(await state(page)).selection.term).toBe(id);
}
test('A86 signed sums, real pointer selection, smoothing and lifecycle',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/signed`,errors);
  let s=await state(page);expect(s.reconstructed).toBe(1);expect(s.summary.agreement).toBe('equal');
  const negative=s.records.find(r=>r.weight<0).id;
  await pickTerm(page,negative);await settled(page);
  await hoverTerm(page,negative);
  expect((await state(page)).summary.direct_queries).toBe(1);
  await capture(page,info,'selected-negative');
  await edit(page,'query','0,0,5,5');await change(page,'apply-query');
  expect((await state(page)).reconstructed).toBe(0);
  await expect(control(page,'hover')).toBeEmpty();
  await edit(page,'query','5,5,0,0');await control(page,'apply-query').click();
  await expect(control(page,'error')).toContainText('p <= q');
  expect((await state(page)).reconstructed).toBe(0);
  await change(page,'reset');
  await edit(page,'smoothing','3,3');await change(page,'apply-smoothing');
  expect((await state(page)).reconstructed).toBe(0);
  expect((await state(page)).summary.agreement).toBe('equal');
  await capture(page,info,'smoothed-rank');
  await change(page,'diagonals');
  const queries=(await state(page)).summary.direct_queries;
  await change(page,'next');expect((await state(page)).summary.direct_queries).toBe(queries);
  await page.setViewportSize({width:480,height:900});await capture(page,info,'signed-narrow');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  const scroll=page.locator('[aria-label="Linked signed contribution charts"]');
  await scrollChart(page,scroll,'right');await capture(page,info,'signed-narrow-right');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
test('A86 grayscale signs and linked viewers',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/signed-grayscale`,errors);
  const linked=await page.context().newPage();await openInspector(linked,`${base}/signed-grayscale`,errors);
  await edit(linked,'term','1');await change(linked,'apply-term');
  await expect.poll(async()=>(await state(page)).selection.term).toBe(1);
  await linked.close();await expect.poll(async()=>(await state(page)).viewer_count,{timeout:45_000}).toBe(1);
  await capture(page,info,'signed-grayscale');
  expect((await state(page)).summary.closed).toBe(false);
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
