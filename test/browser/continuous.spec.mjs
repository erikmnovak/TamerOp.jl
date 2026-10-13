import {test,expect} from '@playwright/test';
import {performance} from 'node:perf_hooks';
import {state,control,openInspector,capture,fullyClosed} from './helpers.mjs';
const base=`http://127.0.0.1:${process.env.TAMEROP_BROWSER_SLICE_PORT||8849}`;
async function idle(page){
  await expect.poll(async()=>{
    const s=await state(page);
    return [s.summary.requests.pending,s.summary.requests.running,s.summary.updating];
  },{timeout:120_000}).toEqual([false,false,false]);
  expect((await state(page)).summary.listener_errors).toEqual([]);
}
async function displayEnvironment(page){
  return {browser:page.context().browser().version(),viewport:page.viewportSize(),
    renderer:await page.evaluate(()=>{
      const gl=document.querySelector('canvas').wglmakie_screen.renderer.getContext();
      const extension=gl.getExtension('WEBGL_debug_renderer_info');
      return extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    })};
}
async function exact(page,value){
  await control(page,'parameter').fill(value);await control(page,'parameter').press('Tab');
  const revision=(await state(page)).selection.revision;
  await control(page,'apply').click();
  await expect.poll(async()=>(await state(page)).selection.revision).toBeGreaterThan(revision);
  await idle(page);
}
async function paintObserver(page,selector,expected,statusSelector='[id$="-status"]'){
  // Start at the real input event, and stop after the matching committed
  // readout and two animation frames. No Julia callback is invoked here.
  await page.evaluate(({selector,expected,statusSelector})=>{
    window.a123Paint=null;
    const element=document.querySelector(selector);
    const eventName=element.tagName==='INPUT' ? 'input' : 'pointerdown';
    const previous=document.querySelector(statusSelector)?.textContent||'';
    element.addEventListener(eventName,()=>{
      const start=performance.now();
      const frames=new Map([...document.querySelectorAll('canvas')].map(c=>[c,c.wglmakie_screen?.renderer?.info.render.frame||0]));
      const poll=()=>{
        const status=document.querySelector(statusSelector)?.textContent||'';
        const canvases=[...document.querySelectorAll('canvas')];
        const painted=canvases.length>=frames.size && canvases.every(c=>c.wglmakie_screen?.renderer?.info.render.frame>0)
          && canvases.some(c=>!frames.has(c)||c.wglmakie_screen?.renderer?.info.render.frame>frames.get(c));
        if((expected===null ? status!==previous : status.includes(expected)) && painted){
          requestAnimationFrame(()=>requestAnimationFrame(()=>window.a123Paint=performance.now()-start));
        }else requestAnimationFrame(poll);
      };
      requestAnimationFrame(poll);
    },{once:true});
  },{selector,expected,statusSelector});
}
async function latency(page,slider,key,expected){
  await paintObserver(page,`#${await slider.getAttribute('id')}`,expected);
  await slider.scrollIntoViewIfNeeded();
  const box=await slider.boundingBox();
  await page.mouse.click(box.x+(key==='End'?box.width-2:2),box.y+box.height/2);
  await page.waitForFunction(()=>window.a123Paint!==null,null,{timeout:120_000});
  const elapsed=await page.evaluate(()=>window.a123Paint);
  await idle(page);return elapsed;
}
async function drag(page,canvas,from,to,steps=18){
  await canvas.scrollIntoViewIfNeeded();let box=await canvas.boundingBox();
  await page.evaluate(y=>window.scrollBy(0,y-innerHeight/2),box.y+from.y*box.height);
  box=await canvas.boundingBox();
  await page.mouse.move(box.x+from.x*box.width,box.y+from.y*box.height);
  await page.waitForTimeout(180); // delivery through WGL's pointer throttle
  await page.mouse.down();
  await page.mouse.move(box.x+to.x*box.width,box.y+to.y*box.height,{steps});
  await page.waitForTimeout(180);await page.mouse.up();
}
for(const [route,max,counts] of [['continuous',5,[16,24,9]],['continuous-large',4,[676,1300,625]]]){
  test(`A123 ${route} scrubbing, exact events, bounded work and painted latency`,async({page},info)=>{
    const errors=[];const opened=performance.now();await openInspector(page,`${base}/${route}`,errors);
    const openMs=performance.now()-opened;
    const environment=await displayEnvironment(page);
    const slider=control(page,'grade-1');
    const timings=[];
    for(let i=0;i<21;i++){
      const atMaximum=i%2===0;
      timings.push(await latency(page,slider,atMaximum?'End':'Home',`Committed parameter (${atMaximum?max:0}//1,)`));
    }
    expect((await state(page)).summary.active_counts).toEqual(counts);
    expect((await state(page)).ui.rebuild_count).toBe(1);
    expect((await state(page)).ui.update_count).toBeGreaterThan(0);
    const warm=timings.slice(1).sort((a,b)=>a-b);
    await info.attach('interaction-latency.json',{body:JSON.stringify({route,environment,openMs,firstQueryMs:timings[0],warmMs:timings.slice(1),medianMs:(warm[9]+warm[10])/2,p95Ms:warm[18],method:'actual pointer range input to committed DOM, an advanced renderer frame and two animation frames; first browser query includes remaining compilation; open includes fixture/scene preparation and accessibility checks'},null,2),contentType:'application/json'});
    if(route==='continuous'){
      await exact(page,'5764607523034234879/1152921504606846976');
      expect((await state(page)).summary.active_counts).toEqual([16,24,8]);
      await control(page,'grade-1-next').click();await idle(page);
      await expect.poll(async()=>(await state(page)).summary.active_counts).toEqual([16,24,9]);
      const linked=await page.context().newPage();await openInspector(linked,`${base}/${route}`,errors);
      await exact(linked,'0');await expect.poll(async()=>(await state(page)).summary.active_counts).toEqual([16,24,8]);
      await expect(slider).toHaveValue('0');await linked.close();
      await slider.focus();
      for(let i=0;i<20;i++)await slider.press(i%2?'Home':'End');
      await slider.press('End');await idle(page);
      await expect.poll(async()=>(await state(page)).summary.active_counts).toEqual([16,24,9]);
      await expect.poll(async()=>(await state(page)).viewer_count).toBe(1);
      const revision=(await state(page)).selection.revision;
      await control(page,'parameter').fill('NaN');await control(page,'parameter').press('Tab');await control(page,'apply').click();
      await expect(control(page,'error')).not.toBeEmpty();expect((await state(page)).selection.revision).toBe(revision);
      await page.setViewportSize({width:480,height:900});
      expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
      await capture(page,info,'continuous-narrow');
    }
    await capture(page,info,route);
    await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
  });
}

test('A123 parameter-plane drag and discrete stored degree threshold',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/continuous-degree`,errors);
  expect((await state(page)).summary.active_counts).toEqual([3,0]);
  await control(page,'grade-2').focus();await control(page,'grade-2').press('End');await idle(page);
  await expect.poll(async()=>(await state(page)).summary.active_counts).toEqual([4,0]);
  await exact(page,'0,1');
  const plane=(await state(page)).ui.parameter_plane;
  // Interior target avoids the frame; k snaps to 3 and diameter exceeds 1.
  const target={x:plane.lower.x+0.96*(plane.upper.x-plane.lower.x),y:plane.lower.y+0.95*(plane.upper.y-plane.lower.y)};
  await drag(page,page.locator('canvas').first(),plane.cursor,target);await idle(page);
  await expect.poll(async()=>(await state(page)).summary.active_counts).toEqual([4,3]);
  expect((await state(page)).selection.parameter[1]).toBe(3);
  const committed=(await state(page)).selection.parameter;
  const invalidPlane=(await state(page)).ui.parameter_plane;
  await drag(page,page.locator('canvas').first(),invalidPlane.cursor,
    {x:invalidPlane.upper.x+0.025,y:invalidPlane.cursor.y},1);
  await expect(control(page,'continuous-status')).toContainText('Parameter unchanged:');
  await idle(page);expect((await state(page)).selection.parameter).toEqual(committed);
  await capture(page,info,'degree-parameter-plane');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});

test('A123 physical radius and integer depth update geometry and homology',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/continuous-geometry`,errors);
  expect((await state(page)).geometry.betti).toEqual([2,0,0]);
  await control(page,'grade-2').focus();await control(page,'grade-2').press('End');await idle(page);
  await expect.poll(async()=>(await state(page)).geometry.betti).toEqual([0,0,0]);
  await control(page,'grade-1-next').click();await idle(page);
  await expect.poll(async()=>(await state(page)).geometry.betti).toEqual([1,0,0]);
  // The event retains its rational radius, rather than the integer text-entry type.
  expect((await state(page)).geometry.panel_parameter).toEqual(['1//1',2]);
  const plane=(await state(page)).ui.parameter_plane;
  await drag(page,page.locator('canvas').first(),plane.cursor,
    {x:plane.lower.x+0.75*(plane.upper.x-plane.lower.x),y:plane.lower.y+0.1*(plane.upper.y-plane.lower.y)});
  await idle(page);await expect.poll(async()=>(await state(page)).geometry.betti).toEqual([2,0,0]);
  await capture(page,info,'continuous-geometry');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});

test('A123 slice handles, continuous keyboard controls and deferred exact Apply',async({page},info)=>{
  const errors=[];await openInspector(page,`${base}/continuous-slices`,errors);
  const environment=await displayEnvironment(page);
  const angle=control(page,'slice-angle');
  const before=(await state(page)).selection.revision;
  await paintObserver(page,`#${await angle.getAttribute('id')}`,null,'[id$="-slice-status"]');
  await angle.focus();await angle.press('ArrowRight');
  await page.waitForFunction(()=>window.a123Paint!==null,null,{timeout:120_000});
  const firstSliceQueryMs=await page.evaluate(()=>window.a123Paint);await idle(page);
  await expect.poll(async()=>(await state(page)).selection.revision).toBeGreaterThan(before);
  expect((await state(page)).slice.line.direction[0]).not.toBe((await state(page)).slice.line.direction[1]);
  const sliceTimings=[];
  for(let i=0;i<12;i++){
    await paintObserver(page,`#${await angle.getAttribute('id')}`,null,'[id$="-slice-status"]');
    await angle.focus();await angle.press('ArrowRight');
    await page.waitForFunction(()=>window.a123Paint!==null,null,{timeout:120_000});
    sliceTimings.push(await page.evaluate(()=>window.a123Paint));await idle(page);
  }
  const ordered=[...sliceTimings].sort((a,b)=>a-b);
  await info.attach('slice-latency.json',{body:JSON.stringify({route:'continuous-slices',environment,firstSliceQueryMs,warmDistinctLinesMs:sliceTimings,
    medianMs:(ordered[5]+ordered[6])/2,p95Ms:ordered.at(-1),cache:(await state(page)).summary,
    method:'actual angle-key input to committed slice readout and two animation frames; distinct new lines after first-use compilation'},null,2),contentType:'application/json'});
  const current=await state(page),handles=current.main_ui.handles;
  const revision=current.selection.revision;
  await drag(page,page.locator('canvas').first(),handles[0],{x:handles[0].x+0.015,y:handles[0].y});
  await idle(page);await expect.poll(async()=>(await state(page)).selection.revision).toBeGreaterThan(revision);
  await control(page,'slice-live').uncheck();await idle(page);
  const committed=(await state(page)).selection.revision;
  await angle.focus();await angle.press('ArrowRight');
  await expect(control(page,'slice-draft')).toContainText('Draft:');
  expect((await state(page)).selection.revision).toBe(committed);
  for(const [suffix,value] of [['base-x','0'],['base-y','0'],['direction-x','1'],['direction-y','1']]){
    const field=control(page,`slice-${suffix}`);await field.fill(value);await field.press('Tab');
  }
  await control(page,'slice-apply').click();await idle(page);
  await expect.poll(async()=>(await state(page)).slice.line.direction).toEqual(['1','1']);
  const records=(await state(page)).records;
  expect(records.map(r=>[r.birth,r.death,r.multiplicity])).toEqual([['0','2',1],['1','3',1]]);
  await capture(page,info,'continuous-slice');
  await control(page,'slice-live').check();await angle.focus();await angle.press('ArrowRight');
  await control(page,'close').click();await fullyClosed(page);expect(errors).toEqual([]);
});
