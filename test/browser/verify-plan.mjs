// Validate actual Playwright collection, including ignored files and separate lanes.
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {collectedAcceptance} from './acceptance-reporter.mjs';
const here=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(here,'../..');
const expected=new Set(execFileSync(process.env.PYTHON || 'python3',
  [path.join(root,'test/visualization_contracts.py'),'--list-browser'],{cwd:root,encoding:'utf8'}).trim().split('\n'));
const selected=new Set();
for(const config of ['playwright.config.mjs','portable-session.config.mjs']) {
  // Collection only: nonempty placeholder paths do not satisfy execution.
  const env={...process.env,TAMEROP_PORTABLE_SESSION:process.env.TAMEROP_PORTABLE_SESSION || '/not-executed/session.html',
    TAMEROP_STATIC_HTML:process.env.TAMEROP_STATIC_HTML || '/not-executed/static.html',
    TAMEROP_SOURCE_PREVIEW_HTML:process.env.TAMEROP_SOURCE_PREVIEW_HTML || '/not-executed/source-preview.html',
    TAMEROP_BATCHED_HTML:process.env.TAMEROP_BATCHED_HTML || '/not-executed/batched.html'};
  const report=JSON.parse(execFileSync(process.execPath,
    [path.join(here,'node_modules/@playwright/test/cli.js'),'test',`--config=${path.join(here,config)}`,'--list','--reporter=json'],
    {cwd:root,env,encoding:'utf8'}));
  const {files,count}=collectedAcceptance(report,config);
  for(const file of files) {
    if(selected.has(file)) throw new Error(`${file}: selected in both lanes`);
    selected.add(file);
  }
  console.log(`${config}: ${count} scenarios in ${files.size} files`);
}
const missing=[...expected].filter(f=>!selected.has(f)),extra=[...selected].filter(f=>!expected.has(f));
if(missing.length || extra.length) throw new Error(`Playwright plan mismatch: missing=${missing}, extra=${extra}`);
console.log('Every declared browser file is selected exactly once.');
