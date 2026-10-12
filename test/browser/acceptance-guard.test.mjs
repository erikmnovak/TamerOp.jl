// Node-only guard tests. The integration fixture never requests page/browser
// fixtures and has no Julia webServer, so it cannot launch either application.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, writeFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';
import AcceptanceReporter, {acceptanceIssues, collectedAcceptance} from './acceptance-reporter.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const completed = (expectedStatus='passed', statuses=['passed'], outcome='expected') => ({
  titlePath: () => ['fixture', 'operation'], expectedStatus,
  results: statuses.map((status, retry) => ({status, retry})), outcome: () => outcome,
});
const plan = expectedStatus => ({errors: [], suites: [{suites: [{specs: [{
  file: '/fixtures/operation.spec.mjs', title: 'operation', tests: [{expectedStatus}],
}]}]}]});

test('collection requires nonempty tests that all expect success', () => {
  const selected = collectedAcceptance(plan('passed'), 'lane');
  assert.equal(selected.count, 1);
  assert.deepEqual([...selected.files], ['operation.spec.mjs']);
  for (const status of ['skipped', 'failed', undefined])
    assert.throws(() => collectedAcceptance(plan(status), 'lane'), /acceptance must expect passed/);
  assert.throws(() => collectedAcceptance({suites: []}, 'lane'), /empty acceptance lane/);
  assert.throws(() => collectedAcceptance({suites: [{specs: [{title: 'empty', tests: []}]}]}, 'lane'), /no collected test/);
  assert.throws(() => collectedAcceptance({errors: [{message: 'bad import'}]}, 'lane'), /bad import/);
});

test('runtime results reject skips, missing attempts and expected failures', () => {
  assert.deepEqual(acceptanceIssues([completed()]), []);
  for (const row of [completed('skipped', ['skipped'], 'skipped'),
    completed('failed', ['failed']), completed('passed', []),
    completed('passed', ['interrupted']), completed('passed', ['timedOut']),
    completed('passed', [undefined]), completed('passed', ['passed'], 'unexpected'),
    completed('passed', ['failed', 'passed'], 'flaky')]) {
    assert.ok(acceptanceIssues([completed(), row]).length > 0);
  }
  assert.ok(acceptanceIssues([]).length > 0);
});

test('reporter preserves existing failures and never certifies malformed results', () => {
  const reporter = new AcceptanceReporter();
  reporter.onBegin({}, {allTests: () => [completed()]});
  assert.deepEqual(reporter.onEnd({status: 'passed'}), {status: 'passed'});
  assert.deepEqual(reporter.onEnd({status: 'interrupted'}), {status: 'interrupted'});
  reporter.tests = [{titlePath: () => ['malformed']}];
  assert.deepEqual(reporter.onEnd({status: 'passed'}), {status: 'failed'});
});

test('actual Playwright exits unsuccessfully on runtime testInfo.skip and expected failure', () => {
  const directory = mkdtempSync(path.join(tmpdir(), 'tamerop-acceptance-guard-'));
  try {
    const config = path.join(directory, 'fixture.config.mjs');
    const fixture = path.join(directory, 'fixture.spec.mjs');
    const moduleURL = pathToFileURL(path.join(here, 'node_modules/@playwright/test/index.mjs')).href;
    writeFileSync(config, `export default ${JSON.stringify({
      testDir: directory, testMatch: 'fixture.spec.mjs', workers: 1, timeout: 10000,
      outputDir: path.join(directory, 'results'), reporter: [['list'], [path.join(here, 'acceptance-reporter.mjs')]],
    })};\n`);
    for (const [body, expectedExit] of [
      [`test('runs', () => {});`, 0],
      [`test('runs', () => {}); test('runtime omission', ({}, info) => { info.skip(); });`, 1],
      [`test('expected failure', () => { test.fail(); throw new Error('not accepted'); });`, 1],
    ]) {
      writeFileSync(fixture, `import {test} from ${JSON.stringify(moduleURL)};\n${body}\n`);
      const result = spawnSync(process.execPath, [path.join(here, 'node_modules/@playwright/test/cli.js'),
        'test', `--config=${config}`], {cwd: directory, encoding: 'utf8', timeout: 30000});
      assert.equal(result.error, undefined);
      assert.equal(result.status, expectedExit, result.stdout + result.stderr);
      if (expectedExit) assert.match(result.stderr, /Visualization acceptance failed/);
    }
  } finally {
    rmSync(directory, {recursive: true, force: true});
  }
});
