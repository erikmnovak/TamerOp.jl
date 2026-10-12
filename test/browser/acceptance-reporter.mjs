// Acceptance requires execution and success, not merely an expected outcome.
// A runtime testInfo.skip() must fail even if no literal test.skip is present.
import path from 'node:path';

export function collectedAcceptance(report, lane) {
  if (report.errors?.length) throw new Error(`${lane}: ${JSON.stringify(report.errors)}`);
  const files = new Set();
  let count = 0;
  function visit(suite) {
    for (const spec of suite.specs || []) {
      if (!spec.tests?.length) throw new Error(`${lane}: ${spec.title} has no collected test`);
      for (const test of spec.tests) {
        if (test.expectedStatus !== 'passed')
          throw new Error(`${lane}: ${spec.title} expects ${test.expectedStatus}; acceptance must expect passed`);
        count++;
      }
      files.add(path.basename(spec.file));
    }
    for (const child of suite.suites || []) visit(child);
  }
  (report.suites || []).forEach(visit);
  if (!count) throw new Error(`${lane}: empty acceptance lane`);
  return {files, count};
}

export function acceptanceIssues(tests) {
  if (!tests.length) return ['No acceptance tests were executed.'];
  const issues = [];
  for (const test of tests) {
    const name = test.titlePath().join(' > ');
    if (test.expectedStatus !== 'passed')
      issues.push(`${name}: expected ${test.expectedStatus}; acceptance must expect passed.`);
    if (!test.results.length) issues.push(`${name}: no execution result.`);
    for (const result of test.results) {
      if (result.status !== 'passed') issues.push(`${name}: attempt ${result.retry ?? 0} was ${result.status}.`);
    }
    if (test.outcome() !== 'expected') issues.push(`${name}: outcome ${test.outcome()}.`);
  }
  return issues;
}

export default class AcceptanceReporter {
  tests = [];
  errors = [];
  onBegin(_config, suite) {
    try { this.tests = suite.allTests(); }
    catch (error) { this.errors.push(`Could not inspect acceptance plan: ${error.message}`); }
  }
  onError(error) { this.errors.push(error.message || String(error)); }
  onEnd(result) {
    // Playwright swallows thrown reporter errors. Return an explicit failed
    // status even if inspecting a future runner's result shape itself fails.
    let issues;
    try { issues = [...this.errors, ...acceptanceIssues(this.tests)]; }
    catch (error) { issues = [`Could not inspect acceptance results: ${error.message}`]; }
    if (issues.length) {
      process.stderr.write(`Visualization acceptance failed:\n${issues.map(s => `  ${s}`).join('\n')}\n`);
      return {status: 'failed'};
    }
    return {status: result.status};
  }
  printsToStdio() { return false; }
}
