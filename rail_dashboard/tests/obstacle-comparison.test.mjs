import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

const source = await readFile(new URL('../app/data-contract.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } });
const { buildRunViews } = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputText).toString('base64')}`);
function fixture() {
  const captures = ['initial', 'rescan'].flatMap((phase) => [1, 2].map((zone) => ({
    capture_id: `${phase}-${zone}`, run_id: 'run', camera_role: 'top', phase,
    logical_zone_number: zone, captured_at_utc: '2026-09-05T00:00:00Z',
    display_timezone: 'Asia/Seoul', processing_status: 'ready',
  })));
  return { runs: [{ run_id: 'run', capture_target: 2, display_timezone: 'Asia/Seoul' }], captures,
    analyses: captures.map((c) => ({ capture_id: c.capture_id, defect_type: 'obstacle', status: 'ready',
      detected: c.phase === 'initial', positive_pixels: c.phase === 'initial' ? 20 : 0,
      inspected_pixels: c.logical_zone_number === 1 ? 100 : 300,
      ratio_fraction: c.phase === 'initial' ? 20 / (c.logical_zone_number === 1 ? 100 : 300) : 0,
    })), artifacts: captures.map((c) => ({ capture_id: c.capture_id, artifact_type: 'obstacle_preview', public_url: `/${c.capture_id}.png` })), run_summaries: [],
  };
}
test('weights by image pixels, keeps clean after images and pairs identical zones', () => {
  const view = buildRunViews(fixture())[0];
  assert.equal(view.beforeRatio, null);
  assert.equal(view.obstacleComparison.beforeRatio, 0.1);
  assert.equal(view.obstacleComparison.afterRatio, 0);
  assert.equal(view.obstacleComparison.reduction, 0.1);
  assert.equal(view.obstacleComparison.improvement, 1);
  assert.equal(view.obstacleComparison.pairs[1].after.maskUrl, '/rescan-2.png');
});
test('missing and failed measurements do not become zero or a complete comparison', () => {
  for (const status of ['error', 'disabled']) {
    const document = fixture(); document.analyses[3].status = status;
    const comparison = buildRunViews(document)[0].obstacleComparison;
    assert.equal(comparison.afterRatio, null); assert.equal(comparison.reduction, null);
    assert.equal(comparison.pairs[1].after, null);
  }
  const document = fixture(); document.analyses = [];
  assert.equal(buildRunViews(document)[0].obstacleComparison.beforeRatio, null);
});
test('excludes side, manual, and ambiguous duplicate captures', () => {
  const document = fixture();
  document.captures[0].camera_role = 'side';
  assert.equal(buildRunViews(document)[0].obstacleComparison.beforeRatio, null);
  document.captures[0].camera_role = 'top'; document.captures.push({ ...document.captures[0], capture_id: 'duplicate' });
  assert.equal(buildRunViews(document)[0].obstacleComparison.beforeRatio, null);
});
test('zero baseline has no relative improvement; increase remains negative', () => {
  const document = fixture();
  document.analyses.forEach((a) => { a.positive_pixels = 0; a.ratio_fraction = 0; });
  assert.equal(buildRunViews(document)[0].obstacleComparison.improvement, null);
  document.analyses[2].positive_pixels = 10; document.analyses[2].ratio_fraction = 0.1;
  assert.equal(buildRunViews(document)[0].obstacleComparison.reduction, -0.025);
});
