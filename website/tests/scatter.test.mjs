import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { logPosition, isoLabel } from '../public/scatter-math.mjs';

const figure = JSON.parse(readFileSync(new URL('../public/data/experiments.json', import.meta.url))).figure1;

test('log axes preserve multiplicative spacing and reversed vertical direction', () => {
  assert.equal(logPosition(1, [1, 100], [0, 300]), 0);
  assert.equal(logPosition(10, [1, 100], [0, 300]), 150);
  assert.equal(logPosition(100, [1, 100], [0, 300]), 300);
  assert.equal(logPosition(10, [1, 100], [300, 0]), 150);
  assert.ok(logPosition(4, [1, 100], [300, 0]) < logPosition(2, [1, 100], [300, 0]));
});

test('every source point obeys the speedup decomposition and fits its panel', () => {
  assert.equal(new Set(figure.rows.map(row => row.id)).size, figure.rows.length);
  for (const row of figure.rows) {
    const ratio = row.baseline_total_s / row.presto_total_s;
    assert.ok(Math.abs(row.call_reduction / row.time_per_call_ratio - ratio) < 1e-12, row.id);
    assert.ok(Math.abs(row.speedup - ratio) < .006, row.id);
    const panel = figure.panels.find(panel => panel.sampler === row.sampler);
    assert.ok(row.call_reduction >= panel.xDomain[0] && row.call_reduction <= panel.xDomain[1], row.id);
    assert.ok(row.per_call_speed_ratio >= panel.yDomain[0] && row.per_call_speed_ratio <= panel.yDomain[1], row.id);
  }
});

test('equal-speedup labels lie on their actual diagonal inside the chart', () => {
  for (const panel of figure.panels) for (const speedup of figure.isoSpeedups) {
    const label = isoLabel(speedup, panel.xDomain, panel.yDomain);
    if (!label) continue;
    assert.ok(Math.abs(label.x * label.y - speedup) < 1e-12);
    assert.ok(label.x > panel.xDomain[0] && label.x < panel.xDomain[1]);
    assert.ok(label.y > panel.yDomain[0] && label.y < panel.yDomain[1]);
  }
});
