import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const data=JSON.parse(readFileSync(new URL('../public/data/case-studies.json',import.meta.url)));
const mean=xs=>xs.reduce((a,b)=>a+b,0)/xs.length;
const sd=xs=>xs.length===1?0:Math.sqrt(xs.reduce((s,x)=>s+(x-mean(xs))**2,0)/(xs.length-1));

test('Case 1 means and sample SDs reproduce the underlying trajectories at every transition',()=>{
  for(const metric of Object.values(data.case1.metrics))for(const series of Object.values(metric))for(const row of series.summary){
    const values=series.traces.map(t=>t.find(p=>p[0]===row.k)?.[1]).filter(v=>v!==undefined);
    assert.ok(values.length>0);
    assert.ok(Math.abs(mean(values)-row.mean)<1e-7,`mean at K=${row.k}`);
    assert.ok(Math.abs(sd(values)-row.sd)<1e-7,`SD at K=${row.k}`);
  }
  const end=data.case1.metrics.totalTime;
  assert.equal(end.PowerMH.summary.at(-1).mean.toFixed(2),'481.02');
  assert.equal(end.PreSTO.summary.at(-1).mean.toFixed(2),'253.29');
  assert.equal(data.case1.prefetchBudget,10);
});
test('Case 2 preserves the budget ordering and the reported optimum',()=>{
  for(const metric of Object.values(data.case2.metrics))assert.equal(metric.values.length,data.case2.budgets.length);
  const values=data.case2.metrics.speedup.values;
  assert.equal(data.case2.budgets[values.indexOf(Math.max(...values))],16);
  assert.equal(Math.max(...values),1.94);
});
test('Case 3 boxes and equivalence shading agree with their prompt samples and intervals',()=>{
  let equivalent=0;
  for(const family of data.case3.families)for(const row of family.rows)for(const result of Object.values(row.metrics)){
    const inside=result.ci90[0]>-result.margin&&result.ci90[1]<result.margin;
    assert.equal(result.pEquiv<.05,inside);
    if(inside)equivalent++;
    for(const group of Object.values(result.groups)){
      assert.equal(group.samples.length,group.n);
      assert.ok(group.whislo<=group.q1&&group.q1<=group.med&&group.med<=group.q3&&group.q3<=group.whishi);
      for(const value of group.fliers)assert.ok(value<group.whislo||value>group.whishi);
    }
  }
  assert.equal(equivalent,29);assert.deepEqual(data.case3.families.map(f=>f.rows.length),[8,9]);
});
test('Case 4 keeps all six settings and uses SD only for the measured time metric',()=>{
  assert.equal(data.case4.rows.length,6);
  for(const row of data.case4.rows){
    assert.ok(row.Predictive.speedup>row.BFS.speedup);
    assert.ok(row.Predictive.totalTime<row.BFS.totalTime);
    assert.ok(row.Predictive.sd>0&&row.BFS.sd>0);
  }
});
