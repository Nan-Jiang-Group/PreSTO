import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const data=JSON.parse(readFileSync(new URL('../public/data/edge-certainty.json',import.meta.url)));
test('All edge-certainty histograms preserve paired edges and source category masses',()=>{
  assert.equal(data.epsilon,.01);assert.equal(data.prefetchBudget,20);
  assert.deepEqual(Object.values(data.figures).map(f=>f.panels.length),[7,28,28]);
  for(const figure of Object.values(data.figures))for(const panel of figure.panels){
    assert.equal(panel.bins.length,5);
    assert.equal(panel.edgeCount,2*panel.proposalNodes);
    assert.equal(panel.bins.reduce((sum,b)=>sum+b.count,0),panel.edgeCount);
    const counts=[panel.bins[0].count,panel.bins.slice(1,4).reduce((sum,b)=>sum+b.count,0),panel.bins[4].count];
    assert.equal(counts[0],counts[2]);
    for(let i=0;i<3;i++)assert.ok(Math.abs(counts[i]/panel.edgeCount-panel.bucketShares[i])<.000001);
    for(const bin of panel.bins)assert.ok(Math.abs(bin.count/panel.edgeCount-bin.share)<.000001);
  }
  assert.deepEqual(data.figures.main.panels.map(p=>p.bucketShares.map(s=>Math.round(s*100))),[[36,27,36],[37,26,37],[42,16,42],[40,19,40],[34,33,34],[41,17,41],[40,20,40]]);
});

const sweep=JSON.parse(readFileSync(new URL('../public/data/epsilon-sweep.json',import.meta.url)));
test('epsilon sweep conserves every edge, reproduces the default and increases certainty',()=>{
  assert.equal(sweep.epsilons[sweep.defaultIndex],.01);
  for(const [kind,figure] of Object.entries(data.figures))for(const panel of figure.panels){
    const snapshots=sweep.panels[`${kind}:${panel.model}:${panel.dataset}`].counts;
    assert.deepEqual(snapshots[sweep.defaultIndex],panel.bins.map(b=>b.count));
    let previous=0;
    for(const counts of snapshots){
      assert.equal(counts.reduce((a,b)=>a+b,0),panel.edgeCount);
      assert.equal(counts[0],counts[4]);
      assert.ok(counts[0]+counts[4]>=previous);previous=counts[0]+counts[4];
    }
  }
});
