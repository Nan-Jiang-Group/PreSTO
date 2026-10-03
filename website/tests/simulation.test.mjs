import test from 'node:test';
import assert from 'node:assert/strict';
import { PRESET, RULES, buildTree, schedule, evaluate, realize, compare, sanitize } from '../public/simulation.mjs';

test('BFS and depth-first scheduling use distinct documented orders',()=>{
 const t=buildTree({...PRESET,depth:3,cutGate:false});
 assert.deepEqual(schedule(t,'bfs',7),[1,2,3,4,5,6,7]);
 assert.deepEqual(schedule(t,'reject',7),[1,3,7,6,2,5,4]);
});
test('all schedules satisfy budget, ancestor closure, and cut feasibility',()=>{
 for(let seed=0;seed<30;seed++) for(const cutGate of [true,false]) {
  const {tree,results}=compare({...PRESET,depth:6,budget:23,seed,cutGate});
  for(const r of results) {
   assert.ok(r.used<=23); assert.equal(new Set(r.order).size,r.used);assert.equal(r.order[0],1);
   const seen=new Set();for(const id of r.order){const n=tree.byId.get(id);assert.ok(n.eligible);if(n.parentId)assert.ok(seen.has(n.parentId));if(cutGate&&n.branch==='A')assert.ok(n.cut<=tree.byId.get(n.parentId).cut);seen.add(id);}
   assert.ok(r.effective>=1&&r.effective<=6);assert.equal(r.used,r.effective+r.wasted);assert.ok(r.expected>=1&&r.expected<=6+1e-10);
  }
 }
});
test('certain acceptance and rejection resolve every scheduled on-path decision',()=>{
 for(const probability of [0,1]) {
  const tree=buildTree({...PRESET,depth:5,budget:5,model:'uniform',probability,cutGate:false});
  const order=schedule(tree,probability?'bfs':'reject',probability?31:5),r=evaluate(tree,order);
  assert.equal(r.effective,5);assert.equal(r.expected,5);assert.equal(r.utilization,5/order.length);assert.equal(r.accepted,probability?5:0);
 }
});
test('analytic expectation matches exhaustive enumeration of possible paths',()=>{
 const tree=buildTree({...PRESET,depth:5,budget:9,model:'uniform',probability:.37,cutGate:false});
 for(const rule of RULES){const order=schedule(tree,rule.id,9),set=new Set(order);let weighted=0;
  function walk(id,weight,count,stopped){const n=tree.byId.get(id);if(n.terminal){weighted+=weight*count;return;}const nextStopped=stopped||!set.has(id),nextCount=count+(!nextStopped?1:0);walk(id*2,weight*n.p,nextCount,nextStopped);walk(id*2+1,weight*(1-n.p),nextCount,nextStopped);}
  walk(1,1,0,false);assert.ok(Math.abs(weighted-evaluate(tree,order).expected)<1e-10);
 }
});
test('resampling leaves proposals and every schedule fixed',()=>{
 const a=compare({...PRESET,pathSeed:17}),b=compare({...PRESET,pathSeed:18});
 assert.deepEqual(a.tree.nodes.map(({u,...n})=>n),b.tree.nodes.map(({u,...n})=>n));
 assert.deepEqual(a.results.map(r=>r.order),b.results.map(r=>r.order));
 assert.notDeepEqual(a.tree.nodes.map(n=>n.u),b.tree.nodes.map(n=>n.u));
 assert.deepEqual(realize(buildTree(PRESET)),realize(buildTree(PRESET)));
});
test('full coverage yields all transitions and bounded budget clamps on resize',()=>{
 const tree=buildTree({...PRESET,depth:5,cutGate:false});
 for(const rule of RULES){const r=evaluate(tree,schedule(tree,rule.id,31));assert.equal(r.effective,5);assert.ok(Math.abs(r.expected-5)<1e-10);}
 assert.equal(sanitize({...PRESET,depth:2,budget:64}).budget,3);
 assert.equal(sanitize({depth:NaN,budget:Infinity,probability:-2}).probability,0);
 for(const rule of ['accept','longest','cut']) assert.equal(sanitize({...PRESET,rule}).rule,'bfs');
});
