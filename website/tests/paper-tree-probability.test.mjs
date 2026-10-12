import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {edgeProbability} from '../public/paper-tree-probability.mjs';
const data=JSON.parse(fs.readFileSync(new URL('../public/data/paper-trees.json',import.meta.url)));
test('MH probabilities use the accept-edge ratio and clamp positive ratios',()=>{
  const t=data.trees[0];
  assert.equal(edgeProbability(t,1),1);
  assert.equal(edgeProbability(t,2),0);
  assert.equal(edgeProbability(t,5),Math.exp(-29.3871));
  assert.equal(edgeProbability({nodes:{}},1),null);
});
test('all recorded sibling probabilities are bounded and sum to one',()=>{
  for(const t of data.trees)for(const key of Object.keys(t.nodes)){
    const id=Number(key);if(!id)continue;
    const p=edgeProbability(t,id);assert(p===null||(p>=0&&p<=1));
    if(id%2&&t.nodes[id+1]&&p!==null)assert(Math.abs(p+edgeProbability(t,id+1)-1)<1e-14);
  }
});
