import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {layoutPaperTree} from '../public/paper-tree-layout.mjs';
const data=JSON.parse(fs.readFileSync(new URL('../public/data/paper-trees.json',import.meta.url)));
test('recorded tree gallery preserves complete 100-transition prompt chains and valid binary paths',()=>{
  assert.equal(data.trees.length,481);
  const prompts=[...new Set(data.trees.map(t=>t.prompt_index))];assert.equal(prompts.length,20);
  for(const prompt of prompts){let end=0;for(const t of data.trees.filter(t=>t.prompt_index===prompt)){assert.equal(t.mh_start,end);end=t.mh_end;assert.equal(t.path_node_ids.length-1,t.mh_end-t.mh_start);for(let i=1;i<t.path_node_ids.length;i++){const child=t.path_node_ids[i];assert.equal(Math.floor((child-1)/2),t.path_node_ids[i-1]);assert.equal(child%2?'accept':'reject',t.decisions[i-1]);}}assert.equal(end,100);}
});
test('every logged node and rounded branch fits the compact tree layout',()=>{
  for(const t of data.trees){const {ids,positions,width,height}=layoutPaperTree(t);assert.equal(ids.length,t.node_count);assert.equal(positions.size,ids.length);for(const id of ids){if(id)assert(t.nodes[Math.floor((id-1)/2)]);const p=positions.get(id);assert(p.x>=22&&p.x<=width-22&&p.y>=15&&p.y<=height-15);}for(const id of [...t.rounded_node_ids,...t.path_node_ids])assert(t.nodes[id]);}
});
test('tidy layout keeps nodes apart and accept children left of reject children',()=>{
  for(const t of data.trees){const {positions}=layoutPaperTree(t),levels=new Map();for(const [id,p] of positions){if(!levels.has(p.level))levels.set(p.level,[]);levels.get(p.level).push(p.x);const a=positions.get(2*id+1),r=positions.get(2*id+2);if(a&&r)assert(a.x<r.x);}for(const xs of levels.values()){xs.sort((a,b)=>a-b);for(let i=1;i<xs.length;i++)assert(xs[i]-xs[i-1]>=63.99);}}
});
