import {hierarchy, tree} from 'd3-hierarchy';
// Tidy layout retains logged heap IDs and acceptance-left/rejection-right order.
export function layoutPaperTree(record) {
  const root=hierarchy({id:0}, ({id})=>[2*id+1,2*id+2].filter(c=>Object.hasOwn(record.nodes,c)).map(id=>({id})));
  tree().nodeSize([96,90]).separation((a,b)=>a.parent===b.parent?1:1.25)(root);
  const nodes=root.descendants(), minX=Math.min(...nodes.map(n=>n.x)), maxX=Math.max(...nodes.map(n=>n.x));
  const width=Math.max(160,maxX-minX+80), height=Math.max(...nodes.map(n=>n.y))+80;
  const positions=new Map(nodes.map(n=>[n.data.id,{x:n.x-minX+40,y:n.y+40,level:n.depth}]));
  return {ids:nodes.map(n=>n.data.id), positions, width, height};
}
