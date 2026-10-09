import {layoutPaperTree} from './paper-tree-layout.mjs';

const host=document.querySelector('.hero-tree-art');
const caption=document.querySelector('.hero-tree figcaption a');
const ns='http://www.w3.org/2000/svg';
let animations=[],visible=true,records,currentIndex=0;
caption.addEventListener('click',()=>{
  const record=records?.[currentIndex];
  const selection={prompt:record?.prompt_index??0,tree:record?.tree_number??1};
  document.documentElement.dataset.heroTreeSelection=JSON.stringify(selection);
  document.dispatchEvent(new CustomEvent('presto-select-record',{detail:selection}));
});
function syncPlayback(){
  const playing=document.documentElement.dataset.motion!=='paused'&&visible&&!document.hidden;
  for(const animation of animations)playing?animation.play():animation.pause();
}
document.addEventListener('presto-motion',syncPlayback);
document.addEventListener('visibilitychange',syncPlayback);
new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;syncPlayback();},{threshold:0}).observe(host);
function element(tag,attributes={},text){
  const node=document.createElementNS(ns,tag);
  for(const [key,value] of Object.entries(attributes))node.setAttribute(key,value);
  if(text!==undefined)node.textContent=text;
  return node;
}
function render(record){
  const layout=layoutPaperTree(record),path=new Set(record.path_node_ids),rounded=new Set(record.rounded_node_ids);
  const svg=element('svg',{viewBox:'0 0 500 480',role:'img','aria-label':`Recorded prefetched tree for prompt ${record.prompt_index}, tree ${record.tree_number}, MH transitions ${record.mh_start} to ${record.mh_end}.`});
  svg.append(element('title',{},`Recorded subtree: prompt ${record.prompt_index}, tree ${record.tree_number}`));
  const scale=Math.min(470/layout.width,450/layout.height);
  const group=element('g',{transform:`translate(${(500-layout.width*scale)/2},${(480-layout.height*scale)/2}) scale(${scale})`});
  svg.append(group);
  const stages=[];
  for(const id of layout.ids){
    if(id===0)continue;
    const parent=Math.floor((id-1)/2),a=layout.positions.get(parent),b=layout.positions.get(id);
    const onPath=path.has(id)&&path.has(parent),reject=id%2===0;
    const edge=element('path',{d:`M${a.x},${a.y+14} L${b.x},${b.y-14}`,class:`hero-edge${onPath?' is-path':''}`,'stroke-width':onPath?3.5:1.7});
    if(reject)edge.setAttribute('stroke-dasharray','5 4');
    group.append(edge);
    if(onPath)stages.push({node:edge,delay:record.path_node_ids.indexOf(id)*650});
  }
  for(const id of layout.ids){
    const p=layout.positions.get(id),node=element('g',{class:`hero-node${path.has(id)?' is-path':''}${rounded.has(id)?' is-rounded':''}`,transform:`translate(${p.x},${p.y})`});
    node.append(element('title',{},`Node v${id}, cut index ${record.nodes[id].cut_idx}`));
    const rect=element('rect',{x:-21,y:-14,width:42,height:28,rx:5});
    if(id!==0&&id%2===0)rect.setAttribute('stroke-dasharray','4 3');
    node.append(rect);
    const text=element('text',{y:4});text.append(element('tspan',{},'v'),element('tspan',{'baseline-shift':'sub','font-size':'8'},id));node.append(text);group.append(node);
    if(path.has(id))stages.push({node,delay:record.path_node_ids.indexOf(id)*650});
  }
  for(const animation of animations)animation.cancel();
  animations=[];
  host.replaceChildren(svg);
  caption.textContent=`Prompt ${record.prompt_index} · Tree ${record.tree_number} ↗`;
  for(const {node,delay} of stages){
    // These animations belong to the current record; no CSS entry animation repeats over them.
    node.style.animation='none';
    animations.push(node.animate([{opacity:.12},{opacity:1}],{duration:600,delay,fill:'both',easing:'ease-out'}));
  }
  const duration=(record.path_node_ids.length-1)*650+2600;
  const clock=svg.animate([{opacity:1},{opacity:1}],{duration});
  animations.push(clock);syncPlayback();
  return clock.finished;
}
async function cycle(){
  try{
    const response=await fetch('./data/paper-trees.json');
    if(!response.ok)throw new Error('Tree records unavailable');
    records=(await response.json()).trees;
    if(!records.length)return;
    // Let the original entry animation finish before beginning the carousel.
    const initialClock=host.animate([{opacity:1},{opacity:1}],{duration:5000});
    animations.push(initialClock);syncPlayback();await initialClock.finished;
    while(true){
      // Uniform random selection without showing the same record twice in a row.
      const offset=records.length>1?1+Math.floor(Math.random()*(records.length-1)):0;
      currentIndex=(currentIndex+offset)%records.length;
      await render(records[currentIndex]);
    }
  }catch(error){
    // The original recorded SVG remains usable if loading fails.
    if(error.name!=='AbortError')console.warn('Hero tree cycle unavailable',error);
  }
}
cycle();
