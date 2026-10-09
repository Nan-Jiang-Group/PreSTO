import {select} from 'd3-selection';
import {zoom, zoomIdentity} from 'd3-zoom';
import {edgeProbability, formatProbability} from './paper-tree-probability.mjs';
import { layoutPaperTree } from './paper-tree-layout.mjs';
const $ = id => document.getElementById(id);
let data, promptsData, records=[], index=0, selected=0, bounds, nodePositions, inspectorVisible=false, selectedEdge=null;
const svgElement=$('paper-tree-svg');
let view=zoomIdentity;
const navigation=zoom().scaleExtent([0.15,4]).on('zoom',e=>{view=e.transform;select(svgElement).select('.paper-tree-content').attr('transform',view);positionInspector();});
select(svgElement).call(navigation).on('dblclick.zoom',null);
function viewport(){return {width:Math.max(280,svgElement.clientWidth),height:500};}
function frame(fit=false){if(!bounds)return;const v=viewport();svgElement.setAttribute('viewBox',`0 0 ${v.width} ${v.height}`);const scale=fit?Math.min(1,(v.width-32)/bounds.width,(v.height-32)/bounds.height):1;const x=(v.width-bounds.width*scale)/2,y=fit?(v.height-bounds.height*scale)/2:20;select(svgElement).call(navigation.transform,zoomIdentity.translate(x,y).scale(scale));}
$('paper-zoom-in').addEventListener('click',()=>select(svgElement).call(navigation.scaleBy,1.25));
$('paper-zoom-out').addEventListener('click',()=>select(svgElement).call(navigation.scaleBy,.8));
$('paper-fit').addEventListener('click',()=>frame(true));
new ResizeObserver(()=>{const v=viewport();svgElement.setAttribute('viewBox',`0 0 ${v.width} ${v.height}`);positionInspector();}).observe(svgElement);
function options() {
  const prompt=promptsData.prompts[$('paper-prompt').value];
  $('paper-prompt-text').value=`${prompt.title}\n\n${prompt.text}`;
  $('paper-prompt-text').scrollTop=0;
  records=data.trees.filter(t=>t.prompt_index===Number($('paper-prompt').value));
  $('paper-tree').innerHTML=records.map((t,i)=>`<option value="${i}">Tree ${t.tree_number} · MH ${t.mh_start} → ${t.mh_end}</option>`).join('');
  index=0; selected=0; inspectorVisible=false; selectedEdge=null; render();frame();
}
function selectHeroRecord(selection) {
  if(!data)return;
  $('paper-prompt').value=String(selection.prompt);options();
  const target=records.findIndex(t=>t.tree_number===selection.tree);
  if(target>=0){$('paper-tree').value=String(target);change(target);frame(true);}
}
document.addEventListener('presto-select-record',e=>selectHeroRecord(e.detail));
function positionInspector() {
  const inspector=$('paper-node-inspector');
  let p=nodePositions?.get(selected);
  if(selectedEdge!==null){const child=nodePositions?.get(selectedEdge), parent=nodePositions?.get(Math.floor((selectedEdge-1)/2));if(child&&parent)p={x:(child.x+parent.x)/2,y:(child.y+parent.y)/2};}
  if(!p||!inspectorVisible)return;
  const v=viewport(), x=view.applyX(p.x), y=view.applyY(p.y);
  inspector.style.left=`${Math.max(8,Math.min(v.width-inspector.offsetWidth-8,x+30*view.k))}px`;
  inspector.style.top=`${Math.max(8,Math.min(v.height-inspector.offsetHeight-8,y-15*view.k))}px`;
}
function inspect() {
  const inspector=$('paper-node-inspector');
  inspector.style.maxWidth=selectedEdge===null?'190px':'220px';
  if(selectedEdge!==null){
    const t=records[index], parent=Math.floor((selectedEdge-1)/2), reject=selectedEdge%2===0;
    const probability=edgeProbability(t,selectedEdge);
    const percent=p=>p===0||p===1?String(p*100):(p*100).toPrecision(3);
    inspector.innerHTML=`<strong>${reject?'Reject':'Accept'} · p ${probability!==0&&probability!==1?'≈':'='} ${formatProbability(probability)}</strong><span>At <math><msub><mi>v</mi><mn>${parent}</mn></msub></math>: ${probability===null?'probability unavailable.':`${probability===0||probability===1?'':'≈'}${percent(probability)}% chance to ${reject?'keep the current continuation':'accept the proposal'}.`}</span>`;
    inspector.hidden=!inspectorVisible;positionInspector();return;
  }
  const n=records[index].nodes[selected];
  $('paper-node-inspector').innerHTML=`<strong>Node ID <math><msub><mi>v</mi><mn>${selected}</mn></msub></math>;</strong><span>cut index: ${n.cut_idx},</span><span>resample suffix: <math aria-label="x sub ${n.cut_idx+1} through x sub ${n.seq_len}"><msub><mi>x</mi><mn>${n.cut_idx+1}</mn></msub><mo>,</mo><mo>…</mo><mo>,</mo><msub><mi>x</mi><mn>${n.seq_len}</mn></msub></math></span>`;
  $('paper-node-inspector').hidden=!inspectorVisible;
  positionInspector();
}
function render() {
  if(!records.length)return;
  const t=records[index], {ids,positions,width,height}=layoutPaperTree(t), path=new Set(t.path_node_ids), rounded=new Set(t.rounded_node_ids);
  const edges=new Set(t.path_node_ids.slice(1).map((id,i)=>`${t.path_node_ids[i]}-${id}`));
  let lines='', nodes='';
  for(const id of ids) {
    const p=positions.get(id), right=id>0&&id%2===0, color=path.has(id)?'#40619f':'#8d969d', fill=rounded.has(id)?'#fcb827':path.has(id)?'#ddeff9':right?'white':'#f5f6f7';
    if(id>0) {
      const parent=Math.floor((id-1)/2), q=positions.get(parent), sampled=edges.has(`${parent}-${id}`), inRounded=rounded.has(parent)&&rounded.has(id), c=sampled?'#40619f':inRounded?'#dca64d':'#b9c1ca', m=sampled?0:inRounded?1:2;
      const probability=edgeProbability(t,id), probabilityText=formatProbability(probability), labelX=(q.x+p.x)/2+(right?8:-8), labelY=(q.y+p.y)/2-8;
      lines+=`<g data-paper-edge="${id}" tabindex="0" role="button" aria-label="Explain MH ${right?'rejection':'acceptance'} probability ${probabilityText}" style="cursor:pointer"><path d="M${q.x+(right?8:-8)} ${q.y+17} L${p.x} ${p.y-19}" fill="none" stroke="transparent" stroke-width="16" pointer-events="stroke"/><path d="M${q.x+(right?8:-8)} ${q.y+17} L${p.x} ${p.y-19}" fill="none" stroke="${c}" stroke-width="${selectedEdge===id?3:sampled?2:1.25}" stroke-linecap="round" stroke-linejoin="round" ${right?'stroke-dasharray="4 4"':''} marker-end="url(#paper-arrow-${m})"/><text x="${labelX}" y="${labelY}" text-anchor="middle" fill="${sampled?'#40619f':inRounded?'#9c751f':'#718094'}" font-size="10" stroke="white" stroke-width="3" paint-order="stroke"><title>MH ${right?'rejection':'acceptance'} probability: ${probabilityText}</title><tspan x="${labelX}">${right?'reject':'accept'}</tspan><tspan x="${labelX}" dy="12" font-size="9">p ${probability!==null&&probability!==0&&probability!==1?'≈':'='} ${probabilityText}</tspan></text></g>`;
    }
    const label=`Node ID v${id}; cut index: ${t.nodes[id].cut_idx}, resample suffix: x_${t.nodes[id].cut_idx+1}, …, x_${t.nodes[id].seq_len}`;
    nodes+=`<g class="paper-node" data-paper-node="${id}" tabindex="0" role="button" aria-label="${label}" aria-pressed="${id===selected}"><title>${label}</title>${id===selected&&selectedEdge===null?`<rect x="${p.x-28}" y="${p.y-21}" width="56" height="42" rx="8" fill="none" stroke="#40619f" opacity=".5"/>`:''}<rect x="${p.x-22}" y="${p.y-15}" width="44" height="30" rx="6" fill="${fill}" stroke="${color}" stroke-width="${path.has(id)?2:1}" ${right?'stroke-dasharray="4 3"':''}/>${id===t.path_node_ids.at(-1)?`<rect x="${p.x-18}" y="${p.y-11}" width="36" height="22" rx="4" fill="none" stroke="${color}"/>`:''}<text x="${p.x}" y="${p.y+4}" text-anchor="middle" fill="#24344e" font-size="13" pointer-events="none"><tspan font-style="italic">v</tspan><tspan baseline-shift="sub" font-size="9">${id}</tspan></text></g>`;
  }
  const svg=svgElement;nodePositions=positions;bounds={width,height};const v=viewport();svg.setAttribute('viewBox',`0 0 ${v.width} ${v.height}`);
  svg.innerHTML=`<defs>${['#40619f','#dca64d','#b9c1ca'].map((c,i)=>`<marker id="paper-arrow-${i}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto" markerUnits="userSpaceOnUse"><path d="M1 1 L7 4 L1 7" fill="none" stroke="${c}" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round"/></marker>`).join('')}</defs><g class="paper-tree-content" transform="${view}">${lines}${nodes}</g>`;
  $('paper-tree-title').textContent=`Prompt ${t.prompt_index} · Tree ${t.tree_number}`;
  $('paper-previous').disabled=index===0;$('paper-next').disabled=index===records.length-1;
  $('paper-tree').value=index;inspect();
}
function change(next) {index=next;selected=0;inspectorVisible=false;selectedEdge=null;render();frame();}
$('paper-prompt').addEventListener('change',options);
$('paper-tree').addEventListener('change',e=>change(Number(e.target.value)));
$('paper-previous').addEventListener('click',()=>change(index-1));$('paper-next').addEventListener('click',()=>change(index+1));
function inspectTarget(target){
  const edge=target.closest('[data-paper-edge]'), node=target.closest('[data-paper-node]');
  if(!edge&&!node)return null;
  selectedEdge=edge?Number(edge.dataset.paperEdge):null;
  if(node)selected=Number(node.dataset.paperNode);
  inspectorVisible=true;render();
  return edge?`[data-paper-edge="${selectedEdge}"]`:`[data-paper-node="${selected}"]`;
}
$('paper-tree-svg').addEventListener('click',e=>inspectTarget(e.target));
$('paper-tree-svg').addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)&&e.target.closest('[data-paper-edge], [data-paper-node]')){e.preventDefault();const target=inspectTarget(e.target);svgElement.querySelector(target)?.focus();}else if(e.key==='Escape'){inspectorVisible=false;$('paper-node-inspector').hidden=true;}});
Promise.all(['paper-trees','paper-prompts'].map(name=>fetch(`./data/${name}.json`).then(r=>{if(!r.ok)throw Error('Tree data unavailable');return r.json();}))).then(([d,p])=>{data=d;promptsData=p;const prompts=[...new Set(d.trees.map(t=>t.prompt_index))];$('paper-prompt').innerHTML=prompts.map(p=>`<option value="${p}">Prompt ${p}${p===0?' · shown in the paper':''}</option>`).join('');options();if(document.documentElement.dataset.heroTreeSelection)selectHeroRecord(JSON.parse(document.documentElement.dataset.heroTreeSelection));}).catch(()=>{$('paper-tree-error').hidden=false;$('paper-tree-error').textContent='Recorded trees could not load. Please refresh the page.';});
