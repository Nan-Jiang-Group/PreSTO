// Animate the user's pipeline.excalidraw directly with the official Excalidraw API.
// Original elements, IDs, seeds, fonts, strokes, fills, and hatching are retained.
import {convertToExcalidrawElements,exportToCanvas} from '@excalidraw/excalidraw';
const clamp=(t,a,b)=>Math.max(0,Math.min(1,(t-a)/(b-a)));
const clone=x=>structuredClone(x);
const state={exportBackground:true,viewBackgroundColor:'#ffffff',exportWithDarkMode:false,exportEmbedScene:false};
let source,subtree,baseline;
const SIZE={subtree:[1440,620],powermh:[1440,410]};
const TIMING={collect:[.6,1.6,2.6],stage2:3.6,proposalBeats:[3.6,4.6,5.6],generate:[4.6,7.4],resolve:[7.4,8.4,9.4],
  accept:[9.8,10.8],reject:[11.2,12.2],activeSeconds:14,totalSeconds:20};
const treeBeats=new Map([
  ...['tz','tz-label','try','try-label','t13','t14','path-z','path-z-label','path-ry','path-ry-label','path-13','path-14','path-reject'].map(id=>[id,1]),
  ...['tw','tw-label','trx2','trx2-label','t25','t26'].map(id=>[id,2]),
]);
function updateText(e,text,{x=e.x,y=e.y,size=e.fontSize,center=null}={}){
  const input={...e,text,originalText:text,x,y,fontSize:size,autoResize:true,width:undefined,height:undefined};
  const [result]=convertToExcalidrawElements([input],{regenerateIds:false});
  if(center!==null)result.x=center-result.width/2;
  return result;
}
function newText(id,text,x,y,size=24,color='#243342'){
  const [e]=convertToExcalidrawElements([{id,type:'text',text,x,y,fontFamily:6,fontSize:size,strokeColor:color,seed:12456}],{regenerateIds:false});
  return e;
}
function bounds(mode){
  const [width,height]=SIZE[mode];
  return convertToExcalidrawElements([{id:'export-bounds',type:'rectangle',x:0,y:0,width,height,
    opacity:0,strokeWidth:0,roughness:0,locked:true,seed:1024}],{regenerateIds:false})[0];
}
function translated(elements,dx,dy){return elements.map(e=>({...e,x:e.x+dx,y:e.y+dy}));}
function completeDecisionTree(){
  const find=id=>subtree.find(e=>e.id===id);
  // Keep only the decision path's left descendants; the right child is a leaf.
  const positions={'path-root':[1298,571],'path-y':[1198,671],'path-rx':[1398,671],
    'path-z':[1146,791],'path-ry':[1250,791]};
  for(const [id,[cx,cy]] of Object.entries(positions)){
    const node=find(id),label=find(id+'-label');
    node.x=cx-node.width/2;node.y=cy-node.height/2;
    node.boundElements=[{id:label.id,type:'text'}];
    label.x=cx-label.width/2;label.y=cy-label.height/2;
  }
  const edges=[['path-01','path-root','path-y'],['path-02','path-root','path-rx'],
    ['path-13','path-y','path-z'],['path-14','path-y','path-ry']];
  for(const [id,parent,child] of edges){
    const edge=find(id);
    const a=positions[parent],b=positions[child],length=Math.hypot(b[0]-a[0],b[1]-a[1]);
    const unit=b.map((v,i)=>(v-a[i])/length);
    const start=a.map((v,i)=>v+35*unit[i]),end=b.map((v,i)=>v-36*unit[i]);
    edge.x=start[0];edge.y=start[1];edge.points=[[0,0],[end[0]-start[0],end[1]-start[1]]];
    edge.width=Math.abs(edge.points[1][0]);edge.height=Math.abs(edge.points[1][1]);
    edge.startBinding={elementId:parent,focus:0,gap:4};edge.endBinding={elementId:child,focus:0,gap:4};
    find(parent).boundElements.push({id,type:'arrow'});find(child).boundElements.push({id,type:'arrow'});
  }
  const reject=find('path-reject');reject.x=1263;reject.y=713;
  const final=find('original');final.x=1250-final.width/2;
}
function proposalDetail({id,center,current,proposal,cut,start,end,call}){
  const left=center-100,width=200,top=174;
  const elements=[];
  const add=e=>{e.customData={...e.customData,revealAt:start,growUntil:end};elements.push(e);return e;};
  const shape=(name,type,x,y,w,h,style={})=>add(convertToExcalidrawElements([{
    id:`${id}-${name}`,type,x,y,width:w,height:h,seed:12456,roughness:1,
    strokeColor:'#adb5bd',strokeWidth:1,...style,
  }],{regenerateIds:false})[0]);
  const text=(name,value,x,y,size=20,color='#243342')=>add(newText(`${id}-${name}`,value,x,y,size,color));
  shape('axis','line',left,top,width,0,{points:[[0,0],[width,0]]});
  for(const [name,offset,label] of [['zero',0,'0'],['cut',cut,call===1?'c1':'c2'],['end',width,'T']]){
    shape(`tick-${name}`,'line',left+offset,top-4,0,10,{points:[[0,0],[0,10]]});
    const e=text(`axis-${name}`,label,0,top-30);e.x=left+offset-e.width/2;
  }
  shape('cut-guide','line',left+cut,top+10,0,88,{points:[[0,0],[0,88]],strokeStyle:'dashed'});
  for(const [name,y,label] of [['current',top+24,current],['proposal',top+67,proposal]]){
    text(`${name}-label`,label,left-32,y+3);
    shape(`${name}-prefix`,'rectangle',left,y,cut-5,28,{fillStyle:'hachure',backgroundColor:'#343a40',strokeColor:'#243342',roundness:{type:3}});
    const suffix=shape(`${name}-suffix`,'rectangle',left+cut+5,y,width-cut-5,28,
      name==='current'?{fillStyle:'cross-hatch',backgroundColor:'#868e96',strokeColor:'#868e96',roundness:{type:3}}:
      {fillStyle:'solid',backgroundColor:'#fff9db',strokeColor:'#f08c00',roundness:{type:3}});
    if(name==='proposal')suffix.customData.growingSuffix=true;
  }
  const note=text('call',`Call ${call} · generate + score`,0,top+121,20);note.x=center-note.width/2;
  text('cached-key','cached',left,top+104,15,'#566575');
  text('new-key','new',left+cut+8,top+104,15,'#f08c00');
  return elements;
}
function prepare(){
  const excluded=new Set(['rule1','rule2','AymR9P4ryFg2pYsp6vQNW','pMQ3H9-wswQiOa75sEEC8']);
  subtree=source.elements.filter(e=>!e.isDeleted&&e.y>=474&&e.y<899&&!excluded.has(e.id)).map(clone);
  subtree=subtree.map(e=>{
    const stage=e.x<550?0:e.x<1095?1:2;
    e.customData={...e.customData,animationStage:stage,sourceElementId:e.id};
    if(['p1','9lQIFGs-vz5BwyVR62KLD','RfH9Y1uV2iPocj-ejq76e'].includes(e.id))e.height=990-e.y;
    else if(stage===1&&e.id!=='p2title')e.y+=40;
    if(e.id==='p1title')e=updateText(e,'1. collect feasible requests',{size:26});
    if(e.id==='p2title')e=updateText(e,'2. generate + score from proposal LLM',{size:26});
    if(e.id==='p2title'&&e.width>475)e=updateText(e,e.text,{size:26*475/e.width});
    if(e.id==='p2title')e.x=817-e.width/2;
    if(e.id==='p3title')e=updateText(e,'3. Follow the MH decisions',{size:26});
    if(e.id==='5Kh74fj2aPD3sgDuAU6uD')e=updateText(e,'Feasibility condition: c2 ≤ c1',{size:24,y:942,center:318});
    if(e.id==='original')e=updateText(e,'final y',{size:26,center:1280});
    return e;
  });
  completeDecisionTree();
  for(const [id,text,x,stage] of [['batch-count','1 (batched) LLM call',817,1],['mh-count','MH transitions: x → y → y',1297,2]]){
    let e=newText(id,text,0,942);e.x=x-e.width/2;e.customData={animationStage:stage};subtree.push(e);
  }
  subtree=translated(subtree,-72,-450);
  // Preserve the separately requested baseline as the original top row.
  baseline=source.elements.filter(e=>!e.isDeleted&&e.y>=300&&e.y<460&&e.id!=='1N3UjHYOMLNZ9SfSvlCco').map(clone);
  baseline=baseline.map(e=>e.id==='serialtitle'?updateText(e,'2 sequential calls',{size:27,center:790,y:650}):e);
  baseline=baseline.filter(e=>!['9rAl8_kJ7KmjeDowsb1ZE','dog5MiFBV6pNBOLwendF4'].includes(e.id));
  baseline=translated(baseline,-72,-285);
  baseline.push(...proposalDetail({id:'sequential-y',center:273,current:'x',proposal:'y',cut:110,start:.6,end:5.3,call:1}),
    ...proposalDetail({id:'sequential-z',center:953,current:'y',proposal:'z',cut:70,start:6.8,end:10.5,call:2}));
}
function fade(e,amount){e.opacity=Math.round((e.opacity??100)*amount);return e;}
function neutral(e){
  e.strokeColor='#8d9baa';
  if(e.type!=='text'&&e.backgroundColor!=='transparent')e.backgroundColor='#ffffff';
}
function growArrow(e,p){
  if(p<=0){neutral(e);return;}
  if(p<1){
    const end=e.points[e.points.length-1];
    e.points=[[0,0],[end[0]*p,end[1]*p]];
    e.width=Math.abs(end[0]*p);e.height=Math.abs(end[1]*p);
    e.startBinding=null;e.endBinding=null;
  }
}
function subtreeFrame(t){
  const result=[bounds('subtree')],starts=[TIMING.collect[0],TIMING.stage2,TIMING.resolve[0]];
  for(const raw of subtree){
    const e=clone(raw),stage=e.customData.animationStage;
    const beat=treeBeats.get(e.id)??0;
    let reveal=stage===0?TIMING.collect[beat]:stage===2?TIMING.resolve[beat]:starts[stage];
    // Row 2 box 2 is a cached block revealed at the second beat.
    if(e.id==='MLozHQRNDAuq_iVpkhB47')reveal=TIMING.proposalBeats[1];
    const visibility=clamp(t,reveal,reveal+.25);if(visibility<=0)continue;
    if(['ty','ty-label'].includes(e.id)&&t<1.2)neutral(e);
    if(['tz','tz-label'].includes(e.id)&&t<2)neutral(e);
    if(['suffix1','suffix2'].includes(e.id)){
      // Row 3 box 2 starts growing first; row 2 box 3 starts one beat later.
      const start=e.id==='suffix2'?TIMING.proposalBeats[1]:TIMING.proposalBeats[2];
      const p=clamp(t,start,TIMING.generate[1]);if(p<=0)continue;e.width=Math.max(1,e.width*p);
    }
    if(['prop1','prop2'].includes(e.id)&&t<TIMING.generate[1])continue;
    if(['path-y','path-y-label','path-accept'].includes(e.id)&&t<TIMING.accept[1])neutral(e);
    if(['path-ry','path-ry-label','path-reject'].includes(e.id)&&t<TIMING.reject[1])neutral(e);
    if(e.id==='original'&&t<TIMING.reject[1])continue;
    if(e.id==='path-01')growArrow(e,clamp(t,...TIMING.accept));
    if(e.id==='path-14')growArrow(e,clamp(t,...TIMING.reject));
    result.push(fade(e,visibility));
  }
  return result;
}
function baselineFrame(t){
  const result=[bounds('powermh')];
  for(const raw of baseline){
    const e=clone(raw),x=e.x+72;
    let start=x<470?.6:x<710?5.3:x<865?6:x<1147?6.8:x<1373?10.5:11.2;
    if(e.id==='serialtitle')start=12.2;
    if(e.customData?.revealAt!==undefined){
      start=e.customData.revealAt;
      if(e.customData.growingSuffix){
        const p=clamp(t,start,e.customData.growUntil);if(p<=0)continue;
        e.width=Math.max(1,e.width*p);
      }
      if(e.id.endsWith('-call'))start=e.customData.growUntil;
    }
    const visibility=clamp(t,start,start+.3);if(visibility>0)result.push(fade(e,visibility));
  }
  return result;
}
function makeScene(t,mode){return mode==='powermh'?baselineFrame(t):subtreeFrame(t);}

const hosts={subtree:document.querySelector('#presto-animation'),powermh:document.querySelector('#powermh-animation')};
const host=hosts.subtree;
let time=0,playing=!matchMedia('(prefers-reduced-motion: reduce)').matches,visible=false,busy=false,pending=false,last=performance.now(),lastPaint=0;
document.addEventListener('presto-motion',e=>{playing=e.detail.playing;last=performance.now();});
if(document.documentElement.dataset.motion==='paused')playing=false;
window.EXCALIDRAW_ASSET_PATH=new URL('./excalidraw-assets/',location.href).href;
async function draw(mode=document.querySelector('#method-panel-presto').hidden?'powermh':'subtree'){
  if(busy){pending=true;return;}
  busy=true;
  try{
    const canvas=await exportToCanvas({elements:makeScene(time,mode),appState:state,files:source.files||{},exportPadding:0,getDimensions:(w,h)=>({width:w*1.5,height:h*1.5,scale:1.5})});
    canvas.setAttribute('role','img');canvas.setAttribute('aria-label',mode==='subtree'?'PreSTO: collect feasible requests, generate and score proposals in one batch, then follow MH decisions.':'Two sequential proposal generation and Metropolis–Hastings decisions.');
    hosts[mode].replaceChildren(canvas);
  }finally{busy=false;if(pending){pending=false;draw().catch(console.error);}}
}
function tick(now){
  const dt=Math.min((now-last)/1000,.2);last=now;
  if(playing&&visible&&!document.hidden){
    time=(time+dt)%TIMING.totalSeconds;
    if(now-lastPaint>100&&!busy){lastPaint=now;draw().catch(console.error);}
  }
  requestAnimationFrame(tick);
}
async function init(){
  source=await(await fetch('./animation/pipeline.excalidraw')).json();
  await exportToCanvas({elements:source.elements,appState:state,files:source.files||{},exportPadding:0});
  prepare();
  if(!playing)time=14;
  await draw('subtree');
  await draw('powermh');
  new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;}).observe(host.closest('.goal-methods'));
  requestAnimationFrame(tick);
}
init().catch(error=>{console.error(error);host.innerHTML='<p>Animation could not load. <a href="./images/subtree-prefetching-excalidraw.gif">View animation</a></p>';});
