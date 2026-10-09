import { PRESET, RULES as AVAILABLE_RULES, sanitize, compare } from './simulation.mjs';
const RULES = AVAILABLE_RULES.filter(r => r.id === 'bfs');
const $ = id => document.getElementById(id);
const STORAGE = 'step-explorer-v1';
let config = { ...PRESET };
try { const saved = JSON.parse(localStorage.getItem(STORAGE)); if (saved && typeof saved === 'object') config = sanitize(saved); } catch {}
config.rule = 'bfs';
let experiment, selectedNode = 1, zoom = 1, toastTimer;
const percent = x => `${(100 * x).toFixed(x === 0 || x === 1 ? 0 : 1)}%`;
const shortP = x => x === 0 || x === 1 ? String(x) : x.toFixed(3).replace(/0+$/, '');
const save = () => { try { localStorage.setItem(STORAGE, JSON.stringify(config)); } catch {} };
function toast(message) { $('toast').textContent = message; $('toast').classList.add('visible'); clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').classList.remove('visible'), 2200); }
function active() { return experiment.results.find(r => r.id === config.rule); }
function renderControls() {
  for (const key of ['depth','budget']) $(key).value = config[key];
  $('depth-value').textContent = config.depth;
  $('budget-value').textContent = config.budget;
  $('budget').max = Math.min(64, 2 ** config.depth - 1);
}
function renderTabs() {
  $('rule-tabs').innerHTML = RULES.map(r => `<button id="tab-${r.id}" class="rule-tab" role="tab" aria-selected="${r.id===config.rule}" tabindex="${r.id===config.rule?0:-1}" data-rule="${r.id}" title="${r.detail}">${r.name}</button>`).join('');
  $('tree-title').textContent = RULES.find(r=>r.id===config.rule).name;
}
function renderTree() {
  if (!experiment) return;
  const { tree, trace } = experiment, result = active(), planned = new Set(result.order);
  const sampledDecisions = trace.decisions.slice(0, result.effective);
  const pathIds = new Set([1, ...sampledDecisions.flatMap(d=>[d.id,d.childId])]);
  const edges = new Set(sampledDecisions.map(d=>d.childId));
  const width = Math.max(280, $('tree-scroll').clientWidth - 12) * zoom;
  const height = config.depth === 7 ? 430 : config.depth === 6 ? 390 : 350;
  const top = 32, bottom = height - 26, pad = 24;
  const pos = n => ({ x: pad + (n.id - 2 ** n.level + .5) / 2 ** n.level * (width - 2*pad), y: top + (bottom-top)*n.level/config.depth });
  const svg = $('tree'); svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.style.width=`${width}px`;svg.style.height=`${height}px`;
  const radius = n => n.terminal ? 3.5 : n.level >= 6 ? 5 : n.level >= 5 ? 7 : n.level >= 4 ? 9 : 12;
  const arrowDefs=`<defs>${['#40619f','#d6b574','#dce2e9'].map((color,i)=>`<marker id="tree-arrow-${i}" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="6" markerHeight="6" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="${color}"/></marker>`).join('')}</defs>`;
  const nodeBox=(x,y,r)=>`x="${x-r*1.25}" y="${y-r}" width="${r*2.5}" height="${r*2}" rx="${Math.min(5,r*.4)}"`;
  let lines = '', nodes = '';
  for (const n of tree.nodes) {
    const {x,y}=pos(n);
    if(n.parentId) {
      const parent=tree.byId.get(n.parentId), p=pos(parent), onPath=edges.has(n.id);
      const blocked=!n.terminal&&!n.eligible;
      const color=onPath?'#40619f':planned.has(n.id)?'#d6b574':'#dce2e9';
      const arrow=onPath?0:planned.has(n.id)?1:2;
      const path=`M${p.x},${p.y+radius(parent)} C${p.x},${p.y+(y-p.y)*.47} ${x},${y-(y-p.y)*.47} ${x},${y-radius(n)}`;
      lines+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="${onPath?2.3:planned.has(n.id)?1.5:1}" marker-end="url(#tree-arrow-${arrow})" ${n.branch==='R'?'stroke-dasharray="4 4"':blocked?'stroke-dasharray="2 3"':''} opacity="${onPath?1:blocked?.45:.9}"/>`;
      if(n.level<=3) lines+=`<text x="${(p.x+x)/2+(n.branch==='A'?-7:7)}" y="${(p.y+y)/2-5}" text-anchor="middle" fill="${onPath?'#40619f':'#a8b1be'}" font-size="8" font-weight="500" style="paint-order:stroke;stroke:white;stroke-width:4;stroke-linejoin:round">${n.branch}</text>`;
    }
    const r=radius(n), isPlanned=planned.has(n.id), onPath=pathIds.has(n.id);
    if(n.terminal) {nodes+=`<rect ${nodeBox(x,y,r)} fill="${onPath?'#40619f':'#dfe4eb'}"/>`;continue;}
    const inspected=n.id===selectedNode;
    const title=`v${n.id-1}${n.id===1?' (root)':''} · ${n.path || 'Root'} · p(accept)=${shortP(n.p)} · cut=${n.cut} · ${isPlanned?'prefetched':n.eligible?'unscheduled':'cut-blocked'}`;
    nodes+=`<g class="tree-node" data-node="${n.id}" role="button" tabindex="0" aria-label="${title}" aria-pressed="${inspected}"><title>${title}</title><rect class="hit-area" ${nodeBox(x,y,Math.max(11,r+4))}/>${inspected?`<rect ${nodeBox(x,y,r+6)} fill="none" stroke="#40619f" stroke-width="1" stroke-dasharray="2 2" opacity=".45"/>`:''}${onPath?`<rect ${nodeBox(x,y,r+3)} fill="white" stroke="#40619f" stroke-width="1.5" ${n.branch==='R'?'stroke-dasharray="3 2"':''}/>`:''}<rect class="node-body" ${n.branch==='R'?'stroke-dasharray="3 2"':''} ${nodeBox(x,y,r)} fill="${isPlanned?'#dca64d':n.eligible?'#f0f3f7':'#f8f9fb'}" stroke="${isPlanned?'#c99642':'#cfd7e2'}" stroke-width="1" opacity="${n.eligible?1:.55}"/>${!n.terminal?`<text x="${x}" y="${y+2.8}" text-anchor="middle" pointer-events="none" font-size="9" font-weight="600" fill="${isPlanned?'#fff':'#93a0b2'}"><tspan font-style="italic">v</tspan><tspan baseline-shift="sub" font-size="6.5">${n.id-1}</tspan></text>`:''}${n.level<3?`<text x="${x}" y="${y-r-8}" text-anchor="middle" pointer-events="none" fill="#8995a5" font-size="8" style="paint-order:stroke;stroke:white;stroke-width:3">${n.id===1?'root · ':''}p=${shortP(n.p)}</text>`:''}</g>`;
  }
  svg.innerHTML=arrowDefs+lines+nodes;
  $('node-count').textContent=`${2**config.depth-1} proposals · ${config.depth} levels`;
  $('zoom-out').disabled=zoom<=1; $('zoom-in').disabled=zoom>=4;
}
function renderInspector() {
  const n=experiment.tree.byId.get(selectedNode) || experiment.tree.byId.get(1), r=active(), ordinal=r.order.indexOf(n.id);
  const status=ordinal>=0?`Prefetched #${ordinal+1}`:!n.eligible?'Blocked by cut eligibility':'Not prefetched';
  $('node-inspector').innerHTML=`<strong><math><msub><mi>v</mi><mn>${n.id-1}</mn></msub></math>${n.id===1?' · Root':` · ${n.path}`}</strong><span><span class="inspector-key">P(A)</span>${shortP(n.p)}</span><span><span class="inspector-key">P(R)</span>${shortP(1-n.p)}</span><span><span class="inspector-key">Cut</span>${n.cut}/128</span><span><span class="inspector-key">u</span>${n.u.toFixed(3)}</span><span class="node-status">${status}</span>`;
}
function render() {
  config=sanitize(config); save(); experiment=compare(config);
  if(!experiment.tree.byId.has(selectedNode)||experiment.tree.byId.get(selectedNode).terminal) selectedNode=1;
  renderControls();renderTabs();renderTree();renderInspector();
}
function chooseRule(rule) {config.rule=rule;save();renderTabs();renderTree();renderInspector();}
function chooseNode(node) {selectedNode=node;renderTree();renderInspector();}
for(const key of ['depth','budget']) $(key).addEventListener('input',e=>{config[key]=Number(e.target.value);$('preset').value='custom';render();});
function loadPreset(name) {
  config={...PRESET}; if(name==='balanced') {config.model='uniform';config.probability=.5;config.cutGate=false;} if(name==='accepting') {config.model='uniform';config.probability=.8;config.cutGate=false;}
  selectedNode=1;zoom=1;$('preset').value=name;render();$('tree-scroll').scrollLeft=0;
}
$('preset').addEventListener('change',e=>loadPreset(e.target.value));
$('reset').addEventListener('click',()=>{loadPreset('step');toast('STeP preset restored');});
$('resample').addEventListener('click',()=>{config.pathSeed=(config.pathSeed+1)%1000000;render();toast(`New path sampled · draw ${config.pathSeed}`);});
document.addEventListener('click',e=>{const r=e.target.closest('[data-rule]');if(r) chooseRule(r.dataset.rule);const n=e.target.closest('[data-node]');if(n) chooseNode(Number(n.dataset.node));});
$('tree').addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.closest('[data-node]')){e.preventDefault();const id=Number(e.target.closest('[data-node]').dataset.node);chooseNode(id);$('tree').querySelector(`[data-node="${id}"]`)?.focus();}});
$('rule-tabs').addEventListener('keydown',e=>{if(!['ArrowRight','ArrowLeft','Home','End'].includes(e.key))return;e.preventDefault();let i=RULES.findIndex(r=>r.id===config.rule);i=e.key==='Home'?0:e.key==='End'?RULES.length-1:(i+(e.key==='ArrowRight'?1:-1)+RULES.length)%RULES.length;chooseRule(RULES[i].id);$(`tab-${RULES[i].id}`).focus();});
$('zoom-in').addEventListener('click',()=>{zoom=Math.min(4,zoom+.5);renderTree();});
$('zoom-out').addEventListener('click',()=>{zoom=Math.max(1,zoom-.5);renderTree();});
$('fit').addEventListener('click',()=>{zoom=1;renderTree();$('tree-scroll').scrollLeft=0;});
const isPreset=JSON.stringify(config)===JSON.stringify(PRESET);$('preset').value=isPreset?'step':'custom';
render();
new ResizeObserver(()=>renderTree()).observe($('tree-scroll'));
