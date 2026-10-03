const NS = 'http://www.w3.org/2000/svg';
const COLORS = { PowerMH: '#8b96ab', PreSTO: '#2e5bea', BFS: '#8b96ab', Predictive: '#2e5bea' };
const fmt = (v, digits = 2) => Number(v).toFixed(digits);
const $ = (s) => document.querySelector(s);
function svg(tag, attrs = {}, text) {
  const n = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) n.setAttribute(key, value);
  if (text !== undefined) n.textContent = text;
  return n;
}
function niceTop(v) {
  const step = 10 ** Math.floor(Math.log10(v / 4 || 1));
  const interval = [1, 2, 2.5, 5, 10].find(n => n * step >= v / 4) * step;
  return Math.ceil(v / interval) * interval;
}
function chart(target, { title, xDomain, yDomain, xTicks, xLabel, height = 285, left = 46, bottom = 48, top = 18, yFormat = v => String(Number(v.toFixed(2))), yTicks = true }) {
  const width = Math.max(270, target.clientWidth);
  const right = 15;
  const x = v => left + (v - xDomain[0]) / (xDomain[1] - xDomain[0]) * (width - left - right);
  const y = v => height - bottom - (v - yDomain[0]) / (yDomain[1] - yDomain[0]) * (height - top - bottom);
  const plot = svg('svg', { viewBox: `0 0 ${width} ${height}`, role: 'group', 'aria-label': title });
  plot.append(svg('title', {}, title));
  if (yTicks) for (let i = 0; i <= 4; i++) {
    const value = yDomain[0] + (yDomain[1] - yDomain[0]) * i / 4;
    plot.append(svg('line', { x1: left, x2: width - right, y1: y(value), y2: y(value), class: 'case-gridline' }));
    plot.append(svg('text', { x: left - 8, y: y(value) + 4, 'text-anchor': 'end' }, yFormat(value)));
  }
  for (const tick of xTicks || []) {
    const value = typeof tick === 'number' ? tick : tick.value;
    plot.append(svg('text', { x: x(value), y: height - bottom + 21, 'text-anchor': 'middle' }, typeof tick === 'number' ? String(Number(value.toFixed(2))) : tick.label));
  }
  if (xLabel) plot.append(svg('text', { x: (left + width - right) / 2, y: height - 4, 'text-anchor': 'middle', class: 'case-axis-label' }, xLabel));
  const tooltip = document.createElement('div');
  tooltip.className = 'case-tooltip'; tooltip.hidden = true; tooltip.setAttribute('role', 'tooltip');
  target.replaceChildren(plot, tooltip);
  function mark(node, label) {
    node.setAttribute('tabindex', '0'); node.setAttribute('role', 'button'); node.setAttribute('aria-label', label); node.classList.add('case-mark');
    node.append(svg('title', {}, label));
    const show = () => { tooltip.textContent = label; tooltip.hidden = false; };
    const hide = () => { tooltip.hidden = true; };
    node.addEventListener('pointerenter', show); node.addEventListener('pointerleave', hide);
    node.addEventListener('focus', show); node.addEventListener('blur', hide); node.addEventListener('click', show);
    node.addEventListener('keydown', e => { if (e.key === 'Escape') hide(); if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); show(); } });
    plot.append(node);
  }
  return { plot, x, y, width, height, left, right, top, bottom, mark };
}
function stepPoints(values, x, y) {
  const out = [];
  values.forEach(([k,v],i) => { if (i) out.push([x(k),y(values[i-1][1])]); out.push([x(k),y(v)]); });
  return out;
}
function polyline(points, attrs = {}) { return svg('polyline', { points: points.map(p => p.join(',')).join(' '), fill: 'none', ...attrs }); }
const timingSpecs = { totalTime: ['Cumulative model-call time', 'seconds'], transitions: ['MH transitions per call', 'transitions/call'], timePerCall: ['Time per model call', 'seconds/call'] };

function renderEfficiency(data) {
  const k = Number($('#case1-k').value);
  $('#case1-k-value').textContent = k;
  const selected = $('#case1-trace').value;
  const selectedIndex = Number(selected);
  const showTraces = selected !== 'mean';
  for (const [key,[title,unit]] of Object.entries(timingSpecs)) {
    const series = data.metrics[key];
    const max = niceTop(Math.max(...Object.values(series).flatMap(s => [...s.summary.map(r => r.mean + r.sd), ...(showTraces ? s.traces.flatMap(t => t.map(p => p[1])) : [])])) * 1.06);
    const c = chart($(`[data-case1="${key}"]`), { title: `${title}: PowerMH and PreSTO, Qwen3.5-9B`, xDomain: [0,100], yDomain: [0,max], xTicks: [0,25,50,75,100], xLabel: 'Realized MH transitions (K)', top:18, height:357 });
    for (const [method, s] of Object.entries(series)) {
      const color = COLORS[method];
      const upper = stepPoints(s.summary.map(r => [r.k,r.mean+r.sd]), c.x,c.y);
      const lower = stepPoints(s.summary.map(r => [r.k,Math.max(0,r.mean-r.sd)]), c.x,c.y);
      c.plot.append(svg('polygon', { points: [...upper,...lower.reverse()].map(p=>p.join(',')).join(' '), fill: color, opacity: .13 }));
      if (showTraces) for (const trace of s.traces) c.plot.append(polyline(stepPoints(trace,c.x,c.y), { stroke: color, 'stroke-width': .8, opacity: .22, class: 'individual-trace' }));
      c.plot.append(polyline(stepPoints(s.summary.map(r=>[r.k,r.mean]),c.x,c.y), { stroke: color, 'stroke-width': 2.3, ...(method==='PowerMH'?{'stroke-dasharray':'5 3'}:{}) }));
      if (selected !== 'mean') {
        const trace = s.traces[selectedIndex] || (key === 'transitions' && method === 'PowerMH' ? s.traces[0] : null);
        c.plot.append(polyline(stepPoints(trace,c.x,c.y), { stroke: color, 'stroke-width': 3, class: 'inspected-trace' }));
        const value = trace.find(p=>p[0]===k)[1];
        c.plot.append(svg('line', { x1:c.x(k),x2:c.x(k),y1:c.top,y2:c.y(0),stroke:'#8797af','stroke-dasharray':'2 4',opacity:.45 }));
        c.mark(svg('circle', {cx:c.x(k),cy:c.y(value),r:4.3,fill:color,stroke:'#fff','stroke-width':1.5}), `${method} · Trace ${selectedIndex+1} · K = ${k}\n${fmt(value)} ${unit}`);
      } else if (selected === 'mean') {
        const row = s.summary.find(r=>r.k===k);
        c.plot.append(svg('line', { x1:c.x(k),x2:c.x(k),y1:c.top,y2:c.y(0),stroke:'#8797af','stroke-dasharray':'2 4',opacity:.35 }));
        c.mark(svg('circle', {cx:c.x(k),cy:c.y(row.mean),r:4.3,fill:color,stroke:'#fff','stroke-width':1}), `${method} · K = ${k}\n${fmt(row.mean)} ± ${fmt(row.sd)} ${unit}\nMean ± SD`);
      }
    }
    const readout=svg('g', {class:'case-inspection-readout','aria-live':'polite'});
    readout.append(svg('text',{x:c.left+12,y:c.top+22,class:'case-readout-heading'},`K = ${k} · ${selected==='mean'?'Mean ± SD':`Trace ${selectedIndex+1}`} · ${unit}`));
    ['PowerMH','PreSTO'].forEach((method,index)=>{
      const seriesData=series[method];
      let value, measured;
      if(selected==='mean'){
        const row=seriesData.summary.find(r=>r.k===k);value=`${fmt(row.mean)} ± ${fmt(row.sd)}`;measured=row.mean;
      } else {
        const trace=seriesData.traces[selectedIndex] || (key==='transitions'&&method==='PowerMH'?seriesData.traces[0]:null);
        measured=trace.find(p=>p[0]===k)[1];value=fmt(measured);
      }
      if(key==='totalTime'||key==='transitions'||key==='timePerCall'){
        const anchor=k>=65?'end':'start';
        const x=anchor==='end'?c.x(k)-8:c.x(k)+8;
        const y=Math.max(c.top+40,Math.min(c.height-c.bottom-8,c.y(measured)-(k<15&&index===0?34:12)));
        readout.append(svg('text',{x,y,'text-anchor':anchor,class:'case-readout-value',style:`fill:${COLORS[method]};paint-order:stroke;stroke:#fff;stroke-width:4px;stroke-linejoin:round`},`${method} ${value}`));
        return;
      }
      const y=c.top+44+index*21;
      readout.append(svg('circle',{cx:c.left+16,cy:y-4,r:3.5,fill:COLORS[method]}));
      readout.append(svg('text',{x:c.left+27,y,class:'case-readout-value'},`${method} ${value}`));
    });
    c.plot.append(readout);
  }
}

const budgetSpecs = {
  speedup: ['Speedup over PowerMH','×'], transitions: ['MH transitions per call',''],
  hitRate: ['Prefix-cache hit rate','%'], kvPool: ['Peak KV-pool use','%'],
};
function renderBudget(data) {
  const body = $('#case2-table-body');
  body.replaceChildren();
  for (const key of ['transitions','speedup','hitRate','kvPool']) {
    const metric=data.metrics[key], [name,unit]=budgetSpecs[key];
    const tr=document.createElement('tr');
    const th=document.createElement('th');th.scope='row';th.textContent=name+(unit==='%'?' (%)':'');tr.append(th);
    [metric.baseline,...metric.values].forEach((value,i)=>{
      const td=document.createElement('td');td.textContent=fmt(value,key==='hitRate'?1:2)+(unit==='×'?'×':'');
      if(key==='speedup'&&i===4){td.className='best-budget';td.setAttribute('aria-label',`${fmt(value)} times, highest measured speedup`);}
      tr.append(td);
    });body.append(tr);
  }
}
function renderQuality(data) {
  const showPrompts = $('#case3-prompts').checked;
  const selectedMethod = $('#case3-method').value;
  const selectedModel = $('#case3-model').value;
  for (const canvas of document.querySelectorAll('[data-case3]')) canvas.closest('figure').hidden = selectedMethod !== 'all' && !canvas.dataset.case3.startsWith(selectedMethod+'-');
  for (const family of data.families) for (const [metric,title] of [['log_likelihood','Token log-likelihood'],['confidence','Token confidence']]) {
    if(selectedMethod !== 'all' && family.baseline !== selectedMethod) continue;
    const values=data.families.flatMap(f=>f.rows).flatMap(row=>Object.values(row.metrics[metric].groups).flatMap(g=>g.samples));
    const low=Math.min(...values),high=Math.max(...values),pad=(high-low)*.08;
    const domain=[low-pad,high+pad+(high-low)*.30];
    const rows=family.rows.filter(row=>selectedModel==='all'||row.model===selectedModel);
    const c=chart($(`[data-case3="${family.baseline}-${metric}"]`),{title:`${title} under p₀: paired prompt distributions`,xDomain:domain,yDomain:[0,rows.length],xTicks:Array.from({length:4},(_,i)=>low-pad+(high-low+2*pad)*i/3),xLabel:`${title} under p₀`,height:rows.length*57+64,left:152,bottom:46,yTicks:false});
    for(const [i,row] of rows.entries()) {
      const result=row.metrics[metric],cy=30+i*57;
      if(result.pEquiv<.05)c.plot.append(svg('rect',{x:2,y:cy-27,width:c.width-8,height:54,rx:4,fill:'#edf2fa',class:'equivalence-shade'}));
      c.plot.append(svg('text',{x:144,y:cy-3,'text-anchor':'end',class:'case-dataset-label'},row.dataset));
      c.plot.append(svg('text',{x:144,y:cy+13,'text-anchor':'end',class:'case-model-label'},row.model));
      c.plot.append(svg('text',{x:c.width-8,y:cy+4,'text-anchor':'end',class:'case-p-value','font-weight':result.pEquiv<.05?600:400},result.pEquiv<.001?'p < 0.001':`p = ${fmt(result.pEquiv,3)}`));
      for(const [j,[method,b]] of Object.entries(result.groups).entries()) {
        const y=cy+(j?10:-10),color=method==='PreSTO'?'#009E73':'#0072B2';
        const group=svg('g');
        group.append(svg('rect',{x:c.x(domain[0]),y:y-9,width:c.x(domain[1])-c.x(domain[0]),height:18,fill:'transparent'}));
        group.append(svg('path',{d:`M ${c.x(b.whislo)} ${y} H ${c.x(b.whishi)} M ${c.x(b.whislo)} ${y-5} V ${y+5} M ${c.x(b.whishi)} ${y-5} V ${y+5}`,stroke:color,'stroke-width':1.2,fill:'none'}));
        group.append(svg('rect',{x:c.x(b.q1),y:y-6,width:Math.max(.7,c.x(b.q3)-c.x(b.q1)),height:12,fill:color,opacity:.8,rx:1}));
        group.append(svg('line',{x1:c.x(b.med),x2:c.x(b.med),y1:y-7,y2:y+7,stroke:'#142d50','stroke-width':1.7}));
        const dots=showPrompts?b.samples:b.fliers;
        dots.forEach((v,index)=>group.append(svg('circle',{cx:c.x(v),cy:y+(showPrompts?((index*7)%9-4)*.8:0),r:showPrompts?1.8:2.2,fill:color,opacity:showPrompts?.65:1,class:showPrompts?'prompt-score':'box-outlier'})));
        const p=result.pEquiv<.001?'< 0.001':fmt(result.pEquiv,3);
        c.mark(group,`${row.dataset} · ${row.model} · ${method} · ${title}\nn = ${b.n}; median ${fmt(b.med,4)}\nMiddle 50%: ${fmt(b.q1,4)} to ${fmt(b.q3,4)}\nPaired TOST p = ${p}: ${result.pEquiv<.05?'equivalence established':'equivalence not established'}`);
      }
    }
  }
}
function renderTraversal(data) {
  const sampler=$('#case4-sampler').value;
  const rows=data.rows.filter(r=>sampler==='all'||r.sampler===sampler);
  const all=sampler==='all',positions=all?[0,1,2.6,3.6,4.6,5.6]:rows.map((_,i)=>i);
  for(const [metric,title,unit] of [['speedup','Speedup over PowerMH','×'],['transitions','MH transitions per call',''],['totalTime','Cumulative model-call time',' s']]) {
    const max=niceTop(Math.max(...rows.flatMap(r=>['BFS','Predictive'].map(m=>r[m][metric]+(metric==='totalTime'?r[m].sd:0))))*1.07);
    const c=chart($(`[data-case4="${metric}"]`),{title,xDomain:[-.6,positions.at(-1)+.6],yDomain:[0,max],xTicks:positions.map((v,i)=>({value:v,label:String(rows[i].budget)})),xLabel:'Prefetch budget',height:310,bottom:68});
    const bw=Math.min(17,(c.x(1)-c.x(0))*.29);
    rows.forEach((row,i)=>['Predictive','BFS'].forEach((method,j)=>{
      const d=row[method],value=d[metric],x=c.x(positions[i])+(j?2:-bw-2),group=svg('g');
      group.append(svg('rect',{x,y:c.y(value),width:bw,height:c.y(0)-c.y(value),fill:COLORS[method],rx:2}));
      if(metric==='totalTime')group.append(svg('path',{d:`M ${x+bw/2} ${c.y(value-d.sd)} V ${c.y(value+d.sd)} M ${x+bw/2-3} ${c.y(value-d.sd)} H ${x+bw/2+3} M ${x+bw/2-3} ${c.y(value+d.sd)} H ${x+bw/2+3}`,stroke:'#253b5b','stroke-width':1.2}));
      c.mark(group,`${row.sampler} · budget ${row.budget}\n${method==='BFS'?'BFS default Traversal':'Predictive traversal'}\n${title}: ${fmt(value)}${metric==='totalTime'?` ± ${fmt(d.sd)}`:''}${unit}`);
    }));
    for(const method of [...new Set(rows.map(r=>r.sampler))]) {
      const indices=rows.map((r,i)=>r.sampler===method?i:-1).filter(i=>i>=0);
      const center=(positions[indices[0]]+positions[indices.at(-1)])/2;
      c.plot.append(svg('text',{x:c.x(center),y:c.height-27,'text-anchor':'middle',class:'case-group-label'},method));
    }
    if(all)c.plot.append(svg('line',{x1:c.x(1.8),x2:c.x(1.8),y1:c.top,y2:c.y(0),stroke:'#aab5c6','stroke-dasharray':'3 4'}));
  }
}
export function mountCaseStudies(data) {
  const efficiency=()=>renderEfficiency(data.case1),budget=()=>renderBudget(data.case2),quality=()=>renderQuality(data.case3),traversal=()=>renderTraversal(data.case4);
  const traceSelect=$('#case1-trace');
  // Compare same-index traces only where both methods have measurements.
  const sharedTraceCount=Math.min(data.case1.prestoTraces,data.case1.baselineTraces);
  for(let i=0;i<sharedTraceCount;i++)traceSelect.append(new Option(`Trace ${i+1} · PreSTO + PowerMH`,String(i)));
  traceSelect.addEventListener('change',efficiency);
  $('#case1-k').addEventListener('input',efficiency);
  const modelSelect=$('#case3-model');
  const updateQualityModels=()=>{
    const previous=modelSelect.value,method=$('#case3-method').value;
    const models=[...new Set(data.case3.families.filter(f=>method==='all'||f.baseline===method).flatMap(f=>f.rows.map(row=>row.model)))];
    modelSelect.replaceChildren(new Option('All models','all'),...models.map(model=>new Option(model,model)));
    modelSelect.value=models.includes(previous)?previous:'all';quality();
  };
  updateQualityModels();
  modelSelect.addEventListener('change',quality);
  $('#case3-prompts').addEventListener('change',quality);$('#case3-method').addEventListener('change',updateQualityModels);$('#case4-sampler').addEventListener('change',traversal);
  const render=()=>{efficiency();budget();quality();traversal();};render();return render;
}
