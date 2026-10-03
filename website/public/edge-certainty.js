const NS='http://www.w3.org/2000/svg';
const colors=['#0066cc','#de4444','#fcb827'];
const names=['certain reject','uncertain','certain accept'];
function svg(tag, attrs={}, text) {
  const node=document.createElementNS(NS,tag);
  for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);
  if(text!==undefined)node.textContent=text;
  return node;
}
function histogram(panel, id, data) {
  const figure=document.createElement('figure');figure.className='certainty-panel';
  const heading=document.createElement('figcaption');heading.textContent=panel.datasetLabel;
  const chart=svg('svg',{viewBox:'0 0 260 218',role:'group','aria-label':`${panel.modelLabel}, ${panel.datasetLabel}: edge transition probability histogram`});
  const defs=svg('defs');
  colors.forEach((color,i)=>{
    const pattern=svg('pattern',{id:`${id}-${i}`,width:6,height:6,patternUnits:'userSpaceOnUse'});
    pattern.append(svg('path',{d:i===1?'M-1,-1 L7,7 M-1,5 L1,7 M5,-1 L7,1':'M-1,1 L1,-1 M-1,7 L7,-1 M5,7 L7,5',stroke:color,'stroke-width':1}));defs.append(pattern);
  });
  chart.append(defs);
  const x=v=>34+v/11*216,y=v=>173-v/50*125;
  for(const tick of [0,25,50]){
    chart.append(svg('line',{x1:34,x2:250,y1:y(tick),y2:y(tick),class:'case-gridline'}));
    chart.append(svg('text',{x:28,y:y(tick)+4,'text-anchor':'end'},`${tick}`));
  }
  chart.append(svg('text',{x:34,y:15,class:'certainty-axis'},'Edge share (%)'));
  for(const boundary of [3.25,7.75])chart.append(svg('line',{x1:x(boundary),x2:x(boundary),y1:43,y2:173,stroke:'#cdd5e0','stroke-dasharray':'3 3'}));
  const tooltip=document.createElement('div');tooltip.className='case-tooltip';tooltip.hidden=true;tooltip.setAttribute('role','tooltip');
  let offset=0, binIndex=0;
  data.binCounts.forEach((n,category)=>{
    const width=data.bucketDisplayWidths[category],bw=Math.min(width/n*.65,1.3);
    const label=svg('text',{x:x(offset+width/2),y:36,'text-anchor':'middle',class:`certainty-share certainty-share-${category}`},`${Math.round(panel.bucketShares[category]*100)}%`);
    chart.append(label);
    for(let i=0;i<n;i++){
      const bin=panel.bins[binIndex++], share=bin.count/panel.edgeCount*100;
      const bar=svg('rect',{x:x(offset+(i+.5)*width/n-bw/2),y:y(share),width:x(bw)-x(0),height:y(0)-y(share),fill:`url(#${id}-${category})`,stroke:colors[category],tabindex:0,role:'button',class:'case-mark'});
      const probability=v=>v<.001?v.toExponential(2):Number(v.toPrecision(4)).toString();
      const label=`${panel.modelLabel} · ${panel.datasetLabel} · ${names[category]}: ${share.toFixed(2)}% (${bin.count.toLocaleString()} of ${panel.edgeCount.toLocaleString()} edges), probability ${probability(bin.left)}–${probability(bin.right)}. Category total: ${(panel.bucketShares[category]*100).toFixed(2)}%.`;
      bar.setAttribute('aria-label',label);bar.append(svg('title',{},label));
      const show=()=>{tooltip.textContent=label;tooltip.hidden=false;};
      const hide=()=>{tooltip.hidden=true;};
      for(const event of ['pointerenter','focus','click'])bar.addEventListener(event,show);
      for(const event of ['pointerleave','blur'])bar.addEventListener(event,hide);
      bar.addEventListener('keydown',e=>{if(e.key==='Escape')hide();if(e.key==='Enter'||e.key===' '){e.preventDefault();show();}});
      chart.append(bar);
    }
    offset+=width;
  });
  for(const [pos,label] of [[3.25,'0.01'],[7.75,'0.99']])chart.append(svg('text',{x:x(pos),y:193,'text-anchor':'middle'},label));
  chart.append(svg('text',{x:142,y:214,'text-anchor':'middle',class:'certainty-axis'},'Edge transition probability p'));
  figure.append(heading,chart,tooltip);return figure;
}
function legend() {
  const el=document.createElement('div');el.className='certainty-legend';
  const ranges=['p ≤ 0.01','0.01 < p < 0.99','p ≥ 0.99'];
  names.forEach((name,i)=>{const item=document.createElement('span'),swatch=document.createElement('i');swatch.style.setProperty('--certainty-color',colors[i]);swatch.className=`certainty-swatch-${i}`;item.append(swatch,document.createTextNode(`${name} · ${ranges[i]}`));el.append(item);});return el;
}
export function mountEdgeCertainty(data) {
  const figures={...data.figures,powerMH:{...data.figures.powerMH,panels:[...data.figures.main.panels,...data.figures.powerMH.panels]}};
  for(const kind of ['appendix']){
    const target=document.querySelector(`[data-certainty="${kind}"]`);
    target.replaceChildren();
    let key=kind==='main'?'main':'powerMH';
    const viewport=document.createElement('div');viewport.className='certainty-compact';
    const axis=document.createElement('p');axis.className='certainty-shared-axis';axis.textContent='Edge share (%)';
    const grid=document.createElement('div');grid.className='certainty-grid';
    const xAxis=document.createElement('p');xAxis.className='certainty-shared-axis certainty-shared-axis-x';xAxis.textContent='Edge transition probability p';
    viewport.append(axis,grid,xAxis);
    const render=model=>{
      const figure=figures[key];
      grid.setAttribute('aria-label',`${figure.method}, ${figure.panels.find(p=>p.model===model).modelLabel}`);
      grid.replaceChildren(...figure.panels.filter(p=>p.model===model).map((panel,index)=>histogram(panel,`${key}-${model}-${index}`,data)));
      const shares=figure.panels.filter(p=>p.model===model).map(p=>100*(p.bucketShares[0]+p.bucketShares[2]));
      document.querySelector('#certainty-appendix-caption').textContent=`Across seven datasets, approximately ${Math.round(Math.min(...shares))}%–${Math.round(Math.max(...shares))}% of edges are ε-certain in the prefetched trees (${figure.method}, ${figure.panels.find(p=>p.model===model).modelLabel}).`;
    };
    if(kind==='appendix'){
      const controls=document.createElement('div');controls.className='certainty-model-controls certainty-picker-row';
      function selector(id,text){
        const field=document.createElement('div');field.className='certainty-picker-field';
        const label=document.createElement('label');label.htmlFor=id;label.textContent=text;
        const select=document.createElement('select');select.id=id;
        const wrapper=document.createElement('div');wrapper.className='filter-select';wrapper.append(select);
        field.append(label,wrapper);controls.append(field);return select;
      }
      const methodSelect=selector('certainty-method','Method');
      for(const method of ['powerMH','entropyCut'])methodSelect.append(new Option(figures[method].method,method));
      const modelSelect=selector('certainty-model','Base model');
      const updateModels=()=>{
        const previous=modelSelect.value,figure=figures[key];
        const models=[...new Set(figure.panels.map(p=>p.model))];
        modelSelect.replaceChildren(...models.map(model=>new Option(figure.panels.find(p=>p.model===model).modelLabel,model)));
        modelSelect.value=models.includes(previous)?previous:models[0];render(modelSelect.value);
      };
      methodSelect.addEventListener('change',()=>{key=methodSelect.value;updateModels();});
      modelSelect.addEventListener('change',()=>render(modelSelect.value));
      target.append(controls);updateModels();
    } else render(data.figures.main.panels[0].model);
    target.append(viewport,legend());
  }
}
