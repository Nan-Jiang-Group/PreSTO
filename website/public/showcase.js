// Enhance the research page without hiding content or changing recorded data.
const root=document.documentElement;
const motionButton=document.querySelector('#motion-control');
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
let paused=reduced.matches;
function setMotion(value){
  paused=value;root.dataset.motion=paused?'paused':'playing';
  motionButton.textContent=paused?'Play motion':'Pause motion';
  motionButton.setAttribute('aria-pressed',String(paused));
  document.dispatchEvent(new CustomEvent('presto-motion',{detail:{playing:!paused}}));
}
motionButton.addEventListener('click',()=>setMotion(!paused));
reduced.addEventListener('change',e=>setMotion(e.matches));
setMotion(paused);
const links=[...document.querySelectorAll('.site-nav a')];
const sectionObserver=new IntersectionObserver(entries=>{
  const visible=entries.filter(e=>e.isIntersecting);
  if(!visible.length)return;
  const section=visible.at(-1).target;
  for(const link of links){if(link.hash===`#${section.id}`)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');}
},{rootMargin:'-105px 0px -55% 0px',threshold:0});
for(const link of links){const section=document.querySelector(link.hash);if(section)sectionObserver.observe(section);}
const revealObserver=new IntersectionObserver(entries=>{
  for(const entry of entries)if(entry.isIntersecting){entry.target.classList.add('reveal-once');revealObserver.unobserve(entry.target);}
},{threshold:.08});
for(const element of document.querySelectorAll('.intro,.experiments-title,.experiment-heading,.citation-heading'))revealObserver.observe(element);
const copy=document.querySelector('#copy-citation');
copy.addEventListener('click',async()=>{
  try{await navigator.clipboard.writeText(document.querySelector('#bibtex-code').textContent);copy.textContent='Copied';setTimeout(()=>copy.textContent='Copy citation',2000);}
  catch{copy.textContent='Select the citation to copy';document.querySelector('.citation-section pre').focus();setTimeout(()=>copy.textContent='Copy citation',3000);}
});

// Load the substantial scientific tools only as their sections approach.
const modules=new Map([
  [document.querySelector('.goal-methods'),'./animation-build/pipeline-animation.js'],
  [document.querySelector('#tree-explorer'),'./tree-build/paper-trees.js'],
  [document.querySelector('#experiments-title'),'./experiments.js?v=quality-method']
]);
const toolObserver=new IntersectionObserver(entries=>{
  for(const entry of entries)if(entry.isIntersecting){
    const url=modules.get(entry.target);toolObserver.unobserve(entry.target);
    import(url).catch(()=>{
      const status=entry.target.querySelector('[role=alert]');
      if(status){status.hidden=false;status.textContent='This visualization could not load. Please reload the page.';}
    });
  }
},{rootMargin:'220px',threshold:0});
for(const section of modules.keys())toolObserver.observe(section);
new IntersectionObserver(([entry])=>{if(entry.isIntersecting)for(const link of links)link.removeAttribute('aria-current');},{threshold:.5}).observe(document.querySelector('.hero-composition'));
