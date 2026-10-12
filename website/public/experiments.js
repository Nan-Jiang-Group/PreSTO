import { mountEdgeCertainty } from './edge-certainty.js?v=epsilon-relative-path';
import { mountFigureOne } from './figure-one.js?v=identity-labels';
import { mountCaseStudies } from './case-studies.js?v=quality-method';
const root = document.querySelector('.experiments-section');
if (root) {
  try {
    const responses = await Promise.all(['./data/experiments.json','./data/case-studies.json?v=appendix-quality','./data/edge-certainty.json'].map(path=>fetch(new URL(path,import.meta.url))));
    if (responses.some(response=>!response.ok)) throw new Error('Experiment data unavailable');
    const [benchmark,cases,certainty] = await Promise.all(responses.map(response=>response.json()));
    await mountEdgeCertainty(certainty);
    const renderBenchmark = mountFigureOne(benchmark.figure1);
    const renderCases = mountCaseStudies(cases);
    let width=root.clientWidth, frame;
    new ResizeObserver(()=>{
      if(root.clientWidth===width)return;
      width=root.clientWidth;cancelAnimationFrame(frame);
      frame=requestAnimationFrame(()=>{renderBenchmark();renderCases();});
    }).observe(root);
  } catch(error) {
    document.querySelector('#experiment-error').hidden=false;
    console.error('Could not load experiment charts',error);
  }
}
