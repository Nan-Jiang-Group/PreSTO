import { logPosition, ticks, isoLabel } from './scatter-math.mjs';

const ns = 'http://www.w3.org/2000/svg';
const fmt = (value) => value.toFixed(2);
function html(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function svg(tag, attributes = {}, text) {
  const node = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
function marker(shape, color, size = 5.5) {
  const attributes = { fill: color, stroke: '#fff', 'stroke-width': .9 };
  const s = size;
  if (shape === 'circle') return svg('circle', { ...attributes, r: s });
  if (shape === 'square') return svg('rect', { ...attributes, x: -s, y: -s, width: s * 2, height: s * 2, rx: .5 });
  const a = s * .4;
  const paths = {
    triangle: `M 0 ${-s * 1.15} L ${s * 1.1} ${s} L ${-s * 1.1} ${s} Z`,
    triangleDown: `M 0 ${s * 1.15} L ${s * 1.1} ${-s} L ${-s * 1.1} ${-s} Z`,
    diamond: `M 0 ${-s * 1.3} L ${s * 1.15} 0 L 0 ${s * 1.3} L ${-s * 1.15} 0 Z`,
    plus: `M ${-a} ${-s} H ${a} V ${-a} H ${s} V ${a} H ${a} V ${s} H ${-a} V ${a} H ${-s} V ${-a} H ${-a} Z`,
  };
  if (shape === 'cross') return svg('path', { ...attributes, d: paths.plus, transform: 'rotate(45)' });
  return svg('path', { ...attributes, d: paths[shape] });
}

export function mountFigureOne(data) {
  const root = document.querySelector('#figure-one');
  const modelSelect = root.querySelector('#figure-model');
  const datasetSelect = root.querySelector('#figure-dataset');
  for (const { name } of data.models) modelSelect.append(new Option(name, name));
  for (const { name } of data.datasets) datasetSelect.append(new Option(name, name));
  const colors = new Map(data.models.map(({ name, color }) => [name, color]));
  const shapes = new Map(data.datasets.map(({ name, shape }) => [name, shape]));
  const shown = data.rows.filter((row) => row.shown);
  let activeId = shown[0].id;
  const allRows = () => shown.filter((row) => (modelSelect.value === 'all' || row.model === modelSelect.value) && (datasetSelect.value === 'all' || row.dataset === datasetSelect.value));

  for (const [kind, values] of [['models', data.models], ['datasets', data.datasets]]) {
    const legend = root.querySelector(`[data-figure-legend="${kind}"]`);
    for (const value of values) {
      const item = html('span', 'scatter-legend-item');
      const icon = svg('svg', { viewBox: '-10 -10 20 20', 'aria-hidden': 'true' });
      icon.append(marker(value.shape || 'circle', value.color || '#59667b', 5));
      item.append(icon, document.createTextNode(value.name));
      legend.append(item);
    }
  }

  function inspect(row, showTooltip = false) {
    activeId = row.id;
    root.querySelectorAll('.scatter-point').forEach((node) => {
      const selected = node.dataset.id === activeId;
      node.classList.toggle('is-selected', selected);
      node.setAttribute('aria-pressed', String(selected));
    });
    root.querySelectorAll('.scatter-tooltip').forEach((tooltip) => { tooltip.hidden = true; });
    if (showTooltip) {
      const point = [...root.querySelectorAll('.scatter-point')].find((node) => node.dataset.id === row.id);
      const panel = point.closest('.scatter-panel');
      const tooltip = panel.querySelector('.scatter-tooltip');
      tooltip.replaceChildren(html('strong', '', `${row.model_label} · ${row.dataset}`), html('span', '', `${fmt(row.speedup)}× speedup`), html('span', '', `${fmt(row.call_reduction)}× fewer calls · ${fmt(row.time_per_call_ratio)}× time/call`));
      const pointRect = point.getBoundingClientRect(), panelRect = panel.getBoundingClientRect();
      tooltip.style.left = `${Math.max(5, Math.min(pointRect.x - panelRect.x - 100, panel.clientWidth - 235))}px`;
      tooltip.style.top = `${Math.max(38, pointRect.y - panelRect.y - 100)}px`;
      tooltip.hidden = false;
    }
    const detail = root.querySelector('#figure-inspector');
    const identity = html('div', 'scatter-identity');
    identity.append(html('span', 'scatter-detail-label', `Improvement over ${row.sampler} base method`), html('strong', '', `Base LLM: ${row.model_label}, Dataset: ${row.dataset}${row.note ? ' †' : ''}`));
    const measures = html('dl', 'scatter-measures');
    for (const [label, value] of [['Fewer calls', row.call_reduction], ['Time per call', row.time_per_call_ratio], ['Speedup', row.speedup]]) {
      const metric = html('div');
      metric.append(html('dt', '', label), html('dd', '', `${fmt(value)}×`));
      measures.append(metric);
    }
    detail.replaceChildren(identity, measures);
    const note = root.querySelector('#figure-point-note');
    note.textContent = row.note;
    note.hidden = !row.note;
  }

  function render() {
    const filtered = allRows();
    const grid = root.querySelector('#figure-panels');
    grid.replaceChildren();
    root.querySelector('#figure-count').textContent = `${filtered.length} ${filtered.length === 1 ? 'comparison' : 'comparisons'}`;
    for (const [index, config] of data.panels.entries()) {
      const rows = filtered.filter((row) => row.sampler === config.sampler);
      const panel = html('section', 'scatter-panel');
      panel.setAttribute('aria-label', `Figure 1 comparison with ${config.sampler}`);
      const title = html('h4', 'scatter-panel-title', `(${String.fromCharCode(97 + index)}) ${config.sampler} with and without PreSTO`);
      const tooltip = html('div', 'scatter-tooltip');
      tooltip.setAttribute('role', 'tooltip');
      tooltip.hidden = true;
      panel.append(title, tooltip);
      grid.append(panel);
      const width = Math.max(panel.clientWidth, 280);
      const height = 310;
      const margin = { left: 49, right: 15, top: 16, bottom: 51 };
      const [x0, x1] = config.xDomain;
      const x = (value) => logPosition(value, config.xDomain, [margin.left, width - margin.right]);
      const y = (value) => logPosition(value, config.yDomain, [height - margin.bottom, margin.top]);
      const plot = svg('svg', { viewBox: `0 0 ${width} ${height}`, role: 'group', 'aria-label': `${config.sampler} with and without PreSTO: call reduction and time per call on logarithmic axes. Arrow keys move between points.` });
      plot.append(svg('title', {}, `Figure 1: ${config.sampler} with and without PreSTO`));
      const clipId = `scatter-clip-${index}`;
      const defs = svg('defs');
      const clip = svg('clipPath', { id: clipId });
      clip.append(svg('rect', { x: margin.left, y: margin.top, width: width - margin.left - margin.right, height: height - margin.top - margin.bottom }));
      defs.append(clip);
      plot.append(defs);
      for (const value of ticks(config.xDomain, [1, .5, .25], 2)) {
        plot.append(svg('text', { x: x(value), y: height - margin.bottom + 20, 'text-anchor': 'middle' }, String(value)));
        plot.append(svg('line', { x1: x(value), x2: x(value), y1: height - margin.bottom, y2: height - margin.bottom + 4, class: 'scatter-axis' }));
      }
      const costDomain = [1 / config.yDomain[1], 1 / config.yDomain[0]];
      for (const value of ticks(costDomain, [.5, .2, .1], 3)) {
        plot.append(svg('text', { x: margin.left - 9, y: y(1 / value) + 4, 'text-anchor': 'end' }, value.toFixed(1)));
        plot.append(svg('line', { x1: margin.left - 4, x2: margin.left, y1: y(1 / value), y2: y(1 / value), class: 'scatter-axis' }));
      }
      const guides = svg('g', { 'clip-path': `url(#${clipId})` });
      for (const speedup of data.isoSpeedups) {
        guides.append(svg('line', { x1: x(x0), x2: x(x1), y1: y(speedup / x0), y2: y(speedup / x1), class: speedup === 1 ? 'scatter-break-even' : 'scatter-guide' }));
        const label = isoLabel(speedup, config.xDomain, config.yDomain);
        if (!label) continue;
        const lx = x(label.x), ly = y(label.y);
        guides.append(svg('rect', { x: lx - 17, y: ly - 9, width: 34, height: 18, fill: '#fff', opacity: .96 }));
        guides.append(svg('text', { x: lx, y: ly + 4, 'text-anchor': 'middle', class: speedup === 1 ? 'scatter-break-label' : 'scatter-guide-label' }, `${speedup}×`));
      }
      plot.append(guides);
      plot.append(svg('path', { d: `M ${margin.left} ${margin.top} V ${height - margin.bottom} H ${width - margin.right}`, class: 'scatter-axis', fill: 'none' }));
      plot.append(svg('text', { x: (margin.left + width - margin.right) / 2, y: height - 6, 'text-anchor': 'middle', class: 'scatter-axis-title' }, '× fewer model calls'));
      plot.append(svg('text', { transform: `translate(13 ${(margin.top + height - margin.bottom) / 2}) rotate(-90)`, 'text-anchor': 'middle', class: 'scatter-axis-title' }, '× longer per call'));
      const points = [];
      rows.forEach((row, pointIndex) => {
        const point = svg('g', { transform: `translate(${x(row.call_reduction)} ${y(row.per_call_speed_ratio)})`, role: 'button', tabindex: pointIndex === 0 ? '0' : '-1', class: 'scatter-point', 'data-id': row.id, 'aria-label': `${row.model_label}, ${row.dataset}${row.note ? ', source version caveat' : ''}, vs. ${row.sampler}: ${fmt(row.call_reduction)} times fewer calls, ${fmt(row.time_per_call_ratio)} times the time per call, ${fmt(row.speedup)} times speedup.`, 'aria-pressed': 'false' });
        point.append(svg('circle', { r: 11, fill: 'transparent', class: 'scatter-hit' }), svg('circle', { r: 9, class: 'scatter-halo' }), marker(shapes.get(row.dataset), colors.get(row.model)));
        const select = () => {
          points.forEach((node) => node.setAttribute('tabindex', node === point ? '0' : '-1'));
          inspect(row, true);
        };
        point.addEventListener('pointerenter', () => inspect(row, true));
        point.addEventListener('pointerleave', () => { tooltip.hidden = true; });
        point.addEventListener('blur', () => { tooltip.hidden = true; });
        point.addEventListener('focus', select);
        point.addEventListener('click', select);
        point.addEventListener('keydown', (event) => {
          if (['ArrowLeft', 'ArrowUp', 'ArrowRight', 'ArrowDown', 'Home', 'End'].includes(event.key)) {
            event.preventDefault();
            const delta = ['ArrowLeft', 'ArrowUp'].includes(event.key) ? -1 : 1;
            const next = event.key === 'Home' ? 0 : event.key === 'End' ? points.length - 1 : (pointIndex + delta + points.length) % points.length;
            points[next].focus();
          } else if (event.key === 'Escape') { tooltip.hidden = true; } else if (['Enter', ' '].includes(event.key)) { event.preventDefault(); select(); }
        });
        points.push(point);
        plot.append(point);
      });
      if (!rows.length) {
        const note = svg('text', { x: (margin.left + width - margin.right) / 2, y: height / 2, 'text-anchor': 'middle', class: 'scatter-empty' }, 'No matching result');
        plot.append(note);
      }
      panel.append(plot);
    }
    if (filtered.length) inspect(filtered.find((row) => row.id === activeId) || filtered[0]);
    else { root.querySelector('#figure-inspector').textContent = 'No matching results. Try another model or dataset.'; root.querySelector('#figure-point-note').hidden = true; }
  }
  modelSelect.addEventListener('change', render);
  datasetSelect.addEventListener('change', render);
  root.querySelector('#figure-reset').addEventListener('click', () => { modelSelect.value = 'all'; datasetSelect.value = 'all'; render(); });
  render();
  return render;
}
