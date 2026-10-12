export function logPosition(value, domain, range) {
  return range[0] + Math.log(value / domain[0]) / Math.log(domain[1] / domain[0]) * (range[1] - range[0]);
}

export function ticks(domain, steps, minimum) {
  for (const step of steps) {
    const values = [];
    for (let value = Math.ceil(domain[0] / step - 1e-9) * step; value <= domain[1] + 1e-9; value += step) values.push(Number(value.toFixed(3)));
    if (values.length >= minimum) return values;
  }
  return domain;
}

// Match the paper's placement where a speedup line meets the rising panel diagonal.
export function isoLabel(speedup, xDomain, yDomain) {
  const [x0, x1] = xDomain.map(Math.log);
  const [y0, y1] = yDomain.map(Math.log);
  const lx = ((x1 - x0) * (Math.log(speedup) - y0) + (y1 - y0) * x0) / (x1 - x0 + y1 - y0);
  const fraction = (lx - x0) / (x1 - x0);
  return fraction >= .04 && fraction <= .96 ? { x: Math.exp(lx), y: speedup / Math.exp(lx) } : null;
}
