export const RULES = [
  { id: 'bfs', name: 'BFS default Traversal', detail: 'Shallowest frontier first; accept before reject at equal depth.' },
  { id: 'reject', name: 'Reject-first', detail: 'Depth-first traversal, exploring the reject branch before the accept branch.' }
];
export const PRESET = { depth: 5, budget: 8, model: 'step', probability: 0.5, seed: 17, pathSeed: 17, cutGate: true, rule: 'bfs' };
export function random(seed) {
  let x = seed >>> 0;
  return () => { x += 0x6D2B79F5; let t = x; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
export function sanitize(input = {}) {
  const number = (x, fallback, low, high, round = false) => { const n = Number(x); const v = Number.isFinite(n) ? Math.max(low, Math.min(high, n)) : fallback; return round ? Math.round(v) : v; };
  const c = { ...PRESET, ...input };
  c.depth = number(c.depth, 5, 2, 7, true);
  c.budget = number(c.budget, 8, 1, Math.min(64, 2 ** c.depth - 1), true);
  c.probability = number(c.probability, .5, 0, 1);
  c.seed = number(c.seed, 17, 0, 999999, true);
  c.pathSeed = number(c.pathSeed, 17, 0, 999999, true);
  c.model = c.model === 'uniform' ? 'uniform' : 'step';
  c.rule = RULES.some(r => r.id === c.rule) ? c.rule : 'bfs';
  c.cutGate = typeof c.cutGate === 'boolean' ? c.cutGate : true;
  return c;
}
export function buildTree(config) {
  const c = sanitize(config);
  const nodes = [];
  const byId = new Map();
  for (let id = 1; id < 2 ** (c.depth + 1); id++) {
    const level = Math.floor(Math.log2(id));
    const terminal = level === c.depth;
    const rng = random(c.seed * 10007 + id * 7919);
    const bucket = rng();
    let p = c.probability, category = 'Uniform';
    if (c.model === 'step') {
      if (bucket < .41) { p = 0; category = 'Certain reject'; }
      else if (bucket < .50) { p = .005; category = 'Near reject'; }
      else if (bucket < .67) { p = .1 + .8 * rng(); category = 'Stochastic'; }
      else if (bucket < .68) { p = .995; category = 'Near accept'; }
      else { p = 1; category = 'Certain accept'; }
    }
    const cut = id === 1 ? 96 : 1 + Math.floor(rng() * 127);
    const parent = byId.get(Math.floor(id / 2));
    const branch = id === 1 ? null : id % 2 === 0 ? 'A' : 'R';
    const localEligible = !parent || !c.cutGate || branch === 'R' || cut <= parent.cut;
    const node = { id, level, terminal, p, cut, category, parentId: parent?.id ?? null, branch,
      path: parent ? parent.path + branch : '',
      reach: parent ? parent.reach * (branch === 'A' ? parent.p : 1 - parent.p) : 1,
      eligible: !terminal && localEligible && (parent?.eligible ?? true), localEligible,
      delta: parent ? Math.abs(cut - parent.cut) : 0,
      u: random(c.pathSeed * 10301 + c.seed * 101 + id * 6271)()
    };
    nodes.push(node); byId.set(id, node);
  }
  return { nodes, byId, config: c };
}
const compareRejectFirst = (a, b) => {
  const key = n => n.path.replaceAll('A', '1').replaceAll('R', '0');
  return key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : a.id - b.id;
};
export function schedule(tree, rule, budget) {
  let frontier = [tree.byId.get(1)];
  const order = [];
  const cmp = {
    bfs: (a, b) => a.id - b.id,
    reject: compareRejectFirst
  }[rule] ?? ((a,b) => a.id-b.id);
  while (frontier.length && order.length < budget) {
    frontier.sort(cmp);
    const n = frontier.shift(); order.push(n.id);
    for (const id of [2 * n.id, 2 * n.id + 1]) {
      const child = tree.byId.get(id);
      if (child && child.eligible) frontier.push(child);
    }
  }
  return order;
}
export function realize(tree) {
  const decisions = [];
  let node = tree.byId.get(1);
  while (!node.terminal) {
    const accepted = node.u < node.p;
    decisions.push({ id: node.id, accepted, childId: 2 * node.id + (accepted ? 0 : 1), p: node.p, u: node.u });
    node = tree.byId.get(decisions.at(-1).childId);
  }
  return { decisions, terminalId: node.id };
}
export function evaluate(tree, order, trace = realize(tree)) {
  const set = new Set(order);
  let effective = 0;
  for (const d of trace.decisions) { if (!set.has(d.id)) break; effective++; }
  const used = order.length;
  return { used, effective, wasted: used - effective, utilization: used ? effective / used : 0,
    expected: order.reduce((sum, id) => sum + tree.byId.get(id).reach, 0),
    accepted: trace.decisions.slice(0, effective).filter(d => d.accepted).length,
    eligible: tree.nodes.filter(n => n.eligible).length };
}
export function compare(config) {
  const tree = buildTree(config), trace = realize(tree);
  return { tree, trace, results: RULES.map(rule => { const order = schedule(tree, rule.id, tree.config.budget); return { ...rule, order, ...evaluate(tree, order, trace) }; }) };
}
