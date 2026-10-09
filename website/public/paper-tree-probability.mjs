// Imported log ratios are stored on the accept child, where the log annotates the edge.
export function edgeProbability(record, childId) {
  const parent=Math.floor((childId-1)/2), raw=record.nodes[2*parent+1]?.log_acc_prob;
  if(raw==null)return null;
  const ratio=Number(raw);
  if(Number.isNaN(ratio))return null;
  const logP=Math.min(0,ratio);
  return childId%2 ? Math.exp(logP) : Math.max(0,-Math.expm1(logP));
}
export function formatProbability(p) {
  if(p==null)return 'unavailable';
  if(p===0||p===1)return String(p);
  if(p<0.001)return p.toExponential(2);
  if(p>0.999)return '1';
  return p.toFixed(3);
}
