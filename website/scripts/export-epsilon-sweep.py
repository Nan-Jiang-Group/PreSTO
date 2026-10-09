"""Run with the experiment repository venv to export Case 2 epsilon snapshots.

Reads original per-proposal logs, never interpolates aggregate histogram bins.
"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path('/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening')
sys.path.insert(0, str(ROOT))
from case_studies.extract.logs.proposal_log import parse_log
BASE = Path(__file__).resolve().parents[1]
data = json.loads((BASE/'public/data/edge-certainty.json').read_text())
epsilons = [float(f'{10 ** (-3 + i / 40):.12g}') for i in range(81)]
result = {'epsilons': epsilons, 'defaultIndex':40, 'panels':{}}
for kind, figure in data['figures'].items():
    rows = list(csv.DictReader((ROOT/'case_studies'/figure['sourceCsv']).open()))
    for panel in figure['panels']:
        row = next(r for r in rows if r['model']==panel['model'] and r['dataset']==panel['dataset'])
        path = Path(row['log_path'])
        assert 'INFO: output saved to' in path.read_text()
        parsed = parse_log(path)
        assert len(parsed.nodes)==panel['proposalNodes']
        assert all({2*n.node_id+1,2*n.node_id+2} <= parsed.tree_node_ids[n.tree_index] for n in parsed.nodes)
        assert sum(len(ids)-1 for ids in parsed.tree_node_ids.values())==panel['edgeCount']
        logs = np.array([n.log_a for n in parsed.nodes])
        probabilities = np.concatenate((np.exp(logs), -np.expm1(logs)))
        counts = []
        for epsilon in epsilons:
            low = int(np.count_nonzero(probabilities<=epsilon))
            high = int(np.count_nonzero(probabilities>=1-epsilon))
            middle = probabilities[(probabilities>epsilon)&(probabilities<1-epsilon)]
            boundaries = np.geomspace(epsilon,1-epsilon,4)
            bins = [low, *np.histogram(middle,bins=boundaries)[0].tolist(), high]
            assert sum(bins)==panel['edgeCount'] and low==high
            counts.append(bins)
        assert counts[40]==[b['count'] for b in panel['bins']]
        result['panels'][f"{kind}:{panel['model']}:{panel['dataset']}"] = {'counts':counts,'sourceLog':str(path.relative_to(ROOT)), 'sourceSha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(BASE/'public/data/epsilon-sweep.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
print(f"Exported {len(result['panels'])} panels at {len(epsilons)} epsilon values; default counts match every original histogram.")
