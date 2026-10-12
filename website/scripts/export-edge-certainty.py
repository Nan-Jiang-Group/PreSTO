"""Export the source histograms used by the manuscript's main and Appendix E.4 figures."""
import csv, json, hashlib, re, subprocess
from collections import defaultdict
from pathlib import Path
ROOT = Path('/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies')
PAPER = Path('/Users/jiangnanhugo/overleaf papers/power_sampling paper/prefetching/exps/epsilon-certainty')
SPECS = [
 ('main', 'PreSTO-PowerMH', 'logs/all-datasets/2026-09-04/vllm/dataset-all.model-qwen3.5-9b.prefetch-budget-20.rank-bfs_accept_first.proposal-certainty.hatched.csv', 'dataset-all.model-qwen3.5-9b.prefetch-budget-20.rank-bfs_accept_first.proposal-certainty.hatched.pdf'),
 ('powerMH', 'PreSTO-PowerMH', 'logs/all-datasets/2026-09-24/vllm/dataset-all.model-all.prefetch-budget-20.rank-bfs_accept_first.all-edges.proposal-certainty.hatched.csv', 'dataset-all.model-all.prefetch-budget-20.rank-bfs_accept_first.all-edges.proposal-certainty.hatched.pdf'),
 ('entropyCut', 'PreSTO-EntropyCut', 'logs-entropycut/all-datasets/2026-09-23/vllm/dataset-all.model-all.prefetch-budget-20.rank-accept_first.cut-entropy4.0.all-edges.proposal-certainty.hatched.csv', 'entropycut/dataset-all.model-all.prefetch-budget-20.rank-accept_first.cut-entropy4.0.all-edges.proposal-certainty.hatched.pdf')]
MODELS = {'qwen3.5-9b':'Qwen3.5-9B','qwen3.5-4b':'Qwen3.5-4B','qwen3-8b':'Qwen3-8B','qwen3-4b':'Qwen3-4B','gemma-12b-it':'Gemma-4-12B-it'}
DATASETS = {'math500':'MATH500','aime':'AIME 24&25','mbpp':'MBPP','gpqa':'GPQA','human_eval':'HumanEval','mmlu':'MMLU','lcb_v6':'LCB V6'}
result = {'epsilon':.01, 'prefetchBudget':20, 'binCounts':[1,3,1], 'bucketDisplayWidths':[3.25,4.5,3.25], 'figures':{}}
for key, method, src, pdf in SPECS:
 groups=defaultdict(list)
 for row in csv.DictReader((ROOT/src).open()):
  assert row['status']=='complete'
  groups[(row['model'],row['dataset'])].append(row)
 panels=[]; labels=[]
 for (model,dataset), rows in groups.items():
  assert len(rows)==5
  count=int(rows[0]['tree_edges']); nodes=int(rows[0]['proposal_nodes'])
  bins=[{'left':float(r['probability_left']),'right':float(r['probability_right']),'count':int(r['edge_count']),'share':float(r['edge_share'])} for r in rows]
  shares=[float(rows[i]['bucket_share']) for i in [0,1,4]]
  assert sum(b['count'] for b in bins)==count==2*nodes
  assert abs(shares[0]-shares[2])<1e-6
  labels.extend(round(s*100) for s in shares)
  panels.append({'model':model,'modelLabel':MODELS[model],'dataset':dataset,'datasetLabel':DATASETS[dataset],'edgeCount':count,'proposalNodes':nodes,'bucketShares':shares,'bins':bins})
 # All 189 printed category percentages must match the actual manuscript PDFs.
 pdf_text=subprocess.check_output(['pdftotext','-layout',str(PAPER/pdf),'-'],text=True)
 assert labels==[int(x) for x in re.findall(r'(\d+)%',pdf_text)], key
 result['figures'][key]={'method':method,'sourceCsv':src,'sourceSha256':hashlib.sha256((ROOT/src).read_bytes()).hexdigest(),'paperFigure':str(pdf),'panels':panels}
Path('public/data/edge-certainty.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
print('Exported 63 histograms; all 189 category percentages match the manuscript figures.')
