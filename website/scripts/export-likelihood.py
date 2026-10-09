"""Export Appendix E.2 box plots using the paper's pairing and scoring code."""
import csv, json, sys
from pathlib import Path
root = Path('/Users/jiangnanhugo/workspace/power-sharpening/Subtree-Prefetching-Power-Sharpening/case_studies')
sys.path.insert(0, str(root / 'draw'))
from draw_likelihood_confidence_boxes import load_pairs, DATASET_DISPLAY, MODEL_LABELS
from draw_rescored_likelihood_and_confidence_summary import LAUNCHER, launcher_jobs, rescoring_agreement, DEFAULT_OUTPUT_DIR
from matplotlib.cbook import boxplot_stats
jobs = launcher_jobs(LAUNCHER)
agreement = rescoring_agreement(jobs, .01)
with (DEFAULT_OUTPUT_DIR / 'likelihood-confidence-summary.csv').open() as f:
    summary = {(r['family'], (r['dataset'], r['model']), r['metric']): r for r in csv.DictReader(f) if r['p_equiv']}
pvalues = {k: float(r['p_equiv']) for k, r in summary.items()}
families = []
for family, baseline in [('uniform', 'PowerMH'), ('entropy', 'EntropyCut')]:
    rows = []
    for (dataset, model), groups in load_pairs(family, ['all'], jobs, agreement, pvalues):
        metrics = {}
        for metric in ['log_likelihood', 'confidence']:
            result = summary[(family, (dataset, model), metric)]
            boxes = {}
            for method, records in zip([baseline, 'PreSTO'], groups.values()):
                values = [float(r[metric]) for r in records]
                stats = boxplot_stats(values)[0]
                boxes[method] = dict(n=len(values), samples=values, **{k: float(stats[k]) for k in ['med','q1','q3','whislo','whishi']}, fliers=stats['fliers'].tolist())
            metrics[metric] = dict(pEquiv=float(result['p_equiv']), ci90=[float(result['ci90_low']),float(result['ci90_high'])], margin=float(result['margin']), groups=boxes)
        rows.append(dict(dataset=DATASET_DISPLAY.get(dataset,dataset),model=MODEL_LABELS.get(model,model),metrics=metrics))
    families.append(dict(baseline=baseline, rows=rows))
assert [len(f['rows']) for f in families] == [8,9]
assert [sum(m['pEquiv'] < .05 for r in f['rows'] for m in r['metrics'].values()) for f in families] == [12,17]
path=Path('public/data/case-studies.json')
data=json.loads(path.read_text());data['case3']=dict(marginBaselineSD=.2, families=families)
path.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
print('Exported 8 PowerMH and 9 EntropyCut pairs; 12/16 and 17/18 equivalent tests.')
