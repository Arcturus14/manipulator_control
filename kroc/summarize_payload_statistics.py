"""Paired statistics and backend equivalence of the 168-run validations.

 python -m kroc.summarize_payload_statistics            # from the tracked per-run values
 python -m kroc.summarize_payload_statistics --import-results   # refresh them from kroc/results
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from kroc.evidence_common import EVIDENCE,write_json

ROOT=EVIDENCE/'validation_20261122'
SOURCES=dict(captured=Path('kroc/results/aligned_circle_20261122/summary.json'),
             kernel=Path('kroc/results/aligned_circle_kernel_20261122/summary.json'))


def holm(p):
    order=np.argsort(p);adjusted=np.empty(len(p));running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(p)-rank)*p[index]);adjusted[index]=min(running,1.)
    return adjusted


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--import-results',action='store_true')
    if p.parse_args().import_results:
        for backend,path in SOURCES.items():
            write_json(ROOT/f'runs_{backend}.json',json.loads(path.read_text())['runs'])
    runs={b:json.loads((ROOT/f'runs_{b}.json').read_text()) for b in SOURCES}
    key=lambda r:(r['name'],r['case'],r['seed'])
    assert [key(r) for r in runs['captured']]==[key(r) for r in runs['kernel']]

    def values(name,case='N0',field='eval_rmse_mm'):
        rows=sorted((r for r in runs['captured'] if r['name']==name and r['case']==case),key=lambda r:r['seed'])
        assert len(rows)==10
        return np.array([r[field] for r in rows])

    # One family of 21 comparisons, the one reported in README 10.3.
    pairs=[('original DOB on - off',values('original_dob_on'),values('original_dob_off')),
           ('aligned DOB on - off',values('aligned_dob_on'),values('aligned_dob_off')),
           ('DOB off original - aligned',values('original_dob_off'),values('aligned_dob_off'))]
    for name,base in (('original_45','original_dob_on'),('aligned_45','aligned_dob_on'),('aligned_47p5','aligned_dob_on')):
        n0,n1,d0=(values(name,c) for c in ('N0','N1','D0'))
        pairs+=[(f'{name} N0 - no disturbance',n0,values(base)),(f'{name} N0 - N1',n0,n1),
                (f'{name} N0 - D0',n0,d0),(f'{name} N1 - D0',n1,d0)]
    for name in ('aligned_45','aligned_47p5'):
        n0,n1,d0=(values(name,c,'eval_max_mm') for c in ('N0','N1','D0'))
        pairs+=[(f'{name} max N0 - N1',n0,n1),(f'{name} max N0 - D0',n0,d0),(f'{name} max N1 - D0',n1,d0)]
    raw=np.array([stats.wilcoxon(x,y).pvalue for _,x,y in pairs]);adjusted=holm(raw)
    rows=[]
    for (name,x,y),p_raw,p_holm in zip(pairs,raw,adjusted):
        d=x-y;half=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
        rows.append(dict(comparison=name,mean_difference_mm=float(d.mean()),ci95_half_width_mm=float(half),
            positive_seeds=int((d>0).sum()),wilcoxon_p=float(p_raw),holm_p=float(p_holm)))
        print(f'{name:38s} {d.mean():7.3f} +- {half:.3f}  {int((d>0).sum())}/10  Holm p={p_holm:.4f}')
    difference={field:float(max(abs(a[field]-b[field]) for a,b in zip(runs['captured'],runs['kernel'])))
                for field in ('eval_rmse_mm','eval_max_mm','q3_min_rad')}
    write_json(ROOT/'statistics.json',dict(family_size=len(rows),test='two-sided Wilcoxon signed-rank, Holm',
        comparisons=rows,backend_max_abs_difference=difference,
        valid=dict((b,f"{sum(r['valid'] for r in v)}/{len(v)}") for b,v in runs.items())))
    print('kernel vs captured, per-run max difference:',difference)


if __name__=='__main__':main()
