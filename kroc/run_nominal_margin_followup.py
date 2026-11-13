"""Two-cycle audit and upper-static sensitivity; reuse compilation, no reports."""
import argparse
import json
from pathlib import Path
import sys
from kroc.benchmark_nominal_margin_pressure import main as benchmark


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    stages=[('repeat5','kroc/results/nominal_margin5_geometry_screen_20261113',61,['.025','.05']),
            ('upper_slow','kroc/results/nominal_margin_geometry_screen_20261113',30,['.025','.05','.075','.10'])]
    (a.output/'plan.json').write_text(json.dumps({'seed':407,'period_s':8,'seconds':16,
        'headroom_enabled':False,'external_disturbance':False,'stages':stages,
        'purpose':'Distinguish first-cycle adjustment from repeated tracking pressure; isolate conservative upper-margin sensitivity.',
        'upper_slow_dynamic_range':[.025,.05],
        'upper_slow_is_sensitivity_not_matched_dynamic_upper_bound_comparison':True},indent=2))
    previous=sys.argv
    try:
        for name,screen,candidate,fractions in stages:
            sys.argv=['benchmark_nominal_margin_pressure','--screen',screen,'--output',str(a.output/name),
                '--candidates',str(candidate),'--seeds','407','--period','8','--seconds','16',
                '--static-fractions',*fractions]
            benchmark()
    finally:sys.argv=previous


if __name__=='__main__':main()
