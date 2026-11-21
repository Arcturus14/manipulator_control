"""Final bounded transition search after valid J5 -12Nm had .109rad slack.

The first qualifying PILOT is frozen for heldout seeds. No heldout retuning.
"""
from pathlib import Path
import json
import hashlib
from kroc import search_n0_other_joint_estimator_margin as study
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight


def main():
    study.ROOT=Path('kroc/results/n0_other_joint_estimator_margin_20261118/final_refinement')
    study.ROOT.mkdir(parents=True,exist_ok=True)
    scenes=[dict(joint=4,radius=.04,amplitudes=[13.,14.,15.]),
            dict(joint=0,radius=.22,amplitudes=[10.,20.,40.]),
            dict(joint=3,radius=.04,amplitudes=[4.,8.,12.])]
    study.SCENES=scenes;valid=study.prefilter()
    cuda=cuda_preflight();print('PREFLIGHT',cuda,flush=True)
    study.write(study.ROOT/'plan.json',dict(scenes=scenes,pilot=study.PILOT,heldout=study.HELDOUT,
        max_pilot_new_runs=14,max_validation_new_runs=10,
        refinement='up to3 bisections after first eligible static failure; only pilot chooses scenario',
        fallback='stop without any parameter changes if no eligible separation',
        no_output_or_rollout_compensation=True,no_headroom=True,
        frozen_n0=True,margin_fractions=[.025,.05],cuda=cuda,
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            [__file__,study.__file__,'kroc/compare_n0_state_margins.py']}))
    history=[];candidate=None;chosen=None
    for scene in valid:
        if study.RUN_COUNT>=12:break
        # Prior J5/J4 nominal controls are reusable; reference and q0 unchanged.
        if scene['joint']==4:
            base=Path('kroc/results/n0_other_joint_estimator_margin_20261118/large_circle_j1/pilot/j5_a12_static25_seed8607.json')
            previous=12.;previous_row=json.loads(base.read_text())
        elif scene['joint']==3:
            base=Path('kroc/results/n0_other_joint_estimator_margin_20261118/pilot/j4_a0_static25_seed8607.json')
            previous=0.;previous_row=json.loads(base.read_text())
        else:
            previous_row=study.run(scene,0.,'static25');history.append(previous_row);previous=0.
        if not previous_row['eligible'] or previous_row['metrics']['rmse_mm']>20.:continue
        bracket=None
        for amplitude in scene['amplitudes']:
            if study.RUN_COUNT>=13:break
            low=study.run(scene,amplitude,'static25');history.append(low)
            if study.static_failure(low):
                dyn=study.run(scene,amplitude,'dynamic_estimate');history.append(dyn)
                if study.separation(low,dyn):candidate=[low,dyn];chosen=scene;break
                bracket=(previous,amplitude);break
            if low['eligible']:previous=amplitude
            study.write(study.ROOT/'progress.json',dict(rows=history,new_runs=study.RUN_COUNT))
        if candidate:break
        if bracket:
            left,right=bracket
            for _ in range(3):
                if study.RUN_COUNT>=13:break
                amplitude=round((left+right)/2,6)
                low=study.run(scene,amplitude,'static25');history.append(low)
                if not low['eligible']:break
                if study.static_failure(low):
                    right=amplitude;dyn=study.run(scene,amplitude,'dynamic_estimate');history.append(dyn)
                    if study.separation(low,dyn):candidate=[low,dyn];chosen=scene;break
                else:left=amplitude
                study.write(study.ROOT/'progress.json',dict(rows=history,new_runs=study.RUN_COUNT))
        if candidate:break
    validation=[];upper=None
    if candidate:
        amplitude=candidate[0]['amplitude_nm'];upper=study.run(chosen,amplitude,'static5')
        if upper['eligible']:study.plot(candidate+[upper],chosen,'pilot_selected')
        for seed in study.HELDOUT:
            rows=[study.run(chosen,amplitude,c,seed,'validation') for c in study.CASES]
            for key in ['circle_reference','d_true','initial_q','sensor_noise_seed']:
                import numpy as np
                with np.load(rows[0]['path']) as a:
                    common=a[key].copy()
                for row in rows[1:]:
                    with np.load(row['path']) as b:np.testing.assert_array_equal(common,b[key])
            pair=dict(seed=seed,primary=study.separation(rows[0],rows[1]),
                secondary=bool(rows[2]['eligible'] and rows[2]['metrics']['rmse_mm']>1.05*rows[1]['metrics']['rmse_mm']),rows=rows)
            validation.append(pair)
            if all(r['eligible'] for r in rows):study.plot(rows,chosen,f'validation_seed{seed}')
            study.write(study.ROOT/'validation_progress.json',validation)
    study.write(study.ROOT/'search_audit.json',dict(rows=history,validation=validation))
    result=dict(status='validated' if validation and all(p['primary'] for p in validation)
        else 'pilot_not_fully_reproduced' if candidate else 'stopped_no_qualifying_case',
        selected_scene=chosen,selected_amplitude_nm=None if not candidate else candidate[0]['amplitude_nm'],
        valid_pilot_rows=[r for r in history if r['eligible']],validation=validation,
        static5_pilot=upper if upper and upper['eligible'] else None,
        new_gpu_runs=study.RUN_COUNT,search_stopped=True,
        scope='No post-DOB compensation; cannot establish failed compensation due to saturation.')
    study.write(study.ROOT/'verdict.json',result)
    print('DONE',result['status'],study.RUN_COUNT,flush=True)


if __name__=='__main__':main()
