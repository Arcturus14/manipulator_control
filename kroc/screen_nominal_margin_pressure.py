"""Offline bounded-IK/mesh prefilter for conservative-margin tracking pressure.

Local IK failure is not a certificate of global task infeasibility. Actual
tracking comparisons must use CUDA MPPI, with unchanged Cartesian-only costs.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from config import DH_PARAMS, DT, Q_MAX, Q_MIN
from kinematics.ur5_kinematics import fk_joints_np
from kroc.collision_geometry import Geometry


def bounded_ik(targets, initial, fraction=0.):
    span = np.asarray(Q_MAX)-np.asarray(Q_MIN)
    lower, upper = np.asarray(Q_MIN)+fraction*span, np.asarray(Q_MAX)-fraction*span
    q = np.clip(np.asarray(initial, dtype=float), lower, upper)
    result, errors = [], []
    for target in targets:
        for _ in range(90):
            tr = np.eye(4); origins = []; axes = []
            for i, (aa, dd, alpha, offset) in enumerate(DH_PARAMS):
                origins.append(tr[:3, 3].copy()); axes.append(tr[:3, 2].copy())
                ct, st = np.cos(q[i]+offset), np.sin(q[i]+offset)
                ca, sa = np.cos(alpha), np.sin(alpha)
                tr = tr @ np.array([[ct,-st*ca,st*sa,aa*ct], [st,ct*ca,-ct*sa,aa*st],
                                    [0,sa,ca,dd], [0,0,0,1]])
            err = target-tr[:3, 3]
            if np.linalg.norm(err) < 2e-6:
                break
            jac = np.stack([np.cross(ax, tr[:3, 3]-o) for ax, o in zip(axes, origins)], axis=1)
            # Remove saturated variables only when the proposed motion points outward.
            delta = jac.T @ np.linalg.solve(jac @ jac.T+1e-5*np.eye(3), err)
            blocked = ((q <= lower+1e-8) & (delta < 0)) | ((q >= upper-1e-8) & (delta > 0))
            if blocked.any():
                free_jac = jac.copy(); free_jac[:, blocked] = 0
                delta = free_jac.T @ np.linalg.solve(free_jac @ free_jac.T+1e-5*np.eye(3), err)
            q = np.clip(q+.8*np.clip(delta, -.05, .05), lower, upper)
        result.append(q.copy())
        errors.append(np.linalg.norm(fk_joints_np(q)[-1]-target))
    return np.array(result), np.array(errors)


def circle(q, ratio, plane='xy', period=4., seconds=8.):
    start = fk_joints_np(q)[-1]
    radial = np.r_[start[:2], 0.]; radial /= np.linalg.norm(radial)
    tangent = np.array([-radial[1], radial[0], 0.])
    radius = ratio*np.linalg.norm(start[:2])
    # Horizontal circle moves toward the base and then back; vertical variant
    # replaces the radial axis but keeps the initial tangent continuous.
    axis = radial if plane == 'xy' else np.array([0., 0., 1.])
    theta = np.arange(round(seconds/DT))*DT*2*np.pi/period
    ref = start+radius*((np.cos(theta)-1)[:, None]*axis+np.sin(theta)[:, None]*tangent)
    return ref, radius


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-fraction', type=float, default=.78)
    p.add_argument('--ratios', type=float, nargs='+', default=[.35,.40,.45])
    p.add_argument('--margins', type=float, nargs='+', default=[.025,.05,.075,.10])
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    geom = Geometry(); rows = []; selected = []
    # Every starting posture is strictly inside the largest tested box.
    postures = list(itertools.product([-1.8,-1.4,-1.0], [.6,1.2], [-1.5,-.5]))
    for pose, (q2,q3,q4) in enumerate(postures):
        q0 = np.array([a.base_fraction*Q_MAX[0], q2,q3,q4,-1.57,0.])
        span=np.asarray(Q_MAX)-np.asarray(Q_MIN)
        assert np.minimum(q0-np.asarray(Q_MIN)-max(a.margins)*span,
                          np.asarray(Q_MAX)-max(a.margins)*span-q0).min()>0
        initial = geom.check(q0)
        if initial['hit_pairs'] or initial['floor_clearance_m'] < .015:
            rows.append({'pose':pose,'rejected':'initial collision/clearance'}); continue
        for plane, ratio in itertools.product(['xy','vertical_tangent'], a.ratios):
            ref, radius = circle(q0, ratio, plane)
            sample = ref[::5]  # 100 ms screen; accepted actual trajectories need 5 ms mesh audit.
            physical, physical_error = bounded_ik(sample, q0)
            record = {'pose':pose,'plane':plane,'ratio':ratio,'radius':radius,'q0':q0.tolist(),
                      'physical_ik_max_error_mm':float(physical_error.max()*1000)}
            if physical_error.max() > 1e-4:
                rows.append({**record,'rejected':'physical local IK error'}); continue
            check = geom.trajectory(physical, stop_on_collision=True)
            record['physical_ik_mesh'] = check
            if check['first_collision_index'] is not None or check['floor_min_m'] < .01:
                rows.append({**record,'rejected':'physical IK collision/clearance'}); continue
            errors = {}
            for fraction in a.margins:
                _, err = bounded_ik(sample, q0, fraction)
                errors[str(fraction)] = {'rmse_mm':float(np.sqrt(np.mean(err**2))*1000),
                                         'max_mm':float(err.max()*1000)}
            record['bounded_local_ik'] = errors
            record['pressure5_mm'] = errors['0.05']['rmse_mm']-errors['0.025']['rmse_mm']
            record['pressure10_mm'] = errors['0.1']['rmse_mm']-errors['0.025']['rmse_mm'] if '0.1' in errors else None
            record['candidate'] = len(selected)
            np.savez_compressed(a.output/f'candidate_{len(selected)}.npz', reference=ref, initial_q=q0)
            selected.append(record); rows.append(record)
            print(json.dumps(record), flush=True)
        (a.output/'screen.json').write_text(json.dumps({'rows':rows,'selected':selected,
            'headroom_enabled':False,'no_external_disturbance':True,
            'arguments':vars(a)|{'output':str(a.output)},
            'local_ik_is_not_global_infeasibility_certificate':True}, indent=2))
    print('SCREEN_COMPLETE', len(selected), '/', len(postures)*6, flush=True)


if __name__ == '__main__':
    main()
