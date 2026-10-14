"""Paired GPU experiment: post-DOB, model-only, and nominal-shift MPPI."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_dob_aware_nominal_shift import cuda_preflight, disturbance_profiles, metrics
from experiment_runner import run_mppi, waypoints
from utils.result_cache import environment_spec, source_fingerprint

MODES = {"post_dob": "output", "model_only": "rollout_model_only",
         "nominal_shift": "rollout_nominal_shift"}
SEEDS = [7, 17, 27]
STEPS = 250


def main():
    preflight = cuda_preflight()
    print("CUDA preflight:", preflight, flush=True)
    spec = {"steps": STEPS, "seeds": SEEDS, "modes": MODES,
            "profiles": ["constant", "step", "slow_sinusoidal", "fast_sinusoidal", "large_step"],
            "large_step_scale": 8.0, "sampling_center": "feedforward_residual",
            "environment": environment_spec(),
            "source": source_fingerprint(Path.cwd(), "run", {}),
            "benchmark_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    output = Path("results/model_only_rollout_gpu") / fingerprint[:12]
    if all((output / name).exists() for name in ("metrics.json", "results.npz", "REPORT.md", "comparison.png")):
        print("REUSE:", output)
        return
    profiles = disturbance_profiles(STEPS, large_step_scale=8.0)
    reference = waypoints("circle", n_steps=STEPS)
    rows, payload, representative = [], {}, {}
    for pi, (profile, disturbance) in enumerate(profiles.items()):
        for si, seed in enumerate(SEEDS):
            order = list(MODES)
            if (pi + si) % 2:
                order.reverse()
            for mode in order:
                # Fresh device check for every run; no CPU fallback.
                cuda_preflight()
                print(f"RUN {profile} seed={seed} {mode}", flush=True)
                data = run_mppi(reference, has_disturbance=True, use_dob=True,
                                dob_compensation_mode=MODES[mode],
                                sampling_center_mode="feedforward_residual",
                                disturbance_function=disturbance, require_cuda=True, seed=seed)
                assert str(data["compute_device"]) == "cuda"
                estimate = data["d_hat"][:-1]
                learned = data["u_mppi"] - data["u_ff"]
                compensation_error = learned - estimate
                predicted = data["u_mppi"] - (estimate if mode != "post_dob" else 0)
                np.testing.assert_allclose(data["u_effective_pred"], predicted, atol=1e-5)
                if mode != "post_dob":
                    np.testing.assert_allclose(data["u_applied"], data["u_mppi"], atol=1e-5)
                if mode == "model_only":
                    np.testing.assert_array_equal(data["u_sampling_center"][0], data["u_ff"][0])
                    np.testing.assert_allclose(data["r_star"], learned, atol=1e-5)
                m = metrics(data)
                m["learned_compensation_error_rms_nm"] = float(np.sqrt(np.mean(compensation_error**2)))
                m["prediction_q_rmse_rad"] = float(np.sqrt(np.mean(data["state_prediction_error"][:, :6]**2)))
                m["prediction_qd_rmse_rad_s"] = float(np.sqrt(np.mean(data["state_prediction_error"][:, 6:]**2)))
                # Residual includes tracking correction, not exclusively disturbance compensation.
                row = {"profile": profile, "seed": seed, "mode": mode, "metrics": m}
                rows.append(row)
                print(json.dumps(row), flush=True)
                prefix = f"{profile}__{mode}__seed_{seed}__"
                for key, value in data.items():
                    if not key.startswith("_"):
                        payload[prefix + key] = value
                payload[prefix + "learned_compensation"] = learned
                payload[prefix + "compensation_error"] = compensation_error
                if seed == SEEDS[0]:
                    representative[(profile, mode)] = data["err_mm"]
                del data
                gc.collect()
                torch.cuda.empty_cache()
    summary = {}
    for profile in profiles:
        summary[profile] = {}
        for mode in MODES:
            selected = [r["metrics"] for r in rows if r["profile"] == profile and r["mode"] == mode]
            summary[profile][mode] = {key: {"mean": float(np.mean([m[key] for m in selected])),
                                                  "std": float(np.std([m[key] for m in selected], ddof=1))}
                                      for key in selected[0] if selected[0][key] is not None
                                      and all(m[key] is not None for m in selected)}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "results.npz", **payload)
    (output / "metrics.json").write_text(json.dumps({"fingerprint": fingerprint, "spec": spec,
        "cuda_preflight": preflight, "rows": rows, "summary": summary}, indent=2))
    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
    for ax, profile in zip(axes, profiles):
        for mode in MODES:
            ax.plot(np.arange(STEPS)*0.02, representative[(profile, mode)], label=mode)
        ax.set_title(profile)
        ax.set_ylabel("EE error [mm]")
        ax.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    lines = ["# Model-only disturbance-aware MPPI GPU 비교", "",
             "원궤적 5초, DT=0.02 s, K=136, H=10, seeds=7/17/27. 파라미터 튜닝 없이 45회 비교.",
             f"GPU: {preflight['gpu_name']}, PyTorch {preflight['torch']}, CUDA {preflight['cuda_runtime']}.", "",
             "```text", "공통: epsilon ~ N(0,Sigma), U_sample = U_center + epsilon",
             "Post-DOB: U_center=U_ff+R, rollout=clamp(U_sample), output=clamp(u_mppi+d_hat)",
             "Model-only: U_center=U_ff+R, rollout=clamp(U_sample)-d_hat, output=clamp(u_mppi)",
             "Nominal-shift: U_center=U_ff+R+d_hat, rollout=clamp(U_sample)-d_hat, output=clamp(u_mppi)",
             "```", "",
             "d_hat은 solve 진입 시 snapshot하여 horizon 전체에 ZOH. 추정은 plant 진행 후 갱신한다.",
             "Model-only residual은 U_star-U_ff로 복원하고 shift한다. 이전 solve가 찾은 보상 성분은 유지한다.",
             "외란 차감 후 effective torque는 다시 clamp하지 않는다. Large-step은 8×D_AMP이며 t=1.24 s에 시작한다.",
             "GUI/main 기본값은 기존 Post-DOB. rollout_model_only는 rollout의 호환 alias다.", "",
             "| 외란 | 모드 | RMSE [mm, 평균±SD] | Torque-rate RMS [Nm/step] | 포화 steps/250 | Mean solve [ms] |",
             "|---|---|---:|---:|---:|---:|"]
    for profile in profiles:
        for mode in MODES:
            s = summary[profile][mode]
            e = s["tracking_rmse_mm"]
            lines.append(f"| {profile} | {mode} | {e['mean']:.3f} ± {e['std']:.3f} | "
                         f"{s['torque_rate_rms_nm_per_step']['mean']:.3f} | {s['saturated_steps']['mean']:.1f} | "
                         f"{s['solve_mean_ms']['mean']:.3f} |")
    lines += ["", "Model-only의 Post-DOB 대비 RMSE 변화:", ""]
    for profile in profiles:
        a = summary[profile]["model_only"]["tracking_rmse_mm"]["mean"]
        b = summary[profile]["post_dob"]["tracking_rmse_mm"]["mean"]
        lines.append(f"- {profile}: {(a/b-1)*100:+.2f}%")
    lines += ["", "`learned_compensation=U_mppi-U_ff`와 `compensation_error=learned_compensation-d_hat`를 NPZ에 저장했다.",
              "이 residual에는 궤적 추종 보정도 포함되므로 compensation_error=0이 최적성의 필요조건은 아니다. Post-DOB에서는 외부 보상이 별도라 동일 지표를 직접 순위화하면 안 된다.",
              "One-step state prediction error는 실제 RK4 plant와 rollout integrator의 차이도 포함한다. 추정 오차만 측정하는 지표가 아니다.",
              "입력 비용은 actuator command를 평가한다. Post-DOB cost에는 후단 보상량이 포함되지 않는 기존 구조를 유지했다.",
              "GPU preflight, CUDA metadata, no-post-compensation 및 residual 항등식을 각 run에서 검증했다. 동일 fingerprint 결과는 재사용한다.",
              "joint position/velocity limit, workspace limit, self-collision penalty는 후속 작업이며 이번 실험에는 추가하지 않았다.", "",
              "실행: `/home/arclab-04/miniconda3/envs/manipulator/bin/python -u benchmark_model_only_mppi.py`", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
