"""Profile MPPI-only CUDA work after compilation and report kernel costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from controllers.mppi_dob_controller import UR5Tracker
from experiment_runner import initial_configuration, waypoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solves", type=int, default=400)
    parser.add_argument("--profile-solves", type=int, default=10)
    parser.add_argument("--compile-rollout-loop", action="store_true")
    parser.add_argument("--pre-benchmark-pause", type=float, default=0.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.solves, args.profile_solves) < 1:
        parser.error("solve counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    reference = waypoints("circle", n_steps=64)
    tracker = UR5Tracker(
        reference, device, dtype=torch.float32, n_samples=136, horizon=10,
        temperature=1., temporal_noise_rho=.8,
        position_weight=50000., terminal_weight=50000.,
        joint_position_limit_weight=50000.,
        joint_velocity_limit_weight=50000.,
        compile_rollout_loop=args.compile_rollout_loop,
    )
    state = torch.cat((initial_configuration(reference, device),
                       torch.zeros(6, device=device)))
    feedforward = tracker._feedforward_torque(state[:6])
    tracker._warm_up_controller(state, feedforward, "feedforward_residual")
    tracker.ctrl.U.copy_(feedforward[:tracker.horizon])
    for _ in range(5):
        tracker._mppi_command(state)
    torch.cuda.synchronize()
    print("READY_FOR_UTILIZATION", flush=True)
    if args.pre_benchmark_pause > 0:
        time.sleep(args.pre_benchmark_pause)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.solves):
        tracker._mppi_command(state)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    throughput_ms = elapsed * 1e3 / args.solves

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(args.profile_solves):
            tracker._mppi_command(state)
        torch.cuda.synchronize()

    summary = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "samples": tracker.n_samples, "horizon": tracker.horizon,
        "compile_rollout_loop": args.compile_rollout_loop,
        "solves": args.solves,
        "continuous_solve_mean_ms": throughput_ms,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    cuda_table = prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=40
    )
    cpu_table = prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=30
    )
    print(json.dumps(summary, indent=2), flush=True)
    print("CUDA_KERNEL_TABLE", flush=True)
    print(cuda_table, flush=True)
    print("CPU_TABLE", flush=True)
    print(cpu_table, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2) + "\n\nCUDA_KERNEL_TABLE\n"
            + cuda_table + "\n\nCPU_TABLE\n" + cpu_table + "\n"
        )


if __name__ == "__main__":
    main()
