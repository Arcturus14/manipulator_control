"""Numerically compare one baseline and complete-loop-compiled MPPI solve."""
import json

import torch

from controllers.mppi_dob_controller import UR5Tracker
from experiment_runner import initial_configuration, waypoints


def make(reference, compiled):
    return UR5Tracker(
        reference, torch.device("cuda"), dtype=torch.float32,
        n_samples=136, horizon=10, temperature=1., temporal_noise_rho=.8,
        position_weight=50000., terminal_weight=50000.,
        joint_position_limit_weight=50000.,
        joint_velocity_limit_weight=50000.,
        compile_rollout_loop=compiled,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    reference = waypoints("circle", n_steps=64)
    state = torch.cat((initial_configuration(reference, device),
                       torch.zeros(6, device=device)))
    baseline, compiled = make(reference, False), make(reference, True)
    ff_baseline = baseline._feedforward_torque(state[:6])
    ff_compiled = compiled._feedforward_torque(state[:6])
    baseline._warm_up_controller(state, ff_baseline, "feedforward_residual")
    compiled._warm_up_controller(state, ff_compiled, "feedforward_residual")
    torch.testing.assert_close(ff_compiled, ff_baseline)
    compiled.ctrl.U.copy_(baseline.ctrl.U)
    rng = torch.cuda.get_rng_state(device)
    action_baseline = baseline._mppi_command(state)
    torch.cuda.synchronize()
    torch.cuda.set_rng_state(rng, device)
    action_compiled = compiled._mppi_command(state)
    torch.cuda.synchronize()
    comparisons = {
        "action": (action_compiled, action_baseline),
        "nominal_sequence": (compiled.ctrl.U, baseline.ctrl.U),
        "perturbed_action": (compiled.ctrl.perturbed_action,
                             baseline.ctrl.perturbed_action),
        "states": (compiled.ctrl.states, baseline.ctrl.states),
        "cost_total": (compiled.ctrl.cost_total, baseline.ctrl.cost_total),
        "weights": (compiled.ctrl.omega, baseline.ctrl.omega),
    }
    result = {}
    for name, (actual, expected) in comparisons.items():
        difference = (actual - expected).abs()
        result[name] = {
            "max_abs": float(difference.max().item()),
            "mean_abs": float(difference.mean().item()),
        }
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    print(json.dumps({"status": "passed", "differences": result}, indent=2))


if __name__ == "__main__":
    main()
