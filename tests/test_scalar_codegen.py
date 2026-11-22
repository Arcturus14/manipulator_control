"""The traced scalar equations must equal the RNEA dynamics in float64."""
import unittest

import numpy as np
import torch

from dynamics.scalar_codegen import Graph, RobotModel, flange_pose
from dynamics.ur5_dynamics import UR5Dynamics
from dynamics.ur5_payload import CubePayload, PayloadUR5Dynamics
from kinematics.payload_tcp import PayloadTCP


class ScalarCodegenTests(unittest.TestCase):
    def test_step_and_flange_match_reference(self):
        device, dtype = torch.device('cpu'), torch.float64
        payload = CubePayload(5., .063)
        random = np.random.default_rng(4)
        q, velocity = random.uniform(-3., 3., (2, 256, 6))
        torque = random.uniform(-25., 25., (256, 6))
        for dynamics in (UR5Dynamics(device, dtype), PayloadUR5Dynamics(device, dtype, payload)):
            graph = Graph()
            q_in, velocity_in, torque_in = ([graph.input(f'{p}{i}') for i in range(6)] for p in 'qwu')
            next_q, next_velocity = RobotModel(dynamics).semi_implicit_step(q_in, velocity_in, torque_in)
            rotation, position = flange_pose(next_q)
            outputs = next_q + next_velocity + rotation[0] + rotation[1] + rotation[2] + position
            scope = {'np': np}
            for prefix, values in zip('qwu', (q, velocity, torque)):
                scope.update({f'{prefix}{i}': values[:, i] for i in range(6)})
            exec('\n'.join(graph.emit({f'out{i}': v for i, v in enumerate(outputs)}, 'np')), scope)
            actual = np.stack([np.broadcast_to(scope[f'out{i}'], (256,)) for i in range(24)], 1)
            state = dynamics.rollout_effective(
                torch.as_tensor(np.c_[q, velocity]), torch.as_tensor(torque))
            pose = PayloadTCP(payload, device, dtype).flange_transform(state[:, :6]).numpy()
            expected = np.c_[state.numpy(), pose[:, :3, :3].reshape(256, 9), pose[:, :3, 3]]
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
