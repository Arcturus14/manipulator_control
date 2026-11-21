"""GUI adapter for the frozen pre-input-study baseline (no tightening)."""
import functools
import types
import numpy as np
import torch
from config import DT, W_ACT, W_VEL, DH_PARAMS, Q_MIN, Q_MAX
from controllers.measured_mppi_controller import MeasuredStateTracker
from kinematics.ur5_kinematics import fk_batch
from kroc.state_margin import MarginConfig, StateMargins


def single_noise(ctrl, shape, original):
    if len(shape) != 2:
        return original(shape)
    k, h = shape
    z = torch.randn(k, 1, ctrl.nu, device=ctrl.d, dtype=ctrl.dtype)
    noise = (z * ctrl._noise_sigma_sqrt_diag if ctrl._diagonal_sigma
             else z @ ctrl._noise_sigma_chol.T) + ctrl.noise_mu
    return noise.expand(k, h, ctrl.nu)


def rollout_cost_only(ctrl):
    ctrl._compute_perturbed_action_and_noise()
    cost, ctrl.states, actions = ctrl._compute_rollout_costs(ctrl.perturbed_action)
    ctrl.actions = actions / ctrl.u_scale if actions is not None else None
    ctrl.cost_total = cost
    return cost


def asynchronous_forward_dynamics(dynamics, state, torque):
    """Same 6x6 solve without per-stage host synchronization for error checks.

    The regularized physical mass matrix is SPD. Nonfinite states remain
    visible to the ordinary simulation diagnostics; this is not a new model.
    """
    velocity = state[...,6:]
    mass,bias = dynamics.mass_matrix_and_bias(state[...,:6],velocity)
    rhs = torque-bias-dynamics.damping*velocity
    solution,_ = torch.linalg.solve_ex(mass,rhs.unsqueeze(-1),check_errors=False)
    return solution.squeeze(-1)


def local_reference(targets, initial):
    """Same continuous, physical-bound IK used by the validated baseline."""
    q = np.array(initial, dtype=float).copy()
    result = []
    for target in targets:
        for _ in range(50):
            tr = np.eye(4); origins = []; axes = []
            for i, (aa, dd, alpha, offset) in enumerate(DH_PARAMS):
                origins.append(tr[:3, 3].copy()); axes.append(tr[:3, 2].copy())
                ct, st = np.cos(q[i]+offset), np.sin(q[i]+offset)
                ca, sa = np.cos(alpha), np.sin(alpha)
                tr = tr @ np.array([[ct,-st*ca,st*sa,aa*ct], [st,ct*ca,-ct*sa,aa*st],
                                    [0,sa,ca,dd], [0,0,0,1]])
            err = target-tr[:3, 3]
            if np.linalg.norm(err) < 2e-7:
                break
            jac = np.stack([np.cross(ax,tr[:3,3]-o) for ax,o in zip(axes,origins)],axis=1)
            delta = jac.T @ np.linalg.solve(jac@jac.T+1e-5*np.eye(3),err)
            q = np.clip(q+.8*np.clip(delta,-.05,.05),Q_MIN,Q_MAX)
        result.append(q.copy())
    return np.array(result)


class ResearchBaselineTracker(MeasuredStateTracker):
    def __init__(self, *args, fixed_margin_fraction=.025, collision_weight=50000.,
                 collision_clearance=.005, collision_backend='triton',
                 fuse_costs=False, compile_auxiliary=False, auxiliary_scope='full', **kwargs):
        asynchronous_solve = kwargs.pop('asynchronous_solve',False)
        super().__init__(*args, **kwargs)
        self.asynchronous_solve = asynchronous_solve
        if asynchronous_solve:
            self.dynamics.forward_dynamics = types.MethodType(asynchronous_forward_dynamics,self.dynamics)
        self.margins = StateMargins(self.dynamics, self.tracking_cost, MarginConfig(
            fixed_fraction=fixed_margin_fraction, min_fraction=fixed_margin_fraction,
            max_fraction=max(.05, fixed_margin_fraction)))
        self.tracking_cost.state_limit_cost = self.margins.cost_value
        self.ctrl._sample_noise = functools.partial(single_noise, self.ctrl,
                                                   original=self.ctrl._sample_noise)
        self.ctrl._compute_total_cost_batch = functools.partial(rollout_cost_only, self.ctrl)
        self.collision_geometry_cost = None
        self.collision_cost = None
        options = {'triton.cudagraphs':False, 'triton.cudagraph_trees':False}
        self.compile_auxiliary = compile_auxiliary
        if auxiliary_scope not in {'matrix','full'}:
            raise ValueError('unknown auxiliary compilation scope')
        self.auxiliary_scope = auxiliary_scope
        if compile_auxiliary:
            # Same RNEA/M(q) equations for RK4 and DOB, not a simplified model.
            self.dynamics.mass_matrix_and_bias = torch.compile(
                self.dynamics.mass_matrix_and_bias, options=options)
            if auxiliary_scope=='full':
                self.dynamics.plant_step_effective = torch.compile(
                    self.dynamics.plant_step_effective, options=options)
                self.dynamics.rollout_effective = torch.compile(
                    self.dynamics.rollout_effective, options=options)
                self.dob.update = torch.compile(self.dob.update, options=options)
        if collision_weight:
            from kroc.self_collision_cost import PolytopeCollisionCost
            self.collision_geometry_cost = PolytopeCollisionCost(
                self.device,self.dtype,collision_weight,collision_clearance,backend=collision_backend)
            self.collision_cost = torch.compile(self.collision_geometry_cost.__call__, options=options)
        self.fuse_costs = fuse_costs
        self._stage = torch.compile(self._stage_value, options=options) if fuse_costs else self._stage_value
        self._end = torch.compile(self._end_value, options=options) if fuse_costs else self._end_value

    def _stage_value(self, state, action, reference, first):
        c = self.tracking_cost
        torque = action.clamp(self.dynamics.u_min,self.dynamics.u_max)
        cost = c.position_weight*(fk_batch(state[:,:6])-reference).square().sum(1)
        cost += W_ACT*torque.square().sum(1)
        cost += W_VEL*state[:,6:].square().sum(1)
        cost += c.state_limit_cost(state)
        if first:
            cost += c.smooth_weight*(torque-c.previous_action).square().sum(1)
        if self.fuse_costs and self.collision_geometry_cost is not None:
            cost += self.collision_geometry_cost(state)
        return cost

    def _running_cost(self, state, action, rollout_step):
        if not self.fuse_costs:
            cost = super()._running_cost(state,action,rollout_step)
            if self.collision_cost is not None:
                cost += self.collision_cost(state)
            return cost
        c = self.tracking_cost
        reference = (c.reference_window[rollout_step] if c.reference_window is not None
                     else self.waypoints[self.step+rollout_step])
        cost = self._stage(state,action,reference,rollout_step==0)
        if self.obstacle_cost is not None:
            cost += self.obstacle_cost.running(state,action,rollout_step)
        return cost

    def _end_value(self, states, actions, reference):
        c = self.tracking_cost
        state = states[...,-1,:].reshape(-1,12)
        cost = c.terminal_weight*(fk_batch(state[:,:6])-reference).square().sum(1)
        cost += c.state_limit_cost(state)
        if self.fuse_costs and self.collision_geometry_cost is not None:
            cost += self.collision_geometry_cost(state)
        cost = cost.reshape(states.shape[:2])
        if self.horizon>1:
            torque = actions.clamp(self.dynamics.u_min,self.dynamics.u_max)
            cost += c.smooth_weight*(torque[...,1:,:]-torque[...,:-1,:]).square().sum((-1,-2))/(self.horizon-1)
        return cost

    def _terminal_cost(self, states, actions):
        if not self.fuse_costs:
            cost = super()._terminal_cost(states,actions)
            if self.collision_cost is not None:
                cost += self.collision_cost(states[...,-1,:].reshape(-1,12)).reshape(states.shape[:2])
            return cost
        c = self.tracking_cost
        reference = (c.reference_window[self.horizon] if c.reference_window is not None
                     else self.waypoints[self.step+self.horizon])
        cost = self._end(states,actions,reference)
        if self.obstacle_cost is not None:
            cost += self.obstacle_cost.terminal(states,actions)
        return cost

    def _feedforward_torque(self, q0):
        count = len(self.waypoints)-self.horizon-2
        joints = local_reference(self.waypoints[:count].detach().cpu().numpy(),q0.detach().cpu().numpy())
        vel = np.gradient(joints,DT,axis=0,edge_order=2)
        acc = np.gradient(vel,DT,axis=0,edge_order=2)
        qr,vr,ar = [torch.as_tensor(v,device=self.device,dtype=self.dtype) for v in (joints,vel,acc)]
        torque = self.dynamics.inverse_dynamics(qr,vr,ar)+self.dynamics.damping*vr
        self.joint_reference = torch.cat((qr,qr[-1:].expand(self.horizon+2,-1)))
        return torch.cat((torque,torque[-1:].expand(self.horizon+2,-1))).clamp(self.dynamics.u_min,self.dynamics.u_max)

    def _warm_up_controller(self, state, feedforward, sampling_center_mode):
        super()._warm_up_controller(state,feedforward,sampling_center_mode)
        if self.compile_auxiliary:
            self.dynamics.mass_matrix_and_bias(state[None,:6],state[None,6:])
            if self.auxiliary_scope=='full':
                torque = feedforward[0][None,:]
                self.dynamics.plant_step_effective(state[None,:],torque)
                self.dynamics.rollout_effective(state[None,:],torque)
                fields=('d_hat','residual','tau_model','raw_acceleration','filtered_acceleration')
                saved={key:getattr(self.dob,key).clone() for key in fields}
                self.dob.update(torque[0],state[:6],state[6:],state[:6],state[6:])
                for key,value in saved.items():
                    setattr(self.dob,key,value)
            if self.device.type=='cuda':
                torch.cuda.synchronize(self.device)

    def export_sensor_data(self):
        result = super().export_sensor_data()
        result.update(mppi_implementation=np.asarray('research_baseline'),
            mppi_weighting=np.asarray('rollout_cost_only'),
            mppi_proposal=np.asarray('single_instance'),
            mppi_fused_costs=np.asarray(self.fuse_costs),
            mppi_compile_auxiliary=np.asarray(self.compile_auxiliary),
            mppi_asynchronous_solve=np.asarray(self.asynchronous_solve),
            mppi_auxiliary_scope=np.asarray(self.auxiliary_scope),
            mppi_fixed_margin_fraction=np.asarray(self.margins.config.fixed_fraction),
            mppi_collision_weight=np.asarray(self.collision_geometry_cost.weight
                if self.collision_geometry_cost is not None else 0.),
            input_tightening_enabled=np.asarray(False))
        return result
