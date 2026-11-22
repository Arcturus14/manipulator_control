"""Opt-in payload posture/rate study. Frozen payload and legacy classes stay intact.

The base--upper_arm mask is a causal diagnostic ONLY, excluded from selection.
No actuator filtering, physical-limit changes, or DOB/filter retuning.
"""
import math
import numpy as np
import torch
from controllers.payload_mppi import PayloadFinalTracker
from controllers.dynamic_bound_mppi import DynamicBoundMixin


class DiagnosticPairMask:
    def __init__(self,base):
        self.base=base
        self.mask=~((base.i==0)&(base.j==2))
    def __getattr__(self,name):return getattr(self.base,name)
    def __call__(self,state):
        gap,floor=self.base.separations(state[...,:6])
        return self.base.weight*((torch.relu(self.base.clearance-gap)/.05).square().mul(self.mask).sum(-1)
            +(torch.relu(self.base.clearance-floor)/.05).square().sum(-1))


class PayloadStabilizedTracker(PayloadFinalTracker):
    def __init__(self,*args,posture_weight=0.,diagnostic_mask_base_upper_arm=False,align_running_reference=False,**kwargs):
        if not isinstance(align_running_reference,bool):raise ValueError('align_running_reference must be true or false')
        self.align_running_reference=align_running_reference
        if not math.isfinite(posture_weight) or posture_weight<0:raise ValueError('Posture weight must be finite and nonnegative')
        self.posture_weight=float(posture_weight)
        self.diagnostic_mask_base_upper_arm=bool(diagnostic_mask_base_upper_arm)
        super().__init__(*args,**kwargs)
        self.posture_window=torch.zeros((self.horizon+1,6),device=self.device,dtype=self.dtype)
        if self.diagnostic_mask_base_upper_arm:
            self.collision_geometry_cost=DiagnosticPairMask(self.collision_geometry_cost)

    def _prepare_solve(self,state,sampling_center):
        self.posture_window.copy_(self.joint_reference[self.step:self.step+self.horizon+1])
        return super()._prepare_solve(state,sampling_center)

    def _running_cost(self,state,action,rollout_step):
        if self.align_running_reference:
            # pytorch_mppi evaluates stage t on x(t+1). Only the references move;
            # the first-action boundary cost and the terminal index stay as they are.
            index=rollout_step+1
            value=self._fast_stage(state,action,self.tracking_cost.reference_window[index],
                self.tracking_cost.previous_action,self._compensation_snapshot,rollout_step==0)
        else:
            index=rollout_step
            value=super()._running_cost(state,action,rollout_step)
        if self.posture_weight:
            value=value+self.posture_weight*(state[:,:6]-self.posture_window[index]).square().sum(-1)
        return value

    def _terminal_cost(self,states,actions):
        value=super()._terminal_cost(states,actions)
        if self.posture_weight:
            value=value+3*self.posture_weight*(states[...,-1,:6]-self.posture_window[-1]).square().sum(-1)
        return value

    def export_sensor_data(self):
        data=super().export_sensor_data()
        data.update(payload_posture_weight=np.asarray(self.posture_weight),
            payload_posture_terminal_weight=np.asarray(3*self.posture_weight),
            payload_diagnostic_mask_base_upper_arm=np.asarray(self.diagnostic_mask_base_upper_arm),
            payload_align_running_reference=np.asarray(self.align_running_reference))
        return data


class PayloadStabilizedD0Tracker(DynamicBoundMixin,PayloadStabilizedTracker):
    pass
