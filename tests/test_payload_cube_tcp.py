"""Physics/kinematics unit checks; no CPU MPPI performance runs."""
import ast
from pathlib import Path
import unittest

import numpy as np
import torch

from config import DH_PARAMS
from dynamics.ur5_dynamics import UR5Dynamics
from dynamics.ur5_payload import CubePayload,PayloadUR5Dynamics
from kinematics.payload_tcp import PayloadTCP


class PayloadPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.device=torch.device('cpu');self.dtype=torch.float64
        self.payload=CubePayload(5.,.10,(.02,-.01,.05))
        self.tcp=PayloadTCP(self.payload,self.device,self.dtype)
        self.q=torch.tensor([[np.pi,-1.2,1.2,-1.5,-1.57,5.592],
                             [.2,-.7,.9,-1.1,.6,.4]],dtype=self.dtype)
        self.base=UR5Dynamics(self.device,self.dtype)
        self.loaded=PayloadUR5Dynamics(self.device,self.dtype,self.payload)

    def angular_jacobian(self,q):
        tr=np.eye(4);axes=[]
        for angle,(a,d,alpha,off) in zip(q,DH_PARAMS):
            axes.append(tr[:3,2].copy());c,s=np.cos(angle+off),np.sin(angle+off);ca,sa=np.cos(alpha),np.sin(alpha)
            tr=tr@np.array([[c,-s*ca,s*sa,a*c],[s,c*ca,-c*sa,a*s],[0,sa,ca,d],[0,0,0,1]])
        return np.stack(axes,axis=1)

    def test_zero_mass_preserves_original_dynamics(self):
        dyn=PayloadUR5Dynamics(self.device,self.dtype,CubePayload(mass_kg=0))
        self.assertTrue(torch.equal(dyn.spatial_inertia,self.base.spatial_inertia))
        v=torch.ones_like(self.q)*.13
        for actual,expected in zip(dyn.mass_matrix_and_bias(self.q,v),self.base.mass_matrix_and_bias(self.q,v)):
            self.assertTrue(torch.equal(actual,expected))

    def test_added_gravity_equals_virtual_work(self):
        z=torch.zeros_like(self.q)
        change=self.loaded.inverse_dynamics(self.q,z,z)-self.base.inverse_dynamics(self.q,z,z)
        expected=[]
        for q in self.q.numpy():
            _,j=self.tcp.position_and_jacobian_np(q)
            expected.append(j.T@np.array([0,0,self.payload.mass_kg*9.81]))
        np.testing.assert_allclose(change.numpy(),expected,atol=1e-12)

    def test_added_mass_equals_payload_kinetic_energy(self):
        z=torch.zeros_like(self.q)
        loaded,_=self.loaded.mass_matrix_and_bias(self.q,z)
        base,_=self.base.mass_matrix_and_bias(self.q,z)
        expected=[]
        for q in self.q.numpy():
            _,j=self.tcp.position_and_jacobian_np(q);w=self.angular_jacobian(q)
            expected.append(self.payload.mass_kg*j.T@j+self.payload.inertia_com_kg_m2*w.T@w)
        np.testing.assert_allclose((loaded-base).numpy(),expected,atol=1e-12)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(loaded)>0))

    def test_bias_gravity_and_mass_are_consistent(self):
        v=torch.ones_like(self.q)*.2;a=torch.ones_like(self.q)*.4
        mass,bias=self.loaded.mass_matrix_and_bias(self.q,v)
        torque=self.loaded.inverse_dynamics(self.q,v,a)
        expected=(mass@a[...,None]).squeeze(-1)+bias-self.loaded.armature*a-1e-5*a
        torch.testing.assert_close(torque,expected,rtol=1e-12,atol=1e-12)

    def test_tcp_jacobian_matches_finite_difference(self):
        for q in self.q.numpy():
            p,j=self.tcp.position_and_jacobian_np(q)
            np.testing.assert_allclose(self.tcp.fk_batch(torch.tensor(q[None],dtype=self.dtype))[0].numpy(),p,atol=1e-15)
            finite=[]
            for k in range(6):
                delta=np.zeros(6);delta[k]=1e-6
                finite.append((self.tcp.position_and_jacobian_np(q+delta)[0]-self.tcp.position_and_jacobian_np(q-delta)[0])/2e-6)
            np.testing.assert_allclose(j,np.stack(finite,axis=1),atol=1e-9)

    def test_tcp_ik_tracks_object_center_and_cube_vertices(self):
        q=self.q.numpy()[0];p,_=self.tcp.position_and_jacobian_np(q)
        targets=p+np.column_stack((np.linspace(0,.003,10),np.linspace(0,-.002,10),np.zeros(10)))
        solved=self.tcp.reference_np(targets,q)
        actual=self.tcp.fk_batch(torch.tensor(solved,dtype=self.dtype)).numpy()
        np.testing.assert_allclose(actual,targets,atol=2e-7)
        vertices=self.tcp.vertices(self.q)
        torch.testing.assert_close(vertices.mean(-2),self.tcp.fk_batch(self.q))
        length=torch.linalg.vector_norm(vertices[:,0]-vertices[:,-1],dim=-1)
        torch.testing.assert_close(length,torch.full_like(length,np.sqrt(3)*self.payload.side_m))

    def test_payload_validation(self):
        for kwargs in [dict(mass_kg=5.1),dict(mass_kg=-1),dict(side_m=0),dict(center_in_flange_m=[0,float('nan'),0])]:
            with self.assertRaises(ValueError):CubePayload(**kwargs)
        self.assertEqual(CubePayload(side_m=.12).center_in_flange_m,(0,0,.06))

    def test_episode_copy_only_substitutes_tcp_fk(self):
        def extract(path,cls):
            source=Path(path).read_text();tree=ast.parse(source)
            c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
            f=next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name=='_run_deferred')
            return '\n'.join(source.splitlines()[f.lineno-1:f.end_lineno])
        old=extract('controllers/fast_final_mppi.py','FastFinalTracker')
        new=extract('controllers/payload_mppi.py','PayloadFinalTracker')
        self.assertEqual(old.replace('fk_batch(','self.tcp.fk_batch('),new)


if __name__=='__main__':unittest.main()
