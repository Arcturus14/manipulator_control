"""Offline numerical checks; closed-loop experiments remain CUDA-only."""
import numpy as np
import torch
import unittest
from kroc.self_collision_cost import PolytopeCollisionCost
from kroc.collision_geometry import Geometry
from kroc.check_circle_mesh_collisions import frames, NAMES


def test_enclosure_and_frames():
    geom=Geometry();cost=PolytopeCollisionCost('cpu',torch.float64)
    for i,(mesh,origin) in enumerate(zip(geom.vertices,geom.origins)):
        v=mesh@origin[:3,:3].T+origin[:3,3]
        signed=v@cost.normals[i].numpy().T+cost.offsets[i].numpy()
        assert signed.max()<=1e-8
    q=np.array([.94*2*np.pi,-2,.9*np.pi,-2,0,0])
    v,n,b=cost.world(torch.tensor(q[None]))
    for i,tr in enumerate(frames(q,geom.kin)):
        expected=cost.vertices[i].numpy()@tr[:3,:3].T+tr[:3,3]
        np.testing.assert_allclose(v[0,i].numpy(),expected,atol=1e-12)
    pair,floor=cost.separations(torch.tensor(q[None]))
    assert floor.min()>0
    assert torch.isfinite(pair).all()


def test_mesh_collision_has_penalty():
    # Previously rejected posture; at least one definite triangle intersection.
    q=np.array([.94*2*np.pi,-1.2,.9*np.pi,-1.5,-1.57,0])
    g=Geometry();cost=PolytopeCollisionCost('cpu',torch.float64)
    rng=np.random.default_rng(12);tested=0
    for _ in range(30):
        pose=q+rng.normal(0,.2,6)
        actual=g.check(pose)
        if actual['hit_pairs']:
            pair,_=cost.separations(torch.tensor(pose[None]))
            pair_names=[f'{NAMES[i]}--{NAMES[j]}' for i,j in g.pairs]
            for name in actual['hit_pairs']:
                assert pair[0,pair_names.index(name)]<=1e-8
            state=torch.tensor(np.r_[pose,np.zeros(6)][None])
            assert cost(state).item()>0
            tested+=1
    assert tested>0


def test_batch_and_weight():
    q=torch.tensor([[5.8,-2,2.8,-2,0,0],[0,-1,1,0,1,0]],dtype=torch.float64)
    c=PolytopeCollisionCost('cpu',torch.float64)
    state=torch.cat((q,torch.zeros_like(q)),-1)
    together=c(state)
    torch.testing.assert_close(together,torch.cat([c(row[None]) for row in state]))
    c.weight=0
    assert (c(state)==0).all()


def test_reference_and_fused_match():
    rng=np.random.default_rng(87)
    q=torch.tensor(rng.uniform(-np.pi,np.pi,(41,6)),dtype=torch.float64)
    c=PolytopeCollisionCost('cpu',torch.float64,backend='fused')
    for actual,expected in zip(c.world(q),c.world_reference(q)):
        torch.testing.assert_close(actual,expected,rtol=0,atol=2e-14)
    for actual,expected in zip(c.separations(q),c.separations_reference(q)):
        torch.testing.assert_close(actual,expected,rtol=0,atol=2e-14)
    with unittest.TestCase().assertRaises(ValueError):
        PolytopeCollisionCost('cpu',torch.float64,backend='invalid')


class CollisionCostChecks(unittest.TestCase):
    test_enclosure_and_frames=staticmethod(test_enclosure_and_frames)
    test_mesh_collision_has_penalty=staticmethod(test_mesh_collision_has_penalty)
    test_batch_and_weight=staticmethod(test_batch_and_weight)
    test_reference_and_fused_match=staticmethod(test_reference_and_fused_match)


if __name__=='__main__':
    unittest.main()
