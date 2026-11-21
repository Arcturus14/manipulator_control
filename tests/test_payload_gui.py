"""Payload GUI/animation contract and reference checks; no CPU control runs."""
import copy
from pathlib import Path
import tkinter as tk
from types import SimpleNamespace
import unittest
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dynamics.ur5_payload import CubePayload
from kinematics.payload_tcp import PayloadTCP
from kroc.payload_gui import DEFAULT_PAYLOAD,payload_command,payload_config_from_form
from kroc.payload_scene import make_reference,SHAPES
from visualization.animator import animate_comparison


class PayloadGUITests(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load(DEFAULT_PAYLOAD.read_text())
        self.tcp=PayloadTCP(CubePayload(**self.cfg['payload']),'cpu',torch.float64)

    def test_reference_start_and_plane(self):
        q=self.cfg['scene']['initial_q'];p=self.tcp.position_and_jacobian_np(q)[0]
        for shape in SHAPES:
            scene=dict(self.cfg['scene'],trajectory=shape)
            r=make_reference(self.tcp,q,301,.02,scene)
            np.testing.assert_allclose(r[0],p,atol=1e-15)
            np.testing.assert_allclose(r[-1],p,atol=1e-15)
            if shape.startswith('circle_'):
                fixed=next(i for i,a in enumerate('xyz') if a not in shape[7:])
                np.testing.assert_allclose(r[:,fixed],p[fixed],atol=1e-15)
        expected=np.column_stack((p[0]+.04*(np.cos(2*np.pi*np.arange(300)*.02/6)-1),p[1]+.04*np.sin(2*np.pi*np.arange(300)*.02/6),np.full(300,p[2])))
        np.testing.assert_array_equal(make_reference(self.tcp,q,300,.02,self.cfg['scene']),expected)

    def test_payload_form_and_command(self):
        interpreter=tk.Tcl();variables={}
        for section in ['payload','scene','controller']:
            for key,value in self.cfg[section].items():
                kind=type(value);var=tk.StringVar(master=interpreter,value=', '.join(map(str,value)) if kind is list else value)
                # BooleanVar mirrors the real Tk checkbutton.
                if kind is bool:var=tk.BooleanVar(master=interpreter,value=value)
                variables[f'{section}.{key}']=(var,kind)
        for key,value,kind in [('seed',6807,int),('execution','rerun',str),('replay_path','',str)]:
            variables[key]=(tk.StringVar(master=interpreter,value=value),kind)
        gui=SimpleNamespace(payload_variables=variables,payload_config=copy.deepcopy(self.cfg))
        config,extras=payload_config_from_form(gui)
        self.assertEqual(config,self.cfg)
        for case in ['N0','N1','D0','N0 vs N1 vs D0']:
            cmd=payload_command('python',Path('/tmp/test.yaml'),extras,case,dict(root='/tmp/out',show=False,export_csv=True))
            self.assertEqual(cmd[cmd.index('--case')+1],case)
            self.assertIn('--no-show',cmd);self.assertIn('--csv',cmd)
        variables['payload.mass_kg'][0].set('5.1')
        with self.assertRaises(ValueError):payload_config_from_form(gui)

    def test_payload_animation_cube_and_center_tcp(self):
        q=np.array([self.cfg['scene']['initial_q']]*2);q[1,0]+=.2
        qt=torch.tensor(q,dtype=torch.float64)
        ee=self.tcp.fk_batch(qt).numpy();vertices=self.tcp.vertices(qt).numpy()
        result=dict(q=q,ee=ee,err_mm=np.zeros(1),payload_cube_vertices=vertices)
        fig,anim=animate_comparison({'D0':result},ee)
        self.assertEqual(len(fig.axes[0].collections),1)
        for frame in [0,1]:
            anim._func(frame);fig.canvas.draw()
            dot=next(line for line in fig.axes[0].lines if line.get_marker()=='D')
            np.testing.assert_allclose(np.array(dot.get_data_3d()).reshape(3),ee[frame])
            for j,limits in enumerate([fig.axes[0].get_xlim(),fig.axes[0].get_ylim(),fig.axes[0].get_zlim()]):
                self.assertLess(limits[0],vertices[...,j].min());self.assertGreater(limits[1],vertices[...,j].max())
        anim._draw_was_started=True;plt.close(fig)


if __name__=='__main__':unittest.main()
