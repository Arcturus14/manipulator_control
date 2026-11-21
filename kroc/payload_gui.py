"""Lightweight Tk payload form/command wiring; no CUDA/model imports at startup."""
import copy
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import yaml

DEFAULT_PAYLOAD=Path(__file__).resolve().parents[1]/'configs/payload_cube_tcp.yaml'


def build_payload_tab(gui):
    gui.payload_config=yaml.safe_load(DEFAULT_PAYLOAD.read_text())
    gui.payload_variables={}
    tab=gui.payload_tab;tab.columnconfigure((0,1),weight=1)
    selection=gui._section(tab,'Payload controller',0,0)
    def field(parent,row,label,path,kind=str,choices=None):
        ttk.Label(parent,text=label).grid(row=row,column=0,sticky='w',pady=3)
        v=tk.BooleanVar() if kind is bool else tk.StringVar()
        gui.payload_variables[path]=(v,kind)
        if kind is bool:
            w=ttk.Checkbutton(parent,variable=v)
        elif choices:
            w=ttk.Combobox(parent,textvariable=v,values=choices,state='readonly',width=30)
        else:
            w=ttk.Entry(parent,textvariable=v,width=43)
        w.grid(row=row,column=1,sticky='ew',padx=8,pady=3)
        parent.columnconfigure(1,weight=1)
    field(selection,0,'Controller','controller.case',choices=('N0','N1','D0','N0 vs N1 vs D0'))
    field(selection,1,'Seed','seed',int)
    field(selection,2,'Execution','execution',choices=('rerun','replay'))
    field(selection,3,'Replay output folder','replay_path')
    object_frame=gui._section(tab,'Rigid object; TCP = object center',0,1)
    field(object_frame,0,'Mass [kg], no gripper','payload.mass_kg',float)
    field(object_frame,1,'Cube side [m]','payload.side_m',float)
    field(object_frame,2,'COM/TCP in flange [x,y,z] m','payload.center_in_flange_m',list)
    field(object_frame,3,'Initial q J1..J6 [rad]','scene.initial_q',list)
    field(object_frame,4,'Enable DOB','scene.use_dob',bool)
    scene=gui._section(tab,'TCP trajectory and disturbance',1,1)
    from kroc.payload_scene import SHAPES
    field(scene,0,'Trajectory','scene.trajectory',choices=SHAPES)
    field(scene,1,'Radius / rectangle half-width [m]','scene.circle_radius_m',float)
    field(scene,2,'Period [s]','scene.period_s',float)
    field(scene,3,'Duration [s]','scene.duration_s',float)
    field(scene,4,'Disturbance joint [1..6]','scene.disturbance_joint',int)
    field(scene,5,'Disturbance final torque [Nm]','scene.disturbance_amplitude_nm',float)
    rectangle=gui._section(tab,'Rectangle geometry / motion start',1,0)
    field(rectangle,0,'Rectangle half-height [m]','scene.rectangle_half_height_m',float)
    field(rectangle,1,'Rectangle corner radius [m]','scene.rectangle_corner_radius_m',float)
    field(rectangle,2,'Reference start speed ramp [s]','scene.reference_start_ramp_s',float)
    ttk.Label(tab,text=('All models include the same known payload. Trajectory and marker refer to the object center.\n'
        'Controller/filter settings come from the payload YAML; existing GUI defaults are unchanged.\n'
        'Output tab selects animation, torque plots, CSV, and sampled rollouts. Replay does not rerun CUDA.'),
        foreground='#555555',wraplength=950).grid(row=2,column=0,columnspan=2,sticky='w',padx=10,pady=10)
    load_payload_form(gui,gui.payload_config)


def load_payload_form(gui,config):
    gui.payload_config=copy.deepcopy(config)
    extras={'seed':config.get('seed',6807),'execution':'rerun','replay_path':''}
    for path,(var,kind) in gui.payload_variables.items():
        if path in extras:value=extras[path]
        else:
            a,b=path.split('.')
            defaults={'trajectory':'circle_xy','rectangle_half_height_m':.75*config['scene']['circle_radius_m'],
                      'rectangle_corner_radius_m':.02,'reference_start_ramp_s':0.}
            value=config[a].get(b,defaults.get(b))
        var.set(', '.join(str(x) for x in value) if kind is list else value)


def payload_config_from_form(gui):
    cfg=copy.deepcopy(gui.payload_config);extras={}
    for path,(var,kind) in gui.payload_variables.items():
        try:
            raw=var.get()
            value=[float(x.strip()) for x in raw.split(',')] if kind is list else kind(raw)
        except (ValueError,TypeError) as exc:
            raise ValueError(f'Invalid payload field:{path}') from exc
        if '.' not in path:extras[path]=value
        else:
            a,b=path.split('.');cfg[a][b]=value
    p,s=cfg['payload'],cfg['scene']
    if not 0<=p['mass_kg']<=5 or p['side_m']<=0:raise ValueError('Payload mass must be0--5kg and cube side positive')
    if len(p['center_in_flange_m'])!=3 or len(s['initial_q'])!=6:raise ValueError('TCP offset needs3 values; initial q needs6')
    numbers=[p['mass_kg'],p['side_m'],*p['center_in_flange_m'],*s['initial_q'],s['circle_radius_m'],s['period_s'],s['duration_s'],s['disturbance_amplitude_nm']]
    if not all(math.isfinite(x) for x in numbers):raise ValueError('Payload and scene values must be finite')
    if min(s['circle_radius_m'],s['period_s'],s['duration_s'])<=0:raise ValueError('Radius, period and duration must be positive')
    if not math.isfinite(s.get('reference_start_ramp_s',0.)) or s.get('reference_start_ramp_s',0.)<0:
        raise ValueError('Reference start speed ramp must be finite and nonnegative')
    if s.get('trajectory','').startswith('rounded_rectangle_'):
        height=s.get('rectangle_half_height_m',.75*s['circle_radius_m']);corner=s.get('rectangle_corner_radius_m',.02)
        if not math.isfinite(height) or not math.isfinite(corner) or not 0<corner<min(height,s['circle_radius_m']):
            raise ValueError('Rectangle corner radius must be positive and below both half-sizes')
    if s['disturbance_joint'] not in range(1,7):raise ValueError('Disturbance joint must be1--6')
    if extras['execution']=='replay' and not extras['replay_path'].strip():raise ValueError('Choose the payload output folder to replay')
    return cfg,extras


def payload_command(python,config_path,extras,case,output):
    command=[python,'-u','-m','kroc.view_payload_tcp','--config',str(config_path),
        '--case',case,'--seed',str(extras['seed']),'--output-root',str(output['root'])]
    if extras['execution']=='replay':command+=['--replay',extras['replay_path']]
    for key,flag in [('show','show'),('torque_plot','torque'),('torque_components_plot','components'),
                     ('disturbance_error_plot','disturbance-error'),('export_csv','csv'),('capture_samples','capture-samples')]:
        command.append('--'+('' if output.get(key,False) else 'no-')+flag)
    return command
