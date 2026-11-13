"""Visualize initial floor penetration and first self-collision witnesses."""
import json
from pathlib import Path
import sys
sys.path.insert(0,'/tmp/kroc_collision_deps')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh
import yaml
from kroc.check_circle_mesh_collisions import frames,collision_origins,NAMES,FILES
from config import DT,Q_MAX,Q_MIN,TAU_MAX


def main():
    root=Path('kroc/results/circle_floor25_mesh_collision_check')
    assets=Path('kroc/assets/ur5_cb3_collision')
    source=Path('kroc/results/circle_floor25_independent_validation')
    result=json.loads((root/'results.json').read_text());row=result['rows'][0]
    z=np.load(source/row['file']);kin=yaml.safe_load((assets/'default_kinematics.yaml').read_text())['kinematics']
    origins=collision_origins(yaml.safe_load((assets/'physical_parameters.yaml').read_text())['offsets'])
    steps=[0,round(row['first_self_collision_s']/DT)]
    fig=plt.figure(figsize=(12,10))
    colors=plt.get_cmap('tab10')(np.arange(7))
    for col,step in enumerate(steps):
        chain=frames(z['q'][step],kin)
        world=[]
        for i,(file,t,o) in enumerate(zip(FILES,chain,origins)):
            mesh=trimesh.load_mesh(assets/f'{file}.stl',process=True);tf=t@o
            v=mesh.vertices@tf[:3,:3].T+tf[:3,3];world.append((v,mesh.faces))
        ax=fig.add_subplot(2,2,col+1,projection='3d')
        for i,(v,f) in enumerate(world):
            ax.add_collection3d(Poly3DCollection(v[f],facecolors=colors[i],edgecolors='none',alpha=.65))
        xyz=np.concatenate([v for v,_ in world]);low=xyz.min(0);high=xyz.max(0)
        span=max(high-low);center=(high+low)/2
        ax.set(xlim=(center[0]-span/2,center[0]+span/2),ylim=(center[1]-span/2,center[1]+span/2),zlim=(min(-.08,low[2]),high[2]+.03),xlabel='x [m]',ylabel='y [m]',zlabel='z [m]')
        ax.set_box_aspect((1,1,1));ax.view_init(25,-60)
        ax.set_title(f't = {step*DT:.2f} s; '+('initial configuration' if step==0 else 'upper_arm -- wrist_3 intersection'))
        xx,yy=np.meshgrid(np.linspace(center[0]-span/2,center[0]+span/2,2),np.linspace(center[1]-span/2,center[1]+span/2,2))
        ax.plot_surface(xx,yy,np.zeros_like(xx),color='gray',alpha=.2)
        side=fig.add_subplot(2,2,col+3)
        for i,(v,f) in enumerate(world):
            from matplotlib.collections import PolyCollection
            side.add_collection(PolyCollection(v[f][:,:,[0,2]],facecolors=colors[i],alpha=.55,label=NAMES[i]))
        side.autoscale();side.axhline(0,color='red',ls='--',label='Mounting plane z=0');side.set_aspect('equal');side.grid(alpha=.2)
        side.set(xlabel='x [m]',ylabel='z [m]',title='All link collision meshes, side view')
        if col==1:side.legend(fontsize=7,ncol=2)
    fig.suptitle('UR5/CB3 collision audit: mounting-plane penetration and nonadjacent link intersection')
    fig.tight_layout();fig.savefig(root/'collision_witnesses.png',dpi=170);plt.close(fig)
    # Headroom feasibility diagnostic only: do not modify or simulate a controller.
    diagnostic=[]
    for path in sorted(source.glob('s1_seed*_directional25_5.npz')):
        d=np.load(path);center=d['kroc_center_horizon']+d['d_hat_used'][:,None,:]
        reserves=(center+np.array(TAU_MAX)[None,None,:]).min(1)
        gain=1+np.maximum(1-reserves/(.2*np.array(TAU_MAX)),0)
        mar=d['kroc_margin'][:,0,1,2]/(Q_MAX[2]-Q_MIN[2]);late=np.arange(len(mar))*DT>=1.2
        diagnostic.append({'file':path.name,'post_ramp_q3_position_margin_at_5pct_fraction':float((mar[late]>=.05-1e-6).mean()),
                           'minimum_postcomp_sampling_center_negative_reserve_nm':float(reserves[:,2].min()),
                           'maximum_proposed_headroom_gain_q3_upper':float(gain[:,2].max())})
    (root/'headroom_offline_diagnostic.json').write_text(json.dumps(diagnostic,indent=2))


if __name__=='__main__':main()
