"""Pure geometric payload TCP reference generation, shared by GUI and studies."""
import numpy as np

SHAPES=('circle_xy','circle_xz','circle_yz','line_radial','line_vertical',
        'rounded_rectangle_xy','rounded_rectangle_xz','rounded_rectangle_yz')


def rounded_rectangle(unit_phase,half_width,half_height,corner_radius):
    """CCW loop starting at right-edge midpoint; exact straight sides, C2 turns.

    Quintic Bezier corners have zero endpoint second derivatives. Segment
    durations match their endpoint tangents, giving continuous velocity and
    acceleration at all joins. Corner length parameter pi*r/2 is a timing
    convention, not a claim of constant speed inside the Bezier corner.
    """
    a,b,r=half_width,half_height,corner_radius
    if not np.isfinite([a,b,r]).all() or not 0<r<min(a,b):
        raise ValueError('Rectangle corner radius must be positive and below both half-sizes')
    c=np.pi*r/10
    segments=[]
    def line(p0,p1):
        p0,p1=np.array(p0),np.array(p1);segments.append((np.linalg.norm(p1-p0),np.array([p0,p1])))
    def corner(p0,p1,t0,t1):
        p0,p1,t0,t1=map(np.array,[p0,p1,t0,t1])
        segments.append((np.pi*r/2,np.array([p0,p0+c*t0,p0+2*c*t0,p1-2*c*t1,p1-c*t1,p1])))
    line((a,0),(a,b-r));corner((a,b-r),(a-r,b),(0,1),(-1,0))
    line((a-r,b),(-a+r,b));corner((-a+r,b),(-a,b-r),(-1,0),(0,-1))
    line((-a,b-r),(-a,-b+r));corner((-a,-b+r),(-a+r,-b),(0,-1),(1,0))
    line((-a+r,-b),(a-r,-b));corner((a-r,-b),(a,-b+r),(1,0),(0,1))
    line((a,-b+r),(a,0))
    lengths=np.array([v[0] for v in segments]);bounds=np.r_[0,np.cumsum(lengths)]
    s=np.mod(np.asarray(unit_phase),1)*bounds[-1];points=np.empty(s.shape+(2,))
    from math import comb
    for i,(length,control) in enumerate(segments):
        mask=(s>=bounds[i])&(s<bounds[i+1]);u=(s[mask]-bounds[i])/length
        degree=len(control)-1
        basis=np.array([comb(degree,k)*u**k*(1-u)**(degree-k) for k in range(degree+1)]).T
        points[mask]=basis@control
    return points


def make_reference(tcp,q0,steps,dt,scene):
    radius=float(scene['circle_radius_m']);period=float(scene['period_s'])
    shape=scene.get('trajectory','circle_xy')
    if shape not in SHAPES:
        raise ValueError(f'Unknown payload trajectory:{shape}')
    if not np.isfinite([radius,period]).all() or radius<=0 or period<=0:
        raise ValueError('Radius/amplitude and period must be finite and positive')
    p0,_=tcp.position_and_jacobian_np(q0)
    elapsed=np.arange(steps)*dt
    ramp=float(scene.get('reference_start_ramp_s',0.))
    if not np.isfinite(ramp) or ramp<0:raise ValueError('Reference start ramp must be finite and nonnegative')
    if ramp:
        s=np.clip(elapsed/ramp,0,1)
        # Integral of quintic smoothstep speed. Starts at rest; C2 path timing.
        elapsed=np.where(elapsed<ramp,ramp*(2.5*s**4-3*s**5+s**6),elapsed-.5*ramp)
    phase=(2*np.pi*elapsed/period if ramp else 2*np.pi*np.arange(steps)*dt/period)
    offset=np.zeros((steps,3))
    if shape.startswith('circle_'):
        first,second={'circle_xy':(0,1),'circle_xz':(0,2),'circle_yz':(1,2)}[shape]
        offset[:,first]=radius*(np.cos(phase)-1)
        offset[:,second]=radius*np.sin(phase)
    elif shape.startswith('rounded_rectangle_'):
        first,second={'xy':(0,1),'xz':(0,2),'yz':(1,2)}[shape[-2:]]
        xy=rounded_rectangle(elapsed/period,radius,float(scene.get('rectangle_half_height_m',.75*radius)),float(scene.get('rectangle_corner_radius_m',.02)))
        offset[:,first]=xy[:,0]-radius;offset[:,second]=xy[:,1]
    elif shape=='line_radial':
        direction=np.r_[p0[:2],0.];direction/=max(np.linalg.norm(direction),1e-12)
        offset=radius*(np.cos(phase)-1)[:,None]*direction
    else:
        offset[:,2]=radius*np.sin(phase)
    return p0+offset
