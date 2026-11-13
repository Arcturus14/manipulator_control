"""Forward-only CUDA pair gaps, with no (batch,pair,plane,vertex) workspace.

The torch custom-op boundary keeps fullgraph compilation supported. Triton
generates code on the target GPU; no GTX-specific binary is shipped.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pair_gap_kernel(V,N,B,I,J,OUT,PAIRS:tl.constexpr,VERTICES:tl.constexpr,
                     PLANES:tl.constexpr,BV:tl.constexpr,BN:tl.constexpr):
    index=tl.program_id(0)
    state=index//(PAIRS*2);pair=(index//2)%PAIRS;direction=index%2
    ii=tl.load(I+pair);jj=tl.load(J+pair)
    source=tl.where(direction==0,ii,jj);target=tl.where(direction==0,jj,ii)
    planes=tl.arange(0,BN)[:,None];vertices=tl.arange(0,BV)[None,:]
    ni=(state*7+source)*PLANES*3+planes*3
    vi=(state*7+target)*VERTICES*3+vertices*3
    nx=tl.load(N+ni,planes<PLANES,0);ny=tl.load(N+ni+1,planes<PLANES,0)
    nz=tl.load(N+ni+2,planes<PLANES,0)
    vx=tl.load(V+vi,vertices<VERTICES,0);vy=tl.load(V+vi+1,vertices<VERTICES,0)
    vz=tl.load(V+vi+2,vertices<VERTICES,0)
    b=tl.load(B+(state*7+source)*PLANES+planes,planes<PLANES,0)
    signed=nx*vx+ny*vy+nz*vz+b
    plane_gap=tl.min(tl.where(vertices<VERTICES,signed,float('inf')),axis=1)
    gap=tl.max(tl.where(tl.arange(0,BN)<PLANES,plane_gap,-float('inf')),axis=0)
    tl.store(OUT+index,gap)


@torch.library.custom_op('kroc_collision::pair_gaps',mutates_args=())
def pair_gaps(vertices:torch.Tensor,normals:torch.Tensor,offsets:torch.Tensor,
              i:torch.Tensor,j:torch.Tensor)->torch.Tensor:
    if not vertices.is_cuda or vertices.dtype!=torch.float32:
        raise ValueError('pair_gaps requires CUDA float32')
    v=vertices.contiguous();n=normals.contiguous();b=offsets.contiguous()
    output=torch.empty((len(v),len(i),2),device=v.device,dtype=v.dtype)
    _pair_gap_kernel[(len(v)*len(i)*2,)](
        v,n,b,i,j,output,len(i),v.shape[2],n.shape[2],
        triton.next_power_of_2(v.shape[2]),triton.next_power_of_2(n.shape[2]),
        num_warps=4,enable_fp_fusion=False)
    return output


@pair_gaps.register_fake
def _(vertices,normals,offsets,i,j):
    return vertices.new_empty((vertices.shape[0],i.shape[0],2))
