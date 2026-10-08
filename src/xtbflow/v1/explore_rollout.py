"""Exploratory joint rollout that records endpoint predictions and takes geometry pushes.

The Euler update, old-state convention and centring are those of
`xtbflow.v1.sampler.rollout` for the joint role, so with `push=None` it
reproduces that rollout. Endpoint predictions come from the network's own
velocity rather than from trace differences, which keeps them defined on
clocks with zero event increments (event delays). A push only moves the
geometry; event variables change only through the coupled velocity field.
"""
import torch

from .clocks import clock_grid
from .guidance import centered


@torch.no_grad()
def recorded_rollout(net,query,state,*,path,n_steps=50,dual_time=True,push=None,window=(.5,.95)):
    """Return endpoint predictions at every step, the final state and push counts.

    `push(k, b_hat, x_hat, x, dt_x) -> (delta, applied)` is called on the OLD
    state of step k when the geometry clock lies in `window`; `delta` is added
    to the geometry update of that step.
    """
    tb,tx=clock_grid(path,n_steps)
    b=state.b.detach().clone();x=state.x.detach().clone();mask=query.atom_mask
    b_hat=[];x_hat=[];trace_b=[b.clone()];trace_x=[x.clone()]
    applied=torch.zeros(len(b),device=b.device)
    for k in range(n_steps):
        bt=torch.full((len(b),),float(tb[k]),device=b.device)
        xt=torch.full((len(b),),float(tx[k]),device=b.device)
        kwargs=dict(z=query.element_index,atom_mask=mask,x_cur=x,x_r=query.x_r,b_cur=b,b_r=query.b_r,t=bt)
        if dual_time:kwargs['t_x']=xt
        out=net(**kwargs)
        if not all(torch.isfinite(v).all() for v in out.values()):raise FloatingPointError('nonfinite velocity')
        b_hat.append(b+float(1-tb[k])*out['b_vel']);x_hat.append(x+float(1-tx[k])*out['x_vel'])
        delta=torch.zeros_like(x)
        if push is not None and window[0]<=float(tx[k])<window[1]:
            delta,ok=push(k,b_hat[-1],x_hat[-1],x,float(tx[k+1]-tx[k]))
            applied+=ok.to(applied.dtype)
        next_b=b+float(tb[k+1]-tb[k])*out['b_vel']
        next_x=x+float(tx[k+1]-tx[k])*out['x_vel']
        next_x=next_x+delta
        if not (torch.isfinite(next_b).all() and torch.isfinite(next_x).all()):raise FloatingPointError('nonfinite state')
        b=next_b.detach();x=centered(next_x,mask).detach()
        trace_b.append(b.clone());trace_x.append(x.clone())
    b_hat.append(b.clone());x_hat.append(x.clone())
    return dict(b_hat=torch.stack(b_hat),x_hat=torch.stack(x_hat),b_trace=torch.stack(trace_b),
                x_trace=torch.stack(trace_x),applied=applied,tb=tb,tx=tx)


def event_direction(b_hat,b_r,x,mask,eps=1e-8):
    """Reaction coordinate implied by the predicted bond-order change, as a unit [B,N,3].

    For a pair with predicted change w_ij = b_hat_ij - b_r_ij, atom i moves by
    w_ij along the unit vector to j: forming bonds (w>0) contract, breaking bonds
    (w<0) stretch. Diagonal (lone-pair) entries carry no direction. Returns the
    centred, Frobenius-normalized direction and a validity flag per sample.
    """
    pair=(mask[:,:,None]&mask[:,None,:]).to(x.dtype)
    eye=torch.eye(x.shape[1],device=x.device,dtype=x.dtype)[None]
    w=(b_hat-b_r)*pair*(1-eye);w=.5*(w+w.transpose(1,2))
    d=x[:,None,:,:]-x[:,:,None,:]  # d[b,i,j] = x_j - x_i
    unit=d/d.norm(dim=-1,keepdim=True).clamp_min(eps)
    u=centered((w[...,None]*unit).sum(2),mask)
    size=u.flatten(1).norm(dim=1)
    return u/size.clamp_min(eps)[:,None,None],size>eps


def saddle_reflect(force,u):
    """(I - 2 u u^T) F per sample, the reflection of `xtbflow.physics.saddle.saddle_guidance`."""
    dot=(force*u).sum((1,2),keepdim=True)
    return force-2*dot*u
