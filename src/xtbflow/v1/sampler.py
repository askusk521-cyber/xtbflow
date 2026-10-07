"""Reactant-only rollout primitives. No catalogue, proxy, reference or label imports."""
from dataclasses import dataclass

import torch

from xtbflow.m0.flow import geometry_noise,symmetric_noise
from .clocks import clock_grid
from .guidance import centered,geometry_step,score_direction
from .interfaces import Query,State
from .rng import addressed_seed


def subset_query(q,indices):
    ix=torch.as_tensor(indices,dtype=torch.long,device=q.x_r.device)
    return Query(tuple(q.query_id[int(i)] for i in ix.tolist()),q.atomic_numbers[ix],
                 q.element_index[ix],q.atom_mask[ix],q.x_r[ix],q.b_r[ix],q.charge,q.multiplicity)


def initial_state(query,proposals,training_seed,sampling_seed=0,namespace='efficiency',
                  sigma_b=1.,sigma_x=.5,group='joint'):
    b=query.b_r.clone();x=query.x_r.clone()
    if len(proposals)!=len(query.query_id):raise ValueError('proposal count mismatch')
    for k,(qid,proposal) in enumerate(zip(query.query_id,proposals)):
        n=int(query.atom_mask[k].sum());mask=torch.ones(1,n,dtype=torch.bool)
        keys=dict(fold=0,query=qid,training_seed=training_seed,sampling_seed=sampling_seed,
                  proposal=int(proposal))
        gb=torch.Generator().manual_seed(addressed_seed(namespace,noise_role=group+'_b',**keys))
        gx=torch.Generator().manual_seed(addressed_seed(namespace,noise_role=group+'_x',**keys))
        b[k,:n,:n]+=sigma_b*symmetric_noise(mask,gb)[0].to(b)
        x[k,:n]+=sigma_x*geometry_noise(mask,gx)[0].to(x)
    zeros=torch.zeros(len(b),device=b.device)
    return State(b,x,zeros,zeros)


@dataclass
class Rollout:
    state: State
    b_trace: torch.Tensor
    x_trace: torch.Tensor
    diagnostics: dict
    failed: torch.Tensor


@torch.no_grad()
def rollout(net,query,state,*,role='joint',path='sync',n_steps=50,start=0,end=None,
            score=None,alpha=0.,guidance_start=0.,guidance_stop=.95,replay=None,
            dual_time=True):
    """All velocities and score gradients use the same OLD state.

    A partial rollout returns a trace indexed relative to `start`; replay accepts
    the complete zero-to-one shadow trace so continuation never shifts its indices.
    """
    end=n_steps if end is None else end
    if not 0<=start<=end<=n_steps:raise ValueError('invalid step range')
    if role not in ('joint','event','geometry'):raise ValueError(role)
    if role!='joint' and path!='sync':raise ValueError('cascade uses its own linear stage clock')
    tb,tx=clock_grid(path,n_steps)
    b=state.b.detach().clone();x=state.x.detach().clone();mask=query.atom_mask
    if role=='event':x=query.x_r.clone()
    trace_b=[b.clone()];trace_x=[x.clone()]
    diag={k:torch.zeros(len(b),device=b.device) for k in [
        'generator_forward','score_member_forward','score_member_backward','eligible_steps',
        'barrier_guidance_steps','skipped_out_of_support_steps','skipped_zero_gradient_steps',
        'guidance_rms_total_angstrom']}
    failed=torch.zeros(len(b),device=b.device,dtype=torch.bool)
    if replay is not None and replay.shape[0]!=n_steps+1:
        raise ValueError('replay must contain every shadow event state')
    for k in range(start,end):
        active=~failed
        bt=torch.full((len(b),),1. if role=='geometry' else float(tb[k]),device=b.device)
        xt=torch.full((len(b),),0. if role=='event' else float(tx[k]),device=b.device)
        old=State(b,x,bt,xt)
        kwargs=dict(z=query.element_index,atom_mask=mask,x_cur=x,x_r=query.x_r,
                    b_cur=b,b_r=query.b_r,t=xt if role=='geometry' else bt)
        if role=='joint' and dual_time:kwargs['t_x']=xt
        out=net(**kwargs)
        diag['generator_forward']+=active
        finite=torch.ones_like(active)
        for value in out.values():finite &= torch.isfinite(value).flatten(1).all(1)
        failed |= ~finite
        active &= finite
        delta=torch.zeros_like(b if role=='event' else x)
        if alpha and score is not None:
            time=float(tb[k] if role=='event' else tx[k])
            if guidance_start<=time<guidance_stop:
                direction,info=score_direction(score,query,old,kind='E' if role=='event' else 'X')
                diag['eligible_steps']+=active
                diag['score_member_forward']+=3*active
                diag['score_member_backward']+=3*active
                applicable=info['applicable']&active
                diag['barrier_guidance_steps']+=applicable
                diag['skipped_out_of_support_steps']+=(~info['supported'])&active
                diag['skipped_zero_gradient_steps']+=(~info['nonzero'])&info['supported']&active
                dt=float(tb[k+1]-tb[k] if role=='event' else tx[k+1]-tx[k])
                if role=='event':
                    delta=direction*alpha*dt
                else:
                    delta,actual=geometry_step(direction,mask,alpha*dt)
                    diag['guidance_rms_total_angstrom']+=actual*applicable
                delta*=applicable[:,None,None]
        next_b=b if role=='geometry' else b+float(tb[k+1]-tb[k])*out['b_vel']
        next_x=x if role=='event' else x+float(tx[k+1]-tx[k])*out['x_vel']
        if role=='event':next_b=next_b+delta
        else:next_x=next_x+delta
        if replay is not None:next_b=replay[k+1].clone()
        finite=torch.isfinite(next_b).flatten(1).all(1)&torch.isfinite(next_x).flatten(1).all(1)
        failed |= ~finite;active &= finite
        b=torch.where(active[:,None,None],next_b,b).detach()
        x=query.x_r.clone() if role=='event' else torch.where(
            active[:,None,None],centered(next_x,mask),x).detach()
        trace_b.append(b.clone());trace_x.append(x.clone())
    bt=torch.full((len(b),),1. if role=='geometry' else float(tb[end]),device=b.device)
    xt=torch.full((len(b),),0. if role=='event' else float(tx[end]),device=b.device)
    return Rollout(State(b,x,bt,xt),torch.stack(trace_b),torch.stack(trace_x),diag,failed)
