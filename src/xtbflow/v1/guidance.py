"""Shared A2/B1/B2 input-gradient guidance and exactly paired pulse amplitudes."""
import torch

from xtbflow.m0.flow import project_conserved


def centered(v,mask):
    m=mask.to(v.dtype)[...,None]
    return (v-(v*m).sum(1,keepdim=True)/m.sum(1,keepdim=True).clamp_min(1))*m


def rms_atom(v,mask):
    return ((v.square().sum(-1)*mask).sum(1)/mask.sum(1).clamp_min(1)).sqrt()


def unit_geometry_direction(gradient,mask,eps=1e-10):
    g=centered(gradient,mask);size=rms_atom(g,mask);valid=size>eps
    unit=-g/size.clamp_min(eps)[:,None,None]
    return torch.where(valid[:,None,None],unit,torch.zeros_like(unit)),valid


def score_direction(score,query,state,kind='X'):
    with torch.enable_grad():
        target=(state.x if kind=='X' else state.b).detach().clone().requires_grad_(True)
        b=state.b.detach() if kind=='X' else target
        x=target if kind=='X' else state.x.detach()
        c=score.components(query,b,x,state.t_b,state.t_x)
        if not torch.isfinite(c['phi']).all():raise FloatingPointError('nonfinite score')
        gradient=torch.autograd.grad(c['phi'].sum(),target)[0]
    if not torch.isfinite(gradient).all():raise FloatingPointError('nonfinite input gradient')
    if kind=='X':
        direction,nonzero=unit_geometry_direction(gradient.detach(),query.atom_mask)
    else:
        g=project_conserved(.5*(gradient+gradient.transpose(1,2)),query.atom_mask).detach()
        n=query.atom_mask.sum(1).clamp_min(1).to(g.dtype)
        size=(g.square().sum((1,2))/(n*n)).sqrt();nonzero=size>1e-10
        direction=-g/size.clamp_min(1e-10)[:,None,None]
    applicable=nonzero & c['supported'].detach()
    direction=torch.where(applicable[:,None,None],direction,torch.zeros_like(direction))
    return direction,dict(applicable=applicable,nonzero=nonzero,supported=c['supported'].detach(),
                          gradient_norm=gradient.detach().flatten(1).norm(dim=1),
                          score=c['phi'].detach(),mean_kcal=c['mean_kcal'].detach(),
                          sd_kcal=c['sd_kcal'].detach())


def geometry_step(unit,mask,amount):
    maxnorm=unit.norm(dim=-1).max(1).values
    factor=torch.clamp(3./maxnorm.clamp_min(1e-10),max=1.)
    actual=torch.as_tensor(amount,dtype=unit.dtype,device=unit.device)*factor
    return unit*actual[:,None,None],actual


def paired_pulses(score_unit,random_raw,mask,amplitude,applicable):
    # unit_geometry_direction negates its input; random isotropy is unchanged.
    random_unit,random_valid=unit_geometry_direction(random_raw,mask)
    common=torch.maximum(score_unit.norm(dim=-1).max(1).values,
                         random_unit.norm(dim=-1).max(1).values)
    actual=amplitude*torch.clamp(3./common.clamp_min(1e-10),max=1.)
    actual=torch.where(applicable&random_valid,actual,torch.zeros_like(actual))
    return score_unit*actual[:,None,None],random_unit*actual[:,None,None],actual
