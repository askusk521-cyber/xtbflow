"""Training-fold catalogue barrier regressors; gradients are guidance, not forces."""
import torch
from torch import nn

from xtbflow.m0.backbone import ReactionTrunk


SCORE_CONFIG=dict(scalar_dim=64,vector_dim=16,edge_dim=32,n_layers=3,
                  n_rbf=32,n_rbf_reactant=16,r_cut=10.)


class BarrierHead(nn.Module):
    def __init__(self,kind,**config):
        super().__init__()
        if kind not in ('E','X'):
            raise ValueError(kind)
        self.kind=kind
        c=dict(SCORE_CONFIG,**config)
        self.trunk=ReactionTrunk(**c,dual_time=kind=='X')
        self.readout=nn.Sequential(nn.Linear(c['scalar_dim'],c['scalar_dim']),nn.SiLU(),
                                   nn.Linear(c['scalar_dim'],1))

    def forward(self,query,b,x,t_b,t_x):
        # h_E has no dependency on the current TS geometry, even in caller graphs.
        h=self.trunk(query.element_index,query.atom_mask,x if self.kind=='X' else query.x_r,
                     query.x_r,b,query.b_r,t_b,t_x=t_x if self.kind=='X' else None)
        mask=query.atom_mask.to(h['s'].dtype)[...,None]
        pooled=(h['s']*mask).sum(1)/mask.sum(1).clamp_min(1)
        return self.readout(pooled).squeeze(-1)


class BarrierEnsemble(nn.Module):
    def __init__(self,members,median,scale,support=None):
        super().__init__()
        if len(members)!=3 or len({m.kind for m in members})!=1:
            raise ValueError('three members of one score kind required')
        if scale<5:
            raise ValueError('normalization scale must be at least 5 kcal/mol')
        self.members=nn.ModuleList(members)
        self.kind=members[0].kind
        self.median=float(median);self.scale=float(scale)
        self.support=support
        self.eval().requires_grad_(False)

    def components(self,query,b,x,t_b,t_x):
        standardized=torch.stack([m(query,b,x,t_b,t_x) for m in self.members])
        barriers=standardized*self.scale+self.median
        mu=barriers.mean(0)
        sigma=((barriers-mu).square().mean(0)+1e-12).sqrt()
        barrier_term=(mu-self.median)/self.scale
        uncertainty_term=.25*torch.clamp(sigma/self.scale,0,2)
        support=torch.isfinite(mu)&torch.isfinite(sigma)
        if self.support is not None:
            support=(support & (mu>=self.support['lower_kcal']) &
                     (mu<=self.support['upper_kcal']) & (sigma<=self.support['max_sd_kcal']))
        return dict(phi=barrier_term+uncertainty_term,mean_kcal=mu,sd_kcal=sigma,
                    barrier_term=barrier_term,uncertainty_term=uncertainty_term,
                    supported=support,member_kcal=barriers)

    def forward(self,query,b,x,t_b,t_x):
        return self.components(query,b,x,t_b,t_x)['phi']
