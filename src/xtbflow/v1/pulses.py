"""Paired, three-amplitude mechanism branches from one frozen shadow trajectory."""
import torch

from .clocks import clock_grid,observation_index
from .guidance import paired_pulses,score_direction
from .interfaces import State
from .rng import addressed_seed
from .sampler import rollout


def random_direction(query,proposals,training_seed,step_index):
    out=torch.zeros_like(query.x_r)
    for k,(qid,j) in enumerate(zip(query.query_id,proposals)):
        n=int(query.atom_mask[k].sum())
        g=torch.Generator().manual_seed(addressed_seed('pulse_random',query=qid,
            training_seed=training_seed,sampling_seed=0,proposal=int(j),t_star=step_index))
        out[k,:n]=torch.randn(n,3,generator=g).to(out)
    return out


@torch.no_grad()
def pulse_branches(net,score,query,initial,proposals,*,training_seed=0,path='sync',
                   requested_t=.35,n_steps=50,shadow=None):
    shadow=shadow or rollout(net,query,initial,path=path,n_steps=n_steps)
    if shadow.failed.any():raise FloatingPointError('numerical failure in paired shadow')
    tb,tx=clock_grid(path,n_steps);k=observation_index(tx,requested_t)
    time_b=torch.full((len(query.x_r),),float(tb[k]),device=query.x_r.device)
    time_x=torch.full_like(time_b,float(tx[k]))
    state=State(shadow.b_trace[k],shadow.x_trace[k],time_b,time_x)
    unit,info=score_direction(score,query,state)
    random=random_direction(query,proposals,training_seed,k)
    results={'F0':shadow};pulse_meta={}
    for amplitude in (.05,.10,.20):
        ds,dr,actual=paired_pulses(unit,random,query.atom_mask,amplitude,info['applicable'])
        after={name:score.components(query,state.b,state.x+delta,time_b,time_x)['phi'].detach()
               for name,delta in [('F_S',ds),('F_R',dr)]}
        if not all(torch.isfinite(value).all() for value in after.values()):
            raise FloatingPointError('nonfinite post-pulse diagnostic score')
        pulse_meta[str(amplitude)]=dict(requested_rms=amplitude,actual_rms=actual,
                                       applicable=info['applicable'],score_after=after)
        for name,delta,replay in [('F_S',ds,None),('F_R',dr,None),('C_S',ds,shadow.b_trace)]:
            changed=State(state.b.clone(),state.x+delta,time_b,time_x)
            branch=rollout(net,query,changed,path=path,n_steps=n_steps,start=k,replay=replay)
            if branch.failed.any():raise FloatingPointError('numerical failure in paired '+name)
            if name=='C_S' and not torch.equal(branch.b_trace,shadow.b_trace[k:]):
                raise RuntimeError('complete event replay mismatch')
            results[name+'_'+str(amplitude)]=branch
    return results,dict(requested_t=requested_t,actual_t_b=float(tb[k]),actual_t_x=float(tx[k]),
                        step_index=k,pulses=pulse_meta,score=info)
