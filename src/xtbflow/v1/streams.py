"""Reactant-only logical five-arm programs, batched across independent parents.

Each coroutine defines one parent's immutable order of operations. The executor
may batch matching requests but cannot alter that order or their charged costs.
Diagnostic snapshot gradients and shadow controls are separately metered, never
read by a program's proposal, ranking, promotion or stopping decisions.
"""
from collections import defaultdict
from dataclasses import dataclass,field

import numpy as np
import torch

from xtbflow.m0.metrics import be_to_mol
from xtbflow.m0.sampler import decode_be
from .clocks import clock_grid,observation_index
from .costs import Ledger
from .guidance import score_direction
from .interfaces import Query,State
from .sampler import initial_state,rollout
from .serial_search import promote,schedule


@dataclass
class Proposal:
    number: int
    state: State
    initial: State
    snapshots: dict=field(default_factory=dict)
    diagnostics: dict=field(default_factory=lambda:defaultdict(float))
    valid: bool=True
    shadow: dict | None=None


@dataclass
class Request:
    query: Query
    proposal: Proposal
    role: str
    start: int=0
    end: int=50
    alpha: float=0.
    path: str='sync'
    guidance_start: float=.5
    guidance_stop: float=.95

    def signature(self):
        return (self.role,self.start,self.end,self.alpha,self.path,self.guidance_start,self.guidance_stop)


def decode_event(query,state):
    n=int(query.atom_mask[0].sum())
    bp,_=decode_be(state.b[0],query.element_index[0],n)
    if bp is None or np.array_equal(bp,query.b_r[0,:n,:n].cpu().numpy()):return None
    try:be_to_mol(query.atomic_numbers[0,:n].cpu().numpy(),bp)
    except (ValueError,RuntimeError):return None
    return bp


def endpoint_record(query,p):
    n=int(query.atom_mask[0].sum());bp=decode_event(query,p.state) if p.valid else None
    return dict(proposal=p.number,b_dec=None if bp is None else bp.tolist(),
                x=p.state.x[0,:n].detach().cpu().tolist(),is_fallback=bp is None,
                decode_status='VALID' if bp is not None else 'DECODE_FAILED',
                snapshots=list(p.snapshots.values()),guidance=dict(p.diagnostics),shadow=p.shadow)


def logical_stream(query,arm,seed,weights,config,output):
    """Output contains all operations/attempts, including partial/eliminated work."""
    if arm not in ('A0','A1','A2','B0','B1'):raise ValueError(arm)
    ledger=Ledger(weights,cap=config.get('cap_units',3200.))
    output.update(candidates=[],attempts=ledger.attempts,operations=ledger.operations,
                  query_id=query.query_id[0],arm=arm,training_seed=seed,sampling_seed=0)
    qid=query.query_id[0];number=0

    def new_proposal(j):
        s=initial_state(query,[j],seed,namespace=config.get('namespace','efficiency'),
                        sigma_b=config['sigma_b'],sigma_x=config['sigma_x'],
                        group='cascade' if arm.startswith('A') else 'joint')
        return Proposal(j,s,s)

    def attempt_id(p):return f'{qid}/{arm}/s{seed}/p{p.number}'

    def segment(p,role,start,end,alpha=0.):
        path=config['path'] if role=='joint' else 'sync'
        tb,tx=clock_grid(path);clock=tb if role=='event' else tx
        kind='E' if role=='event' else 'X'
        begin=config.get('event_start',.5) if role=='event' else config['guidance_start']
        stop=config.get('event_stop',.95) if role=='event' else config['guidance_stop']
        last=start
        for k in range(start,end):
            guided=bool(alpha and begin<=float(clock[k])<stop)
            if not ledger.step(attempt_id(p),{'joint':'f','event':'g','geometry':'h'}[role],guided,kind):break
            last=k+1
        if last>start:
            yield Request(query,p,role,start,last,alpha,path,begin,stop)
        return last==end

    def finish(p,status='COMPLETE'):
        cid=attempt_id(p)+'/endpoint'
        record=endpoint_record(query,p)
        record.update(ledger.complete(attempt_id(p),cid,status))
        output['candidates'].append(record)

    def terminate(p,status):
        ledger.complete(attempt_id(p),None,status)

    def event(p):
        complete=yield from segment(p,'event',0,50,config['alpha_b'] if arm=='A1' else 0.)
        if not complete:return False
        bp=decode_event(query,p.state)
        if bp is None:
            p.valid=False;terminate(p,'INVALID_EVENT');return True
        b=torch.zeros_like(p.state.b);n=len(bp);b[0,:n,:n]=torch.as_tensor(bp,device=b.device,dtype=b.dtype)
        p.state=State(b,p.initial.x.clone(),torch.ones_like(p.state.t_b),torch.zeros_like(p.state.t_x))
        return True

    if arm!='A2':
        while ledger.spent<ledger.cap-1e-8:
            p=new_proposal(number);number+=1
            if arm.startswith('A'):
                if not (yield from event(p)):
                    terminate(p,'BUDGET_PARTIAL_EVENT');break
                if not p.valid:continue
                complete=yield from segment(p,'geometry',0,50)
            else:
                complete=yield from segment(p,'joint',0,50,config['alpha_x'] if arm=='B1' else 0.)
            if not complete:
                terminate(p,'BUDGET_PARTIAL');break
            finish(p)
    else:
        round_id=0;exhausted=False
        while not exhausted and ledger.spent<ledger.cap-1e-8:
            plan=schedule(round_id);pool=[]
            for _ in range(plan['proposals']):
                p=new_proposal(number);number+=1
                if not (yield from event(p)):
                    terminate(p,'BUDGET_PARTIAL_EVENT');exhausted=True;break
                if p.valid:pool.append(p)
            if exhausted:
                for p in pool:terminate(p,'BUDGET_PENDING_GEOMETRY')
                break
            checkpoints=[observation_index(clock_grid('sync')[1],float(t)) for t in config['a2_times']]+[50]
            start=0
            for stage,end in enumerate(checkpoints):
                for i,p in enumerate(pool):
                    if not (yield from segment(p,'geometry',start,end,config['alpha_x'])):
                        terminate(p,'BUDGET_PARTIAL_GEOMETRY');exhausted=True
                        for other in pool[i+1:]:terminate(other,'BUDGET_PENDING_GEOMETRY')
                        for other in pool[:i]:
                            if stage<2:terminate(other,'BUDGET_PENDING_PROMOTION')
                        break
                    if stage==2:finish(p)
                if exhausted or stage==2:break
                ranks={}
                for i,p in enumerate(pool):
                    if not ledger.charge(attempt_id(p),'X',3):
                        exhausted=True
                        for other in pool:terminate(other,'BUDGET_PENDING_PROMOTION')
                        break
                    rank=yield Request(query,p,'rank',end,end)
                    ranks[p.number]=rank
                if exhausted:break
                best,random=plan['promotions'][stage]
                keep=promote([p.number for p in pool],ranks,best,random,query=qid,
                             training_seed=seed,round_id=round_id,stage=stage)
                for p in pool:
                    if p.number not in keep:terminate(p,f'ELIMINATED_STAGE_{stage}')
                pool=[p for p in pool if p.number in keep];start=end
            round_id+=1
    output.update(spent_units=ledger.spent,cap_units=ledger.cap,n_proposals=number,
                  terminal_score_policy='Diagnostic endpoint scores only; no online terminal ranking or selection.')


def combine_queries(queries):
    return Query(tuple(q.query_id[0] for q in queries),
                 *[torch.cat([getattr(q,k) for q in queries]) for k in
                   ('atomic_numbers','element_index','atom_mask','x_r','b_r')])


def combine_states(states):
    return State(*[torch.cat([getattr(s,k) for s in states]) for k in ('b','x','t_b','t_x')])


def slice_state(state,i):
    return State(*[getattr(state,k)[i:i+1].detach().clone() for k in ('b','x','t_b','t_x')])


@torch.no_grad()
def execute_batch(requests,networks,scores,*,diagnostics=False,drift=False):
    request=requests[0];q=combine_queries([r.query for r in requests])
    state=combine_states([r.proposal.state for r in requests])
    if request.role=='rank':
        c=scores['X'].components(q,state.b,state.x,state.t_b,state.t_x)
        return [float(c['phi'][i]) if c['supported'][i] else None for i in range(len(requests))]
    result=rollout(networks[request.role],q,state,role=request.role,path=request.path,
                   start=request.start,end=request.end,score=scores['E' if request.role=='event' else 'X'],
                   alpha=request.alpha,guidance_start=request.guidance_start,guidance_stop=request.guidance_stop)
    if result.failed.any():raise FloatingPointError('numeric rollout failure; preserve job for repair')
    for i,r in enumerate(requests):
        r.proposal.state=slice_state(result.state,i)
        for key,value in result.diagnostics.items():r.proposal.diagnostics[key]+=float(value[i])
    if diagnostics and request.role!='event':
        tb,tx=clock_grid(request.path)
        for t in (.2,.35,.5,.65,.8,1.):
            k=observation_index(tx,t)
            if not request.start<=k<=request.end:continue
            bt=torch.full_like(state.t_b,1. if request.role=='geometry' else float(tb[k]))
            xt=torch.full_like(bt,float(tx[k]))
            snap=State(result.b_trace[k-request.start],result.x_trace[k-request.start],bt,xt)
            _,info=score_direction(scores['X'],q,snap)
            c=scores['X'].components(q,snap.b,snap.x,bt,xt)
            for i,r in enumerate(requests):
                r.proposal.snapshots[str(t)]=dict(requested_t=t,actual_t_x=float(tx[k]),actual_t_b=float(bt[i]),
                    **{key:float(info[name][i]) for key,name in [('score','score'),('mean_kcal','mean_kcal'),
                        ('sd_kcal','sd_kcal'),('gradient_norm','gradient_norm')]},
                    supported=bool(info['supported'][i]),applicable=bool(info['applicable'][i]),
                    member_kcal=c['member_kcal'][:,i].cpu().tolist())
    if drift and request.alpha and request.end==50 and request.role in ('geometry','joint'):
        if request.role=='geometry':
            un=State(result.state.b,torch.cat([r.proposal.initial.x for r in requests]),
                     torch.ones_like(state.t_b),torch.zeros_like(state.t_x))
        else:un=combine_states([r.proposal.initial for r in requests])
        shadow=rollout(networks[request.role],q,un,role=request.role,path=request.path)
        if shadow.failed.any():raise FloatingPointError('numeric drift shadow failure')
        for i,r in enumerate(requests):
            p=Proposal(r.proposal.number,slice_state(shadow.state,i),r.proposal.initial)
            r.proposal.shadow=endpoint_record(r.query,p)
    return [None]*len(requests)


def run_programs(programs,networks,scores,batch_size=64,diagnostics=False,drift=False):
    pending={}
    for i,program in enumerate(programs):
        try:pending[i]=next(program)
        except StopIteration:pass
    while pending:
        buckets=defaultdict(list)
        for i,r in pending.items():buckets[r.signature()].append(i)
        # Stable choice affects physical batching only; no shared per-parent RNG/state.
        key=max(buckets,key=lambda k:len(buckets[k]));indices=buckets[key][:batch_size]
        results=execute_batch([pending[i] for i in indices],networks,scores,
                              diagnostics=diagnostics,drift=drift)
        for i,value in zip(indices,results):
            try:pending[i]=programs[i].send(value)
            except StopIteration:del pending[i]
