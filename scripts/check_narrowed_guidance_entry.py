#!/usr/bin/env python3
"""Check actual archived guidance entries with synthetic networks, never weights/QC."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time


SOURCES = {'score': 'e84e330390a519bf4ac12048b297403c09717646',
           'physical_update': 'ffca40f9dc9b8f72f67c695a7b99fa38e83794e3'}


def check(entry: str) -> dict:
    import torch
    from xtbflow.m0.flow import project_conserved
    from xtbflow.v1.interfaces import Query, State
    from xtbflow.v1.sampler import rollout
    torch.set_num_threads(1)
    mask = torch.tensor([[True, True, False]])
    x = torch.tensor([[[-1.,0.,0.],[1.,0.,0.],[0.,0.,0.]]])
    b = torch.zeros(1,3,3)
    q = Query(('synthetic',),torch.tensor([[1,1,0]]),torch.tensor([[1,1,0]]),mask,x,b,0,1)
    st = State(b,x,torch.zeros(1),torch.zeros(1))
    network_calls = []
    class Network:
        def __init__(self):
            self.inputs = []
        def __call__(self, **kwargs):
            if len(network_calls) >= 16:
                raise RuntimeError('synthetic forward cap')
            network_calls.append(1)
            self.inputs.append(kwargs['x_cur'].clone())
            return {'b_vel':project_conserved(torch.diag_embed(kwargs['x_cur'][:,:,0]),mask),
                    'x_vel':torch.zeros_like(kwargs['x_cur'])}
    class Guide:
        def __init__(self, fail=False):
            self.calls, self.backwards, self.fail = 0, 0, fail
        def components(self, query, current_b, current_x, tb, tx):
            self.calls += 1
            if self.fail:
                raise RuntimeError('recording guide reached')
            def backward(gradient):
                self.backwards += 1
                return gradient
            current_x.register_hook(backward)
            phi = (current_x*torch.tensor([[[1.,0.,0.],[-1.,0.,0.],[7.,0.,0.]]])).sum((1,2))
            return dict(phi=phi, supported=torch.ones(1,dtype=torch.bool),mean_kcal=phi,sd_kcal=torch.zeros(1))

    if entry == 'score':
        net = Network()
        base = rollout(net,q,st,n_steps=2)
        observations = []
        for fail in (False,True):
            guide, installed_net = Guide(fail), Network()
            zero = rollout(installed_net,q,st,n_steps=2,score=guide,alpha=0.,guidance_start=0.,guidance_stop=1.)
            assert torch.equal(base.b_trace,zero.b_trace) and torch.equal(base.x_trace,zero.x_trace)
            assert guide.calls == guide.backwards == 0
            assert len(installed_net.inputs) == len(net.inputs) == 2
            assert all(torch.equal(zero.diagnostics[k],base.diagnostics[k]) for k in base.diagnostics)
            assert zero.diagnostics['score_member_forward'].item() == zero.diagnostics['score_member_backward'].item() == 0
            observations.append(dict(failing_guide=fail,trajectory_bitwise_equal=True,guide_calls=0,
                                     guide_backwards=0,synthetic_forwards=2,extra_charged_work=0))
        guide = Guide()
        active = rollout(Network(),q,st,n_steps=2,score=guide,alpha=.1,guidance_start=0.,guidance_stop=1.)
        assert guide.calls == guide.backwards == 2
        assert not torch.equal(active.x_trace,base.x_trace)
        failing = Guide(True)
        try:
            rollout(Network(),q,st,n_steps=2,score=failing,alpha=.1,guidance_start=0.,guidance_stop=1.)
        except RuntimeError as exc:
            assert str(exc) == 'recording guide reached'
        else:
            raise AssertionError('nonempty failing guide was never reached at positive alpha')
        assert failing.calls == 1
        return dict(entry='v1.sampler.rollout -> score_direction -> Guide.components',zero_strength=observations,
                    positive_control=dict(guide_calls=2,aggregate_backwards=2,
                                          declared_member_forwards=active.diagnostics['score_member_forward'].item(),
                                          declared_member_backwards=active.diagnostics['score_member_backward'].item()),
                    positive_failing_guide_reached=True,synthetic_forwards=len(network_calls))
    from xtbflow.v1.explore_rollout import recorded_rollout
    off_net, on_net, observed = Network(), Network(), []
    def push(k,b_hat,x_hat,current_x,dt_x):
        observed.append(dict(step=k,dt_x=dt_x))
        assert torch.equal(current_x,st.x)
        return torch.tensor([[[.1,0.,0.],[-.1,0.,0.],[0.,0.,0.]]]),torch.tensor([True])
    off = recorded_rollout(off_net,q,st,path='sync',n_steps=2,push=None,window=(0.,.5))
    on = recorded_rollout(on_net,q,st,path='sync',n_steps=2,push=push,window=(0.,.5))
    assert len(observed)==1 and on['applied'].item()==1
    assert torch.equal(on['b_trace'][1],off['b_trace'][1])
    assert not torch.equal(on['x_trace'][1],off['x_trace'][1])
    assert torch.equal(on_net.inputs[1],on['x_trace'][1])
    assert not torch.equal(on_net.inputs[1],off_net.inputs[1])
    assert not torch.equal(on['b_trace'][2],off['b_trace'][2])
    assert torch.equal(on['x_trace'][:,:,2],torch.zeros_like(on['x_trace'][:,:,2]))
    assert torch.allclose(on['x_trace'].sum(2),torch.zeros_like(on['x_trace'].sum(2)))
    assert torch.allclose(on['b_trace'].sum((2,3)),torch.zeros_like(on['b_trace'].sum((2,3))))
    return dict(entry='v1.explore_rollout.recorded_rollout actual push callback and Euler update',
                push_observations=observed,applied_pushes=1,same_step_event_equal=True,
                next_geometry_consumed=True,next_step_event_changed=True,mask_and_centering_preserved=True,
                synthetic_forwards=len(network_calls),physical_calls=0)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry',choices=SOURCES,required=True)
    parser.add_argument('--research-source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('receipt exists')
    root=Path(__file__).resolve().parents[1]
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**2,)*2)
    start,cpu=time.monotonic(),time.process_time()
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(schema='xtbflow-guidance-entry-check/v1',host=socket.gethostname(),python=platform.python_version(),
                driver_sha256=sha(Path(__file__)),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),
                new_trained_model_forwards=0,new_qc_calls=0,training_steps=0,gpu_seconds=0,
                caps=dict(cpu_seconds=60,wall_seconds=60,synthetic_forwards=16,output_bytes=4*1024**2,retries=0,concurrency=1))
    try:
        observed=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.research_source).decode().strip()
        if observed!=SOURCES[args.entry] or subprocess.check_output(['git','status','--porcelain'],cwd=args.research_source).strip():
            raise ValueError('exact clean research source required')
        sys.path.insert(0,str(args.research_source/'src'))
        report['research_commit']=observed
        files=['src/xtbflow/v1/sampler.py','src/xtbflow/v1/guidance.py','src/xtbflow/v1/clocks.py']
        if args.entry=='physical_update':files.append('src/xtbflow/v1/explore_rollout.py')
        report['source_file_sha256']={p:sha(args.research_source/p) for p in files}
        report['checks']=check(args.entry)
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',failure_type=type(exc).__name__,failure=str(exc))
    report.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
                  claim_limits=['Synthetic network and nonempty guides exercise real pinned code entries; no chemical/physical efficacy.',
                                'This adds boundary coverage; the previous 90 trained forwards and derivative evidence are unchanged.'])
    with args.out.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
    print(report['status'])
    return int(report['status']!='passed')


if __name__=='__main__':
    raise SystemExit(main())
