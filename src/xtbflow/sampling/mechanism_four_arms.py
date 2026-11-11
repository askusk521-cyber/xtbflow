"""Frozen N/T/F/C mechanism pilot. C is a counterfactual event-clamped replay."""
from __future__ import annotations

from dataclasses import replace

import torch

from .amplitude_backtracking import bounded_backtrack
from .integrated_guidance import FlowState, Injection, apply_update, model_values, prepare_injection
from .physical_diagnostic import physical_fields


PULSES=(10,25,40)
STEPS=50


def endpoint_jvp(model,query,base,direction,meter,context):
    """One JVP on the NEW actual base, no unbudgeted finite differences."""
    unit=direction/direction.norm().clamp_min(1e-30)
    def values(x):
        y,vb,_=model_values(model,query,replace(base,x=x))
        return y,vb
    receipt=meter.begin('jvp',context)
    try:
        (y0,vb0),(jx,jb)=torch.autograd.functional.jvp(values,base.x,unit,strict=True)
    except Exception:
        meter.end(receipt,'execution_failed')
        raise
    meter.end(receipt)
    finite=bool(torch.isfinite(jx).all() and torch.isfinite(jb).all())
    floor=100*torch.finfo(base.x.dtype).eps*(float(y0.norm()+vb0.norm())+1)/1e-4
    projection=float((jx*unit).sum())
    qualified=finite and float((jb-jb.transpose(1,2)).abs().max())<1e-7 and abs(float(jb.sum()))<1e-6
    return dict(unit=unit.detach(),jx=jx.detach(),jb=jb.detach(),y0=y0.detach(),vb0=vb0.detach()),dict(
        qualified=qualified,endpoint_response_resolved=bool(jx.norm()>floor),endpoint_projection=projection,
        endpoint_gain=float(jx.norm()),endpoint_cosine=projection/max(float(jx.norm()),1e-30),
        event_response_norm=float(jb.norm()),numerical_floor=floor,
        qualification='finite/invariant AD plus each trial linearization; FD implementation qualified in #198')


def generate_four_arms(model,query,initial: FlowState,physical,meter) -> dict:
    """physical(arm,Y,label) is the only E/F boundary; all trained calls are metered."""
    states={};traces={};slots={'F':[],'T':[]};pushes={};work={};signals={}
    for arm in ('N','F','C'):
        before=meter.summary()['counts']
        state=initial
        b_trace=[state.b.clone()];x_trace=[state.x.clone()]
        for k in range(STEPS):
            with torch.no_grad():
                injection=prepare_injection(model,query,state,(k+1)/STEPS,((k+1)/STEPS)**2)
            state=injection.base
            if arm=='F' and k in PULSES:
                row=dict(step=k,status='slot_started',old_t_x=float(injection.observed.t_x[0]),
                         next_t_x=float(injection.base.t_x[0]),dt_x=injection.dt_x)
                slots['F'].append(row)
                receipt=physical('F',injection.predicted_x,f'F:k{k}:old_endpoint')
                row['observation_status']=receipt['status']
                if receipt['status']=='completed':
                    f=torch.as_tensor(receipt['force'],dtype=state.x.dtype)[None]
                    f,g,u,valid=physical_fields(f,injection.predicted_b,query.b_r,injection.predicted_x,query.atom_mask)
                    signals[f'F:{k}']=dict(predicted_b=injection.predicted_b,predicted_x=injection.predicted_x,
                                            base_x=injection.base.x,base_b=injection.base.b,force=f,G=g,u=u)
                    if valid and float(g.norm())>1e-10:
                        derivative,qualification=endpoint_jvp(model,query,injection.base,g,meter,f'F:k{k}:G')
                        signals[f'F:{k}']['derivative']=derivative
                        row['derivative']=qualification
                        state,_=bounded_backtrack(model,query,injection,g,derivative,qualification,
                                                   lambda y,label:physical('F',y,label),context=f'F:k{k}',audit=row)
                    else:
                        row.update(status='no_push',reason='event_direction_or_force_unavailable')
                else:
                    row.update(status='no_push',reason='physical_observation_unavailable')
                pushes[k]=(state.x-injection.base.x).detach().clone()
                row['actual_push_l2_angstrom']=float(pushes[k].norm())
            if arm=='C':
                # Clamp both input (previous step) and next BE to N, while its x velocity stays live.
                if not torch.equal(injection.observed.b,traces['N']['b'][k]):
                    raise ValueError('counterfactual_input_BE_mismatch')
                injection=replace(injection,base=replace(injection.base,b=traces['N']['b'][k+1].clone()))
                delta=pushes.get(k)
                state=(apply_update(injection,query,delta) if delta is not None and bool(delta.any()) else injection.base)
            b_trace.append(state.b.clone());x_trace.append(state.x.clone())
        states[arm]=state
        traces[arm]=dict(b=torch.stack(b_trace),x=torch.stack(x_trace))
        after=meter.summary()['counts'];work[arm]={key:after[key]-before[key] for key in before}
    before=meter.summary()['counts'];state=states['N']
    for k in PULSES:
        row=dict(step=k,status='slot_started');slots['T'].append(row)
        y0=state.x
        baseline=physical('T',y0,f'T:k{k}:baseline')
        row['observation_status']=baseline['status']
        if baseline['status']!='completed':
            row.update(status='no_push',reason='physical_observation_unavailable');continue
        force=torch.as_tensor(baseline['force'],dtype=state.x.dtype)[None]
        f,g,u,valid=physical_fields(force,state.b,query.b_r,y0,query.atom_mask)
        signals[f'T:{k}']=dict(geometry=y0,BE=state.b,force=f,G=g,u=u)
        if not valid or float(g.norm())<=1e-10:
            row.update(status='no_push',reason='event_direction_or_force_unavailable');continue
        unit=g/g.norm()
        derivative=dict(y0=y0,vb0=torch.zeros_like(state.b),jx=unit)
        qualification=dict(qualified=True,endpoint_response_resolved=True,endpoint_projection=1.)
        # Explicit registered terminal dose, never a zero dt derived from t_x=1.
        dt=((k+1)/STEPS)**2-(k/STEPS)**2
        injection=Injection(state,state,state.b,y0,dt)
        def terminal_physical(y,label):
            if label.endswith(':unperturbed_next_endpoint'):
                if not torch.equal(y,y0):raise ValueError('terminal_baseline_cache_mismatch')
                return baseline
            return physical('T',y,label)
        state,_=bounded_backtrack(None,query,injection,g,derivative,qualification,terminal_physical,
                                   context=f'T:k{k}',audit=row,
                                   endpoint_function=lambda s:(s.x,torch.zeros_like(s.b),None))
        if not torch.equal(state.b,states['N'].b):raise ValueError('terminal_event_changed')
    states['T']=state
    after=meter.summary()['counts'];work['T']={key:after[key]-before[key] for key in before}
    return dict(states=states,traces=traces,slots=slots,pushes=pushes,work=work,signals=signals)
