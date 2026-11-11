"""One-factor residual backtracking at the shared post-Euler injection base."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

import torch

from .integrated_guidance import Injection, apply_update, centered, model_values
from .physical_diagnostic import displacement_metrics, rms_atom


ALPHA_CAP = .1  # Angstrom per unit dimensionless geometry clock.
LAMBDAS = (1., .5, .25)
FORCE_FLOOR = 1e-10  # Hartree/Angstrom flattened L2.


def geometry_step(direction, mask, amount):
    """Unchanged e84e330 common max-atom scaling (3 times the nominal RMS cap)."""
    maxnorm=direction.norm(dim=-1).max(1).values
    factor=torch.clamp(3./maxnorm.clamp_min(1e-10),max=1.)
    actual=torch.as_tensor(amount,dtype=direction.dtype,device=direction.device)*factor
    return direction*actual[:,None,None],actual


def residual(force, mask):
    return float(.5*rms_atom(centered(force,mask),mask)[0].square())


def bounded_backtrack(model, query, injection: Injection, reflected, derivative: dict,
                      qualification: dict, physical: Callable, *, context: str,
                      audit: dict | None = None, endpoint_function: Callable | None = None) -> tuple[object,dict]:
    """Freeze direction, event, clocks and base; return a state with selected push or no push.

    physical(Y, label) returns a settled finite E/F receipt or an explicit
    calculation failure. Unknown termination/accounting must raise and abort.
    Cached JVP is per unit flattened L2; its scaling uses the actual delta.
    """
    zero=torch.zeros_like(injection.base.x)
    report={} if audit is None else audit
    report.update(status='executing',selected_lambda=None,trials=[],alpha_cap=ALPHA_CAP,
                  lambdas=list(LAMBDAS),actual_displacement=displacement_metrics(zero,query.atom_mask))
    if (not qualification.get('qualified') or not qualification.get('endpoint_response_resolved') or
            qualification.get('endpoint_projection',0)<=0 or float(reflected.norm())<=FORCE_FLOOR):
        report.update(status='no_push',reason='amplitude_only_repair_not_supported')
        return injection.base,report
    d=reflected/rms_atom(reflected,query.atom_mask)[:,None,None]
    delta0,actual=geometry_step(d,query.atom_mask,ALPHA_CAP*injection.dt_x)
    y0=derivative['y0'];vb0=derivative['vb0'];jx=derivative['jx']
    report['nominal_displacement']=displacement_metrics(delta0,query.atom_mask)
    report['baseline_status']='started'
    baseline=physical(y0,context+':unperturbed_next_endpoint')
    report['baseline_status']=baseline['status']
    if baseline['status']!='completed':
        report.update(status='no_push',reason='baseline_physical_unavailable')
        return injection.base,report
    phi0=residual(torch.as_tensor(baseline['force'],dtype=y0.dtype)[None],query.atom_mask)
    report['baseline_phi']=phi0
    for scale in LAMBDAS:
        delta=delta0*scale
        row=dict(lambda_value=scale,physical_status='not_run',model_status='started',
                 displacement=displacement_metrics(delta,query.atom_mask),accepted=False)
        report['trials'].append(row)
        with torch.no_grad():
            trial_state=apply_update(injection,query,delta)
            y,vb,_=(model_values(model,query,trial_state) if endpoint_function is None
                    else endpoint_function(trial_state))
        row.update(model_status='completed',physical_status='started',next_event_velocity_delta_l2=float((vb-vb0).norm()))
        receipt=physical(y,context+f':lambda={scale}')
        row['physical_status']=receipt['status']
        if receipt['status']!='completed':
            row['reason']='trial_physical_unavailable'
            continue
        phi=residual(torch.as_tensor(receipt['force'],dtype=y.dtype)[None],query.atom_mask)
        move=y-y0
        linear=jx*delta.norm()
        error=float((move-linear).norm()/linear.norm().clamp_min(1e-12))
        projection=float((move*reflected).sum())
        flags=dict(residual=phi<=.99*phi0,positive_projection=projection>0,
                   linearization=error<=.10,displacement=float(rms_atom(move,query.atom_mask)[0])<=float(rms_atom(delta0,query.atom_mask)[0]))
        row.update(phi=phi,endpoint_projection=projection,linearization_relative_error=error,
                   endpoint_displacement=displacement_metrics(move,query.atom_mask),checks=flags,
                   endpoint_gain=float(move.norm()/delta.norm()),accepted=all(flags.values()))
        if row['accepted']:
            report.update(status='accepted',selected_lambda=scale,accepted_phi=phi,
                          actual_displacement=row['displacement'],reason='first_acceptable_trial')
            return trial_state,report
    report.update(status='no_push',reason='no_trial_satisfies_registered_rule')
    return injection.base,report
