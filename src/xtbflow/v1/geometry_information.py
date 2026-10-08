"""Exploratory: does TS geometry rank channels beyond the event, and does it arrive in time?

Check A ranks a parent's known channels by barrier predictors that see the event
only, the event plus the reference TS geometry, or GFN2-xTB on that geometry.
Check B follows unguided joint rollouts and compares when the decoded event
commits with when the predicted endpoint geometry becomes energetically
informative. Development split only; nothing here feeds a gate.
"""
import numpy as np
from scipy.stats import spearmanr

from xtbflow.calculators.xtb_oracle import BOHR_IN_ANGSTROM,KELVIN_TO_HARTREE

HARTREE_TO_KCAL=627.509474


def event_table(rows,key,pool='min'):
    """One parent's records -> (channels, target, prediction) at the channel level.

    The target is the channel's lowest catalogue barrier. With pool='min' the
    prediction is the lowest predicted barrier among the channel's reference
    geometries, which uses only the predictor, never the catalogue label.
    """
    by={}
    for r in rows:by.setdefault(r['channel_id'],[]).append(r)
    channels=sorted(by)
    agg={'min':np.min,'mean':np.mean}[pool]
    target=np.array([min(r['catalog_barrier_kcal'] for r in by[c]) for c in channels],dtype=float)
    pred=np.array([agg([r[key] for r in by[c]]) for c in channels],dtype=float)
    return channels,target,pred


def ranking_metrics(target,pred,pair_gap=2.,tie=1e-6):
    """Within-parent ranking quality; None when the parent cannot rank (constant target)."""
    t=np.asarray(target,dtype=float);p=np.asarray(pred,dtype=float)
    if t.shape!=p.shape or len(t)<2 or not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise ValueError('need >= 2 finite paired channel values')
    if np.ptp(t)<=tie:return None
    rho=0. if np.ptp(p)<=1e-12 else float(spearmanr(t,p).statistic)
    best=t<=t.min()+tie;chosen=p<=p.min()+1e-12
    top1=float(best[chosen].mean())  # tied predictions share the hit
    i,j=np.triu_indices(len(t),1);keep=np.abs(t[i]-t[j])>=pair_gap
    if keep.any():
        s=np.sign(t[i]-t[j])[keep]*np.sign(p[i]-p[j])[keep]
        concordance=float(np.where(s>0,1.,np.where(s<0,0.,.5)).mean())
    else:concordance=None
    return dict(rho=rho,top1=top1,concordance=concordance,n_channels=len(t),n_pairs=int(keep.sum()))


def predicted_endpoints(trace,clock):
    """Unguided Euler trace [K+1,...] on clock [K+1] -> endpoint predictions at every step.

    The flow is trained on linear interpolants, so the velocity is a1-a0 and
    a1_hat(k) = a_k + (1-t_k)*v_k with v_k recovered from the next state. Only
    valid when no guidance or replay altered the step; step K is the endpoint.
    """
    clock=np.asarray(clock,dtype=float)
    if len(clock)!=len(trace) or np.any(np.diff(clock)<=0):raise ValueError('clock/trace mismatch')
    out=trace.clone()
    for k in range(len(clock)-1):
        v=(trace[k+1]-trace[k])/float(clock[k+1]-clock[k])
        out[k]=trace[k]+float(1-clock[k])*v
    return out


def commit_index(events):
    """First step after which every decoded event equals the final one (None if final invalid)."""
    events=list(events)
    if not events or events[-1] is None:return None
    k=len(events)-1
    while k>0 and events[k-1]==events[-1]:k-=1
    return k


def min_distance(x):
    x=np.asarray(x,dtype=float);d=np.linalg.norm(x[:,None]-x[None],axis=-1)
    return float(d[np.triu_indices(len(x),1)].min()) if len(x)>1 else np.inf


def xtb_energy_kcal(z,x,cfg):
    """GFN2-xTB single point in kcal/mol, or (nan, reason) for collapsed/failed geometries."""
    if not np.isfinite(np.asarray(x,dtype=float)).all():return float('nan'),'nonfinite'
    if min_distance(x)<cfg['min_distance_angstrom']:return float('nan'),'collapsed'
    from tblite.interface import Calculator
    try:
        calc=Calculator('GFN2-xTB',np.asarray(z,dtype=np.int32),np.asarray(x,dtype=float)/BOHR_IN_ANGSTROM,
                        charge=cfg['charge'],uhf=cfg['uhf'])
        calc.set('verbosity',0);calc.set('accuracy',float(cfg['accuracy']))
        calc.set('max-iter',int(cfg['max_iterations']))
        calc.set('temperature',float(cfg['electronic_temperature_kelvin'])*KELVIN_TO_HARTREE)
        e=float(calc.singlepoint().get('energy'))
    except (RuntimeError,ValueError) as exc:
        return float('nan'),'scf_failed:'+type(exc).__name__
    return (e*HARTREE_TO_KCAL,'ok') if np.isfinite(e) else (float('nan'),'nonfinite_energy')


def within_parent_rho(pred,target,parents,min_n=4):
    """Parent -> Spearman over its finite (pred, target) pairs; constant predictions give 0."""
    pred=np.asarray(pred,dtype=float);target=np.asarray(target,dtype=float);parents=np.asarray(parents)
    out={}
    for p in np.unique(parents):
        m=(parents==p)&np.isfinite(pred)&np.isfinite(target)
        if m.sum()<min_n or np.ptp(target[m])<=1e-6:continue
        out[str(p)]=0. if np.ptp(pred[m])<=1e-12 else float(spearmanr(pred[m],target[m]).statistic)
    return out


def first_informative(grid,mean_curve,final_lower,fraction=.5):
    """First grid time whose curve reaches `fraction` of its t=1 value, if that value is credibly > 0."""
    if final_lower is None or not final_lower>0:return None
    goal=fraction*mean_curve[-1]
    for t,v in zip(grid,mean_curve):
        if v is not None and np.isfinite(v) and v>=goal:return float(t)
    return None
