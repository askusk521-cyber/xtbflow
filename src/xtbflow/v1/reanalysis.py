"""Exploratory V1a re-analysis (task R): candidate-count and verification-cost budgets.

Offline only: reads sealed, re-matched V1a streams and never feeds generation.
The formal screen set has already been analysed once, so nothing computed here
can confirm or change the V1a decision.

Terms used throughout
- completed candidate: one entry of a stream's ``candidates`` list, in logical
  completion order; decode failures and fallbacks count, as in V1a.
- verification unit: a candidate that would be sent on to quantum
  verification. Invalid or fallback candidates never open one: they are
  discarded before any calculation and only cost generation units.
- kappa: generation cost units charged per opened verification unit. One
  unguided B0 candidate costs 50 units, so budget point k means
  T_kappa(k) = k * (50 + kappa), "k B0-equivalent candidates verified once".
- cell: one (parent, arm, training seed) stream; censored cells are unknown,
  never imputed.
"""
import numpy as np

from xtbflow.m0.metrics import batched_kabsch_rmsd
from .metrics import cluster_summary

AUC_WEIGHTS=(.125,.25,.25,.25,.125)
B0_UNITS=50.
TOL=1e-8


def mode_key(spec):
    """Stable name of a verification-unit mode, e.g. 'D1' or 'D2_0.30'."""
    if spec['mode']=='D1':return 'D1'
    if spec['mode']=='D2':return f"D2_{float(spec['rmsd_cut_angstrom']):.2f}"
    raise ValueError(f"unknown verification-unit mode {spec['mode']}")


def same_event_rmsd(perms,b_ref,x_ref,b,x):
    """Minimum RMSD (Å) between two candidates that decode to the same channel.

    Same rule as CatalogueMatcher with the earlier candidate as the reference:
    only reactant automorphisms sigma with b_ref[sigma, sigma] == b are allowed,
    the SAME sigma permutes the coordinates, and Kabsch uses proper rotations.
    """
    perms=np.asarray(perms,dtype=np.int64);rb=np.asarray(b_ref)
    maps=perms[np.all(rb[perms[:,:,None],perms[:,None,:]]==np.asarray(b),axis=(1,2))]
    if len(maps)==0:raise RuntimeError('channel quotient disagrees with exact atom mapping')
    return float(batched_kabsch_rmsd(np.asarray(x_ref,dtype=float)[maps],np.asarray(x,dtype=float)).min())


def opens_verification_unit(cand,earlier_units,mode,rmsd_cut=0.30,rmsd=None):
    """earlier_units: units already opened in the same (parent, arm, seed) stream.

    Invalid or fallback candidates return False: they are discarded before any
    quantum calculation and only cost generation units.
    D1: a new unit opens only the first time a channel_id appears.
    D2: a new unit opens when the channel_id is new, or when the minimum RMSD to
        the earlier units of that channel exceeds rmsd_cut (Å). ``rmsd(unit,
        cand)`` must follow the V1a proxy-matching rule (see same_event_rmsd).
    """
    channel=cand.get('predicted_channel_id')
    if channel is None or cand.get('proxy_status')=='INVALID_OUTPUT':return False
    same=[u for u in earlier_units if u['predicted_channel_id']==channel]
    if mode=='D1':return not same
    if mode!='D2':raise ValueError(f'unknown verification-unit mode {mode}')
    if rmsd is None:raise ValueError('D2 needs an rmsd(unit, candidate) function')
    return not same or min(rmsd(u,cand) for u in same)>rmsd_cut


def verification_flags(cands,spec,rmsd=None):
    """Per-candidate 'opens a new unit' flags for one stream, in completion order."""
    units=[];flags=[]
    for c in cands:
        new=opens_verification_unit(c,units,spec['mode'],spec.get('rmsd_cut_angstrom',.30),rmsd)
        flags.append(bool(new))
        if new:units.append(c)
    return flags


def count_prefix(n_completed,n):
    """Analysis A, main axis: the first n completed candidates, or None if censored."""
    return n if n_completed>=n else None


def unit_prefix(opens,n):
    """Analysis A, unit axis: prefix ending at the opener of the n-th unit, or None."""
    seen=0
    for i,o in enumerate(opens):
        seen+=bool(o)
        if seen==n:return i+1
    return None


def total_costs(completion_units,opens,kappa):
    """total_cost(j) = completion_units(j) + kappa * units opened up to and including j."""
    return np.asarray(completion_units,dtype=float)+kappa*np.cumsum(np.asarray(opens,dtype=float))


def budget_prefix(completion_units,opens,kappa,budget,cap_units):
    """Analysis B: number of candidates within ``budget`` total cost, or None if censored.

    The stream stopped at its generation cap. Every unobserved continuation
    candidate completes after cap_units and has opened at least the units seen
    so far, so it costs more than H = cap_units + kappa * V_total. The prefix is
    exact iff budget <= H; with kappa = 0 this is V1a's completion_units <= 50k.
    """
    opens=np.asarray(opens,dtype=bool)
    horizon=cap_units+kappa*float(opens.sum())
    if budget>horizon+TOL:return None
    cost=total_costs(completion_units,opens,kappa)
    if np.any(np.diff(cost)<=0):raise ValueError('completion order is not strictly increasing in cost')
    return int(np.searchsorted(cost,budget+TOL,side='right'))


def included_points(censored_fraction,arms,limit=.05):
    """Budget points whose censored-cell fraction is <= limit for EVERY compared arm.

    censored_fraction[arm] lists, per budget point, the fraction of that arm's
    (parent, seed) cells that are censored.
    """
    n=len(next(iter(censored_fraction.values())))
    return [all(censored_fraction[a][i]<=limit for a in arms) for i in range(n)]


def renormalized_auc(hits,included,weights=AUC_WEIGHTS):
    """AUC over the included budget points, weights renormalized to sum to one."""
    hits=np.asarray(hits,dtype=float);included=np.asarray(included,dtype=bool)
    if hits.shape!=included.shape or not included.any():raise ValueError('no included budget point')
    w=np.asarray(weights,dtype=float)[included]
    return float(hits[included]@w/w.sum())


def summarize(values,groups):
    """cluster_summary when it is defined; otherwise an explicit status."""
    if not len(values):return dict(estimate=None,n_parents=0,n_formula_groups=0,status='NO_AVAILABLE_PARENTS')
    if len(values)<2 or len(set(groups))<2:
        return dict(estimate=float(np.mean(values)),n_parents=len(values),
                    n_formula_groups=len(set(groups)),status='TOO_FEW_GROUPS_FOR_INTERVAL')
    return cluster_summary(values,groups)


def parent_mean(cells,order,groups,seeds):
    """Arm marginal: mean over available seeds in each parent, then over parents.

    cells[(parent, seed)] is a number or None (censored). Returns
    (mean or None, parent values, their groups, dropped parents).
    """
    values=[];kept=[];dropped=[]
    for p,g in zip(order,groups):
        v=[cells[p,s] for s in seeds if cells.get((p,s)) is not None]
        if v:values.append(float(np.mean(v)));kept.append(g)
        else:dropped.append(p)
    return (float(np.mean(values)) if values else None),values,kept,dropped


def paired_difference(x_cells,y_cells,order,groups,seeds):
    """Parent-paired X - Y using only seeds available in both arms.

    Returns (summary, coverage fraction of parents, dropped parents).
    """
    values=[];kept=[];dropped=[]
    for p,g in zip(order,groups):
        d=[x_cells[p,s]-y_cells[p,s] for s in seeds
           if x_cells.get((p,s)) is not None and y_cells.get((p,s)) is not None]
        if d:values.append(float(np.mean(d)));kept.append(g)
        else:dropped.append(p)
    return summarize(values,kept),len(values)/len(order),dropped


def _sign(value):
    # Paired means of 0/1 data can be -1e-18 instead of 0; treat that as zero.
    return float(np.sign(round(float(value),12)))


def kappa_star(path):
    """First grid kappa whose estimate changes sign relative to kappa = 0.

    ``path`` is [(kappa, summary or None)] in grid order starting at kappa 0.
    Kappas without an estimate are skipped and listed. Reaching exactly zero
    counts as a change.
    """
    (k0,s0),*rest=path
    if k0!=0 or s0 is None or s0.get('estimate') is None:raise ValueError('kappa = 0 estimate required')
    sign=_sign(s0['estimate']);previous=(k0,s0);skipped=[]
    for kappa,s in rest:
        if s is None or s.get('estimate') is None:
            skipped.append(kappa);continue
        if _sign(s['estimate'])!=sign:
            return dict(kappa_star=kappa,kappa_before=previous[0],at_kappa_star=s,
                        before=previous[1],sign_at_zero=sign,skipped_kappas=skipped)
        previous=(kappa,s)
    return dict(kappa_star=None,kappa_before=None,sign_at_zero=sign,skipped_kappas=skipped,
                last_estimated_kappa=previous[0],at_last=previous[1])


def atom_summary(counts):
    if not counts:return dict(n=0)
    c=np.asarray(counts,dtype=float)
    return dict(n=len(c),mean=float(c.mean()),quantiles_0_25_50_75_100=np.quantile(c,[0,.25,.5,.75,1]).tolist())
