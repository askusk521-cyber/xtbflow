"""Frozen distinct-event recall, exact random ordering, and censor-aware AUC."""
import hashlib
import math
import numpy as np
from .metrics import cluster_summary

GRID=(1,2,4,8,16,32,64)
WEIGHTS={1:1,2:2,4:2,8:2,16:1}


def random_recall(n,m,k):
    if not 0<=m<=n or k<0:raise ValueError('invalid hypergeometric counts')
    k=min(k,n)
    if n==0 or m==0:return 0.
    if k>n-m:return 1.
    return 1-math.comb(n-m,k)/math.comb(n,k)


def generator_curve(channels,best):
    unique=list(dict.fromkeys(c for c in channels if c is not None))
    return {k:None if len(unique)<k else float(bool(set(unique[:k])&set(best))) for k in GRID}


def ranked_curve(channels,scores,best):
    if len(channels)!=len(scores) or not np.isfinite(scores).all():raise ValueError('invalid scores')
    order=sorted(range(len(channels)),key=lambda i:(scores[i],hashlib.sha256(channels[i].encode()).hexdigest()))
    return {k:float(bool({channels[i] for i in order[:k]}&set(best))) for k in GRID}


def paired_auc(enumeration,generation,groups):
    """Generation maps parent -> list(seed -> k -> hit or None)."""
    ids=sorted(enumeration.keys()&generation.keys())
    denominator=sum(len(generation[p]) for p in ids)
    censored={k:sum(row[k] is None for p in ids for row in generation[p])/denominator for k in GRID}
    kept=[k for k in WEIGHTS if censored[k]<=.05]
    if not kept:raise ValueError('no prespecified AUC point survives censoring')
    used=[];differences=[]
    for p in ids:
        means={k:[r[k] for r in generation[p] if r[k] is not None] for k in kept}
        if any(not x for x in means.values()):continue
        used.append(p)
        differences.append(sum(WEIGHTS[k]*(enumeration[p][k]-np.mean(means[k])) for k in kept)/sum(WEIGHTS[k] for k in kept))
    summary=cluster_summary(differences,[groups[p] for p in used])
    return dict(summary=summary,kept_k=kept,excluded_k=[k for k in WEIGHTS if k not in kept],
                censoring=censored,included_parents=used,excluded_parents=sorted(set(ids)-set(used)))
