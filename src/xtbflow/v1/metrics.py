"""V1a v2.1 pre-specified parent aggregation and formula-cluster inference."""
import numpy as np

AUC_WEIGHTS = np.array([1, 2, 2, 2, 1], dtype=float) / 8.0


def parent_mechanism(score,none,random):
    """Inputs [parent, training_seed, proposal, amplitude], not independent rows."""
    s,n,r=(np.asarray(v,dtype=float) for v in (score,none,random))
    if s.shape!=n.shape or s.shape!=r.shape or s.ndim!=4 or s.shape[1:]!=(3,8,3):
        raise ValueError('expected complete [parent,3,8,3] paired utilities')
    if not all(np.isfinite(v).all() and np.all((0<=v)&(v<=1)) for v in (s,n,r)):
        raise ValueError('missing or invalid utility')
    return (s-n).mean((1,2,3)),(s-r).mean((1,2,3))


def cluster_bootstrap(values,formula_of,repeats=10000,seed=20261007):
    d=np.asarray(values,dtype=float);groups=np.asarray(formula_of,dtype=str)
    if d.shape!=groups.shape or not np.isfinite(d).all():raise ValueError('incomplete parent values')
    keys=np.unique(groups)
    if len(keys)<2:raise ValueError('at least two groups required')
    totals=np.array([d[groups==k].sum() for k in keys])
    counts=np.array([(groups==k).sum() for k in keys])
    indices=np.random.default_rng(seed).integers(0,len(keys),size=(repeats,len(keys)))
    means=totals[indices].sum(1)/counts[indices].sum(1)
    return dict(ci_two95=np.quantile(means,[.025,.975]).tolist(),repeats=repeats,seed=seed,
                method='whole_formula_groups_preserve_parent_weights_and_pairing')

def prefix_auc(hits):
    y = np.asarray(hits, dtype=float)
    if y.shape[-1] != 5:
        raise ValueError("expected budgets 4,8,16,32,64")
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("invalid hit values")
    if np.any(np.diff(y, axis=-1) < 0):
        raise ValueError("cumulative recall must be monotone")
    return y @ AUC_WEIGHTS

def parent_auc_difference(a2_hits, b1_hits):
    a = np.asarray(a2_hits, dtype=float)
    b = np.asarray(b1_hits, dtype=float)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (3, 5):
        raise ValueError("shape must be [parents,3 training seeds,5 budgets]")
    return (prefix_auc(b) - prefix_auc(a)).mean(axis=1)


from scipy.stats import t
import numpy as np

def cluster_summary(values, formula_of):
    d = np.asarray(values, dtype=float)
    groups = np.asarray(formula_of, dtype=str)
    if d.ndim != 1 or groups.shape != d.shape or len(d) < 2:
        raise ValueError("one complete value per parent required")
    if not np.isfinite(d).all():
        raise ValueError("missing values must be resolved upstream")
    keys = np.unique(groups)
    g = len(keys)
    if g < 2:
        raise ValueError("at least two independent formula groups required")
    mean = float(d.mean())
    scores = np.array([(d[groups == key] - mean).sum() for key in keys])
    se = float(np.sqrt(g / (g - 1) * np.square(scores).sum()) / len(d))
    if se <= 1e-12:
        return {
            "estimate": mean, "se_cluster": se,
            "lower_one95": -1.0, "upper_one95": 1.0,
            "ci_two95": [-1.0, 1.0],
            "n_parents": len(d), "n_formula_groups": g,
            "status": "DEGENERATE_VARIANCE",
            "scope": "conservative support bounds; t inference unresolved"
        }
    q_one = float(t.ppf(0.95, g - 1))
    q_two = float(t.ppf(0.975, g - 1))
    return {
        "estimate": mean, "se_cluster": se,
        "lower_one95": mean - q_one * se,
        "upper_one95": mean + q_one * se,
        "ci_two95": [mean - q_two * se, mean + q_two * se],
        "n_parents": len(d), "n_formula_groups": g,
        "scope": "conditional on frozen models and specified evaluation design"
    }


