"""Development-only nuisance estimation and complete-gate working simulations.

The supplied illustrative simulator is not used as experimental evidence. Here
the group sizes come from a reserved inventory and nuisance estimates from dev.
"""
import numpy as np
from scipy.stats import t


def nuisance(values,groups):
    x=np.asarray(values,dtype=float);groups=np.asarray(groups)
    if x.ndim!=2 or x.shape[1]!=3 or len(groups)!=len(x) or not np.isfinite(x).all():
        raise ValueError('complete development [AUC,M0,MR] parent summaries required')
    keys=np.unique(groups);n=len(x);g=len(keys)
    if g<3 or n<=g:raise ValueError('insufficient clustered development replication')
    mean=x.mean(0);sd=x.std(0,ddof=1)
    if (sd<=1e-10).any():raise ValueError('degenerate nuisance; power cannot be inferred')
    sizes=np.array([(groups==k).sum() for k in keys]);n0=(n-np.square(sizes).sum()/n)/(g-1)
    msb=sum(len(x[groups==k])*np.square(x[groups==k].mean(0)-mean) for k in keys)/(g-1)
    msw=sum(np.square(x[groups==k]-x[groups==k].mean(0)).sum(0) for k in keys)/(n-g)
    icc=(msb-msw)/(msb+(n0-1)*msw)
    return dict(n_parents=n,n_formula_groups=g,means=mean.tolist(),sd=sd.tolist(),
                icc_raw=icc.tolist(),icc_nonnegative=np.maximum(0,icc).tolist(),
                correlation=np.corrcoef(x.T).tolist(),group_sizes=sizes.tolist(),
                seed_averaging_discount=1.,
                scope='Seed-0 development nuisance; no assumed variance reduction from averaging three training seeds.')


def complete_gate_batch(x,sizes):
    """x [replicate,parent,3], same ordered complete formula groups each draw."""
    sizes=np.asarray(sizes,dtype=int);g=len(sizes);n=int(sizes.sum())
    if x.ndim!=3 or x.shape[1:]!=(n,3) or g<2 or (sizes<1).any():
        raise ValueError('invalid simulated group shape')
    mean=x.mean(1);starts=np.r_[0,np.cumsum(sizes)[:-1]]
    scores=np.add.reduceat(x-mean[:,None,:],starts,axis=1)
    se=np.sqrt(g/(g-1)*np.square(scores).sum(1))/n
    q=float(t.ppf(.95,g-1));lower=mean-q*se;upper=mean+q*se
    degenerate=se<=1e-12;lower=np.where(degenerate,-1,lower);upper=np.where(degenerate,1,upper)
    nr=upper[:,0]<.05
    nm=~nr & ((upper[:,1]<.02)|(upper[:,2]<.02))
    go=~nr & ~nm & (lower>0).all(1)
    return np.stack([go,nr,nm,~(go|nr|nm)],axis=1)


def simulate_gate(values,groups,sizes,effects,*,repeats=20000,seed=20261007,
                  sd_multiplier=1.,icc_increment=0.,distribution='normal'):
    x=np.asarray(values,dtype=float);nu=nuisance(x,groups)
    sizes=np.asarray(sizes,dtype=int);g=len(sizes);n=int(sizes.sum())
    group_index=np.repeat(np.arange(g),sizes)
    sd=np.array(nu['sd'])*sd_multiplier
    icc=np.clip(np.array(nu['icc_nonnegative'])+icc_increment,0,.95)
    corr=np.array(nu['correlation']);eigen,vec=np.linalg.eigh(corr)
    factor=vec@np.diag(np.sqrt(np.maximum(eigen,0)))
    empirical=(x-x.mean(0))/x.std(0,ddof=0)
    rng=np.random.default_rng(seed);counts=np.zeros(4,dtype=np.int64);outside=0;mean_sum=np.zeros(3)
    for start in range(0,repeats,512):
        b=min(512,repeats-start)
        if distribution=='normal':
            group_z=rng.standard_normal((b,g,3))@factor.T
            parent_z=rng.standard_normal((b,n,3))@factor.T
        elif distribution=='empirical_skew':
            # Resample whole metric vectors to preserve the observed skewness,
            # sparse mechanisms and cross-metric dependence; impose planned ICC.
            group_z=empirical[rng.integers(0,len(x),(b,g))]
            parent_z=empirical[rng.integers(0,len(x),(b,n))]
        else:raise ValueError('unknown working distribution')
        samples=np.array(effects)+(np.sqrt(icc)*group_z[:,group_index]+np.sqrt(1-icc)*parent_z)*sd
        outside+=int((np.abs(samples)>1).sum())
        mean_sum+=samples.mean(1).sum(0)
        counts+=complete_gate_batch(samples,sizes).sum(0)
    rates=counts/repeats
    return dict(**dict(zip(('GO','NO_GO_RESOURCE','NO_GO_MECHANISM','HOLD'),rates.tolist())),
                mcse_GO=float(np.sqrt(rates[0]*(1-rates[0])/repeats)),repeats=repeats,seed=seed,
                effects=list(effects),n_parents=n,n_formula_groups=g,group_sizes=sizes.tolist(),
                sd_used=sd.tolist(),icc_used=icc.tolist(),distribution=distribution,
                mean_simulated_effect=(mean_sum/repeats).tolist(),
                outside_support_fraction=outside/(repeats*n*3),
                scope='Location-shift parent-summary working model, untruncated. Out-of-support draws disclosed; no claimed physical simulation.')
