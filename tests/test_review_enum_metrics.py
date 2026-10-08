import itertools
import numpy as np
from xtbflow.v1.review_enum_metrics import random_recall,generator_curve,ranked_curve,paired_auc


def test_random_matches_exhaustive_subsets():
    for n in range(1,9):
        for m in range(n+1):
            for k in range(n+2):
                subsets=list(itertools.combinations(range(n),min(k,n)))
                exact=sum(any(v<m for v in s) for s in subsets)/len(subsets)
                assert abs(random_recall(n,m,k)-exact)<1e-12
    assert random_recall(0,0,4)==0


def test_distinct_nonnull_and_censoring():
    r=generator_curve([None,'a','a','b'],{'b'})
    assert r[1]==0 and r[2]==1 and r[4] is None
    assert ranked_curve(['a'],np.array([0.]),{'a'})[64]==1


def test_auc_excludes_high_censoring():
    enum={p:{k:1. for k in (1,2,4,8,16,32,64)} for p in ('a','b','c')}
    gen={p:[generator_curve(['x','y'],{'z'})]*3 for p in enum}
    out=paired_auc(enum,gen,{'a':'F1','b':'F2','c':'F3'})
    assert out['kept_k']==[1,2]
    assert out['excluded_k']==[4,8,16]
    assert out['summary']['estimate']==1
