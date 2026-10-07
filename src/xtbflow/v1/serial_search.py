"""A2 promotion: fixed exploitation plus a distinct, addressed exploration item."""
import numpy as np
from .rng import addressed_seed


def promote(candidates,scores,keep_best,keep_random,*,query,training_seed,round_id,stage):
    if len(candidates)!=len(set(candidates)) or set(candidates)!=set(scores):
        raise ValueError('promotion candidate identities must be unique and complete')
    # Unsupported/nonfinite scores are ranked last, never replaced with true barriers.
    def rank(cid):
        score=scores[cid]
        return (float(score) if score is not None and np.isfinite(score) else float('inf'),cid)
    ranked=sorted(candidates,key=rank)
    best=ranked[:keep_best]
    others=sorted(set(candidates)-set(best))
    rng=np.random.default_rng(addressed_seed('a2_explore',query=query,training_seed=training_seed,
        sampling_seed=0,round=round_id,promotion_stage=stage))
    random=[] if not others else rng.choice(others,size=min(keep_random,len(others)),replace=False).tolist()
    return sorted(best+random)


def schedule(round_id):
    return dict(proposals=4 if round_id==0 else 8,
                promotions=[(2,1),(1,1)] if round_id==0 else [(4,1),(2,1)])
