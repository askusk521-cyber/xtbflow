"""Physical numerical-Hessian check on optimized water, no DFT."""
import argparse
from pathlib import Path
import numpy as np
from xtbflow.v1.xtb_chain import XTB, optimize
from xtbflow.v1.qc_protocol import harmonic, classify_minimum
from xtbflow.v1.data import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    z=np.array([8,1,1]);x=np.array([[0.,0.,0.],[.9572,0.,0.],[-.239987,.927297,0.]])
    engine=XTB(z)
    frames=optimize(engine,x,a.out/'water',maxiter=200)
    assert frames, 'water optimization failed'
    h=engine.hessian(frames[-1]);freq,_,rank=harmonic(z,frames[-1],h)
    assert rank==6 and len(freq)==3 and np.all(freq>0)
    assert np.max(np.abs(h-h.T))<1e-12
    write_json(a.out/'hessian_check.json',dict(passed=True,tr_rank=rank,frequencies_cm1=freq.tolist(),
        status=classify_minimum(freq),symmetry_error=float(np.max(np.abs(h-h.T))),gradient_calls=engine.calls))


if __name__=='__main__':main()
