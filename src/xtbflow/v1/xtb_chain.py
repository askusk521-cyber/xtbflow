"""CPU-only exploratory GFN2-xTB certification; never invokes DFT."""
import os
from pathlib import Path
import time

import numpy as np

from .data import write_json
from .qc_protocol import (harmonic, classify_saddle, classify_minimum, endpoint_identity,
                          TIGHT, IRC, SYMBOLS, BOHR_PER_ANGSTROM, CLEAR_NEGATIVE_CM1)


def numerical_hessian(gradient, coords, step=0.005):
    """Central differences of Hartree/bohr gradients; coordinates in bohr."""
    coords = np.asarray(coords, dtype=float).ravel()
    h = np.empty((len(coords), len(coords)))
    for j in range(len(coords)):
        dx = np.zeros_like(coords)
        dx[j] = step
        h[:, j] = (np.asarray(gradient(coords+dx)).ravel()-np.asarray(gradient(coords-dx)).ravel())/(2*step)
    return (h+h.T)/2


class XTB:
    def __init__(self, z):
        self.z = np.asarray(z, dtype=np.int32)
        self.calls = 0
        self.rows = []
        self.stage = 'initial'

    def evaluate(self, coords):
        from tblite.interface import Calculator
        start = time.monotonic()
        calc = Calculator('GFN2-xTB', self.z, np.asarray(coords).reshape(-1,3), charge=0, uhf=0)
        calc.set('verbosity', 0)
        calc.set('accuracy', 0.1)
        calc.set('max-iter', 250)
        result = calc.singlepoint()
        self.calls += 1
        self.rows.append(dict(stage=self.stage, seconds=time.monotonic()-start))
        return float(result.get('energy')), np.asarray(result.get('gradient')).ravel()

    def hessian(self, x):
        coords = np.asarray(x).ravel()*BOHR_PER_ANGSTROM
        return numerical_hessian(lambda c: self.evaluate(c)[1], coords)


def optimize(engine, x, work, **params):
    from geometric.engine import Engine
    from geometric.molecule import Molecule
    from geometric.optimize import run_optimizer
    from geometric.errors import GeomOptNotConvergedError

    class Adapter(Engine):
        def __init__(self):
            mol = Molecule()
            mol.elem = [SYMBOLS[int(v)] for v in engine.z]
            mol.xyzs = [np.asarray(x)]
            mol.comms = ['']
            mol.build_topology()
            super().__init__(mol)

        def calc_new(self, coords, dirname):
            e, g = engine.evaluate(coords)
            return dict(energy=e, gradient=g)

    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(work)
        progress = run_optimizer(customengine=Adapter(), input='geometric.in', prefix='opt',
                                 coordsys='tric', qdata=False, **TIGHT, **params)
        return [np.asarray(frame) for frame in progress.xyzs]
    except GeomOptNotConvergedError:
        return []
    finally:
        os.chdir(cwd)


def certification_chain(z, x, br, perms, bp, work):
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    engine = XTB(z)
    start = time.monotonic()
    out = dict(protocol='review-xtb-o1-1', method='GFN2-xTB', charge=0, uhf=0,
               hessian_step_bohr=0.005, clear_negative_cm1=CLEAR_NEGATIVE_CM1)
    def finish(status):
        out.update(status=status, wall_seconds=time.monotonic()-start, gradient_calls=engine.calls,
                   stage_timings={stage:sum(r['seconds'] for r in engine.rows if r['stage']==stage)
                                  for stage in sorted({r['stage'] for r in engine.rows})})
        write_json(work/'verdict.json', out)
        return out
    try:
        engine.stage='ts_initial_hessian'
        h = engine.hessian(x)
        initial = work/'initial.hess'
        np.savetxt(initial, h)
        engine.stage='ts_optimization'
        frames = optimize(engine, x, work/'ts', transition=True, hessian='file:'+str(initial), maxiter=200)
        if not frames:
            return finish('TS_OPT_LIMIT')
        xts = frames[-1]
        engine.stage='ts_final_hessian'
        h = engine.hessian(xts)
        freq, _, rank = harmonic(z, xts, h)
        e, _ = engine.evaluate(xts.ravel()*BOHR_PER_ANGSTROM)
        out['ts'] = dict(x=xts.tolist(), energy_hartree=e, frequencies_cm1=freq.tolist(), tr_rank=rank)
        status = classify_saddle(freq)
        if status!='TS_OPTFREQ_PASS':
            return finish(status)
        final = work/'final.hess'
        np.savetxt(final, h)
        engine.stage='irc'
        frames = optimize(engine, xts, work/'irc', irc=True, irc_direction='both',
                          hessian='file:'+str(final), **IRC)
        if not frames:
            return finish('IRC_LIMIT')
        ends = []
        for i, end in enumerate((frames[0], frames[-1])):
            engine.stage=f'endpoint_{i}_optimization'
            ef = optimize(engine, end, work/f'end_{i}', maxiter=200)
            if not ef:
                return finish('MIN_OPT_LIMIT')
            xm = ef[-1]
            engine.stage=f'endpoint_{i}_hessian'
            f, _, _ = harmonic(z, xm, engine.hessian(xm))
            mstatus = classify_minimum(f)
            ends.append(dict(x=xm.tolist(), frequencies_cm1=f.tolist(), status=mstatus))
            out['endpoints'] = ends
            if mstatus!='MINIMUM_PASS':
                return finish(mstatus)
        out['identity'] = endpoint_identity(z, br, perms, bp, np.array(ends[0]['x']), np.array(ends[1]['x']))
        return finish(out['identity']['status'])
    except Exception as exc:
        out['error'] = type(exc).__name__+': '+str(exc)
        out['failed_stage'] = engine.stage
        return finish('EVALUATION_UNRESOLVED')
