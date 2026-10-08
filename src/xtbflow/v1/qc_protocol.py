"""V1b H-level certification chain on gpu4pyscf + geomeTRIC (guide Sections 6 and 8).

Protocol substitution, frozen with the owner's choice: the guide's ORCA 6.1
templates are replaced by gpu4pyscf (RKS wB97X/6-31G(d), electronic energy,
analytic Hessian) driving geomeTRIC for OptTS, IRC and endpoint minimisation.
Convergence thresholds mirror ORCA TightOpt (max grad 1e-4, RMS grad 3e-5
Eh/bohr, energy 1e-6 Eh, max/RMS step 1e-3/6e-4 bohr). Units are explicit:
coordinates Angstrom at every API boundary, energies Hartree internally,
Hessians Hartree/bohr^2. Every network of calls is timed per stage so the
records double as a gpu4pyscf speed benchmark.

This module never reads proxy labels or reference energies; it receives a
query, a starting geometry and the predicted event, and returns evidence.
"""
from dataclasses import dataclass,field
import json
import os
from pathlib import Path
import time

import numpy as np

HARTREE_TO_KCAL=627.509474
EV_TO_KCAL=23.060548
BOHR_PER_ANGSTROM=1/0.52917721092
# sqrt(Hartree / (bohr^2 amu)) expressed in cm^-1.
AU_TO_WAVENUMBER=5140.48714
CLEAR_NEGATIVE_CM1=-30.
TIGHT=dict(convergence_energy=1e-6,convergence_grms=3e-5,convergence_gmax=1e-4,
           convergence_drms=6e-4,convergence_dmax=1e-3)
VERY_TIGHT=dict(convergence_energy=2e-7,convergence_grms=8e-6,convergence_gmax=3e-5,
                convergence_drms=1e-4,convergence_dmax=2e-4)
SYMBOLS={1:'H',6:'C',7:'N',8:'O'}
# D1 revision 2: 80 steps of the default 0.1 trust left 3/8 IRCs short of their minima;
# applied identically to every arm and reference. Per-direction iteration cap.
IRC=dict(maxiter=300,trust=0.2,tmax=0.2)  # revision 3: one D1 backward IRC was still descending at step 150
PROTOCOL_VERSION='v1b-h-gpu4pyscf-3'  # 3 = revision-2 IRC + one uniform endpoint retry


@dataclass
class Method:
    xc: str='wb97x'
    basis: str='6-31g*'
    charge: int=0
    spin: int=0
    grids_level: int=5
    conv_tol: float=1e-10
    max_cycle: int=300
    device: str='gpu'

    def as_dict(self):
        return dict(program='gpu4pyscf' if self.device=='gpu' else 'pyscf',xc=self.xc,basis=self.basis,
                    spherical=True,charge=self.charge,spin=self.spin,wavefunction='RKS',
                    grids_level=self.grids_level,conv_tol=self.conv_tol,max_cycle=self.max_cycle,
                    density_fitting=False,dispersion=None,solvent=None,energy_kind='electronic')


def to_np(a):
    return np.asarray(a.get() if hasattr(a,'get') else a)


@dataclass
class Meter:
    """Per-stage wall time and call counts; written as benchmark rows."""
    rows: list=field(default_factory=list)

    def record(self,stage,kind,seconds,**extra):
        self.rows.append(dict(stage=stage,kind=kind,wall_s=seconds,**extra))


def build_mol(z,x_ang,method):
    from pyscf import gto
    atom=[(SYMBOLS[int(a)],tuple(float(v) for v in p)) for a,p in zip(z,x_ang)]
    return gto.M(atom=atom,basis=method.basis,charge=method.charge,spin=method.spin,unit='Angstrom',verbose=0)


def make_scf(mol,method,dm0=None):
    if method.device=='gpu':
        from gpu4pyscf.dft import rks
        mf=rks.RKS(mol,xc=method.xc)
    else:
        from pyscf import dft
        mf=dft.RKS(mol,xc=method.xc)
    mf.grids.level=method.grids_level
    mf.conv_tol=method.conv_tol
    mf.max_cycle=method.max_cycle
    return mf


class SCFError(RuntimeError):
    pass


def single_point(z,x_ang,method,meter,stage,dm0=None,gradient=True):
    mol=build_mol(z,x_ang,method);mf=make_scf(mol,method)
    t=time.perf_counter();e=mf.kernel(dm0=dm0)
    ts=time.perf_counter()-t
    meter.record(stage,'scf',ts,n_atoms=len(z),nao=int(mol.nao),cycles=int(getattr(mf,'cycles',-1) or -1),
                 converged=bool(mf.converged))
    if not mf.converged:raise SCFError('SCF_LIMIT')
    g=None
    if gradient:
        t=time.perf_counter();g=to_np(mf.nuc_grad_method().kernel())
        meter.record(stage,'gradient',time.perf_counter()-t,n_atoms=len(z),nao=int(mol.nao))
    return float(e),g,mf


def hessian(z,x_ang,method,meter,stage):
    e,_,mf=single_point(z,x_ang,method,meter,stage,gradient=False)
    t=time.perf_counter();h=to_np(mf.Hessian().kernel())
    meter.record(stage,'hessian',time.perf_counter()-t,n_atoms=len(z),nao=int(mf.mol.nao))
    n=len(z)
    return e,h.transpose(0,2,1,3).reshape(3*n,3*n)


def harmonic(z,x_ang,hess):
    """Internal frequencies (cm^-1, imaginary as negative) and mass-weighted modes.

    Translations/rotations are removed by diagonalising in the orthogonal
    complement of their exact span; no lowest-|nu| modes are discarded by count.
    """
    from pyscf.data.elements import MASSES
    z=np.asarray(z);x=np.asarray(x_ang)*BOHR_PER_ANGSTROM;n=len(z)
    m=np.array([MASSES[int(a)] for a in z]);sm=np.repeat(np.sqrt(m),3)
    hmw=hess/np.outer(sm,sm)
    com=(m[:,None]*x).sum(0)/m.sum();r=x-com
    tr=[]
    for k in range(3):
        v=np.zeros((n,3));v[:,k]=1;tr.append((v*np.sqrt(m)[:,None]).ravel())
    for k in range(3):
        axis=np.zeros(3);axis[k]=1
        tr.append((np.cross(axis,r)*np.sqrt(m)[:,None]).ravel())
    tr=np.array(tr).T
    u,s,_=np.linalg.svd(tr,full_matrices=True)
    rank=int((s>1e-6*s.max()).sum())
    q=u[:,rank:]
    lam,vec=np.linalg.eigh(q.T@hmw@q)
    freq=np.sign(lam)*np.sqrt(np.abs(lam))*AU_TO_WAVENUMBER
    modes=(q@vec)/sm[:,None]
    return freq,modes.T.reshape(-1,n,3),rank


def classify_saddle(freq):
    neg=freq[freq<0];clear=neg[neg<CLEAR_NEGATIVE_CM1];shallow=neg[neg>=CLEAR_NEGATIVE_CM1]
    if len(clear)==1 and len(shallow)==0:return 'TS_OPTFREQ_PASS'
    if len(clear)>=2:return 'MULTIPLE_NEGATIVE_MODES'
    if len(neg)==0:return 'NO_NEGATIVE_MODE'
    return 'FREQUENCY_GRAY_ZONE'


def classify_minimum(freq):
    neg=freq[freq<0]
    if (neg<CLEAR_NEGATIVE_CM1).any():return 'MINIMUM_HAS_NEGATIVE_MODE'
    return 'MINIMUM_PASS' if len(neg)==0 else 'MINIMUM_GRAY_ZONE'


def make_engine(z,method,meter,stage):
    from geometric.engine import Engine
    from geometric.errors import EngineError
    from geometric.molecule import Molecule

    class GPU4PySCFEngine(Engine):
        def __init__(self,x_ang):
            M=Molecule();M.elem=[SYMBOLS[int(a)] for a in z];M.xyzs=[np.asarray(x_ang,dtype=float)]
            M.comms=['']
            M.build_topology()  # IRC in tric coordinates needs fragment topology (M.molecules)
            super().__init__(M);self.dm=None;self.calls=0

        def calc_new(self,coords,dirname):
            x=np.asarray(coords).reshape(-1,3)/BOHR_PER_ANGSTROM
            try:e,g,mf=single_point(z,x,method,meter,stage,dm0=self.dm)
            except SCFError as err:raise EngineError(str(err))
            self.dm=mf.make_rdm1();self.calls+=1
            return dict(energy=e,gradient=g.ravel())
    return GPU4PySCFEngine


def geometric_run(z,x_ang,method,meter,stage,workdir,**params):
    """Run one geomeTRIC job; returns (frames Angstrom, energies, converged, iterations)."""
    from geometric.optimize import run_optimizer
    from geometric.errors import GeomOptNotConvergedError
    workdir=Path(workdir);workdir.mkdir(parents=True,exist_ok=True)
    Engine=make_engine(z,method,meter,stage);engine=Engine(x_ang)
    t=time.perf_counter();converged=True
    cwd=os.getcwd();os.chdir(workdir)
    try:
        progress=run_optimizer(customengine=engine,input='geometric.in',prefix=stage,coordsys='tric',
                               qdata=False,**params)
        frames=[np.asarray(x) for x in progress.xyzs];energies=list(progress.qm_energies)
    except GeomOptNotConvergedError:
        converged=False;frames=[];energies=[]
    finally:
        os.chdir(cwd)
    meter.record(stage,'geometric',time.perf_counter()-t,n_atoms=len(z),engine_calls=engine.calls,
                 converged=converged,frames=len(frames))
    return frames,energies,converged,engine.calls


def save_hessian(path,h):
    np.savetxt(path,h)
    return f'file:{path}'


def ts_stage(z,x_raw,method,meter,workdir,tight=TIGHT,label='ts',maxiter=200):
    """Initial exact Hessian -> OptTS -> final full Hessian/frequencies."""
    workdir=Path(workdir);workdir.mkdir(parents=True,exist_ok=True)
    _,h0=hessian(z,x_raw,method,meter,label+'_initial_hessian')
    hfile=save_hessian(workdir/f'{label}_initial.hess',h0)
    frames,_,conv,calls=geometric_run(z,x_raw,method,meter,label+'_optts',workdir,transition=True,
                                      hessian=hfile,maxiter=maxiter,**tight)
    if not conv:return dict(status='TS_OPT_LIMIT',engine_calls=calls)
    x_ts=frames[-1]
    e,h=hessian(z,x_ts,method,meter,label+'_final_hessian')
    freq,modes,rank=harmonic(z,x_ts,h)
    np.savetxt(workdir/f'{label}_final.hess',h)
    return dict(status=classify_saddle(freq),x_ts=x_ts.tolist(),energy_hartree=e,frequencies_cm1=freq.tolist(),
                reaction_mode=modes[0].tolist(),tr_rank=rank,engine_calls=calls,optts_iterations=len(frames),
                hessian_file=str(workdir/f'{label}_final.hess'))


MIN_RETRY_DISPLACEMENT=0.10  # Angstrom, largest atom; D1 revision 2, all arms alike


def minimum_stage(z,x,method,meter,workdir,label,retry=True):
    """Opt+Freq to a minimum. One uniform retry if a clear negative mode remains:
    displace along that mode (largest atom MIN_RETRY_DISPLACEMENT) and re-optimize
    at VeryTight before the final frequency check (D1 revision 2)."""
    frames,_,conv,calls=geometric_run(z,x,method,meter,label+'_opt',workdir,maxiter=200,**TIGHT)
    if not conv:return dict(status='MIN_OPT_LIMIT',engine_calls=calls)
    xm=frames[-1];e,h=hessian(z,xm,method,meter,label+'_freq')
    freq,modes,_=harmonic(z,xm,h)
    out=dict(status=classify_minimum(freq),x=xm.tolist(),energy_hartree=e,
             frequencies_cm1=freq.tolist(),engine_calls=calls)
    if out['status']=='MINIMUM_HAS_NEGATIVE_MODE' and retry:
        mode=np.asarray(modes[0]);step=MIN_RETRY_DISPLACEMENT/np.linalg.norm(mode,axis=1).max()
        frames,_,conv,more=geometric_run(z,xm+step*mode,method,meter,label+'_retry_opt',Path(workdir)/'retry',
                                         maxiter=200,**VERY_TIGHT)
        first=out
        if not conv:return dict(status='MIN_OPT_LIMIT',engine_calls=calls+more,first_attempt=first)
        xm=frames[-1];e,h=hessian(z,xm,method,meter,label+'_retry_freq')
        freq,_,_=harmonic(z,xm,h)
        out=dict(status=classify_minimum(freq),x=xm.tolist(),energy_hartree=e,frequencies_cm1=freq.tolist(),
                 engine_calls=calls+more,first_attempt=first)
    return out


def irc_stage(z,ts,method,meter,workdir,irc=IRC):
    """Bidirectional IRC from the SAME optimized TS and its final Hessian."""
    frames,energies,conv,calls=geometric_run(z,np.asarray(ts['x_ts']),method,meter,'irc',workdir,irc=True,
                                             irc_direction='both',hessian='file:'+ts['hessian_file'],
                                             **irc,**TIGHT)
    if not frames:return dict(status='IRC_LIMIT',engine_calls=calls)
    return dict(status='IRC_COMPLETE' if conv else 'IRC_LIMIT',first=frames[0].tolist(),last=frames[-1].tolist(),
                n_frames=len(frames),energies_hartree=energies,engine_calls=calls)


def perceive_be(z,x_ang):
    """Bond/electron matrix of an optimized endpoint (neutral closed shell), or None if ambiguous."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    valence={1:1,6:4,7:5,8:6}
    block=f"{len(z)}\n\n"+''.join(f"{SYMBOLS[int(a)]} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n" for a,p in zip(z,x_ang))
    mol=Chem.MolFromXYZBlock(block)
    try:
        rdDetermineBonds.DetermineBonds(mol,charge=0)
        Chem.Kekulize(mol,clearAromaticFlags=True)
    except Exception:
        return None
    n=len(z);b=np.zeros((n,n),dtype=int)
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx();o=bond.GetBondTypeAsDouble()
        if o not in (1.,2.,3.):return None
        b[i,j]=b[j,i]=int(o)
    for i,atom in enumerate(mol.GetAtoms()):
        if atom.GetNumRadicalElectrons():return None
        b[i,i]=valence[int(z[i])]-atom.GetFormalCharge()-int(b[i].sum())
    return b


def endpoint_identity(z,b_r,perms,b_predicted,x_end1,x_end2):
    """Section 8.3: one end must be the input reactant graph; the other gives the actual event."""
    from .data import canonical_event
    b_r=np.asarray(b_r);perms=np.asarray(perms)
    ends=[perceive_be(z,x_end1),perceive_be(z,x_end2)]
    if any(b is None for b in ends):return dict(status='EVALUATION_UNRESOLVED',reason='endpoint_bond_perception')
    is_r=[bool(np.all(b[perms[:,:,None],perms[:,None,:]]==b_r,axis=(1,2)).any()) for b in ends]
    if not any(is_r):return dict(status='IRC_REACTANT_MISMATCH')
    product=ends[1] if is_r[0] else ends[0]
    if np.array_equal(product,b_r):return dict(status='STRICT_EVENT_MISMATCH',reason='both_ends_reactant',
                                               actual_event_id=None,reactant_end=0 if is_r[0] else 1)
    actual=canonical_event(b_r,product,perms);predicted=canonical_event(b_r,np.asarray(b_predicted),perms)
    return dict(status='STRICT_JOINT_GRAPH_VALID' if actual==predicted else 'STRICT_EVENT_MISMATCH',
                actual_event_id=actual,predicted_event_id=predicted,reactant_end=0 if is_r[0] else 1,
                product_be=product.tolist())


def certification_chain(z,x_raw,b_r,perms,b_predicted,method,workdir,meter=None,energy_exclusion_hartree=None,
                        reuse_ts=None,reuse_irc=None):
    """Full strict chain for one start structure. Returns a JSON-serialisable verdict.

    `energy_exclusion_hartree` is the Stage-B pre-registered short circuit: an
    absolute TS energy above which the candidate cannot be a best_cert_hit. It
    only ends the chain with ENERGY_EXCLUDED_FOR_BEST; strict_joint_graph stays
    unknown (None), never 0. `reuse_ts` continues from an already certified TS
    stage (same structure and final Hessian); its cost stays in the earlier record.
    """
    meter=meter or Meter();workdir=Path(workdir);workdir.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter();out=dict(method=method.as_dict(),n_atoms=len(z),protocol_version=PROTOCOL_VERSION)
    try:
        ts=reuse_ts if reuse_ts is not None else ts_stage(z,x_raw,method,meter,workdir)
        if ts['status']=='FREQUENCY_GRAY_ZONE' and reuse_ts is None:
            out['gray_zone_first']=ts
            ts=ts_stage(z,np.asarray(ts['x_ts']),method,meter,workdir/'retry',tight=VERY_TIGHT,label='ts_retry')
            if ts['status']=='FREQUENCY_GRAY_ZONE':ts['status']='FREQUENCY_GRAY_AT_CAP'
        out['ts']=ts;out['ts_optfreq_success']=int(ts['status']=='TS_OPTFREQ_PASS')
        if ts['status']!='TS_OPTFREQ_PASS':
            out['strict_status']='PROTOCOL_FAILURE';out['strict_joint_graph']=0
            return finish(out,meter,t0,workdir)
        if energy_exclusion_hartree is not None and ts['energy_hartree']>energy_exclusion_hartree:
            out['strict_status']='ENERGY_EXCLUDED_FOR_BEST';out['strict_joint_graph']=None
            return finish(out,meter,t0,workdir)
        irc=reuse_irc if reuse_irc is not None else irc_stage(z,ts,method,meter,workdir/'irc');out['irc']=irc
        if irc['status']!='IRC_COMPLETE':
            out['strict_status']='PROTOCOL_FAILURE';out['strict_joint_graph']=0
            return finish(out,meter,t0,workdir)
        ends=[minimum_stage(z,np.asarray(irc[k]),method,meter,workdir/k,f'end_{k}') for k in ('first','last')]
        out['endpoints']=ends
        if any(e['status'] not in ('MINIMUM_PASS','MINIMUM_GRAY_ZONE') for e in ends):
            out['strict_status']='PROTOCOL_FAILURE';out['strict_joint_graph']=0
            return finish(out,meter,t0,workdir)
        ident=endpoint_identity(z,b_r,perms,b_predicted,np.asarray(ends[0]['x']),np.asarray(ends[1]['x']))
        out['identity']=ident;out['strict_status']=ident['status']
        out['strict_joint_graph']=None if ident['status']=='EVALUATION_UNRESOLVED' else int(ident['status']=='STRICT_JOINT_GRAPH_VALID')
        if 'reactant_end' in ident:
            r=ends[ident['reactant_end']]
            out['local_barrier_kcal']=(ts['energy_hartree']-r['energy_hartree'])*HARTREE_TO_KCAL
    except SCFError as err:
        out['strict_status']='PROTOCOL_FAILURE';out['failure']=str(err);out['strict_joint_graph']=0
    return finish(out,meter,t0,workdir)


def finish(out,meter,t0,workdir):
    out['wall_s']=time.perf_counter()-t0;out['stage_timings']=meter.rows
    (Path(workdir)/'verdict.json').write_text(json.dumps(out,indent=1,default=float))
    return out
