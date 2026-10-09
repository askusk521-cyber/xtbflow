"""Pure Cartesian/rigid-mode Hessian diagnostics; kcal/mol/Angstrom^2."""
import numpy as np

MASS={1:1.00782503,6:12.,7:14.0030740,8:15.9949146}
# sqrt((kcal/mol)/Angstrom^2/amu)/(2*pi*c), c in cm/s.
WAVENUMBER=np.sqrt(4184/6.02214076e23/1e-20/1.66053906660e-27)/(2*np.pi*2.99792458e10)


def internal_basis(x, masses=None):
    x=np.asarray(x,dtype=float);n=len(x)
    m=np.ones(n) if masses is None else np.asarray(masses)
    r=x-(m[:,None]*x).sum(0)/m.sum()
    rigid=[]
    for axis in np.eye(3):
        rigid.append((np.broadcast_to(axis,(n,3))*np.sqrt(m)[:,None]).ravel())
    for axis in np.eye(3):rigid.append((np.cross(axis,r)*np.sqrt(m)[:,None]).ravel())
    vec,s,_=np.linalg.svd(np.array(rigid).T,full_matrices=True)
    rank=int(np.count_nonzero(s>1e-8*s.max()))
    return vec[:,rank:],rank


def hessian_metrics(z,x,h,u):
    h=np.asarray(h,dtype=float)
    if not np.isfinite(h).all():raise ValueError('Nonfinite Hessian')
    asymmetry=float(np.max(np.abs(h-h.T)))
    h=(h+h.T)/2
    q,rank=internal_basis(x)
    eigenvalues,vectors=np.linalg.eigh(q.T@h@q)
    v=q@vectors
    u=np.asarray(u,dtype=float).ravel()
    if np.linalg.norm(u)<1e-8:raise ValueError('Undefined event direction')
    u=u/np.linalg.norm(u)
    overlap=np.abs(v.T@u)
    masses=np.array([MASS[int(a)] for a in z]);sm=np.repeat(np.sqrt(masses),3)
    qm,mrank=internal_basis(x,masses)
    massvalues=np.linalg.eigvalsh(qm.T@(h/np.outer(sm,sm))@qm)
    freq=np.sign(massvalues)*np.sqrt(np.abs(massvalues))*WAVENUMBER
    projected=q@(q.T@h@q)@q.T
    return dict(c_u=float(u@projected@u),overlap=float(overlap[0]),
                best_neg_overlap=float(overlap[eigenvalues<0].max()) if np.any(eigenvalues<0) else 0.,
                imaginary_count=int(np.count_nonzero(freq < -30)),frequencies_cm1=freq.tolist(),
                tr_rank=rank,mass_tr_rank=mrank,raw_max_asymmetry=asymmetry,
                symmetric_max_asymmetry=float(np.max(np.abs(h-h.T))))


def finite_difference_hessian(x,gradient,step=.005):
    x=np.asarray(x,dtype=float)
    n=x.size;h=np.empty((n,n))
    for j in range(n):
        dx=np.zeros_like(x);dx.flat[j]=step
        h[:,j]=(np.asarray(gradient(x+dx)).ravel()-np.asarray(gradient(x-dx)).ravel())/(2*step)
    return (h+h.T)/2
