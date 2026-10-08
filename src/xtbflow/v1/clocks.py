import numpy as np


def clock_grid(path, n_steps=50):
    if n_steps<1:
        raise ValueError('positive n_steps required')
    s=np.linspace(0.,1.,n_steps+1)
    if path=='sync':
        return s,s.copy()
    if path=='event_lead2':
        return s,s*s
    # Exploratory mirrors of event_lead2: geometry matures first. Frozen V1a
    # programs never request these; the joint generator was trained on
    # independent (t_b, t_x), so both stay inside its training distribution.
    if path=='geometry_lead2':
        return s*s,s.copy()
    if path=='geometry_lead3':
        return s**3,s.copy()
    raise ValueError(path)


def observation_index(clock,target):
    if not 0<=target<=1:
        raise ValueError('target outside [0,1]')
    return min(int(np.searchsorted(clock,target,side='left')),len(clock)-1)


def increments(path,n_steps=50):
    tb,tx=clock_grid(path,n_steps)
    return np.diff(tb),np.diff(tx)
