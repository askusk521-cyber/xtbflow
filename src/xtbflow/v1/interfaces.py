"""The production input whitelist contains reactants and generated state only."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Query:
    query_id: Any
    atomic_numbers: Any
    element_index: Any
    atom_mask: Any
    x_r: Any
    b_r: Any
    charge: int = 0
    multiplicity: int = 1


@dataclass(frozen=True)
class State:
    b: Any
    x: Any
    t_b: Any
    t_x: Any


def assert_query_fields(payload):
    allowed = set(Query.__dataclass_fields__)
    extra = set(payload)-allowed
    if extra:
        raise ValueError('forbidden query fields: '+repr(sorted(extra)))


def query_from_parents(parents, device='cpu'):
    import torch
    from xtbflow.m0.batching import ELEMENT_INDEX
    n = max(len(p['atomic_numbers']) for p in parents)
    bsz = len(parents)
    z = torch.zeros(bsz,n,dtype=torch.long,device=device)
    elements = torch.zeros_like(z)
    mask = torch.zeros(bsz,n,dtype=torch.bool,device=device)
    x = torch.zeros(bsz,n,3,device=device)
    b = torch.zeros(bsz,n,n,device=device)
    for k,p in enumerate(parents):
        if p['charge']!=0 or p['multiplicity']!=1:
            raise ValueError('unsupported electronic state')
        m=len(p['atomic_numbers'])
        z[k,:m]=torch.tensor(p['atomic_numbers'],device=device)
        elements[k,:m]=torch.tensor([ELEMENT_INDEX[v] for v in p['atomic_numbers']],device=device)
        mask[k,:m]=True
        x[k,:m]=torch.tensor(p['x_r'],device=device)
        b[k,:m,:m]=torch.tensor(p['b_r'],device=device)
    return Query(tuple(p['query_id'] for p in parents),z,elements,mask,x,b)
