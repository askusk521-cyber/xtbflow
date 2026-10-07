import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from xtbflow.m0.t1x_data import be_matrix
from xtbflow.v1.data import (EV_TO_KCAL_MOL, canonical_event, graph_identity,
                             legal_maps, reserve_split)
from xtbflow.v1.interfaces import assert_query_fields, query_from_parents


def test_stereo_identity_and_mapping():
    m=Chem.AddHs(Chem.MolFromSmiles('C[C@H](O)N'))
    assert AllChem.EmbedMolecule(m,randomSeed=10)==0
    z=np.array([a.GetAtomicNum() for a in m.GetAtoms()]);b=be_matrix(m)
    x=m.GetConformer().GetPositions()
    p,_,mol=graph_identity(z,b,x)
    mirrored=x.copy();mirrored[:,0]*=-1
    p2,_,mol2=graph_identity(z,b,mirrored)
    assert p!=p2
    with pytest.raises(ValueError,match='no_exact'):
        legal_maps(mol,mol2,z,z,b,b)
    pi=np.arange(len(z))[::-1]
    p3,_,mol3=graph_identity(z[pi],b[np.ix_(pi,pi)],x[pi])
    assert p==p3
    assert len(legal_maps(mol,mol3,z,z[pi],b,b[np.ix_(pi,pi)]))>0


def test_directed_bond_order_event():
    r=np.array([[2,1],[1,2]])
    p=np.array([[0,2],[2,2]])
    perms=np.array([[0,1]])
    assert canonical_event(r,p,perms)!=canonical_event(p,r,perms)
    assert canonical_event(r,p,perms)!=canonical_event(r,r,perms)


def test_split_uses_complete_capped_formula_groups_and_excludes_all_reserved_parents():
    parents=[dict(parent_id=f'p{g}_{i}',official_split=0,known_channels=2,
                  split_group=f'f{g}',cache_indices=[g*10+i]) for g in range(50) for i in range(6)]
    s=reserve_split(parents,cap=4,development=12,screen_reserve=51)
    assert len(s['screen_reserve']['parent_ids'])==48
    sets=[set(s[k]['formula_groups']) for k in ['train','development','screen_reserve']]
    assert not any(a&b for i,a in enumerate(sets) for b in sets[i+1:])
    assert s==reserve_split(list(reversed(parents)),cap=4,development=12,screen_reserve=51)


def test_query_is_reactant_only_and_keeps_atomic_number_distinct():
    for key in ['b_p','x_ts','energies','known_channels','reference_catalog']:
        with pytest.raises(ValueError,match='forbidden'):
            assert_query_fields({key:None})
    q=query_from_parents([dict(query_id='q',atomic_numbers=[6,8],x_r=[[0,0,0],[1,0,0]],
                              b_r=[[2,1],[1,4]],charge=0,multiplicity=1)])
    assert q.atomic_numbers.tolist()==[[6,8]]
    assert q.element_index.tolist()==[[2,4]]
    assert_query_fields(q.__dict__)
    assert 1.0*EV_TO_KCAL_MOL==pytest.approx(23.06055,abs=1e-5)
