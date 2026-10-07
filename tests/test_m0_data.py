import numpy as np
import pytest

pytest.importorskip('rdkit')
from rdkit import Chem
from rdkit.Chem import AllChem

from xtbflow.m0.t1x_data import (
    T1xCache, be_matrix, connectivity_change, kabsch_align, perceive_mol, write_cache,
)
from xtbflow.m0.batching import collate


def test_kabsch_proper_rotation_and_no_reflection():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    assert np.sqrt(np.mean(np.sum((kabsch_align(x @ q + 2, x) - x)**2, axis=1))) < 1e-6
    mirror = x.copy()
    mirror[:, 0] *= -1
    assert np.sqrt(np.mean(np.sum((kabsch_align(mirror, x) - x)**2, axis=1))) > 0.1


def test_be_matrix_acetaldehyde():
    mol = Chem.AddHs(Chem.MolFromSmiles('CC=O'))
    assert AllChem.EmbedMolecule(mol, randomSeed=0) == 0
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    b = be_matrix(perceive_mol(z, mol.GetConformer().GetPositions()))
    assert np.array_equal(b, b.T)
    assert np.all(np.diag(b) % 2 == 0)
    assert b.sum() == 18


def test_connectivity_change_ignores_order_and_diagonal():
    r = np.zeros((4, 4), dtype=int)
    r[0, 1] = r[1, 0] = 1
    r[1, 2] = r[2, 1] = 1
    p = r.copy()
    p[0, 1] = p[1, 0] = 2
    p[1, 2] = p[2, 1] = 0
    p[2, 3] = p[3, 2] = 1
    p[0, 0] = 4
    assert np.array_equal(connectivity_change(r, p), [[1, 2], [2, 3]])


def records():
    out = []
    for i, n in enumerate((2, 3)):
        out.append(dict(z=np.ones(n, dtype=np.int8),
                        x_r=np.arange(n*3, dtype=np.float32).reshape(n, 3),
                        x_ts=np.ones((n, 3), dtype=np.float32),
                        x_p=np.zeros((n, 3), dtype=np.float32),
                        b_r=np.zeros((n, n), dtype=np.int8),
                        b_p=np.eye(n, dtype=np.int8)*2,
                        energies=np.array([1., 2., 3.]), split=i,
                        parent_id=str(i), rxn_key=str(i), formula=f'H{n}',
                        smiles_r='[H]', smiles_p='[H]'))
    return out


def test_cache_roundtrip(tmp_path):
    rs = records()
    path = tmp_path/'cache.npz'
    write_cache(rs, path)
    cache = T1xCache(path)
    assert len(cache) == 2
    for i, r in enumerate(rs):
        got = cache.reaction(i)
        for key in ('z', 'x_r', 'x_ts', 'x_p', 'b_r', 'b_p'):
            assert np.array_equal(got[key], r[key])
        assert got['index'] == i
        assert got['parent_id'] == r['parent_id']
        for key in ('energies', 'split', 'parent_id', 'rxn_key', 'formula', 'smiles_r', 'smiles_p'):
            assert np.array_equal(cache.arrays[key][i], r[key])


def test_collate_padding(tmp_path):
    path = tmp_path/'cache.npz'
    write_cache(records(), path)
    cache = T1xCache(path)
    b = collate([cache.reaction(0), cache.reaction(1)])
    assert b['atom_mask'].tolist() == [[True, True, False], [True, True, True]]
    assert b['z'].tolist() == [[1, 1, 0], [1, 1, 1]]
    for key in ('x_r', 'x_ts', 'x_p'):
        assert (b[key][0, 2] == 0).all()
    for key in ('b_r', 'b_p'):
        assert (b[key][0, 2] == 0).all()
        assert (b[key][0, :, 2] == 0).all()
