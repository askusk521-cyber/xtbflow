"""Geometry-only tests; run with the approved aimnet RDKit environment too."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('thermo',Path(__file__).parents[1]/'scripts/followup_thermo.py')
thermo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(thermo)


def water():
    return np.array([8,1,1]), np.array([[4,1,1],[1,0,0],[1,0,0]])


def test_fragment_seed_exact():
    expected = int(thermo.sha256(b'p|reactant|0').hexdigest()[:8],16) % 2**31
    assert thermo.fragment_seed('p','reactant',0) == expected
    assert 0 <= expected < 2**31


def test_water_geometry_deterministic():
    z,b = water()
    z1,x1,s1 = thermo.product_geometry(z,b,'p','event')
    z2,x2,s2 = thermo.product_geometry(z,b,'p','event')
    assert np.array_equal(z1,z2)
    assert np.array_equal(x1,x2)
    assert s1 == s2
    assert np.allclose(x1.mean(0),0)
    assert len(x1)==3


def test_fragments_centroid_separation():
    z,b = water()
    b2 = np.zeros((6,6),dtype=int)
    b2[:3,:3] = b
    b2[3:,3:] = b
    zz,xx,states = thermo.product_geometry(np.tile(z,2),b2,'p','event')
    assert len(states)==2
    assert np.allclose(xx[:3].mean(0),[0,0,0])
    assert np.allclose(xx[3:].mean(0),[10,0,0])
    assert sorted(zz.tolist())==sorted(np.tile(z,2).tolist())


def test_energy_failure_is_retained(monkeypatch):
    monkeypatch.setattr(thermo,'product_geometry',lambda *a: (_ for _ in ()).throw(ValueError('embedding_failed')))
    z,b=water()
    out=thermo.single_energy(z,b,'p','event',{})
    assert out['energy_kcal'] is None
    assert out['status']=='ValueError:embedding_failed'
