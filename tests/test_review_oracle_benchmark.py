import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('review_oracle',Path(__file__).parents[1]/'scripts/review_oracle_benchmark.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_gfn_collapsed_geometry_is_not_evaluated():
    cfg={'min_distance_angstrom':.5}
    value,status=m.energy('GFN1-xTB',[1,1],np.zeros((2,3)),cfg)
    assert np.isnan(value) and status=='collapsed'


def test_gxtb_parses_hartree_and_uses_restricted_neutral(tmp_path,monkeypatch):
    executable=tmp_path/'mock-xtb'
    executable.write_text('#!/bin/sh\nprintf "TOTAL ENERGY -1.25 Eh\\n"\n')
    executable.chmod(0o755)
    monkeypatch.setenv('REVIEW_GXTB_BINARY',str(executable))
    value,status=m.energy('g-xTB',[1,1],[[0,0,0],[1,0,0]],{'min_distance_angstrom':.5})
    assert status=='ok'
    assert value==-1.25*m.HARTREE_TO_KCAL
