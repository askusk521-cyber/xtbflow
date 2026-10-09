import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('gate', Path(__file__).parents[1]/'scripts/followup_enum_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_counts_upper_triangle_bond_order_not_pairs():
    br = np.array([[0,3,0],[3,0,0],[0,0,0]], dtype=np.int8)
    b = np.repeat(br[None], 4, axis=0)
    b[0,0,1] = b[0,1,0] = 1
    b[1,0,1] = b[1,1,0] = 0
    b[2,0,2] = b[2,2,0] = 2
    b[3,0,2] = b[3,2,0] = 3
    assert gate.b2f2_mask(b,br).tolist() == [True,False,True,False]


def test_diagonals_ignored_and_no_int8_overflow():
    br = np.zeros((2,2),dtype=np.int8)
    b = np.array([[[100,2],[2,100]]],dtype=np.int8)
    assert gate.b2f2_mask(b,br).tolist() == [True]
