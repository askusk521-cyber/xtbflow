"""Verify pruning bounds against the actual strict endpoint sanitizer."""
import numpy as np
from rdkit import Chem, RDLogger
from xtbflow.v1.enumeration import MAX_BONDS
from xtbflow.m0.metrics import be_to_mol


def test_charge_specific_valence_bounds():
    RDLogger.DisableLog('rdApp.error')
    valence={1:1,6:4,7:5,8:6}
    for z in (1,6,7,8):
        for charge in (-1,0,1):
            for degree in range(9):
                # Sanitizer uses formal charge and bond degree; probe all degrees.
                mol=Chem.RWMol();atom=Chem.Atom(z);atom.SetFormalCharge(charge);atom.SetNoImplicit(True);mol.AddAtom(atom)
                for j in range(degree):
                    h=Chem.Atom(1);h.SetNoImplicit(True);mol.AddAtom(h);mol.AddBond(0,j+1,Chem.BondType.SINGLE)
                try:Chem.SanitizeMol(mol.GetMol());valid=True
                except (ValueError,RuntimeError):valid=False
                if valid:assert degree<=MAX_BONDS[z][charge],(z,charge,degree)
