#!/usr/bin/env python3
"""RDKit geometry identity check in its isolated n2 Python environment."""
from __future__ import annotations
import json
import socket
import sys


def identify(point, coordinates):
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdDetermineBonds
    xyz = str(len(point["symbols"])) + "\nexplicit model identity\n" + "\n".join(
        s + " " + " ".join(f"{v:.12f}" for v in x) for s, x in zip(point["symbols"], coordinates)) + "\n"
    result = {"rdkit_version": rdBase.rdkitVersion, "status": "identity_unresolved"}
    try:
        mol = Chem.MolFromXYZBlock(xyz)
        rdDetermineBonds.DetermineBonds(mol, charge=point["charge"], allowChargedFragments=True, embedChiral=True)
        observed = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
        expected = Chem.MolToSmiles(Chem.MolFromSmiles(point["stereo_smiles"]), isomericSmiles=True)
        result.update(observed_smiles=observed, expected_smiles=expected,
                      status="identity_passed" if observed == expected else "identity_changed")
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    return result


if __name__ == "__main__":
    if socket.gethostname() != "node2":
        raise RuntimeError("molecular data checks require n2")
    request = json.load(sys.stdin)
    print(json.dumps(identify(request["point"], request["coordinates_angstrom"])))
