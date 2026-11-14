#!/usr/bin/env python3
"""Construct explicit mapped hypothesis graphs and finite task packets on n2.

This is double-ended mechanistic screening, not a reactant-only ML benchmark.
Coordinates and complete graph packets remain in the n2 run directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from gqc_tautomer_pilot import ROOT, digest, write_json
from qc_campaign_geometry import identify


def canonical(mol):
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True)


def explicit(smiles):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    Chem.Kekulize(m, clearAromaticFlags=True)
    for i, a in enumerate(m.GetAtoms()):
        a.SetAtomMapNum(i + 1)
    return m


def edited(mol, remove=(), change=(), add=()):
    rw = Chem.RWMol(mol)
    for i, j in remove:
        rw.RemoveBond(i, j)
    for i, j, order in change:
        rw.GetBondBetweenAtoms(i, j).SetBondType(Chem.BondType.values[order])
    for i, j, order in add:
        rw.AddBond(i, j, Chem.BondType.values[order])
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    Chem.Kekulize(out, clearAromaticFlags=True)
    assert [a.GetAtomicNum() for a in out.GetAtoms()] == [a.GetAtomicNum() for a in mol.GetAtoms()]
    assert [a.GetAtomMapNum() for a in out.GetAtoms()] == list(range(1, out.GetNumAtoms() + 1))
    assert sum(a.GetFormalCharge() for a in out.GetAtoms()) == 0
    assert not any(a.GetNumRadicalElectrons() for a in out.GetAtoms())
    return out


def embed(mol, seed):
    m = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(m, params) != 0:
        raise RuntimeError("deterministic embedding failed")
    if not AllChem.MMFFHasAllMoleculeParams(m):
        raise RuntimeError("MMFF parameters unavailable")
    status = AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    xyz = np.asarray(m.GetConformer().GetPositions())
    fragments = Chem.GetMolFrags(m)
    # Disconnected RDKit fragments are embedded at overlapping origins. Place
    # them deterministically at >= 2.8 A without a product-derived orientation.
    placed = list(fragments[0])
    for fragment in fragments[1:]:
        frag = list(fragment)
        xyz[frag] += xyz[placed].mean(0) - xyz[frag].mean(0)
        for _ in range(100):
            if np.linalg.norm(xyz[frag, None] - xyz[placed][None], axis=2).min() >= 2.8:
                break
            xyz[frag, 0] += 0.15
        else:
            raise RuntimeError("fragment placement cap")
        placed += frag
    for i, x in enumerate(xyz):
        m.GetConformer().SetAtomPosition(i, x)
    Chem.AssignStereochemistryFrom3D(m)
    return m, xyz, status


def align(x, reference, symbols):
    ix = [i for i, s in enumerate(symbols) if s != "H"]
    a, b = x[ix].mean(0), reference[ix].mean(0)
    u, _, vt = np.linalg.svd((x[ix] - a).T @ (reference[ix] - b))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return (x - a) @ (u @ correction @ vt) + b


def packet(pid, mol, xyz, group, role, mmff_status=None):
    return {"id": pid, "group": group, "role": role,
            "symbols": [a.GetSymbol() for a in mol.GetAtoms()],
            "coordinates_angstrom": np.asarray(xyz).tolist(), "charge": 0, "multiplicity": 1,
            "electronic_state_source": "explicit neutral closed-shell model hypothesis",
            "formula": rdMolDescriptors.CalcMolFormula(mol), "stereo_smiles": canonical(mol),
            "mapped_smiles": Chem.MolToSmiles(mol, allHsExplicit=True),
            "atom_map": [a.GetAtomMapNum() for a in mol.GetAtoms()],
            "bonds": sorted([min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), b.GetBondTypeAsDouble()] for b in mol.GetBonds()),
            "coordinate_unit": "angstrom", "environment": "gas phase finite molecule or cluster",
            "mmff_status": mmff_status, "source": "independently constructed graph; no dataset TS or product coordinates"}


def da_graph(diene, partner):
    m = explicit(diene + "." + partner)
    fragments = Chem.GetMolFrags(m)
    first, second = set(fragments[0]), set(fragments[1])
    matches = [t for t in m.GetSubstructMatches(Chem.MolFromSmarts("[C]=[C]-[C]=[C]")) if set(t) <= first]
    if not matches:
        raise ValueError("no designated diene")
    a, b, c, d = min(matches)
    doubles = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in m.GetBonds()
               if bond.GetBondTypeAsDouble() == 2 and {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} <= second
               and all(m.GetAtomWithIdx(i).GetSymbol() == "C" for i in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))]
    e, f = min(doubles)
    p = edited(m, change=[(a, b, 1), (b, c, 2), (c, d, 1), (e, f, 1)], add=[(a, e, 1), (d, f, 1)])
    return m, p, (e, f)


def tautomer_graph():
    enol = explicit("C=CC=CO")
    o = next(a for a in enol.GetAtoms() if a.GetSymbol() == "O")
    h = next(a.GetIdx() for a in o.GetNeighbors() if a.GetSymbol() == "H")
    c = next(a.GetIdx() for a in o.GetNeighbors() if a.GetSymbol() == "C")
    beta = next(b.GetOtherAtomIdx(c) for b in enol.GetAtomWithIdx(c).GetBonds() if b.GetBondTypeAsDouble() == 2)
    keto = edited(enol, remove=[(o.GetIdx(), h)], change=[(o.GetIdx(), c, 2), (c, beta, 1)], add=[(beta, h, 1)])
    return keto, enol, (beta, o.GetIdx(), h)


def opening_graph(adduct):
    for carbon in adduct.GetAtoms():
        oxygens = [a for a in carbon.GetNeighbors() if a.GetSymbol() == "O"]
        if len(oxygens) != 2:
            continue
        for donor in oxygens:
            hs = [a.GetIdx() for a in donor.GetNeighbors() if a.GetSymbol() == "H"]
            if hs:
                acceptor = next(a for a in oxygens if a.GetIdx() != donor.GetIdx())
                c, o, a, h = carbon.GetIdx(), donor.GetIdx(), acceptor.GetIdx(), hs[0]
                p = edited(adduct, remove=[(c, a), (o, h)], change=[(c, o, 2)], add=[(a, h, 1)])
                p.GetAtomWithIdx(c).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                return p, (o, a, h)
    raise ValueError("hemiacetal motif not found")


def michael_graph(reactant, alkene):
    n = next(a for a in reactant.GetAtoms() if a.GetSymbol() == "N" and sum(b.GetSymbol() == "H" for b in a.GetNeighbors()) == 2)
    h = next(a.GetIdx() for a in n.GetNeighbors() if a.GetSymbol() == "H")
    alpha, beta = alkene
    return edited(reactant, remove=[(n.GetIdx(), h)], change=[(alpha, beta, 1)], add=[(n.GetIdx(), beta, 1), (alpha, h, 1)])


def water_pair(r, p, xr, xp, transfer):
    donor, acceptor, moved_h = transfer
    water = Chem.AddHs(Chem.MolFromSmiles("O"))
    n = r.GetNumAtoms()
    rr, pp = Chem.CombineMols(r, water), Chem.CombineMols(p, water)
    for m in (rr, pp):
        for i, atom in enumerate(m.GetAtoms()):
            atom.SetAtomMapNum(i + 1)
    # A product water H reaches the acceptor, while the original donor H
    # becomes a water H. The explicit atom map encodes a proton relay.
    pp = edited(pp, remove=[(acceptor, moved_h), (n, n + 1)], add=[(acceptor, n + 1, 1), (n, moved_h, 1)])
    xyzs = []
    for x in (xr, xp):
        axis = x[acceptor] - x[donor]
        axis /= np.linalg.norm(axis)
        perp = np.cross(axis, [0, 0, 1])
        if np.linalg.norm(perp) < 0.1:
            perp = np.cross(axis, [0, 1, 0])
        perp /= np.linalg.norm(perp)
        perpendicular2 = np.cross(axis, perp)
        placement = None
        heavy = [i for i, atom in enumerate(r.GetAtoms()) if atom.GetSymbol() != "H"]
        for radius in (2.4, 2.8, 3.2):
            for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                wo = (x[donor] + x[acceptor]) / 2 + radius * (np.cos(angle) * perp + np.sin(angle) * perpendicular2)
                wh1 = x[acceptor] - wo
                wh1 /= np.linalg.norm(wh1)
                ortho = np.cross(wh1, axis)
                ortho /= np.linalg.norm(ortho)
                wh2 = np.cos(np.deg2rad(104.5)) * wh1 + np.sin(np.deg2rad(104.5)) * ortho
                candidate = np.array([wo, wo + 0.97 * wh1, wo + 0.97 * wh2])
                if np.linalg.norm(candidate[:, None] - x[None], axis=2).min() >= 1.45 and np.linalg.norm(wo - x[heavy], axis=1).min() >= 2.4:
                    placement = candidate
                    break
            if placement is not None:
                break
        if placement is None:
            raise RuntimeError("bounded clash-free water placement failed")
        xyzs.append(np.vstack([x, placement]))
    xyzs[1][[moved_h, n + 1]] = xyzs[1][[n + 1, moved_h]]
    return rr, pp, *xyzs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "wave1", "out"):
        ap.add_argument("--" + key, type=Path, required=True)
    a = ap.parse_args()
    if socket.gethostname() != "node2":
        raise RuntimeError("real molecular preparation requires n2")
    a.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text())
    first = json.loads(a.wave1.read_text())
    points, paths, failures = {}, [], []
    graphs = {}

    def add_point(pid, mol, group, role, seed, xyz=None):
        if pid in points:
            return graphs[pid], np.asarray(points[pid]["coordinates_angstrom"])
        if xyz is None:
            mol, xyz, mmff_status = embed(mol, seed)
        else:
            mmff_status = None
        points[pid] = packet(pid, mol, xyz, group, role, mmff_status)
        graphs[pid] = mol
        return mol, xyz

    def pair(name, r, p, group, seed, rid=None, pid=None):
        rid, pid = rid or name + "_R", pid or name + "_P"
        r, xr = add_point(rid, r, group, "path_reactant", seed)
        p, xp = add_point(pid, p, group, "path_product_hypothesis", seed + 1)
        xp = align(xp, xr, points[rid]["symbols"])
        points[pid]["coordinates_angstrom"] = xp.tolist()
        assert points[rid]["formula"] == points[pid]["formula"]
        assert points[rid]["atom_map"] == points[pid]["atom_map"]
        assert points[rid]["symbols"] == points[pid]["symbols"]
        paths.append({"id": name, "reactant": rid, "product": pid, "group": group,
                      "map_evidence": "explicit graph edits preserve every atom including transferring hydrogens"})
        return r, p, xr, xp

    r, p, transfer = tautomer_graph()
    r, p, xr, xp = pair("A_R6_tautomer", r, p, "R6", 1700)
    rw, pw, xrw, xpw = water_pair(r, p, xr, xp, transfer)
    add_point("AD_R6_water_R", rw, "R6_water", "cluster_reactant", 0, xrw)
    add_point("AD_R6_water_P", pw, "R6_water", "cluster_product", 0, xpw)
    pair("AD_R6_water", rw, pw, "R6_water", 1702)
    r, p, _ = da_graph("C=CC=CO", "C=C")
    pair("A_R6_DA", r, p, "R6", 1710)
    r, p, _ = da_graph("Cc1ccc(O)o1", "C=C")
    r, p, xr, xp = pair("B_R51_DA", r, p, "R51", 1720)
    opened, transfer = opening_graph(p)
    r, p, xr, xp = pair("B_R51_open", p, opened, "R51", 1730, rid="B_R51_DA_P")
    rw, pw, xrw, xpw = water_pair(r, p, xr, xp, transfer)
    add_point("D_R51_water_R", rw, "R51_water", "cluster_reactant", 0, xrw)
    add_point("D_R51_water_P", pw, "R51_water", "cluster_product", 0, xpw)
    pair("D_R51_water", rw, pw, "R51_water", 1732)
    r, p, alkene = da_graph("Cc1ccc(N)o1", "O=C1C=CC(=O)N1")
    r, _, _, _ = pair("C_amino_DA", r, p, "amino", 1740)
    pair("C_amino_Michael", r, michael_graph(r, alkene), "amino", 1750, rid="C_amino_DA_R")

    # One independently seeded conformer per selected old graph; no selection
    # using the reference energy. These are additional starts, not a partition function.
    for index, (pid, row) in enumerate(first["points"].items()):
        if pid.endswith("_distorted"):
            continue
        add_point("conformer_" + pid, explicit(row["stereo_smiles"]), row["group"], "additional_conformer", 1800 + index)
    for index, (group, smiles) in enumerate([
        ("R6", "CC=CC=O"), ("R30", "CCC=CC=O"), ("R30", "C=CCCC=O"),
        ("R51", "CC1C=CC(=O)O1"), ("R51", "C=C1CCC(=O)O1")]):
        stereos = sorted({Chem.MolToSmiles(m, isomericSmiles=True) for m in EnumerateStereoisomers(
            Chem.MolFromSmiles(smiles), options=StereoEnumerationOptions(onlyUnassigned=True, unique=True, maxIsomers=8))})
        for j, stereo in enumerate(stereos):
            pid = f"alternative_{group}_{index}_{j}"
            try:
                add_point(pid, explicit(stereo), group, "additional_carbonyl_isomer", 1900 + 10 * index + j)
            except Exception as exc:
                failures.append({"id": pid, "smiles": stereo, "status": "preparation_failed", "error_type": type(exc).__name__})

    for point in points.values():
        point["initial_geometry_identity"] = identify(point, point["coordinates_angstrom"])
    invalid = [pid for pid, point in points.items() if point["initial_geometry_identity"]["status"] != "identity_passed"]
    if invalid:
        write_json(a.out / "preparation_failure.json", {"status": "input_identity_failed", "invalid_points": invalid})
        raise ValueError("initial geometry identity check failed")

    tasks = []
    for pid in ("AD_R6_water_R", "C_amino_DA_R"):
        tasks.append({"id": "audit_" + pid, "point": pid, "kind": "audit", "method": "b3lyp-rks", "xc": "b3lyp", "basis": "def2-svp", "gpu": 1, "cpus": 4, "mem_gib": 16, "minutes": 30, "calls": 8})
    if len(points) > 36 or len(paths) != 8:
        raise ValueError("frozen point/path cap violated")
    for pid, point in points.items():
        if point["group"] == "amino":
            gate = "wave2/audit_C_amino_DA_R"
        elif "water" in point["group"]:
            gate = "wave2/audit_AD_R6_water_R"
        else:
            gate = "wave1/audit_" + point["group"] + "_enol_def2-svp"
        common = {"point": pid, "xc": "b3lyp", "basis": "def2-svp", "cpus": 4, "mem_gib": 16, "gate": gate}
        tasks.append({**common, "id": "opt_" + pid, "kind": "opt", "gpu": 1, "minutes": 30, "calls": 180})
        tasks.append({**common, "id": "freq_" + pid, "kind": "frequency", "gpu": 0, "minutes": 45, "calls": 4, "upstream": "opt_" + pid})
    for path in paths:
        tasks.append({"id": "path_" + path["id"], "point": path["reactant"], "path": path, "kind": "path", "xc": "b3lyp", "basis": "def2-svp",
                      "gate": next(t["gate"] for t in tasks if t["id"] == "opt_" + path["reactant"]),
                      "gpu": 1, "cpus": 4, "mem_gib": 24, "minutes": 150, "calls": 1600})
    # Matched gas single points isolate the PCM effect. R6 cc-pVTZ checks
    # whether the cc-pVDZ reference gap survives a larger basis.
    for pid, point in first["points"].items():
        if pid.endswith("_distorted"):
            continue
        points["old_" + pid] = point
        tasks.append({"id": "gas_tz_" + pid, "point": "old_" + pid, "kind": "single_point", "xc": "b3lyp", "basis": "def2-tzvp", "gpu": 0, "cpus": 4, "mem_gib": 16, "minutes": 30, "calls": 2})
        if pid.startswith("R6_"):
            tasks.append({"id": "ccsdt_tz_" + pid, "point": "old_" + pid, "kind": "ccsd_t", "basis": "cc-pvtz", "gpu": 0, "cpus": 8, "mem_gib": 32, "minutes": 240, "calls": 5})
    manifest = {"schema": "xtbflow-qc-tasks/v1", "config": cfg, "config_sha256": digest(a.config),
                "preparation_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "preparation_script_sha256": digest(Path(__file__)),
                "wave1_manifest_sha256": digest(a.wave1), "points": points, "paths": paths, "tasks": tasks,
                "preparation_failures": failures, "rdkit_version": rdBase.rdkitVersion,
                "input_scope": "exposed, explicitly mapped double-ended model hypotheses; no mechanistic completeness claim"}
    write_json(a.out / "tasks.json", manifest)
    summary = {"task_count": len(tasks), "kind_counts": {k: sum(t["kind"] == k for t in tasks) for k in sorted({t["kind"] for t in tasks})},
               "gpu_hour_ceiling": sum(t["gpu"] * t["minutes"] / 60 for t in tasks),
               "cpu_core_hour_ceiling": sum(t["cpus"] * t["minutes"] / 60 for t in tasks),
               "operation_ceiling": sum(t["calls"] for t in tasks), "manifest_sha256": digest(a.out / "tasks.json"), "preparation_failures": failures}
    for budget_key, summary_key in [("gpu_allocation_hours_ceiling", "gpu_hour_ceiling"), ("cpu_core_hours_ceiling", "cpu_core_hour_ceiling"), ("quantum_operations_ceiling", "operation_ceiling")]:
        used = sum(t["gpu"] * t["minutes"] / 60 if summary_key == "gpu_hour_ceiling" else t["cpus"] * t["minutes"] / 60 if summary_key == "cpu_core_hour_ceiling" else t["calls"] for t in first["tasks"])
        if used + summary[summary_key] > cfg["resources"][budget_key]:
            raise ValueError("aggregate budget exceeded: " + budget_key)
        summary["campaign_" + summary_key] = used + summary[summary_key]
    write_json(a.out / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
