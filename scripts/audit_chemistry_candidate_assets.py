#!/usr/bin/env python3
"""Bounded n2-only inventory for choosing chemistry questions, with zero QC.

Inputs: existing public DA caches and the pinned development manifest.
Output: path-free source hashes, graph-level coverage, selected seed locators,
environment metadata and resource receipt. No training data are admitted and no
source coordinate mapping, charge/spin, energy units or chemical validity are
inferred from a graph. Tautomer enumeration proposes graphs, not populations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import resource
import signal
import socket
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CAPS = {"wall_seconds": 180, "cpu_seconds": 180,
        "input_bytes": 512 * 1024**2, "output_bytes": 8 * 1024**2,
        "model_forwards": 0, "quantum_chemistry_calls": 0,
        "gpu_seconds": 0, "retries": 0}
FULL_HASH = "4df57289edecdfc518da6aab147b704745a9f42901893671aab05a3f72bd73ed"
SEEDS = ("R0", "R1", "R6", "R30", "R51", "R85", "R671", "R3066", "R9173")
PATTERNS = {
    "furan": "c1ccoc1", "maleimide": "O=C1C=CC(=O)N1",
    "maleic_anhydride": "O=C1OC(=O)C=C1", "enol": "[OX2H]-[C,c]=[C,c]",
    "aromatic_hydroxy": "[OX2H]-[c]", "hydroxy": "[OX2H]",
    "nh": "[NX3;H1,H2]",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to replace existing evidence")
    if socket.gethostname() != "node2":
        raise RuntimeError("real source parsing must execute on node2")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CPU, (180, 181))
    started, cpu_started = time.monotonic(), time.process_time()
    read_bytes = 0
    report: dict[str, Any] = {
        "schema": "xtbflow-chemistry-question-inventory/v1",
        "host": socket.gethostname(), "observed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_clean": not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(json.dumps({"caps": CAPS, "seeds": SEEDS, "patterns": PATTERNS}, sort_keys=True).encode()).hexdigest(),
        "caps": CAPS, "sources": {}, "status": "started", "phase": "inventory",
        "claim_limits": [
            "Graph coverage and file availability only; no chemical result or new QC.",
            "Canonical graph grouping is not atom-map/coordinate proof or reaction-family independence.",
            "Source reported labels retain unresolved per-record environment and protocol qualifications.",
            "All selected seeds are exposed hypothesis-development examples, not blind model evaluation.",
            "RDKit tautomers are finite graph proposals; populations and conversion rates are unknown.",
            "Installed packages do not establish an operational or chemically qualified DFT backend.",
        ],
    }

    def stop(signum: int, frame: Any) -> None:
        raise TimeoutError(f"bounded audit interrupted by signal {signum}")

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGXCPU, stop)
    signal.alarm(CAPS["wall_seconds"])

    def data(path: Path) -> bytes:
        nonlocal read_bytes
        if read_bytes + path.stat().st_size > CAPS["input_bytes"]:
            raise RuntimeError("input byte cap exceeded")
        value = path.read_bytes()
        read_bytes += len(value)
        return value

    def fingerprint(path: Path, label: str, retain: bool = False) -> bytes:
        nonlocal read_bytes
        if retain:
            value = data(path)
            digest = hashlib.sha256(value).hexdigest()
        else:
            value = b""
            h = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024**2):
                    read_bytes += len(chunk)
                    if read_bytes > CAPS["input_bytes"]:
                        raise RuntimeError("input byte cap exceeded")
                    h.update(chunk)
            digest = h.hexdigest()
        report["sources"][label] = {"bytes": path.stat().st_size, "sha256": digest}
        return value

    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.MolStandardize import rdMolStandardize
        RDLogger.DisableLog("rdApp.warning")
        if not report["source_clean"]:
            raise RuntimeError("audit source must be clean")
        cache = args.cache_root.expanduser()
        full = cache / "dft-da-full-29118509-v1"
        dft = cache / "dft-da-dft-source-28768301-v1"
        full_dir = full / "extracted/DATASET_DA_F"
        dft_dir = dft / "extracted/DFT_DA_DATASET"
        fingerprint(full / "DATASET_DA_F.tar.gz", "figshare_29118509_archive")
        if report["sources"]["figshare_29118509_archive"]["sha256"] != FULL_HASH:
            raise RuntimeError("full DA archive hash differs from pinned source")
        fingerprint(dft / "DFT_DA_DATASET.tar.gz", "figshare_28768301_archive")
        full_raw = fingerprint(full_dir / "dataset_xtb_final.csv", "paired_source_csv", True)
        dft_raw = fingerprint(dft_dir / "da_dataset_dft_new.csv", "dft_source_csv", True)
        manifest_raw = fingerprint(ROOT / "data/manifests/paired_pilot_v1/manifest.jsonl", "development_manifest", True)
        full_rows = list(csv.DictReader(io.StringIO(full_raw.decode())))
        dft_rows = list(csv.DictReader(io.StringIO(dft_raw.decode())))
        admitted = {r["source_record_id"]: r for r in map(json.loads, manifest_raw.decode().splitlines())}
        dft_by_r = {r["R"]: r for r in dft_rows}
        patterns = {name: Chem.MolFromSmarts(s) for name, s in PATTERNS.items()}
        groups: dict[str, dict[str, Any]] = {}
        parsed: dict[str, Any] = {}
        parse_failures: list[str] = []

        def canonical(smiles: str) -> tuple[Any, str]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("SMILES parsing failed")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            mol = Chem.RemoveHs(mol)
            return mol, Chem.MolToSmiles(mol, isomericSmiles=True)

        for row in full_rows:
            try:
                mol, key = canonical(row["smiles"].split(">>")[0])
            except ValueError:
                parse_failures.append(row["R"])
                continue
            parsed[row["R"]] = (mol, key)
            group = groups.setdefault(key, {"rows": [], "admitted": [], "heavy_atoms": mol.GetNumHeavyAtoms(),
                "chno": all(a.GetAtomicNum() in (1, 6, 7, 8) for a in mol.GetAtoms()),
                "motifs": [name for name, pat in patterns.items() if mol.HasSubstructMatch(pat)]})
            group["rows"].append(row["R"])
            if row["R"] in admitted:
                group["admitted"].append(row["R"])
        report["coverage"] = {
            "paired_rows": len(full_rows), "dft_rows": len(dft_rows),
            "canonical_reactant_groups": len(groups), "parse_failures": parse_failures,
            "chno_groups": sum(g["chno"] for g in groups.values()),
            "admitted_rows": len(admitted), "admitted_parent_ids": len({r["parent_reaction_id"] for r in admitted.values()}),
            "exact_R_and_mapped_reaction_join": sum(r["R"] in dft_by_r and dft_by_r[r["R"]]["reaction"] == r["smiles"] for r in full_rows),
            "motif_groups": {n: sum(n in g["motifs"] for g in groups.values()) for n in patterns},
            "motif_groups_with_development_rows": {n: sum(n in g["motifs"] and bool(g["admitted"]) for g in groups.values()) for n in patterns},
            "label_columns": {"paired": list(full_rows[0]), "dft": list(dft_rows[0])},
            "label_interpretation": "No energies compared; source labels are not new independent calculations.",
        }
        report["seeds"] = []
        enum = rdMolStandardize.TautomerEnumerator()
        enum.SetMaxTautomers(12)
        enum.SetMaxTransforms(100)
        by_r = {r["R"]: r for r in full_rows}
        for rid in SEEDS:
            row = by_r[rid]
            mol, key = parsed[rid]
            ta = []
            for component in Chem.GetMolFrags(mol, asMols=True):
                result = enum.Enumerate(component)
                ta.append({"input": Chem.MolToSmiles(component),
                    "proposed_graphs": sorted(Chem.MolToSmiles(x) for x in result),
                    "enumeration_status": str(result.status)})
            files = {}
            for label, directory, names in [
                ("paired_archive", full_dir / row["R_dir"], [f"{rid}_reactant.xyz", f"{rid}_ts.xyz", "monomer_1.log", "monomer_2.log"]),
                ("dft_archive", dft_dir / row["R_dir"], [f"{rid}_ts_dft.xyz", f"{rid}_product_dft.xyz"]),
            ]:
                files[label] = {name: (directory / name).is_file() for name in names}
            report["seeds"].append({"record": rid, "source_parent": row["ID"],
                "reactant_graph": key, "heavy_atoms": mol.GetNumHeavyAtoms(),
                "motifs": groups[key]["motifs"], "development_admission": admitted.get(rid, {}).get("admission", "not_in_admitted_manifest"),
                "source_electronic_state": "see existing manifest evidence" if rid in admitted else "unknown",
                "source_environment": "unknown", "geometry_files_present": files,
                "tautomer_proposals": ta})
        report["environment"] = {"python": platform.python_version(), "packages": {}}
        for name in ("torch", "numpy", "scipy", "rdkit", "tblite", "xtb", "pyscf", "gpu4pyscf"):
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                version = None
            report["environment"]["packages"][name] = version
        report["status"], report["phase"] = "completed_inventory_only", "complete"
    except Exception as exc:
        report["status"] = "execution_failed"
        report["error_type"] = type(exc).__name__
        report["error_summary"] = "Source/parse/budget failure; inspect private stderr for details."
        import traceback
        traceback.print_exc()
    finally:
        signal.alarm(0)
        report["resources"] = {"wall_seconds": time.monotonic() - started,
            "process_cpu_seconds": time.process_time() - cpu_started,
            "input_bytes": read_bytes, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "qc_calls": 0, "model_forwards": 0, "gpu_seconds": 0}
        rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if len(rendered.encode()) > CAPS["output_bytes"]:
            raise RuntimeError("output byte cap exceeded; report not written")
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
        print(json.dumps({"status": report["status"], "resources": report["resources"]}))
    return 0 if report["status"] == "completed_inventory_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
