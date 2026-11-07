#!/usr/bin/env python3
"""Bounded fresh joint/serial generation through pinned research assets on n2."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xtbflow.governance.generation_bridge import (  # noqa: E402
    certification_candidate, legacy_query, validate_journal, validate_query, validate_generation_config,
)
from research_generation_worker import bind_parent_lifetime  # noqa: E402
from xtbflow.governance.research_replay import check_source, git  # noqa: E402
from xtbflow.runtime import LedgerEvent, RunLedger, StageBudget  # noqa: E402
from xtbflow.validation.certification import file_hash, identity, write_json  # noqa: E402


def prepare_queries(cache: Path, cfg: dict) -> tuple[list[dict], dict]:
    """Select using training admission, reactant atom count and record ID only."""
    import numpy as np
    from rdkit import Chem
    from xtbflow.data.paired_loader import load_paired_pilot
    manifest = ROOT / cfg["source_manifest"]
    if file_hash(manifest) != cfg["source_manifest_sha256"]:
        raise ValueError("source_manifest_hash_mismatch")
    samples = load_paired_pilot(manifest, cache).samples
    eligible = sorted((s for s in samples if s.record.admission == "development_train"),
                      key=lambda s: (len(s.atomic_numbers), s.record.record_id))
    queries, excluded, parents = [], [], set()
    for sample in eligible:
        record = sample.record
        if record.parent_reaction_id in parents:
            continue
        if len(sample.atomic_numbers) > cfg["max_atoms"]:
            excluded.append({"record_id": record.record_id, "reason": "atom_cap"})
            continue
        try:
            params = Chem.SmilesParserParams()
            params.removeHs = False
            mol = Chem.MolFromSmiles(sample.reactant_smiles, params)
            positions = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
            mol = Chem.RenumberAtoms(mol, [positions[i] for i in sample.map_ids])
            Chem.Kekulize(mol, clearAromaticFlags=True)
            n = len(sample.atomic_numbers)
            if [a.GetAtomicNum() for a in mol.GetAtoms()] != list(sample.atomic_numbers):
                raise ValueError("mapped_atom_inventory_mismatch")
            be = np.zeros((n, n), dtype=np.int64)
            for bond in mol.GetBonds():
                order = bond.GetBondTypeAsDouble()
                if order not in (1., 2., 3.):
                    raise ValueError("noninteger_reactant_lewis_graph")
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                be[i, j] = be[j, i] = int(order)
            valence = {1: 1, 6: 4, 7: 5, 8: 6}
            for i, atom in enumerate(mol.GetAtoms()):
                if atom.GetNumRadicalElectrons():
                    raise ValueError("open_shell_reactant")
                be[i, i] = valence[atom.GetAtomicNum()] - atom.GetFormalCharge() - int(be[i].sum())
                atom.SetAtomMapNum(0)
            matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=True, maxMatches=cfg["max_automorphisms"] + 1)
            if len(matches) > cfg["max_automorphisms"]:
                raise ValueError("reactant_automorphism_cap")
            permutations = [list(p) for p in matches if np.array_equal(be, be[np.ix_(p, p)])]
            x = np.asarray(sample.reactant_coordinates)
            x = x - x.mean(0)
            content = dict(atomic_numbers=list(sample.atomic_numbers), x_r=x.tolist(), b_r=be.tolist(),
                           atom_ids=[f"{record.record_id}:map:{m}" for m in sample.map_ids],
                           reactant_automorphisms=permutations, charge=record.reactant["charge"],
                           multiplicity=record.reactant["multiplicity"], environment=cfg["environment"])
            query = dict(query_id=record.record_id, parent_id=record.parent_reaction_id, **content,
                         provenance={"input_sha256": identity(content),
                                     "source_record_id": record.record_id, "family_id": record.family_id,
                                     "source_asset_sha256": record.source_asset_sha256,
                                     "atom_mapping_evidence": "Audited DFT-DA mapped SMILES/XYZ correspondence; reactant-only Kekule graph",
                                     "electronic_state_evidence": record.reactant["charge_spin_evidence"],
                                     "environment_origin": cfg["environment_origin"],
                                     "input_origin": record.reactant["input_origin"],
                                     "role": "development_functional_integration"})
            validate_query(query)
        except (ValueError, RuntimeError, KeyError) as exc:
            excluded.append({"record_id": record.record_id, "reason": str(exc)})
            continue
        queries.append(query)
        parents.add(record.parent_reaction_id)
        if len(queries) == cfg["query_count"]:
            break
    if len(queries) != cfg["query_count"]:
        raise ValueError("insufficient_eligible_reactant_queries")
    return queries, dict(training_rows=len(eligible), exclusions=excluded,
                         selection=cfg["input_policy"], selected=[{"record_id": q["query_id"],
                         "parent_id": q["parent_id"], "atoms": len(q["atomic_numbers"]),
                         "input_sha256": q["provenance"]["input_sha256"]} for q in queries])


def main() -> int:
    bind_parent_lifetime()
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--cache", type=Path)
    inputs.add_argument("--reactants", type=Path)
    parser.add_argument("--conditioned-events", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/generation_bridge_20261107.json")
    parser.add_argument("--research-source", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "node2" or not os.environ.get("SLURM_JOB_ID"):
        parser.error("run through ssh n2 with a finite Slurm GPU allocation")
    if git(ROOT, "status", "--porcelain"):
        parser.error("clean committed source required")
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    validate_generation_config(cfg)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    self_before = resource.getrusage(resource.RUSAGE_SELF)
    cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    registry = json.loads((ROOT / "configs/research/assets_20261107.json").read_text())
    asset = next(a for a in registry["assets"] if a["pr"] == 119)
    report = dict(schema="xtbflow-generation-bridge-report/v1", status="preflight", host=socket.gethostname(),
                  source_commit=git(ROOT, "rev-parse", "HEAD"), config_sha256=file_hash(config_path),
                  research_commit=cfg["research_commit"], slurm_job_id=os.environ["SLURM_JOB_ID"],
                  started_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  packages={p: importlib.metadata.version(p) for p in ("torch", "numpy", "rdkit")},
                  budget=cfg["generation"], claim_limits=cfg["claim_limits"], new_calculator_calls=0,
                  training_steps=0, candidates=[], call_summary={"forward_evaluations": 0})
    write_json(args.out / "report.json", report)
    raw = args.out / "raw"
    raw.mkdir()
    try:
        report["research_source"] = check_source(args.research_source, asset)
        if asset["commit"] != cfg["research_commit"]:
            raise ValueError("research_commit_not_in_registry")
        if args.reactants is None:
            queries, report["input_audit"] = prepare_queries(args.cache, cfg)
        else:
            if args.reactants.stat().st_size > 1024**2:
                raise ValueError("reactant_packet_too_large")
            queries = json.loads(args.reactants.read_text())["queries"]
            if len(queries) != cfg["query_count"]:
                raise ValueError("reactant_query_count_mismatch")
            for query in queries:
                validate_query(query)
            report["input_audit"] = dict(selection="explicit_hashed_reactant_packet",
                packet_sha256=file_hash(args.reactants), selected=[dict(record_id=q["query_id"], parent_id=q["parent_id"],
                    atoms=len(q["atomic_numbers"]), input_sha256=q["provenance"]["input_sha256"]) for q in queries])
        write_json(raw / "reactants.json", {"queries": queries})
        packet = dict(config=cfg, queries=[legacy_query(q) for q in queries])
        if "conditioned_geometry" in cfg["methods"]:
            if args.conditioned_events is None or args.conditioned_events.stat().st_size > 1024**2:
                raise ValueError("explicit_bounded_conditioned_events_required")
            events = json.loads(args.conditioned_events.read_text())
            if set(events) != {q["query_id"] for q in queries}:
                raise ValueError("conditioned_event_query_mismatch")
            for q in queries:
                certification_candidate(q, dict(query_id=q["query_id"], candidate_id="condition-check", status="generated",
                    method="conditioned_geometry", proposal=0, b_dec=events[q["query_id"]], x=q["x_r"]),
                    generation_protocol={"role": "validate_generated_condition"})
            packet["conditioned_events"] = events
        elif args.conditioned_events is not None:
            raise ValueError("unexpected_conditioned_event")
        write_json(raw / "worker-packet.json", packet)
        report["worker_packet_sha256"] = file_hash(raw / "worker-packet.json")
        report["status"] = "executing"
        write_json(args.out / "report.json", report)
        env = dict(os.environ, PYTHONPATH=str(args.research_source.resolve() / "src"),
                   PYTHONDONTWRITEBYTECODE="1", XTBFlow_PARENT_PID=str(os.getpid()))
        command = [sys.executable, str(ROOT / "scripts/research_generation_worker.py"),
                   "--research-source", str(args.research_source.resolve()), "--model-root", str(args.model_root.resolve()),
                   "--packet", str(raw / "worker-packet.json"), "--out", str(raw)]
        with (raw / "worker.log").open("wb") as log:
            completed = subprocess.run(command, env=env, cwd=args.out, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=cfg["generation"]["max_wall_seconds"], check=False)
        journal = json.loads((raw / "calls.json").read_text())
        report["call_summary"] = validate_journal(journal, cap=cfg["generation"]["max_forward_evaluations"])
        generated = json.loads((raw / "generation.json").read_text())
        report["models"] = generated.get("models")
        report["environment"] = generated.get("environment")
        report["worker_status"] = generated["status"]
        report["forbidden_imports"] = generated.get("forbidden_imports")
        report["candidates"] = [{k: row[k] for k in ("candidate_id", "query_id", "method", "status", "forward_evaluations", "wall_seconds")}
                                for row in generated["candidates"]]
        if completed.returncode or generated["status"] != "completed" or generated.get("forbidden_imports"):
            raise ValueError("generation_worker_failed")
        protocol = dict(config_sha256=report["config_sha256"], research_commit=cfg["research_commit"],
                        worker_source_sha256=file_hash(ROOT / "scripts/research_generation_worker.py"), models=report["models"])
        candidates = []
        by_id = {q["query_id"]: q for q in queries}
        for row, summary in zip(generated["candidates"], report["candidates"]):
            if row["status"] == "generated":
                candidate = certification_candidate(by_id[row["query_id"]], row, generation_protocol=protocol)
                candidates.append(dict(candidate=asdict(candidate), method=row["method"],
                                       generation_forward_evaluations=row["forward_evaluations"],
                                       generation_wall_seconds=row["wall_seconds"]))
                summary["candidate_content_sha256"] = identity(candidate.content())
        plan = dict(schema="xtbflow-generated-certification-plan/v1", source_commit=report["source_commit"],
                    generation_protocol=protocol, generator_journal_sha256=file_hash(raw / "calls.json"),
                    candidates=candidates, scope="development_functional_integration")
        write_json(raw / "certification-plan.json", plan)
        report["certification_plan_sha256"] = file_hash(raw / "certification-plan.json")
        report["certification_candidates"] = len(candidates)
        check_source(args.research_source, asset)
        if file_hash(raw / "worker-packet.json") != report["worker_packet_sha256"]:
            raise ValueError("worker_input_changed")
        report["status"] = "generation_completed"
    except subprocess.TimeoutExpired:
        report.update(status="execution_timeout", failure_type="worker_wall_limit")
    except Exception as exc:
        report.update(status="execution_failed", failure_type=type(exc).__name__)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    accounting_resolved = True
    if (raw / "calls.json").exists():
        try:
            report["call_summary"] = validate_journal(json.loads((raw / "calls.json").read_text()),
                                                     cap=cfg["generation"]["max_forward_evaluations"])
        except (ValueError, KeyError, TypeError) as exc:
            accounting_resolved = False
            report.update(status="accounting_unresolved", accounting_failure_type=type(exc).__name__)
    usage, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    wall = time.monotonic() - started
    cpu = (usage.ru_utime + usage.ru_stime - self_before.ru_utime - self_before.ru_stime
           + child.ru_utime + child.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime)
    artifacts = [{"path": str(p.relative_to(args.out)), "bytes": p.stat().st_size, "sha256": file_hash(p)}
                 for p in sorted(raw.iterdir()) if p.is_file()]
    report.update(wall_seconds=wall, cpu_seconds=cpu, artifacts=artifacts,
                  cost_basis="Program wall and process CPU; scheduler/import/setup overhead must also be reported for full-cost comparisons")
    if sum(a["bytes"] for a in artifacts) > cfg["generation"]["max_output_bytes"]:
        report["status"] = "storage_budget_exhausted"
    budget = StageBudget(phase="W01-fresh-generation-20261107", max_gpu_hours=cfg["generation"]["max_gpu_hours"],
                         max_cpu_core_hours=cfg["generation"]["max_cpu_core_hours"],
                         max_system_evaluations=cfg["generation"]["max_forward_evaluations"],
                         max_calculator_calls=1, max_storage_gb=cfg["generation"]["max_storage_gb"],
                         max_retries_per_job=0, max_wall_minutes_per_job=12)
    ledger = RunLedger(budget)
    try:
        if not accounting_resolved:
            raise ValueError("unresolved_forward_accounting_requires_reconciliation")
        ledger.append(LedgerEvent(run_id=report["config_sha256"], phase=budget.phase, calculator="trained_generator",
            protocol_id=cfg["schema"], calculator_build_hash=identity(cfg["checkpoints"]),
            system_id="generation-worker", input_hash=report.get("worker_packet_sha256", report["config_sha256"]),
            operation="other", status="success" if report["status"] == "generation_completed" else "failure",
            calculator_calls=0, system_evaluations=report["call_summary"]["forward_evaluations"],
            cpu_seconds=cpu, cpu_cores=1, gpu_seconds=wall, storage_bytes=sum(a["bytes"] for a in artifacts)),
            persist_path=args.out / "ledger.json")
        report["ledger"] = ledger.summary()
        report["settlement_status"] = "settled"
    except Exception as exc:
        report.update(settlement_status="settlement_failed", settlement_failure_type=type(exc).__name__)
    write_json(args.out / "report.json", report)
    print(json.dumps({"status": report["status"], "forwards": report["call_summary"]["forward_evaluations"]}))
    return 0 if report["status"] == "generation_completed" and report["settlement_status"] == "settled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
