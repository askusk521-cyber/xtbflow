#!/usr/bin/env python3
"""Score frozen reactant-only candidates and report parent-weighted metrics.

This command is deliberately separate from ``run_reactant_generation.py``.
It reads reference product/TS labels only after the generated candidate file
exists, so a scoring change cannot alter the generated candidate set.  The
result is a development diagnostic; the Diels--Alder adapter's geometry-level
provenance still needs independent per-file verification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch

from xtbflow.data.dft_da import load_dft_da_samples, with_splits
from xtbflow.evaluation.generation import (
    GeneratedCandidate,
    ReactantView,
    centered_reactant_coordinates,
    reactant_packed_state,
    reactant_view_from_sample,
    reference_labels_from_sample,
    score_candidates,
)
from xtbflow.evaluation.grouped import paired_group_comparison, summarize_by_group
from xtbflow.models import pack_be, unpack_be
from xtbflow.proposals import CoreRuleConfig, enumerate_core_events
from xtbflow.proposals.roles import infer_roles
from mechai.data.events import LewisState


METRICS = ("event_exact_match", "event_bond_mse", "geometry_endpoint_mse")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_locator(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _candidate_from_mapping(value: dict[str, Any]) -> GeneratedCandidate:
    if value.get("decode_status") not in {None, "accepted"}:
        raise ValueError("cannot score a rejected candidate")
    coordinates = value.get("coordinates_angstrom")
    packed = value.get("packed_event")
    if not isinstance(coordinates, list) or not isinstance(packed, list):
        raise ValueError("candidate output is missing coordinates_angstrom or packed_event")
    candidate = GeneratedCandidate(
        record_id=value.get("record_id"),
        control_mode=value.get("control_mode"),
        coordinates=tuple(tuple(float(item) for item in row) for row in coordinates),
        packed_event=tuple(int(item) for item in packed),
        steps=value.get("steps"),
        candidate_index=value.get("candidate_index", 0),
    )
    declared_fingerprint = value.get("candidate_fingerprint")
    if not isinstance(declared_fingerprint, str) or declared_fingerprint != candidate.fingerprint():
        raise ValueError(f"candidate fingerprint mismatch for {candidate.record_id}")
    return candidate


def _zero_change_candidate(view: ReactantView) -> GeneratedCandidate:
    return GeneratedCandidate(
        record_id=view.record_id,
        control_mode="zero_change",
        coordinates=centered_reactant_coordinates(view),
        packed_event=reactant_packed_state(view),
        steps=0,
    )


def _reactant_state(view: ReactantView) -> LewisState:
    packed = torch.tensor(reactant_packed_state(view), dtype=torch.float64)
    matrix = unpack_be(packed, len(view.atomic_numbers)).to(torch.int64)
    return LewisState(
        symbols=view.symbols,
        be=tuple(tuple(int(item) for item in row) for row in matrix.tolist()),
        charge=view.charge,
        multiplicity=view.multiplicity,
    )


def _rule_candidates(view: ReactantView, cap: int) -> tuple[GeneratedCandidate, ...]:
    state = _reactant_state(view)
    proposals = enumerate_core_events(
        state,
        infer_roles(state),
        config=CoreRuleConfig(max_candidates=cap),
    )
    geometry = centered_reactant_coordinates(view)
    candidates: list[GeneratedCandidate] = []
    for index, proposal in enumerate(proposals.candidates[:cap]):
        endpoint = proposal.event.apply(state)
        packed = pack_be(
            torch.tensor(endpoint.be, dtype=torch.float64)
        ).tolist()
        candidates.append(
            GeneratedCandidate(
                record_id=view.record_id,
                control_mode="reactant_rule",
                coordinates=geometry,
                packed_event=tuple(int(item) for item in packed),
                steps=0,
                candidate_index=index,
            )
        )
    return tuple(candidates)


def _record_metrics(
    view: ReactantView,
    labels: Any,
    candidates: tuple[GeneratedCandidate, ...],
    *,
    method: str,
) -> tuple[dict[str, Any] | None, tuple[dict[str, Any], ...]]:
    scored = score_candidates(view, labels, candidates)
    if not scored:
        return None, scored
    return (
        {
            "method": method,
            "record_id": view.record_id,
            "parent_reaction_id": view.parent_reaction_id,
            "family_id": view.family_id,
            "candidate_count": len(scored),
            "decode_accepted": 1,
            "event_exact_match": int(any(row["event_exact_match"] for row in scored)),
            "event_bond_mse": (
                float(min(row["event_bond_mse"] for row in scored if row["event_bond_mse"] is not None))
                if any(row["event_bond_mse"] is not None for row in scored)
                else None
            ),
            "geometry_endpoint_mse": float(min(row["geometry_endpoint_mse"] for row in scored)),
            "candidate_fingerprints": [row["candidate_fingerprint"] for row in scored],
        },
        scored,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260930)
    parser.add_argument("--allow-partial", action="store_true", help="score a subset while marking coverage_complete=false")
    return parser


def main() -> int:
    args = _parser().parse_args()
    stage_start = time.monotonic()
    if args.candidate_cap < 1 or args.bootstrap_samples < 1:
        raise SystemExit("candidate-cap and bootstrap-samples must be positive")
    output = args.output.expanduser()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    root = Path(__file__).resolve().parents[1]
    generated = json.loads(args.generated.read_text(encoding="utf-8"))
    if generated.get("schema") != "xtbflow-reactant-generation-evidence/v2":
        raise SystemExit("unsupported generated-candidate evidence schema")
    if generated.get("status") != "completed":
        raise SystemExit("cannot score an incomplete generation run")
    if generated.get("target_fields_read") is not False:
        raise SystemExit("generated evidence does not prove a target-free input")
    generated_list = generated.get("records", [])
    if not isinstance(generated_list, list) or any(not isinstance(row, dict) for row in generated_list):
        raise SystemExit("generated evidence records must be a list of objects")
    generated_ids = [row.get("record_id") for row in generated_list]
    if any(not isinstance(record_id, str) or not record_id for record_id in generated_ids):
        raise SystemExit("generated evidence contains a malformed record_id")
    if len(set(generated_ids)) != len(generated_ids):
        raise SystemExit("generated evidence contains duplicate record_id values")
    generated_rows = dict(zip(generated_ids, generated_list))
    samples, source_audit = load_dft_da_samples(args.cache_root)
    samples = [sample for sample in with_splits(samples) if sample.record.admission == f"development_{args.split}"]
    by_id = {sample.record.record_id: sample for sample in samples}
    if set(generated_rows) - set(by_id):
        unknown = sorted(set(generated_rows) - set(by_id))
        raise SystemExit(f"generated records are absent from the selected source split: {unknown[:3]}")
    missing = sorted(set(by_id) - set(generated_rows))
    if missing and not args.allow_partial:
        raise SystemExit(
            f"generated evidence is incomplete for the selected split ({len(missing)} missing); use --allow-partial for a diagnostic"
        )

    method_rows: dict[str, list[dict[str, Any]]] = {"learned": [], "zero_change": [], "reactant_rule": []}
    raw_scores: list[dict[str, Any]] = []
    generation_accounting: list[dict[str, Any]] = []
    for record_id, generated_row in sorted(generated_rows.items()):
        sample = by_id[record_id]
        view = reactant_view_from_sample(sample)
        if generated_row.get("input_fingerprint") != view.input_fingerprint:
            raise SystemExit(f"input fingerprint mismatch for generated record {record_id}")
        labels = reference_labels_from_sample(sample)
        candidates = tuple(_candidate_from_mapping(item) for item in generated_row.get("candidates", []))
        learned_summary, learned_scores = _record_metrics(view, labels, candidates, method="learned")
        if learned_summary is not None:
            method_rows["learned"].append(learned_summary)
        raw_scores.extend({"method": "learned", **row} for row in learned_scores)
        generation_accounting.append(
            {
                "record_id": record_id,
                "parent_reaction_id": view.parent_reaction_id,
                "attempted": int(generated_row.get("attempted", 0)),
                "accepted": int(generated_row.get("accepted", 0)),
                "rejected": int(generated_row.get("rejected", 0)),
                "candidate_cap": generated.get("candidate_cap"),
            }
        )
        zero_summary, zero_scores = _record_metrics(view, labels, (_zero_change_candidate(view),), method="zero_change")
        if zero_summary is not None:
            method_rows["zero_change"].append(zero_summary)
        raw_scores.extend({"method": "zero_change", **row} for row in zero_scores)
        rule_candidates = _rule_candidates(view, args.candidate_cap)
        rule_summary, rule_scores = _record_metrics(view, labels, rule_candidates, method="reactant_rule")
        if rule_summary is not None:
            method_rows["reactant_rule"].append(rule_summary)
        raw_scores.extend({"method": "reactant_rule", **row} for row in rule_scores)

    summaries: dict[str, Any] = {}
    for method, rows in method_rows.items():
        summaries[method] = (
            summarize_by_group(rows, group_key="parent_reaction_id", metric_keys=METRICS)
            if rows
            else {"record_count": 0, "group_count": 0, "record_weighted": {}, "parent_equal": {}, "groups": []}
        )
    paired = {}
    if method_rows["learned"] and method_rows["zero_change"]:
        paired["learned_minus_zero_change"] = paired_group_comparison(
            method_rows["learned"],
            method_rows["zero_change"],
            metric_keys=METRICS,
            left_label="learned",
            right_label="zero_change",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
    if method_rows["learned"] and method_rows["reactant_rule"]:
        paired["learned_minus_reactant_rule"] = paired_group_comparison(
            method_rows["learned"],
            method_rows["reactant_rule"],
            metric_keys=METRICS,
            left_label="learned",
            right_label="reactant_rule",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + 1,
        )
    report = {
        "schema": "xtbflow-reactant-generation-scoring/v1",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_host": platform.node(),
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__},
        "generated_locator": _safe_locator(args.generated, root),
        "generated_sha256": _sha256(args.generated),
        "source_url": source_audit.get("source_url"),
        "source_revision": source_audit.get("source_revision"),
        "source_archive_sha256": source_audit.get("source_archive_sha256"),
        "source_csv_sha256": source_audit.get("source_csv_sha256"),
        "split": args.split,
        "record_count": len(generation_accounting),
        "parent_count": len({row["parent_reaction_id"] for row in generation_accounting}),
        "expected_record_count": len(by_id),
        "missing_record_ids": missing,
        "coverage_complete": not missing,
        "generation_accounting": generation_accounting,
        "method_summaries": summaries,
        "method_summary_population": "records with at least one accepted decoded candidate; generation_accounting contains all attempted records",
        "paired_parent_bootstrap": paired,
        "raw_candidate_scores": raw_scores,
        "elapsed_seconds": time.monotonic() - stage_start,
        "claim_limit": "label-assisted development diagnostic; no DFT geometry certification, physical refinement, or discovery claim",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(output, report)
    print(json.dumps({"status": "completed", "records": len(generation_accounting), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
