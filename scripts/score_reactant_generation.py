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

from xtbflow.evaluation.generation import (
    GeneratedCandidate,
    ReactantView,
    centered_reactant_coordinates,
    deduplicate_generated_candidates,
    generated_candidate_key,
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


def _deduplicate_with_stats(
    candidates: tuple[GeneratedCandidate, ...],
    *,
    max_slots: int,
    mode: str,
) -> tuple[tuple[GeneratedCandidate, ...], int, int, int]:
    """Return kept candidates and explicit duplicate/cap accounting."""

    unique_keys = {generated_candidate_key(candidate, mode=mode) for candidate in candidates}
    unique_count = len(unique_keys)
    kept = deduplicate_generated_candidates(candidates, max_slots=max_slots, mode=mode)
    return kept, len(candidates) - unique_count, unique_count, max(0, unique_count - max_slots)


def _record_metrics(
    view: ReactantView,
    labels: Any,
    candidates: tuple[GeneratedCandidate, ...],
    *,
    event_candidates: tuple[GeneratedCandidate, ...] | None = None,
    method: str,
    max_slots: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Score one method while retaining an evaluation row for every parent.

    Geometry and event metrics have different slot contracts.  A baseline may
    propose multiple events at one geometry, so event metrics use a
    geometry-plus-event key while geometry metrics use a geometry key.
    """

    raw_candidates = tuple(candidates)
    raw_event_candidates = raw_candidates if event_candidates is None else tuple(event_candidates)
    geometry_candidates, geometry_duplicates, geometry_unique, geometry_cap = _deduplicate_with_stats(
        raw_candidates, max_slots=max_slots, mode="geometry"
    )
    event_slots, event_duplicates, event_unique, event_cap = _deduplicate_with_stats(
        raw_event_candidates, max_slots=max_slots, mode="geometry_event"
    )
    geometry_scores = score_candidates(view, labels, geometry_candidates)
    event_scores = score_candidates(view, labels, event_slots)
    scored: list[dict[str, Any]] = []
    for scope, rows in (("geometry", geometry_scores), ("event", event_scores)):
        for row in rows:
            annotated = dict(row)
            annotated["metric_scope"] = scope
            scored.append(annotated)

    accepted_event = [row for row in event_scores if row["decode_status"] == "accepted"]
    event_exact_values = [row["event_exact_match"] for row in accepted_event if row["event_exact_match"] is not None]
    event_bond_values = [row["event_bond_mse"] for row in accepted_event if row["event_bond_mse"] is not None]
    geometry_values = [row["geometry_endpoint_mse"] for row in geometry_scores]
    if not raw_candidates:
        status = "empty"
    elif not event_scores:
        status = "empty"
    elif not accepted_event:
        status = "decode_failed"
    else:
        status = "accepted"
    summary = {
        "method": method,
        "record_id": view.record_id,
        "parent_reaction_id": view.parent_reaction_id,
        "family_id": view.family_id,
        "status": status,
        "raw_candidate_count": len(raw_candidates),
        # Keep candidate_count as the event-slot count for compatibility, but
        # expose both slot populations so a report cannot hide a collapse.
        "candidate_count": len(event_slots),
        "geometry_candidate_count": len(geometry_candidates),
        "event_candidate_count": len(event_slots),
        "raw_event_candidate_count": len(raw_event_candidates),
        "decode_accepted": len(accepted_event),
        "decode_unresolved": len(event_scores) - len(accepted_event),
        "event_exact_match": None if not event_exact_values else int(any(event_exact_values)),
        "event_bond_mse": None if not event_bond_values else float(min(event_bond_values)),
        "geometry_endpoint_mse": None if not geometry_values else float(min(geometry_values)),
        "candidate_fingerprints": sorted({row["candidate_fingerprint"] for row in scored}),
        "deduplication": {
            "geometry_unique_before_cap": geometry_unique,
            "event_unique_before_cap": event_unique,
            "geometry_duplicates_collapsed": geometry_duplicates,
            "event_duplicates_collapsed": event_duplicates,
            "geometry_slots_truncated": geometry_cap,
            "event_slots_truncated": event_cap,
        },
    }
    return summary, tuple(scored)


def _metric_coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int | float | bool]]:
    """Report missing label-assisted metrics without converting them to zero."""

    denominator = len(rows)
    parent_ids = {row.get("parent_reaction_id") for row in rows}
    return {
        metric: {
            "observed": sum(row.get(metric) is not None for row in rows),
            "missing": sum(row.get(metric) is None for row in rows),
            "denominator": denominator,
            "coverage": (sum(row.get(metric) is not None for row in rows) / denominator if denominator else 0.0),
            "parent_observed": len({row.get("parent_reaction_id") for row in rows if row.get(metric) is not None}),
            "parent_missing": len(parent_ids) - len({row.get("parent_reaction_id") for row in rows if row.get(metric) is not None}),
            "parent_denominator": len(parent_ids),
            "parent_coverage": (len({row.get("parent_reaction_id") for row in rows if row.get(metric) is not None}) / len(parent_ids) if parent_ids else 0.0),
            "complete": bool(rows) and all(row.get(metric) is not None for row in rows),
        }
        for metric in METRICS
    }


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize every observed metric without dropping unresolved parents."""

    coverage = _metric_coverage(rows)
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        group_id = row["parent_reaction_id"]
        entry = groups.setdefault(group_id, {"group_id": group_id, "record_count": 0, "metric_means": {}})
        entry["record_count"] += 1
    record_weighted: dict[str, float] = {}
    parent_equal: dict[str, float] = {}
    for metric in METRICS:
        observed_rows = [row for row in rows if row.get(metric) is not None]
        if not observed_rows:
            continue
        partial = summarize_by_group(observed_rows, group_key="parent_reaction_id", metric_keys=(metric,))
        record_weighted[metric] = partial["record_weighted"][metric]
        parent_equal[metric] = partial["parent_equal"][metric]
        for group in partial["groups"]:
            groups[group["group_id"]]["metric_means"][metric] = group["metric_means"][metric]
    summary = {
        "group_key": "parent_reaction_id",
        "record_count": len(rows),
        "group_count": len(groups),
        "record_weighted": record_weighted,
        "parent_equal": parent_equal,
        "groups": [groups[group_id] for group_id in sorted(groups)],
        "status_counts": {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted({row.get("status", "unknown") for row in rows})
        },
    }
    summary["metric_coverage"] = coverage
    summary["unavailable_metrics"] = [metric for metric in METRICS if coverage[metric]["observed"] == 0]
    return summary


def _paired_summary(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], *,
    left_label: str, right_label: str, bootstrap_samples: int, seed: int,
) -> dict[str, Any]:
    """Pair arms on the fixed parent population and retain unresolved metrics."""

    coverage = {
        "left": _metric_coverage(left_rows),
        "right": _metric_coverage(right_rows),
    }
    left_groups = {row.get("parent_reaction_id") for row in left_rows}
    right_groups = {row.get("parent_reaction_id") for row in right_rows}
    if left_groups != right_groups:
        return {
            "status": "unavailable",
            "left_label": left_label,
            "right_label": right_label,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "fixed_parent_set": False,
            "paired_parent_count": len(left_groups.intersection(right_groups)),
            "fixed_parent_ids": sorted(left_groups.intersection(right_groups)),
            "unmatched_left": sorted(left_groups - right_groups),
            "unmatched_right": sorted(right_groups - left_groups),
            "metrics": {},
            "metric_coverage": coverage,
            "unavailable_metrics": list(METRICS),
        }
    metrics: dict[str, Any] = {}
    available: list[str] = []
    for index, metric in enumerate(METRICS):
        if not coverage["left"][metric]["complete"] or not coverage["right"][metric]["complete"]:
            continue
        comparison = paired_group_comparison(
            left_rows,
            right_rows,
            metric_keys=(metric,),
            left_label=left_label,
            right_label=right_label,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index,
        )
        metrics[metric] = comparison["metrics"][metric]
        available.append(metric)
    return {
        "status": "completed" if available else "unavailable",
        "left_label": left_label,
        "right_label": right_label,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "fixed_parent_set": True,
        "paired_parent_count": len(left_groups),
        "fixed_parent_ids": sorted(left_groups),
        "metrics": metrics,
        "metric_coverage": coverage,
        "unavailable_metrics": [metric for metric in METRICS if metric not in available],
    }


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
    # Keep the post-hoc scorer importable for contract tests on a checkout
    # that has only the target-free generation module.  The public DA adapter
    # is needed only when the full source-label scoring stage is launched.
    try:
        from xtbflow.data.dft_da import load_dft_da_samples, with_splits
    except ImportError as exc:
        raise SystemExit(
            "Diels--Alder source scoring requires the admitted dft_da adapter; "
            "target-free generation remains available without it"
        ) from exc
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
    evaluation_ids = sorted(by_id) if args.allow_partial else sorted(generated_rows)
    for record_id in evaluation_ids:
        generated_row = generated_rows.get(record_id)
        sample = by_id[record_id]
        view = reactant_view_from_sample(sample)
        if generated_row is not None and generated_row.get("input_fingerprint") != view.input_fingerprint:
            raise SystemExit(f"input fingerprint mismatch for generated record {record_id}")
        labels = reference_labels_from_sample(sample)
        candidates = (
            tuple(_candidate_from_mapping(item) for item in generated_row.get("candidates", []))
            if generated_row is not None
            else ()
        )
        event_candidates = (
            tuple(
                _candidate_from_mapping(item)
                for item in generated_row.get("event_candidates", generated_row.get("candidates", []))
            )
            if generated_row is not None
            else ()
        )
        learned_summary, learned_scores = _record_metrics(
            view,
            labels,
            candidates,
            event_candidates=event_candidates,
            method="learned",
            max_slots=args.candidate_cap,
        )
        if generated_row is None:
            learned_summary["status"] = "missing_from_generation"
        elif not candidates:
            if int(generated_row.get("execution_failed", 0)):
                learned_summary["status"] = "generation_failed"
            elif int(generated_row.get("decode_failed", 0)):
                learned_summary["status"] = "decode_failed"
            elif int(generated_row.get("invalid", 0)):
                learned_summary["status"] = "invalid"
        method_rows["learned"].append(learned_summary)
        raw_scores.extend({"method": "learned", **row} for row in learned_scores)
        generation_accounting.append(
            {
                "record_id": record_id,
                "parent_reaction_id": view.parent_reaction_id,
                "status": "missing_from_generation" if generated_row is None else generated_row.get("status", "completed"),
                "failure_stage": None if generated_row is None else generated_row.get("failure_stage"),
                "attempted": int(generated_row.get("attempted", 0)) if generated_row is not None else 0,
                "accepted": int(generated_row.get("accepted", 0)) if generated_row is not None else 0,
                "decode_accepted": int(generated_row.get("decode_accepted", generated_row.get("accepted", 0))) if generated_row is not None else 0,
                "decode_failed": int(generated_row.get("decode_failed", 0)) if generated_row is not None else 0,
                "invalid": int(generated_row.get("invalid", 0)) if generated_row is not None else 0,
                "execution_failed": int(generated_row.get("execution_failed", 0)) if generated_row is not None else 0,
                "duplicate_geometry": int(generated_row.get("duplicate_geometry", 0)) if generated_row is not None else 0,
                "duplicate_event": int(generated_row.get("duplicate_event", 0)) if generated_row is not None else 0,
                "unique_candidates": int(generated_row.get("unique_candidates", generated_row.get("accepted", 0))) if generated_row is not None else 0,
                "event_unique_candidates": int(generated_row.get("event_unique_candidates", len(event_candidates))) if generated_row is not None else 0,
                "candidate_slot_contract": (
                    generated_row.get(
                        "candidate_slot_contract",
                        {"geometry": "candidates", "geometry_event": "event_candidates"},
                    )
                    if generated_row is not None
                    else {"geometry": "candidates", "geometry_event": "event_candidates"}
                ),
                "rejected": int(generated_row.get("rejected", 0)) if generated_row is not None else 0,
                "rejection_reasons": list(generated_row.get("rejection_reasons", ())) if generated_row is not None else ["missing_from_generation"],
                "candidate_cap": generated.get("candidate_cap"),
            }
        )
        zero_summary, zero_scores = _record_metrics(view, labels, (_zero_change_candidate(view),), method="zero_change", max_slots=args.candidate_cap)
        method_rows["zero_change"].append(zero_summary)
        raw_scores.extend({"method": "zero_change", **row} for row in zero_scores)
        rule_candidates = _rule_candidates(view, args.candidate_cap)
        rule_summary, rule_scores = _record_metrics(view, labels, rule_candidates, method="reactant_rule", max_slots=args.candidate_cap)
        method_rows["reactant_rule"].append(rule_summary)
        raw_scores.extend({"method": "reactant_rule", **row} for row in rule_scores)

    summaries: dict[str, Any] = {method: _summary_for_rows(rows) for method, rows in method_rows.items()}
    paired = {}
    if method_rows["learned"] and method_rows["zero_change"]:
        paired["learned_minus_zero_change"] = _paired_summary(
            method_rows["learned"],
            method_rows["zero_change"],
            left_label="learned",
            right_label="zero_change",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
    if method_rows["learned"] and method_rows["reactant_rule"]:
        paired["learned_minus_reactant_rule"] = _paired_summary(
            method_rows["learned"],
            method_rows["reactant_rule"],
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
        "method_summary_population": "all predeclared source records; unresolved metrics remain null and generation_accounting preserves attempt/dedup statuses",
        "fixed_parent_population": sorted({row["parent_reaction_id"] for row in generation_accounting}),
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
