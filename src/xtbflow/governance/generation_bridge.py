"""Whitelist boundary between research generators and current certification."""
from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

from xtbflow.validation.certification import CertificationCandidate, identity


QUERY_FIELDS = frozenset({"query_id", "parent_id", "atomic_numbers", "x_r", "b_r", "atom_ids",
                          "reactant_automorphisms", "charge", "multiplicity", "environment", "provenance"})
LEGACY_FIELDS = ("query_id", "atomic_numbers", "x_r", "b_r", "charge", "multiplicity")


def validate_generation_config(cfg: dict) -> None:
    """An external action config cannot turn a bounded worker into an unlimited run."""
    methods = {"joint_generation": 1, "independent_event_then_geometry": 2, "conditioned_geometry": 1}
    if not cfg["methods"] or len(cfg["methods"]) != len(set(cfg["methods"])) or any(m not in methods for m in cfg["methods"]):
        raise ValueError("unregistered or duplicated generation methods")
    for name, cap in (("query_count", 8), ("proposals_per_query", 8), ("steps_per_branch", 100)):
        if type(cfg[name]) is not int or not 1 <= cfg[name] <= cap:
            raise ValueError("finite generation dimensions required")
    if cfg["joint_clock"] != "event_lead2" or type(cfg["sampling_seed"]) is not int or cfg["sampling_seed"] < 0:
        raise ValueError("unregistered clock or invalid seed")
    g = cfg["generation"]
    for name, cap in (("max_forward_evaluations", 10000), ("cpu_cores", 8), ("max_output_bytes", 64 * 1024**2)):
        if type(g[name]) is not int or not 1 <= g[name] <= cap:
            raise ValueError("finite worker resources required")
    for name, cap in (("max_wall_seconds", 600), ("max_gpu_hours", 1), ("max_cpu_core_hours", 8), ("max_storage_gb", 1)):
        v = g[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= cap:
            raise ValueError("finite worker time/storage required")
    needed = cfg["query_count"] * cfg["proposals_per_query"] * cfg["steps_per_branch"] * sum(methods[m] for m in cfg["methods"])
    if needed > g["max_forward_evaluations"]:
        raise ValueError("requested rollouts exceed forward envelope")
    if cfg.get("candidate_namespace") and (len(cfg["candidate_namespace"]) != 64 or any(c not in "0123456789abcdef" for c in cfg["candidate_namespace"])):
        raise ValueError("candidate namespace must be a SHA256")


def validate_query(query: dict) -> CertificationCandidate:
    if set(query) != QUERY_FIELDS:
        raise ValueError("reactant query fields are missing or include forbidden information")
    # The identity product is used only to reuse fixed-pool validation. It is
    # never sent as a generated hypothesis or used to claim a reaction.
    return CertificationCandidate(
        candidate_id=query["query_id"], parent_id=query["parent_id"],
        atomic_numbers=query["atomic_numbers"], coordinates_angstrom=query["x_r"],
        atom_ids=query["atom_ids"], reactant_be=query["b_r"], planned_product_be=query["b_r"],
        reactant_automorphisms=query["reactant_automorphisms"], charge=query["charge"],
        multiplicity=query["multiplicity"], environment=query["environment"], provenance=query["provenance"])


def legacy_query(query: dict) -> dict:
    validate_query(query)
    return {key: query[key] for key in LEGACY_FIELDS}


def certification_candidate(query: dict, output: dict, *, generation_protocol: dict) -> CertificationCandidate:
    seed = validate_query(query)
    if output["query_id"] != query["query_id"] or output["status"] != "generated":
        raise ValueError("only a finite, decoded candidate from this query can be certified")
    provenance = {**seed.provenance, "generation_protocol_sha256": identity(generation_protocol),
                  "method": output["method"], "proposal": output["proposal"],
                  "generated_output_sha256": identity(output)}
    payload = asdict(seed)
    payload.update(candidate_id=output["candidate_id"], coordinates_angstrom=output["x"],
                   planned_product_be=output["b_dec"], provenance=provenance)
    return CertificationCandidate(**payload)


def validate_journal(journal: dict, *, cap: int) -> dict[str, Any]:
    """Retain started failed calls; never count only completed forwards."""
    rows = journal["calls"]
    if len(rows) > cap or journal["consumed_forward_evaluations"] != sum(r["systems"] for r in rows):
        raise ValueError("generation call journal does not reconcile")
    if journal["consumed_forward_evaluations"] > cap:
        raise ValueError("generation forward cap exceeded")
    by_role: dict[str, int] = {}
    for index, row in enumerate(rows, 1):
        if row["sequence"] != index or row["status"] not in {"started", "completed"} or type(row["systems"]) is not int or row["systems"] < 1:
            raise ValueError("malformed generator call")
        if row["role"] not in {"joint", "event", "geometry"}:
            raise ValueError("undeclared generator role")
        by_role[row["role"]] = by_role.get(row["role"], 0) + row["systems"]
    return {"forward_evaluations": journal["consumed_forward_evaluations"], "kernel_invocations": len(rows),
            "by_role": by_role, "unfinished_calls": sum(r["status"] == "started" for r in rows)}
