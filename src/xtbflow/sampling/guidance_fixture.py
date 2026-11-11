#!/usr/bin/env python3
"""Bounded synthetic chemical-BE handoff; fake/cached modes execute no scientific work."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from xtbflow.runtime import RunLedger, StageBudget
from xtbflow.sampling.integrated_guidance import FlowState, guided_step, candidate_from_state
from xtbflow.validation.certification import CertificationCandidate, CertificationService, identity, write_json


class SyntheticCoupledModel:
    """Explicit software fixture; not a trained generator or the rule four-score decoder."""
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append({k: kw[k].detach().clone() for k in ("x_cur", "b_cur", "t", "t_x")})
        x = kw["x_cur"]
        a = x[:, 1, 0]-x[:, 0, 0]
        pattern = x.new_tensor([[1., -1.], [-1., 1.]])
        return dict(b_vel=a[:, None, None]*pattern, x_vel=-.1*x)


class FixtureReference:
    identity = {"engine": "software_fixture", "device": "cpu", "version": "integrated-v1"}

    def run(self, candidate, meter, work):
        meter.before("software_fixture", "scf")
        meter.record("software_fixture", "scf", .001)
        return {"ts": {"status": "SOFTWARE_FIXTURE_UNRESOLVED"}}


def fixture_query():
    b = torch.tensor([[[0., 1.], [1., 0.]]], dtype=torch.float64)
    x = torch.tensor([[[-.4, 0., 0.], [.4, 0., 0.]]], dtype=torch.float64)
    query = SimpleNamespace(element_index=torch.ones((1, 2), dtype=torch.long),
                            atom_mask=torch.ones((1, 2), dtype=torch.bool), x_r=x, b_r=b)
    packet = dict(query_id="synthetic:H2", parent_id="synthetic", atomic_numbers=[1, 1],
                  x_r=x[0].tolist(), b_r=b[0].int().tolist(), atom_ids=["map:1", "map:2"],
                  reactant_automorphisms=[[0, 1], [1, 0]], charge=0, multiplicity=1,
                  environment={"kind": "gas", "periodic": False},
                  provenance={"input_sha256": identity(x.tolist()), "atom_mapping_evidence": "fixture row IDs",
                              "electronic_state_evidence": "explicit synthetic neutral singlet"})
    return query, packet, FlowState(b, x, torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))


def run_fixture(out: Path, mode: str) -> dict:
    cfg = json.loads((Path(__file__).resolve().parents[3]/"configs/research/integrated_entry_20261111.json").read_text())
    budget = StageBudget(**cfg["software_budget"])
    generated = out/"generated.json"
    if mode == "fake":
        out.mkdir(parents=True, exist_ok=False)
        query, packet, state = fixture_query()
        model = SyntheticCoupledModel()
        observations = []
        def guide(injection, query):
            observations.append({"old_clock": injection.observed.t_x.tolist(),
                                 "injection_clock": injection.base.t_x.tolist(),
                                 "observed_endpoint": injection.predicted_x.tolist()})
            return -.01*injection.predicted_x
        state = guided_step(model, query, state, next_t_b=.5, next_t_x=.25, guide=guide, alpha=1.)
        state = guided_step(model, query, state, next_t_b=1., next_t_x=1.)
        upstream = dict(status="completed", termination="confirmed", settlement="settled", reservations={})
        candidate = candidate_from_state(packet, state, lambda b: torch.round(b).to(torch.int64),
                                         candidate_id="integrated-fixture", protocol=cfg, upstream=upstream)
        payload = dict(candidate=asdict(candidate), observations=observations, model_forwards=len(model.calls), upstream=upstream)
        write_json(generated, dict(payload=payload, sha256=identity(payload)))
    else:
        payload = json.loads(generated.read_text())
        if identity(payload["payload"]) != payload["sha256"]:
            raise ValueError("cached_generated_identity_mismatch")
        candidate = CertificationCandidate(**payload["payload"]["candidate"])
        if not (out/"fake-receipt.json").exists() or not list((out/"reference").glob("*/attempt-*/settlement.json")):
            raise ValueError("incomplete_cached_fixture")
    service = CertificationService(out/"reference", budget)
    result = service.run(candidate, FixtureReference(), max_calls=2, max_wall_seconds=10,
                         cpu_cores=1, max_raw_bytes=1024**2)
    if mode == "cached" and not result["cache_hit"]:
        raise ValueError("cached_mode_did_not_replay")
    report = dict(evidence_level="software_demo", mode=mode, candidate_sha256=identity(candidate.content()),
                  result=result, ledger=RunLedger.load(service.ledger_path).summary(),
                  new_scientific_calculations=0, trained_forwards=0)
    target = out/(mode+"-receipt.json")
    if target.exists():
        raise FileExistsError(target)
    write_json(target, report)
    return report

