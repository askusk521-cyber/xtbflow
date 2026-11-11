from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.research_generation_worker import ForwardMeter
from scripts.certify_generated_candidates import load_generation
from xtbflow.validation.certification import file_hash, identity, write_json
from xtbflow.governance.generation_bridge import (
    certification_candidate, legacy_query, validate_journal, validate_query,
)


def query():
    return dict(query_id="software-fixture", parent_id="parent", atomic_numbers=[1, 1],
                x_r=[[0., 0., -.4], [0., 0., .4]], b_r=[[0, 1], [1, 0]],
                atom_ids=["map:1", "map:2"], reactant_automorphisms=[[0, 1], [1, 0]],
                charge=0, multiplicity=1, environment={"kind": "gas", "periodic": False},
                provenance={"input_sha256": "a" * 64, "atom_mapping_evidence": "fixture mapping",
                            "electronic_state_evidence": "fixture explicit state"})


def output():
    return dict(query_id="software-fixture", candidate_id="generated-fixture", method="joint_generation",
                proposal=0, status="generated", b_dec=[[1, 0], [0, 1]], x=[[0, 0, -.5], [0, 0, .5]])


@pytest.mark.parametrize("key", ["product", "x_ts", "reference_mode", "energy", "sealed_label"])
def test_query_rejects_hidden_information(key):
    payload = query()
    payload[key] = "hidden"
    with pytest.raises(ValueError, match="forbidden"):
        legacy_query(payload)


def test_model_query_excludes_provenance_and_candidate_hypotheses():
    payload = query()
    altered = deepcopy(payload)
    altered["provenance"]["audit_only_note"] = "different source bookkeeping"
    assert legacy_query(payload) == legacy_query(altered)
    assert set(legacy_query(payload)) == {"query_id", "atomic_numbers", "x_r", "b_r", "charge", "multiplicity"}


def test_missing_or_unknown_electronic_state_is_not_defaulted():
    for value in (None, "unknown", True, 1):
        payload = query()
        payload["charge"] = value
        with pytest.raises(ValueError):
            validate_query(payload)


def test_generated_candidate_preserves_origin_and_proposed_graph():
    result = certification_candidate(query(), output(), generation_protocol={"version": "fixture"})
    assert result.parent_id == "parent" and result.atom_ids == ["map:1", "map:2"]
    assert result.coordinates_angstrom == output()["x"]
    assert result.planned_product_be == output()["b_dec"]
    assert "observed_event" not in result.provenance
    assert result.provenance["generation_protocol_sha256"]


@pytest.mark.parametrize("change", [{"status": "decode_failed"}, {"query_id": "different"},
                                    {"x": [[float("nan"), 0, 0], [0, 0, 0]]},
                                    {"b_dec": [[0, 0], [0, 0]]}])
def test_failed_or_misaligned_generator_output_cannot_enter_certification(change):
    with pytest.raises(ValueError):
        certification_candidate(query(), {**output(), **change}, generation_protocol={"version": "fixture"})


def test_forward_meter_is_durable_before_kernel_and_keeps_unfinished_calls(tmp_path, monkeypatch):
    path = tmp_path / "calls.json"
    meter = ForwardMeter(path, 2)
    meter.context = dict(method="fixture", query_id="q", proposal=0)
    kwargs = {"z": SimpleNamespace(shape=(1, 2))}
    meter.before("joint", None, (), kwargs)
    assert json.loads(path.read_text())["calls"][0]["status"] == "started"
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    meter.after(None, (), {})
    meter.before("event", None, (), kwargs)
    with pytest.raises(RuntimeError, match="budget_exhausted"):
        meter.before("geometry", None, (), kwargs)
    summary = validate_journal(json.loads(path.read_text()), cap=2)
    assert summary == {"forward_evaluations": 2, "kernel_invocations": 2,
                       "by_role": {"joint": 1, "event": 1}, "unfinished_calls": 1}


def test_journal_rejects_dropped_failed_calls_and_duplicate_sequences(tmp_path):
    meter = ForwardMeter(tmp_path / "calls.json", 3)
    meter.before("joint", None, (), {"z": SimpleNamespace(shape=(2, 2))})
    payload = json.loads(meter.path.read_text())
    assert validate_journal(payload, cap=3)["forward_evaluations"] == 2
    payload["consumed_forward_evaluations"] = 0
    with pytest.raises(ValueError, match="reconcile"):
        validate_journal(payload, cap=3)


def generation_fixture(folder):
    raw = folder / "raw"
    raw.mkdir()
    config = folder / "config.json"
    write_json(config, {"generation": {"max_forward_evaluations": 2}})
    protocol = {"version": "fixture"}
    candidate = certification_candidate(query(), output(), generation_protocol=protocol)
    journal = {"calls": [{"sequence": 1, "role": "joint", "systems": 1, "status": "completed"}],
               "consumed_forward_evaluations": 1}
    write_json(raw / "calls.json", journal)
    plan = dict(schema="xtbflow-generated-certification-plan/v1", source_commit="a"*40,
        generator_journal_sha256=file_hash(raw / "calls.json"), generation_protocol=protocol,
        candidates=[dict(candidate=asdict(candidate), method="joint_generation", generation_forward_evaluations=1)])
    write_json(raw / "certification-plan.json", plan)
    report = dict(status="generation_completed", settlement_status="settled", config_sha256=file_hash(config),
        artifacts=[dict(path="raw/calls.json", sha256=file_hash(raw / "calls.json"), bytes=(raw / "calls.json").stat().st_size)],
        certification_plan_sha256=file_hash(raw / "certification-plan.json"), source_commit="a"*40,
        call_summary=validate_journal(journal, cap=2), certification_candidates=1,
        candidates=[dict(status="generated", candidate_id=candidate.candidate_id, method="joint_generation",
                        forward_evaluations=1, candidate_content_sha256=identity(candidate.content()))])
    write_json(folder / "report.json", report)
    return config, report, plan


def test_complete_generation_chain_retains_candidate_identity(tmp_path):
    config, _, _ = generation_fixture(tmp_path)
    report, candidates = load_generation(tmp_path, config)
    assert report["call_summary"]["forward_evaluations"] == 1
    assert candidates[0][1].candidate_id == "generated-fixture"


@pytest.mark.parametrize("failure", ["unsettled", "tampered_journal", "dropped_candidate", "different_geometry"])
def test_invalid_generation_chain_cannot_enter_reference_stage(tmp_path, failure):
    config, report, plan = generation_fixture(tmp_path)
    if failure == "unsettled":
        report["settlement_status"] = "settlement_failed"
    elif failure == "tampered_journal":
        write_json(tmp_path / "raw/calls.json", {"consumed_forward_evaluations": 0, "calls": []})
    elif failure == "dropped_candidate":
        plan["candidates"] = []
    else:
        plan["candidates"][0]["candidate"]["coordinates_angstrom"][0][0] = 5.
    if failure in {"dropped_candidate", "different_geometry"}:
        write_json(tmp_path / "raw/certification-plan.json", plan)
        report["certification_plan_sha256"] = file_hash(tmp_path / "raw/certification-plan.json")
    write_json(tmp_path / "report.json", report)
    with pytest.raises(ValueError):
        load_generation(tmp_path, config)
