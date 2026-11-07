"""Decision/recovery software checks using synthetic scientific-tool fixtures."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from xtbflow.runtime import feedback_loop as module
from xtbflow.runtime.feedback_loop import (
    CachedReferenceTools, FeedbackLoop, candidate_view, choose, initial_state, reduce_outcome,
)
from xtbflow.validation.certification import CertificationService, file_hash, identity
from test_certification_service import FixtureBackend, budget, candidate, run


def config(**changes):
    root = Path(__file__).resolve().parents[1]
    value = json.loads((root / "configs/agent/feedback_policy_20261107.json").read_text())
    value.update(changes)
    return value


def setup(root, *, mode="alternative", two=False, alias=False):
    service = CertificationService(root / "service", budget())
    candidates = [candidate()]
    if two:
        candidates.append(candidate(candidate_id="second", coordinates_angstrom=((0., 0., 0.), (0., 0., .9))))
    attempts, pins, views = {}, {}, []
    for item in candidates:
        result = run(service, item=item, backend=FixtureBackend(mode=mode))
        work = service.root / result["record"]["cache_key"] / "attempt-0001"
        attempts[item.candidate_id], pins[item.candidate_id] = work, file_hash(work / "result.json")
        views.append(candidate_view(item, identity(result["record"]["protocol"])))
    if alias:
        item = replace(candidates[0], candidate_id="alias")
        views.append(candidate_view(item, views[0]["protocol_sha256"]))
        attempts["alias"], pins["alias"] = attempts["fixture"], pins["fixture"]
    return views, CachedReferenceTools(attempts, pins)


def test_real_tool_feedback_changes_only_predeclared_sampler_and_is_replayable(tmp_path):
    views, tools = setup(tmp_path)
    loop = FeedbackLoop(tmp_path / "loop", views, config(), tools)
    result = loop.run()
    assert result == loop.run()
    assert result["sampler"] == "independent_serial"
    assert result["sampler_changes"][0]["executed_generation"] is False
    assert len(result["channels"]) == 1 and result["attributed_reference_cost"]["reference_primitive_calls"] == 2
    assert result["step"] == 3
    fixed = FeedbackLoop(tmp_path / "fixed", views, config(policy="fixed"), tools).run()
    assert fixed["step"] == 2 and fixed["sampler"] == "joint_event_lead2"
    assert fixed["attributed_reference_cost"] == result["attributed_reference_cost"]


def test_cache_reuse_is_not_an_independent_duplicate_channel_or_second_charge(tmp_path):
    views, tools = setup(tmp_path, mode="success", alias=True)
    state = FeedbackLoop(tmp_path / "loop", views, config(), tools).run()
    assert state["reused_evidence"] == 1 and state["duplicate_channels"] == 0
    assert len(state["evidence"]) == 1 and state["attributed_reference_cost"]["reference_primitive_calls"] == 2
    assert state["sampler"] == "joint_event_lead2"


def test_independent_geometries_reaching_same_event_trigger_feedback_without_new_channel(tmp_path):
    views, tools = setup(tmp_path, mode="success", two=True)
    state = FeedbackLoop(tmp_path / "loop", views, config(), tools).run()
    assert state["duplicate_channels"] == 1 and len(state["channels"]) == 1
    assert state["attributed_reference_cost"]["reference_primitive_calls"] == 4
    assert state["sampler"] == "independent_serial"


def test_failure_keeps_consumed_calls_lower_layers_and_no_certified_channel(tmp_path):
    views, tools = setup(tmp_path, mode="failure")
    state = FeedbackLoop(tmp_path / "loop", views, config(), tools).run()
    row = next(iter(state["evidence"].values()))
    assert row["execution_status"] == "execution_failed"
    assert row["feedback"]["evidence_status"] == "attempted_unresolved"
    assert row["feedback"]["first_order_saddle"]
    assert state["attributed_reference_cost"]["reference_primitive_calls"] == 3
    assert row["provenance"]["unfinished_primitives"] == 1
    assert not state["channels"] and not state["sampler_changes"]


def test_complete_envelope_is_required_before_reading_any_future_observation(tmp_path):
    views, tools = setup(tmp_path)
    cfg = config()
    cfg["budget"]["reference_primitive_calls"] = 2999
    state = FeedbackLoop(tmp_path / "loop", views, cfg, tools).run()
    assert state["stop_reason"] == "insufficient_complete_reference_envelope"
    assert not state["evidence"] and state["attributed_reference_cost"]["reference_primitive_calls"] == 0


def test_no_future_results_or_label_fields_can_enter_decision_state(tmp_path):
    views, tools = setup(tmp_path)
    cfg = config()
    state = initial_state(views, cfg)
    before = choose(state, cfg)
    tools.pins = {key: "f" * 64 for key in tools.pins}  # Opaque future outcome identity changes.
    assert choose(state, cfg) == before
    assert not state["evidence"] and not state["channels"]
    changed = deepcopy(views)
    changed[0]["sealed_test_label"] = "forbidden"
    with pytest.raises(ValueError, match="unexpected"):
        initial_state(changed, cfg)
    changed = deepcopy(views)
    changed[0]["pool"]["reference_product"] = "forbidden"
    changed[0]["pool_sha256"] = identity(changed[0]["pool"])
    with pytest.raises(ValueError, match="unexpected"):
        initial_state(changed, cfg)


def test_interrupted_state_settlement_recovers_without_rebilling_or_lost_endpoints(tmp_path, monkeypatch):
    views, tools = setup(tmp_path)
    loop = FeedbackLoop(tmp_path / "loop", views, config(), tools)
    original = module.write_json

    def fail_state(path, value):
        if path.name == "state.json":
            raise OSError("injected state settlement failure")
        original(path, value)

    monkeypatch.setattr(module, "write_json", fail_state)
    with pytest.raises(OSError):
        loop.run()
    failure = json.loads(next((tmp_path / "loop").glob("step-*/failure-*.json")).read_text())
    assert failure["status"] == "state_settlement_failed"
    monkeypatch.setattr(module, "write_json", original)
    result = loop.run()
    assert result["attributed_reference_cost"]["reference_primitive_calls"] == 2
    assert next(iter(result["evidence"].values()))["feedback"]["endpoint_be_sha256"]
    assert result == loop.run()


@pytest.mark.parametrize("target", ["raw", "action", "outcome", "manifest"])
def test_corrupted_artifacts_or_decision_history_cannot_be_resumed(tmp_path, target):
    views, tools = setup(tmp_path)
    loop = FeedbackLoop(tmp_path / "loop", views, config(), tools)
    loop.run()
    if target == "raw":
        path = next(iter(tools.attempts.values())) / "raw/calls.json"
    elif target == "manifest":
        path = tmp_path / "loop/manifest.json"
    else:
        path = tmp_path / "loop/step-0000" / f"{target}.json"
    path.write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        loop.run()


def test_wrong_candidate_or_protocol_and_unavailable_actions_are_rejected(tmp_path):
    views, tools = setup(tmp_path)
    cfg = config()
    state = initial_state(views, cfg)
    action = choose(state, cfg)
    outcome = tools.execute(action)
    outcome["observation"]["candidate"]["protocol_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="differs from candidate"):
        reduce_outcome(state, action, outcome, cfg)
    with pytest.raises(ValueError, match="unavailable"):
        tools.execute({**action, "name": "expand_intermediate"})
    with pytest.raises(ValueError, match="catalogue"):
        initial_state(views, config(samplers={"invented": {}}))
