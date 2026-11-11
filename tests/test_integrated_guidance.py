from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest
import torch

from xtbflow.sampling.integrated_guidance import guided_step, require_settled

from xtbflow.sampling import guidance_fixture as fixture

CONFIG = Path(__file__).parents[1]/"configs/research/integrated_entry_20261111.json"


def test_installed_guide_zero_and_actual_next_forward_order():
    query, _, initial = fixture.fixture_query()
    calls = []
    def guide(injection, query):
        calls.append(injection)
        assert torch.equal(injection.observed.t_x, initial.t_x)
        assert injection.base.t_x.item() == .25
        return -.01*injection.predicted_x
    plain_model, guided_model = fixture.SyntheticCoupledModel(), fixture.SyntheticCoupledModel()
    plain = guided_step(plain_model, query, initial, next_t_b=.5, next_t_x=.25)
    zero = guided_step(guided_model, query, initial, next_t_b=.5, next_t_x=.25, guide=guide, alpha=0)
    assert not calls and torch.equal(plain.x, zero.x) and torch.equal(plain.b, zero.b)
    changed = guided_step(guided_model, query, initial, next_t_b=.5, next_t_x=.25, guide=guide, alpha=1)
    assert len(calls) == 1 and torch.equal(plain.b, changed.b) and not torch.equal(plain.x, changed.x)
    end_plain = guided_step(plain_model, query, plain, next_t_b=1, next_t_x=1)
    end_changed = guided_step(guided_model, query, changed, next_t_b=1, next_t_x=1)
    assert torch.equal(guided_model.calls[-1]["x_cur"], changed.x)
    assert torch.equal(guided_model.calls[-1]["b_cur"], changed.b)
    assert not torch.equal(end_plain.b, end_changed.b)
    assert end_changed.b.sum() == pytest.approx(initial.b.sum())
    def broken(*args):
        raise RuntimeError("guide_reached")
    with pytest.raises(RuntimeError, match="guide_reached"):
        guided_step(guided_model, query, initial, next_t_b=.5, next_t_x=.25, guide=broken, alpha=1)


@pytest.mark.parametrize("change", [dict(status="budget_exhausted"), dict(termination="unknown"),
                                     dict(settlement="unsettled"), dict(reservations={"active": 1})])
def test_certification_boundary_blocks_unsettled_or_stopped_guidance(change):
    upstream = dict(status="completed", termination="confirmed", settlement="settled", reservations={})
    with pytest.raises(ValueError, match="uncertified_upstream"):
        require_settled(upstream | change)


def test_integrated_candidate_reference_ledger_and_zero_cost_cache(tmp_path):
    out = tmp_path/"run"
    first = fixture.run_fixture(out, "fake", CONFIG)
    second = fixture.run_fixture(out, "cached", CONFIG)
    assert first["candidate_sha256"] == second["candidate_sha256"]
    assert first["result"]["new_calculator_calls"] == 1
    assert second["result"]["cache_hit"] and second["result"]["new_calculator_calls"] == 0
    assert first["ledger"]["totals"] == second["ledger"]["totals"]
    assert first["trained_forwards"] == first["new_scientific_calculations"] == 0


def test_incomplete_replay_preserves_generated_evidence(tmp_path):
    out = tmp_path/"run"
    fixture.run_fixture(out, "fake", CONFIG)
    before = (out/"generated.json").read_bytes()
    (out/"generated.json").write_bytes(before.replace(b'"completed"', b'"incomplete"'))
    altered = (out/"generated.json").read_bytes()
    with pytest.raises(ValueError, match="identity_mismatch"):
        fixture.run_fixture(out, "cached", CONFIG)
    assert (out/"generated.json").read_bytes() == altered
