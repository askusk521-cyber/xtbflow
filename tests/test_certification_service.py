"""Software contract fixtures, not chemical validation or reference data."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from xtbflow.runtime import BudgetExceeded, RunLedger, StageBudget
from xtbflow.validation import certification as service_module
from xtbflow.validation.certification import (
    CallMeter, CertificationCandidate, CertificationService, feedback,
)
from xtbflow.validation import reference_protocol as numerical


def test_storage_monitor_tolerates_atomic_journal_replacement(tmp_path):
    stable = tmp_path / "calls.json"
    stable.write_bytes(b"retained")
    symlink = tmp_path / "link"
    symlink.symlink_to(stable)

    class DisappearingEntry:
        def is_file(self):
            return True  # Entry existed when the directory scan reached it.

        def is_symlink(self):
            return False

        def stat(self):
            raise FileNotFoundError("temporary journal was atomically replaced")

        lstat = stat

    folder = SimpleNamespace(rglob=lambda pattern: iter([DisappearingEntry(), stable, symlink]))
    assert service_module._directory_bytes(folder) == len(b"retained")


def test_storage_monitor_does_not_hide_permission_errors(tmp_path, monkeypatch):
    (tmp_path / "calls.json").write_bytes(b"retained")

    def denied(path):
        raise PermissionError("unreadable metadata")

    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(PermissionError):
        service_module._directory_bytes(tmp_path)


def candidate(**changes):
    base = CertificationCandidate(
        candidate_id="fixture", parent_id="synthetic-parent", atomic_numbers=(1, 1),
        coordinates_angstrom=((0., 0., 0.), (0., 0., .8)), atom_ids=("a", "b"),
        reactant_be=((0, 1), (1, 0)), planned_product_be=((1, 0), (0, 1)),
        reactant_automorphisms=((0, 1), (1, 0)), charge=0, multiplicity=1,
        environment={"kind": "gas", "periodic": False},
        provenance={"input_sha256": "a" * 64, "atom_mapping_evidence": "synthetic fixture IDs",
                    "electronic_state_evidence": "synthetic fixture declaration"})
    return replace(base, **changes)


def evidence(*, alternative=False):
    gradient = [[0., 0., 0.], [0., 0., 0.]]
    return {"ts": {"gradient_hartree_bohr": gradient, "frequencies_cm1": [-400.],
                   "reaction_mode": [[1., 0., 0.], [-1., 0., 0.]]},
            "irc": {"status": "IRC_COMPLETE"},
            "endpoints": [{"status": "MINIMUM_PASS", "gradient_hartree_bohr": gradient,
                           "frequencies_cm1": [200.]} for _ in range(2)],
            "identity": {"status": "STRICT_EVENT_MISMATCH" if alternative else "STRICT_JOINT_GRAPH_VALID",
                         "actual_event_id": "observed", "predicted_event_id": "planned" if alternative else "observed",
                         "reactant_end": 0, "product_be": [[1, 0], [0, 1]],
                         "endpoint_be": [[[0, 1], [1, 0]], [[1, 0], [0, 1]]]}}


class FixtureBackend:
    def __init__(self, *, mode="success", version="fixture-v1"):
        self.mode, self.version = mode, version

    @property
    def identity(self):
        return {"engine": "fixture", "device": "cpu", "version": self.version, "mode": self.mode}

    def run(self, candidate, meter, work):
        for _ in range(2):
            meter.before("fixture", "scf")
            meter.record("fixture", "scf", .001)
        raw = evidence(alternative=self.mode == "alternative")
        meter.checkpoint(raw)
        if self.mode == "failure":
            meter.before("failed-kernel", "gradient")
            raise RuntimeError("injected SCF failure")
        if self.mode == "timeout":
            time.sleep(30)
        if self.mode == "oversized":
            (work / "oversized").write_bytes(b"x" * 100000)
        return raw


def budget(**changes):
    values = dict(phase="certification-fixture", max_calculator_calls=30, max_gpu_hours=1,
                  max_cpu_core_hours=1, max_storage_gb=1, max_system_evaluations=10,
                  max_retries_per_job=1, max_wall_minutes_per_job=1)
    values.update(changes)
    return StageBudget(**values)


def run(service, item=None, backend=None, **changes):
    limits = dict(max_calls=5, max_wall_seconds=10, cpu_cores=1, max_raw_bytes=1024**2)
    limits.update(changes)
    return service.run(item or candidate(), backend or FixtureBackend(), **limits)


@pytest.mark.parametrize("changes", [
    {"charge": None}, {"multiplicity": 3}, {"environment": {"kind": "unknown"}},
    {"atom_ids": ("a", "a")}, {"provenance": {}},
    {"coordinates_angstrom": ((float("nan"), 0, 0), (0, 0, 1))},
    {"planned_product_be": ((0, 0), (0, 0))},
    {"reactant_automorphisms": ((0, 0),)},
])
def test_candidate_fails_closed(changes):
    with pytest.raises(ValueError):
        candidate(**changes)


def test_layers_preserve_alternative_and_reject_incomplete_evidence():
    raw = evidence(alternative=True)
    info = feedback(raw)
    assert info["evidence_status"] == "valid_alternative"
    assert info["observed_event_id"] == "observed" != info["planned_event_id"]
    assert info["planned_event_realized"] is False
    for mutate in (
        lambda r: r["ts"].update(gradient_hartree_bohr=[[.1, 0, 0], [0, 0, 0]]),
        lambda r: r["ts"].update(frequencies_cm1=[-400, -.01]),
        lambda r: r["ts"].update(reaction_mode=[]),
        lambda r: r["endpoints"][0].update(frequencies_cm1=[-.01]),
        lambda r: r["endpoints"][0].update(status="MINIMUM_GRAY_ZONE"),
        lambda r: r["irc"].update(status="IRC_LIMIT"),
        lambda r: r["identity"].update(status="EVALUATION_UNRESOLVED"),
    ):
        changed = deepcopy(raw)
        mutate(changed)
        assert feedback(changed)["evidence_status"] == "attempted_unresolved"
    raw["identity"] = {"status": "STRICT_EVENT_MISMATCH", "reason": "both_ends_reactant"}
    assert feedback(raw)["evidence_status"] == "same_endpoint"
    raw["identity"] = {"status": "IRC_REACTANT_MISMATCH"}
    assert feedback(raw)["evidence_status"] == "reactant_mismatch"


def test_cache_and_channel_identity_survive_process_restart(tmp_path):
    service = CertificationService(tmp_path, budget())
    first = run(service)
    assert first["new_calculator_calls"] == 2 and first["new_channel"]
    cached = run(CertificationService(tmp_path, budget()), candidate(candidate_id="renamed"))
    assert cached["cache_hit"] and cached["new_calculator_calls"] == 0 and not cached["new_channel"]
    changed = run(service, candidate(coordinates_angstrom=((0., 0., 0.), (0., 0., .9))))
    assert not changed["cache_hit"] and not changed["new_channel"]
    version = run(service, backend=FixtureBackend(version="fixture-v2"))
    assert not version["cache_hit"] and version["new_channel"]
    assert RunLedger.load(service.ledger_path).summary()["totals"]["calculator_calls"] == 6


def test_call_budget_and_failed_kernel_are_charged_before_invocation(tmp_path):
    service = CertificationService(tmp_path, budget())
    limited = run(service, max_calls=1)
    assert limited["record"]["execution_status"] == "budget_exhausted"
    assert limited["new_calculator_calls"] == 1
    failed = run(service, backend=FixtureBackend(mode="failure"))
    assert failed["new_calculator_calls"] == 3
    assert failed["record"]["execution_status"] == "execution_failed"
    assert failed["record"]["feedback"]["evidence_status"] == "attempted_unresolved"
    assert failed["record"]["feedback"]["first_order_saddle"]
    assert not failed["new_channel"]


def test_explicit_retry_is_bounded_and_not_a_cache_side_effect(tmp_path):
    service = CertificationService(tmp_path, budget())
    backend = FixtureBackend(mode="failure")
    run(service, backend=backend)
    assert run(service, backend=backend)["new_calculator_calls"] == 0
    assert run(service, backend=backend, retry=True)["new_calculator_calls"] == 3
    with pytest.raises(BudgetExceeded, match="retry"):
        run(service, backend=backend, retry=True)


def test_corrupt_artifact_fails_closed_without_recomputation(tmp_path):
    service = CertificationService(tmp_path, budget())
    run(service)
    path = next(tmp_path.glob("*/attempt-*/raw/calls.json"))
    path.write_text("tampered")
    with pytest.raises(ValueError, match="artifact mismatch"):
        run(service)
    assert RunLedger.load(service.ledger_path).summary()["totals"]["calculator_calls"] == 2


def test_report_write_failure_recovers_consumption_without_rerun(tmp_path, monkeypatch):
    service = CertificationService(tmp_path, budget())
    real_write = service_module.write_json
    def broken(path, data):
        if path.name == "result.json":
            raise OSError("injected disk error")
        real_write(path, data)
    monkeypatch.setattr(service_module, "write_json", broken)
    with pytest.raises(OSError):
        run(service)
    pending = RunLedger.load(service.ledger_path)
    assert sum(v["consumed_calls"] for v in pending.reservations.values()) == 2
    monkeypatch.setattr(service_module, "write_json", real_write)
    recovered = run(service)
    assert recovered["cache_hit"] and recovered["new_calculator_calls"] == 0
    assert recovered["record"]["execution_status"] == "execution_interrupted"
    assert recovered["record"]["resources"]["basis"] == "conservative_upper_bound"
    assert recovered["record"]["evidence"]["failure"] == "report_write_failed"
    assert RunLedger.load(service.ledger_path).summary()["totals"]["calculator_calls"] == 2


def test_settlement_retry_does_not_double_bill(tmp_path, monkeypatch):
    service = CertificationService(tmp_path, budget())
    original = RunLedger.append
    monkeypatch.setattr(RunLedger, "append", lambda *a, **k: (_ for _ in ()).throw(OSError("settlement failed")))
    with pytest.raises(OSError):
        run(service)
    assert json.loads(next(tmp_path.glob("*/attempt-*/settlement.json")).read_text())["status"] == "settlement_failed"
    monkeypatch.setattr(RunLedger, "append", original)
    recovered = run(service)
    assert recovered["cache_hit"] and recovered["new_calculator_calls"] == 0
    summary = RunLedger.load(service.ledger_path).summary()["totals"]
    assert summary["calculator_calls"] == 2 and summary["attempts"] == 1


def test_wall_kills_worker_and_keeps_started_calls(tmp_path):
    service = CertificationService(tmp_path, budget())
    started = time.monotonic()
    result = run(service, backend=FixtureBackend(mode="timeout"), max_wall_seconds=3)
    assert time.monotonic() - started < 8
    assert result["record"]["execution_status"] == "execution_timeout"
    assert result["new_calculator_calls"] == 2


def test_resource_envelope_rejected_before_any_calculation(tmp_path):
    service = CertificationService(tmp_path, budget(max_cpu_core_hours=0))
    result = run(service)
    assert result["record"]["execution_status"] == "not_started_budget_exhausted"
    assert result["new_calculator_calls"] == 0
    assert result["record"]["ledger_event"]["system_evaluations"] == 0


def test_process_lock_blocks_overlapping_service_calls(tmp_path):
    service = CertificationService(tmp_path, budget())
    with service._locked(), pytest.raises(BlockingIOError):
        run(CertificationService(tmp_path, budget()))


def test_numerical_kernel_cannot_run_beyond_durable_token(tmp_path, monkeypatch):
    class MF:
        converged = True
        mol = type("Mol", (), {"nao": 1})()
        def kernel(self, **kw):
            calls.append("scf")
            return 0.
        def nuc_grad_method(self):
            return type("Gradient", (), {"kernel": lambda self: calls.append("gradient") or np.zeros((2, 3))})()
        def Hessian(self):
            raise AssertionError("must stop before constructing Hessian kernel")
    calls = []
    monkeypatch.setattr(numerical, "build_mol", lambda *args: MF.mol)
    monkeypatch.setattr(numerical, "make_scf", lambda *args: MF())
    ledger = RunLedger(budget())
    token = ledger.issue_calculator_token("kernel", 2, persist_path=tmp_path / "ledger.json")
    meter = CallMeter(token, tmp_path, 10)
    with pytest.raises(BudgetExceeded):
        numerical.hessian([1, 1], [[0, 0, 0], [0, 0, 1]], numerical.Method(), meter, "test")
    assert calls == ["scf", "gradient"] and token.consumed_calls == 2


def test_event_identity_uses_observed_endpoint_not_prediction(monkeypatch):
    reactant = np.array([[0, 1], [1, 0]])
    product = np.array([[1, 0], [0, 1]])
    outcomes = iter((reactant, product))
    monkeypatch.setattr(numerical, "perceive_be", lambda *args: next(outcomes))
    result = numerical.endpoint_identity([1, 1], reactant, [[0, 1], [1, 0]], reactant, [], [])
    assert result["status"] == "STRICT_EVENT_MISMATCH"
    assert result["actual_event_id"] != result["predicted_event_id"]
    assert result["product_be"] == product.tolist()
