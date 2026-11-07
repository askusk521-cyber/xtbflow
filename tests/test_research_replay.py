"""Provenance failures must not turn into successful chemical replays."""
import json
from pathlib import Path
import subprocess
import time

import pytest

from xtbflow.governance.research_replay import (
    ReplayError, audit_inputs, candidate_links, check_source, content_id,
    digest, logical_path,
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_input_hash_failure_retains_completed_io_and_relocated_identity(tmp_path):
    (tmp_path / "a").write_bytes(b"original")
    (tmp_path / "b").write_bytes(b"changed")
    rows = []
    manifest = {"files": [
        {"path": "/private/xtbflow-runs/a", "sha256": digest(tmp_path / "a")},
        {"path": "/private/xtbflow-runs/b", "sha256": "0" * 64},
    ]}
    with pytest.raises(ReplayError, match="raw_input_hash_mismatch"):
        audit_inputs(manifest, tmp_path, rows, max_bytes=100, deadline=time.monotonic() + 10)
    assert [r["status"] for r in rows] == ["verified", "hash_mismatch"]
    assert sum(r["bytes"] for r in rows) == 15
    assert "/private" not in json.dumps(rows)


@pytest.mark.parametrize("path", ["/elsewhere/a", "/xtbflow-runs/../secret", "/xtbflow-runs", "/xtbflow-runs/x/xtbflow-runs/a"])
def test_archived_locations_cannot_escape_declared_root(path):
    with pytest.raises(ReplayError):
        logical_path(path)


def test_symlink_escape_and_missing_input_are_unavailable(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    (tmp_path / "secret").write_bytes(b"private")
    (root / "escape").symlink_to(tmp_path / "secret")
    for name in ("escape", "missing"):
        rows = []
        with pytest.raises(ReplayError, match="missing_or_outside_run_root"):
            audit_inputs({"files": [{"path": "/xtbflow-runs/" + name, "sha256": "0" * 64}]},
                         root, rows, max_bytes=100, deadline=time.monotonic() + 10)
        assert rows[0]["status"] == "unavailable"


def test_budget_exhaustion_does_not_hash_unaffordable_file(tmp_path):
    (tmp_path / "a").write_bytes(b"1234")
    rows = []
    with pytest.raises(ReplayError, match="budget_exhausted"):
        audit_inputs({"files": [{"path": "/xtbflow-runs/a", "sha256": digest(tmp_path / "a")}]},
                     tmp_path, rows, max_bytes=3, deadline=time.monotonic() + 10)
    assert rows == [{"asset": "a", "expected_sha256": digest(tmp_path / "a"), "status": "budget_exhausted"}]


def test_source_requires_exact_clean_head(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init", "-q")
    (tmp_path / "source.py").write_text("# pinned\n")
    git("add", "source.py")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
    asset = {"commit": git("rev-parse", "HEAD"), "files": [{"path": "source.py", "sha256": digest(tmp_path / "source.py")}]}
    assert check_source(tmp_path, asset)["dirty"] is False
    with pytest.raises(ReplayError, match="revision_mismatch"):
        check_source(tmp_path, {**asset, "commit": "0" * 40})
    (tmp_path / "source.py").write_text("# changed\n")
    with pytest.raises(ReplayError, match="source_dirty"):
        check_source(tmp_path, asset)


def candidate_fixture(root):
    item = dict(kind="candidate", item_id="one", candidate_id="parent/B0/p0", parent_id="parent", query_id="query",
                z=[1, 1], x_start=[[0, 0, 0], [0, 0, 1]], charge=0, multiplicity=1, coordinate_unit="Angstrom",
                b_r=[[0, 1], [1, 0]], b_predicted=[[1, 0], [0, 1]], perms=[[0, 1]])
    work = root / "o2-run/candidate_one"
    dump(root / "o2-plan.json", {"items": [item]})
    dump(work / "manifest.json", dict(item=item, slurm_job_id="17", protocol_version="fixture", source_commit="a" * 40))
    dump(work / "verdict.json", dict(identity=dict(predicted_event_id="planned", actual_event_id="observed"),
         ts=dict(x_ts=item["x_start"], status="TS_OPTFREQ_PASS"), irc=dict(status="IRC_COMPLETE"),
         endpoints=[dict(status="MINIMUM_PASS", x=item["x_start"])], method=dict(charge=0, spin=0)))
    cost = dict(item_id="one", parent_id="parent", kind="candidate", status="STRICT_EVENT_MISMATCH",
                calls=dict(scf=10, gradient=8, hessian=2), slurm_job_id="17", slurm_runtime_s=20, wall_s=18)
    return {"items": [cost]}, work


def test_linkage_preserves_distinct_events_and_cost_without_promoting_status(tmp_path):
    result, _ = candidate_fixture(tmp_path)
    link = candidate_links(tmp_path, result)[0]
    assert link["planned_event_id"] == "planned"
    assert link["observed_event_id"] == "observed"
    assert link["archived_status"] == "STRICT_EVENT_MISMATCH"
    assert link["evidence_kind"] == "archival_linkage_only"
    assert link["historical_cost"]["calls"] == result["items"][0]["calls"]
    assert "x_start" not in link and len(link["input_sha256"]) == 64


def test_linkage_does_not_guess_observed_event_after_chain_exception(tmp_path):
    result, work = candidate_fixture(tmp_path)
    (work / "verdict.json").unlink()
    result["items"][0]["status"] = "CHAIN_EXCEPTION:EngineError:SCF_LIMIT"
    link = candidate_links(tmp_path, result)[0]
    assert link["observed_event_id"] is None and link["planned_event_id"] is None
    assert link["historical_cost"]["calls"]["scf"] == 10
    assert link["archived_status"].startswith("CHAIN_EXCEPTION")


def test_linkage_rejects_misbound_plan_and_cost(tmp_path):
    result, work = candidate_fixture(tmp_path)
    result["items"][0]["slurm_job_id"] = "18"
    with pytest.raises(ReplayError, match="job_identity_mismatch"):
        candidate_links(tmp_path, result)
    result["items"][0]["slurm_job_id"] = "17"
    manifest = json.loads((work / "manifest.json").read_text())
    manifest["item"]["charge"] = 1
    dump(work / "manifest.json", manifest)
    with pytest.raises(ReplayError, match="manifest_identity_mismatch"):
        candidate_links(tmp_path, result)


def test_content_identity_is_order_independent_and_rejects_nan():
    assert content_id({"a": 1, "b": 2}) == content_id({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        content_id({"x": float("nan")})
