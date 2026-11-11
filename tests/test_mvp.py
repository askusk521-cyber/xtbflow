from __future__ import annotations

import json
from pathlib import Path

from xtbflow.mvp import run_mvp


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "mvp_fixture.yaml"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]


def test_mvp_mock_end_to_end_writes_auditable_bundle(tmp_path: Path) -> None:
    output = tmp_path / "mvp"
    manifest = run_mvp(CONFIG, output_dir=output)

    assert manifest["status"] == "completed"
    assert manifest["backend"] == "mock"
    assert manifest["evidence_level"] == "software_demo"
    assert manifest["counts"] == {"candidate_count": 5, "accepted": 3, "rejected": 2, "failure": 0}
    assert manifest["failure_reason_counts"] == {"QC_FORCE_THRESHOLD": 1, "QC_MIN_DISTANCE": 1}
    assert manifest["budget"]["totals"]["system_evaluations"] == 5
    assert manifest["budget"]["totals"]["calculator_calls"] == 4
    assert manifest["budget"]["totals"]["failures"] == 1

    for name in ("manifest.json", "candidates.jsonl", "ledger.json", "summary.md", "summary.png", "input.json"):
        assert (output / name).is_file()
    assert (output / "summary.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    rows = _rows(output)
    assert [row["candidate_index"] for row in rows] == list(range(5))
    required = {"status", "reason_code", "attempt", "wall_seconds", "cpu_hours", "gpu_hours"}
    assert all(required <= row.keys() for row in rows)
    assert rows[3]["status"] == "rejected"
    assert rows[3]["reason_code"] == "QC_FORCE_THRESHOLD"
    assert rows[4]["reason_code"] == "QC_MIN_DISTANCE"
    assert all(row["backend"] == "mock" and row["evidence_level"] == "software_demo" for row in rows)


def test_mvp_replays_qc_failure_and_candidate_order(tmp_path: Path) -> None:
    first = run_mvp(CONFIG, output_dir=tmp_path / "first")
    second = run_mvp(CONFIG, output_dir=tmp_path / "second")

    assert first["replay"]["deterministic_summary_sha256"] == second["replay"]["deterministic_summary_sha256"]
    assert first["replay"]["deterministic_summary"] == second["replay"]["deterministic_summary"]
    first_rows = _rows(tmp_path / "first")
    second_rows = _rows(tmp_path / "second")
    assert [
        (row["candidate_fingerprint"], row["status"], row["reason_code"])
        for row in first_rows
    ] == [
        (row["candidate_fingerprint"], row["status"], row["reason_code"])
        for row in second_rows
    ]
    assert any(row["reason_code"] == "QC_MIN_DISTANCE" for row in first_rows)


def test_xtb_request_fails_closed_without_mock_fallback(tmp_path: Path, monkeypatch) -> None:
    # Ordinary CPU validation must not launch real QC merely because n2 has xTB.
    import xtbflow.mvp as mvp
    monkeypatch.setattr(mvp, "XTBOracleAdapter", None)
    manifest = run_mvp(CONFIG, output_dir=tmp_path / "xtb", backend_override="xtb")

    assert manifest["backend"] == "xtb"
    assert manifest["status"] == "blocked_backend"
    assert all(row["backend"] == "xtb" for row in _rows(tmp_path / "xtb"))
    if manifest["status"] == "blocked_backend":
        assert manifest["failure_reason_counts"].get("BLOCKED_BACKEND", 0) == 4
        assert all(row["reason_code"] == "BLOCKED_BACKEND" for row in _rows(tmp_path / "xtb") if row["status"] == "failure")
