from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_kingfisher_gfn2_preflight.py"
_SPEC = importlib.util.spec_from_file_location("kingfisher_preflight", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _row(candidate_id: str = "candidate-1") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_record_id": "source-1",
        "symbols": ["H", "H"],
        "coordinates_angstrom": [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        "charge": 0,
        "multiplicity": 1,
        "initial_mode": [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    }


def test_manifest_requires_explicit_electronic_state_and_initial_mode(tmp_path):
    row = _row()
    row.pop("multiplicity")
    path = tmp_path / "candidates.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing.*multiplicity"):
        _MODULE.load_candidate_manifest(path)


def test_manifest_rejects_duplicate_candidate_ids(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in (_row(), _row())),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate candidate_id"):
        _MODULE.load_candidate_manifest(path)

