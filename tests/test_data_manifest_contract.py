from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_spice_openff_pilot import _source_collection_id


ROOT = Path(__file__).resolve().parents[1]
STABLE_COLLECTIONS = {
    "spice2_openff_dipeptides_v1.2",
    "spice2_openff_solvated_amino_acids_v1.1",
}


def _rows(path: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_spice_manifests_use_stable_source_collection_ids() -> None:
    expected_counts = {
        "data/manifests/spice2_openff_pilot_20260929.jsonl": 64,
        "data/manifests/spice2_openff_256_20260929.jsonl": 256,
    }
    for relative_path, expected_count in expected_counts.items():
        rows = _rows(relative_path)
        assert len(rows) == expected_count
        assert {row["source_collection"] for row in rows} <= STABLE_COLLECTIONS
        assert all(not Path(row["source_collection"]).is_absolute() for row in rows)

    assert _source_collection_id(
        "/home/runner/SPICE Dipeptides Single Points Dataset v1.2"
    ) == "spice2_openff_dipeptides_v1.2"
    assert _source_collection_id(
        r"C:\\datasets\\SPICE Solvated Amino Acids Single Points Dataset v1.1"
    ) == "spice2_openff_solvated_amino_acids_v1.1"


def test_public_manifest_marks_untracked_source_manifests_external_only() -> None:
    text = (ROOT / "configs/public_data_manifest.yaml").read_text(encoding="utf-8")
    blocks = text.split("\n  - id: ")[1:]
    by_id = {block.splitlines()[0].strip(): block for block in blocks}

    assert "manifest_status: tracked" in by_id["spice2_openff_v1.1_pilot"]
    assert "source_manifest: data/manifests/spice2_openff_256_20260929.jsonl" in by_id[
        "spice2_openff_v1.1_pilot"
    ]
    for dataset_id in (
        "units_lib_formula_oos",
        "uspto_15k_reactant_edits",
        "transition1x_preprocessed",
        "flower_v2",
        "kingfisher_ch2o",
        "rgd1_aggregate_features",
    ):
        assert "source_manifest: null" in by_id[dataset_id]
        assert "manifest_status: external_only" in by_id[dataset_id]


def test_manifest_readme_states_admitted_and_quarantined_counts() -> None:
    readme = (ROOT / "data/manifests/README.md").read_text(encoding="utf-8")
    assert "262 rows" in readme
    assert "six inherited source-audit rows" in readme
    assert "256 SPICE2 OpenFF configurations" in readme
    assert "48/8/8" in readme
