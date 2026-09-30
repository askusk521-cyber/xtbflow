from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path("scripts/audit_dft_da_provenance.py")
SPEC = importlib.util.spec_from_file_location("audit_dft_da_provenance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


REACTION = "[C:1]-[C:2].[H:3]-[H:4]>>[C:1]-[C:2].[H:3]-[H:4]"
ROW = {
    "ID": "ID1",
    "R": "R0",
    "R_dir": "reaction_R0",
    "smiles": REACTION,
    "DG_act": "35.0",
    "DrG": "-22.0",
    "DG_act_xtb": "21.0",
    "DrG_xtb": "-44.0",
}


def _xyz(coordinates: list[tuple[float, float, float]], *, comment: str = "") -> str:
    rows = [str(len(coordinates)), comment]
    rows.extend(f"{symbol} {x:.8f} {y:.8f} {z:.8f}" for symbol, (x, y, z) in coordinates)
    return "\n".join(rows) + "\n"


MAIN_COORDINATES = [
    ("C", (0.0, 0.0, 0.0)),
    ("C", (1.4, 0.0, 0.0)),
    ("H", (0.0, 3.0, 0.0)),
    ("H", (1.4, 3.0, 0.0)),
]
REFERENCE_COORDINATES = [
    ("C", (0.0, 0.0, 0.0)),
    ("C", (1.5, 0.0, 0.0)),
    ("H", (0.0, 3.0, 0.0)),
    ("H", (1.5, 3.0, 0.0)),
]


def _write_cache(tmp_path: Path, *, include_logs: bool = True, ts_symbols: list[str] | None = None):
    cache = tmp_path / "main-cache"
    root = cache / "extracted" / "DATASET_DA_F" / "reaction_R0"
    root.mkdir(parents=True)
    (cache / "DATASET_DA_F.tar.gz").write_bytes(b"main-archive")
    csv_path = cache / "extracted" / "DATASET_DA_F" / "dataset_xtb_final.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW))
        writer.writeheader()
        writer.writerow(ROW)
    for role in ("reactant", "product", "ts", "monomer_1", "monomer_2"):
        symbols = ts_symbols if role == "ts" and ts_symbols is not None else [item[0] for item in MAIN_COORDINATES]
        coordinates = [(symbol, coords) for symbol, (_, coords) in zip(symbols, MAIN_COORDINATES)]
        root.joinpath(f"R0_{role}.xyz").write_text(_xyz(coordinates, comment="test" if role == "reactant" else ""), encoding="utf-8")
    if include_logs:
        log = """#opt freq external=\"xtb_external.py\"\nCharge =  0 Multiplicity = 1\nxtb mol.xyz --chrg 0 --uhf 0 --gfn 2 --grad\nNormal termination of Gaussian 16\n"""
        (root / "monomer_1.log").write_text(log, encoding="utf-8")
        (root / "monomer_2.log").write_text(log, encoding="utf-8")
    return cache


def _write_reference(tmp_path: Path) -> Path:
    cache = tmp_path / "dft-cache"
    root = cache / "extracted" / "DFT_DA_DATASET" / "reaction_R0"
    root.mkdir(parents=True)
    (cache / "DFT_DA_DATASET.tar.gz").write_bytes(b"dft-archive")
    csv_path = cache / "extracted" / "DFT_DA_DATASET" / "da_dataset_dft_new.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "R", "R_dir", "reaction", "G(TS)", "DrG"])
        writer.writeheader()
        writer.writerow({"ID": "ID1", "R": "R0", "R_dir": "reaction_R0", "reaction": REACTION, "G(TS)": "35.0", "DrG": "-22.0"})
    for role in ("monomer_1", "monomer_2", "product", "ts"):
        suffix = "product_dft" if role == "product" else "ts_dft" if role == "ts" else role
        coordinates = [(symbol, coords) for symbol, (_, coords) in zip([item[0] for item in REFERENCE_COORDINATES], REFERENCE_COORDINATES)]
        root.joinpath(f"R0_{suffix}.xyz").write_text(_xyz(coordinates), encoding="utf-8")
    return cache


def _patch_archive_constants(monkeypatch: pytest.MonkeyPatch, *caches: Path) -> None:
    for cache, archive_name, size_name, md5_name, sha_name in (
        (caches[0], "DATASET_DA_F.tar.gz", "SOURCE_ARCHIVE_SIZE_BYTES", "SOURCE_ARCHIVE_MD5", "SOURCE_ARCHIVE_SHA256"),
        (caches[1], "DFT_DA_DATASET.tar.gz", "DFT_REFERENCE_ARCHIVE_SIZE_BYTES", "DFT_REFERENCE_ARCHIVE_MD5", "DFT_REFERENCE_ARCHIVE_SHA256"),
    ):
        raw = (cache / archive_name).read_bytes()
        monkeypatch.setattr(MODULE, size_name, len(raw))
        monkeypatch.setattr(MODULE, md5_name, hashlib.md5(raw).hexdigest())
        monkeypatch.setattr(MODULE, sha_name, hashlib.sha256(raw).hexdigest())


def test_audit_separates_energy_level_from_geometry_level(tmp_path, monkeypatch):
    main_cache = _write_cache(tmp_path)
    dft_cache = _write_reference(tmp_path)
    _patch_archive_constants(monkeypatch, main_cache, dft_cache)

    report = MODULE.build_report(main_cache, dft_cache_root=dft_cache, repository_base_commit="abc123", audit_date="2026-09-30")

    assert report["source"]["archive"]["matches_expected"] is True
    assert len(report["source"]["geometry_protocol_sha256"]) == 64
    assert report["summary"]["rows_with_all_xyz"] == 1
    assert report["summary"]["rows_with_complete_gfn2_log_evidence"] == 1
    assert report["summary"]["rows_with_neutral_singlet_state"] == 1
    assert report["energy_audit"]["dft_energy_claim_status"] == "cross_archive_values_match"
    row = report["rows"][0]
    assert row["geometry_method_status"] == "source_declared_xTB_log_supported_for_monomers"
    assert row["ts_mapping"]["status"] == "element_order_only"
    assert row["dft_reference_geometry_status"] == "all_present_and_distinct"
    assert report["scientific_claim_allowed"] is False
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)


def test_audit_flags_ts_order_mismatch_and_missing_log(tmp_path, monkeypatch):
    main_cache = _write_cache(tmp_path, include_logs=False, ts_symbols=["C", "C", "H", "C"])
    raw = (main_cache / "DATASET_DA_F.tar.gz").read_bytes()
    monkeypatch.setattr(MODULE, "SOURCE_ARCHIVE_SIZE_BYTES", len(raw))
    monkeypatch.setattr(MODULE, "SOURCE_ARCHIVE_MD5", hashlib.md5(raw).hexdigest())
    monkeypatch.setattr(MODULE, "SOURCE_ARCHIVE_SHA256", hashlib.sha256(raw).hexdigest())

    report = MODULE.build_report(main_cache)

    assert report["hard_failure_count"] == 1
    assert report["hard_failure_counts"] == {"ts_atom_order_mismatch": 1}
    row = report["rows"][0]
    assert row["monomer_geometry_method_evidence"] == "partial_or_missing"
    assert row["mapping_status"] == "unresolved"


def test_parser_rejects_duplicate_or_unmapped_atoms():
    with pytest.raises(MODULE.AuditError, match="atom maps"):
        MODULE.parse_reaction_smiles("[C:1]-[C:1]>>[C:1]-[C:1]")
    with pytest.raises(MODULE.AuditError, match="empty middle"):
        MODULE.parse_reaction_smiles("[C:1]>foo>[C:1]")


def test_manifest_claims_are_flagged_without_becoming_file_evidence(tmp_path, monkeypatch):
    main_cache = _write_cache(tmp_path)
    raw = (main_cache / "DATASET_DA_F.tar.gz").read_bytes()
    monkeypatch.setattr(MODULE, "SOURCE_ARCHIVE_SIZE_BYTES", len(raw))
    monkeypatch.setattr(MODULE, "SOURCE_ARCHIVE_MD5", hashlib.md5(raw).hexdigest())
    monkeypatch.setattr(MODULE, "SOURCE_ARCHIVE_SHA256", hashlib.sha256(raw).hexdigest())
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "source_record_id": "R0",
                "reference_protocol": {"method": "M06-2X/def2-TZVP"},
                "ts_geometry": {"evidence": "derived_under_contract"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = MODULE.build_report(main_cache, manifest=manifest)

    reasons = {flag["reason"] for flag in report["manifest_audit"]["flags"]}
    assert reasons == {
        "manifest_reference_protocol_can_be_read_as_DFT_geometry",
        "TS_mapping_is_contract_derived_without_source_sidecar",
    }
    assert report["provenance_gate"]["dft_geometry_claim"] == "quarantine"
