#!/usr/bin/env python3
"""Audit file-level provenance for the public Diels--Alder archive.

The v5 ``DATASET_DA_F`` archive deliberately contains two kinds of labels:
DFT energies and GFN2-xTB reaction-profile geometries.  The XYZ files do not
carry a method tag, so a CSV column name or a DFT article description must not
silently promote every geometry to DFT.  This audit records the evidence that
is actually present for every source row and, when supplied, compares each
file with the separately published M06-2X/def2-TZVP geometry archive.

The command is intentionally standard-library only.  It can therefore run on
the n2 host before the scientific Python environment is loaded.  Raw cache
paths are never written to the JSON report; locators are archive-relative.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


SOURCE_URL = (
    "https://figshare.com/articles/dataset/"
    "Diels-Alder_reaction_space_for_self-healing_polymer/29118509"
)
SOURCE_DOI = "10.6084/m9.figshare.29118509.v5"
SOURCE_ARCHIVE_NAME = "DATASET_DA_F.tar.gz"
SOURCE_ARCHIVE_SIZE_BYTES = 358677850
SOURCE_ARCHIVE_MD5 = "d37dbd26be16b63038537f8f04868fad"
SOURCE_ARCHIVE_SHA256 = "4df57289edecdfc518da6aab147b704745a9f42901893671aab05a3f72bd73ed"
SOURCE_GEOMETRY_PROTOCOL = {
    "workflow": "TS-tools reaction-profile archive",
    "geometry_level": "GFN2-xTB (source workflow declaration)",
    "source_paper_doi": "10.1039/d5dd00340g",
    "file_level_method_metadata": "absent_in_main_archive",
}

# The independently deposited reference geometry archive is small and public.
# It is optional at runtime because the main audit must also be useful when
# only the project-approved full archive is present on n2.
DFT_REFERENCE_URL = (
    "https://figshare.com/articles/dataset/"
    "DFT_dataset_M06-2X_def2-TZVP_/28768301"
)
DFT_REFERENCE_DOI = "10.6084/m9.figshare.28768301.v1"
DFT_REFERENCE_ARCHIVE_NAME = "DFT_DA_DATASET.tar.gz"
DFT_REFERENCE_ARCHIVE_SIZE_BYTES = 3357107
DFT_REFERENCE_ARCHIVE_MD5 = "244bb9316918110afd18178e7d12e5e8"
DFT_REFERENCE_ARCHIVE_SHA256 = (
    "8644db068a769f2b9e9054594b6299f57eaa154172f40853b93177d4d418cedd"
)

SOURCE_CSV_RELATIVE = Path("extracted/DATASET_DA_F/dataset_xtb_final.csv")
SOURCE_EXTRACTED_RELATIVE = Path("extracted/DATASET_DA_F")
DFT_CSV_RELATIVE = Path("extracted/DFT_DA_DATASET/da_dataset_dft_new.csv")
DFT_EXTRACTED_RELATIVE = Path("extracted/DFT_DA_DATASET")

# Atom-map labels can follow stereochemical markers (for example ``[C@:1]``),
# so the map separator is not necessarily the first character after the
# element symbol.
_ATOM_RE = re.compile(r"\[([A-Za-z][a-z]?)[^\]:]*:(\d+)[^\]]*\]")
_LOG_STATE_RE = re.compile(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)")
_XTB_STATE_RE = re.compile(r"--chrg\s+(-?\d+)\s+--uhf\s+(\d+)")
_XTB_COMMAND_RE = re.compile(r"xtb\s+mol\.xyz[^\n]*--gfn\s*2", re.IGNORECASE)
_EXTERNAL_XTB_RE = re.compile(r"external=.*xtb_external(?:_script)?", re.IGNORECASE)
_METHOD_RE = re.compile(
    r"(?:M06[- ]?2X|def2[- ]?TZVP|GFN2[- ]?xTB|--gfn\s+2|xtb_external)",
    re.IGNORECASE,
)
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?$")


class AuditError(ValueError):
    """Raised for an invalid audit invocation or source table."""


@dataclass(frozen=True)
class XYZ:
    relative_path: str
    atom_count: int
    symbols: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    comment: str
    sha256: str
    size_bytes: int


def hash_file(path: Path, *, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    """Hash a JSON protocol without depending on project package imports."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative(path: Path, root: Path) -> str:
    """Return a portable archive-relative locator, never an absolute path."""

    return path.relative_to(root).as_posix()


def _parse_mapped_side(text: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Read mapped atom order from one SMILES side without constructing a graph."""

    atoms = tuple(_ATOM_RE.findall(text))
    if not atoms:
        raise AuditError("mapped SMILES has no bracketed atoms")
    # Lower-case aromatic SMILES symbols still denote the same element as the
    # upper-case XYZ symbol (``c`` -> ``C``, ``o`` -> ``O``).  Preserve the
    # atom order while normalising only this representation detail.
    symbols = tuple(atom[0].capitalize() for atom in atoms)
    maps = tuple(int(atom[1]) for atom in atoms if atom[1])
    if len(maps) != len(atoms) or len(set(maps)) != len(maps):
        raise AuditError("mapped SMILES has missing or duplicate atom maps")
    return symbols, maps


def parse_reaction_smiles(value: str) -> dict[str, Any]:
    parts = value.split(">")
    if len(parts) != 3 or parts[1]:
        raise AuditError("reaction SMILES must have one empty middle field")
    reactant_text, _, product_text = parts
    reactant_components = reactant_text.split(".")
    reactant_atoms: list[tuple[str, int]] = []
    for component in reactant_components:
        symbols, maps = _parse_mapped_side(component)
        reactant_atoms.extend(zip(symbols, maps))
    product_symbols, product_maps = _parse_mapped_side(product_text)
    reactant_maps = tuple(map_id for _, map_id in reactant_atoms)
    if set(reactant_maps) != set(product_maps):
        raise AuditError("reactant and product atom-map sets differ")
    by_map = dict(zip(product_maps, product_symbols))
    return {
        "reactant_symbols": tuple(symbol for symbol, _ in reactant_atoms),
        "reactant_map_ids": reactant_maps,
        "product_symbols_in_reactant_order": tuple(by_map[map_id] for map_id in reactant_maps),
        "product_map_ids": product_maps,
    }


def _number(value: str) -> float:
    if not _FLOAT_RE.fullmatch(value):
        raise ValueError(f"non-numeric XYZ coordinate: {value!r}")
    result = float(value.replace("D", "E").replace("d", "e"))
    if not math.isfinite(result):
        raise ValueError("non-finite XYZ coordinate")
    return result


def read_xyz(path: Path, root: Path) -> XYZ:
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ has fewer than two header lines")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("XYZ atom count is not an integer") from exc
    if atom_count < 1 or len(lines) < atom_count + 2:
        raise ValueError("XYZ atom count does not match rows")
    symbols: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    for line in lines[2 : 2 + atom_count]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError("XYZ coordinate row is incomplete")
        symbols.append(fields[0])
        coordinates.append(tuple(_number(value) for value in fields[1:4]))
    return XYZ(
        relative_path=_relative(path, root),
        atom_count=atom_count,
        symbols=tuple(symbols),
        coordinates=tuple(coordinates),
        comment=lines[1].strip(),
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
    )


def _pairwise_distances(xyz: XYZ) -> tuple[float, ...]:
    distances: list[float] = []
    for left in range(xyz.atom_count):
        x1, y1, z1 = xyz.coordinates[left]
        for right in range(left + 1, xyz.atom_count):
            x2, y2, z2 = xyz.coordinates[right]
            distances.append(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2))
    return tuple(distances)


def geometry_difference(first: XYZ, second: XYZ, *, tolerance: float = 1e-5) -> dict[str, Any]:
    """Compare geometry independent of rigid translation/rotation.

    Pairwise distances preserve the atom order supplied by the archive.  This
    is enough to distinguish the independently published DFT coordinates from
    a byte-identical or merely reoriented copy without claiming a chemical
    equivalence threshold.
    """

    if first.symbols != second.symbols or first.atom_count != second.atom_count:
        return {
            "status": "atom_order_or_count_differs",
            "max_pairwise_distance_delta": None,
            "pairwise_rms_delta": None,
        }
    left = _pairwise_distances(first)
    right = _pairwise_distances(second)
    deltas = [a - b for a, b in zip(left, right)]
    maximum = max((abs(value) for value in deltas), default=0.0)
    rms = math.sqrt(sum(value * value for value in deltas) / len(deltas)) if deltas else 0.0
    return {
        "status": "same_geometry" if maximum <= tolerance else "distinct_geometry",
        "max_pairwise_distance_delta": maximum,
        "pairwise_rms_delta": rms,
    }


def _log_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "missing",
            "relative_path": path.name,
            "sha256": None,
            "size_bytes": None,
            "charge_multiplicity": [],
            "xtb_states": [],
            "route_has_external_xtb": False,
            "gfn2_command_count": 0,
            "normal_termination": False,
            "error_termination": False,
            "method_evidence": "missing",
        }
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    states = sorted({tuple(int(value) for value in match) for match in _LOG_STATE_RE.findall(text)})
    xtb_states = sorted({tuple(int(value) for value in match) for match in _XTB_STATE_RE.findall(text)})
    route = bool(_EXTERNAL_XTB_RE.search(text))
    gfn2_count = len(_XTB_COMMAND_RE.findall(text))
    normal = "Normal termination of Gaussian" in text
    error = "Error termination" in text
    if route and gfn2_count and normal and not error:
        method = "GFN2-xTB_external_supported"
    elif route or gfn2_count:
        method = "GFN2-xTB_external_partial"
    else:
        method = "unresolved"
    return {
        "status": "present",
        "relative_path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "charge_multiplicity": [list(state) for state in states],
        "xtb_states": [list(state) for state in xtb_states],
        "route_has_external_xtb": route,
        "gfn2_command_count": gfn2_count,
        "normal_termination": normal,
        "error_termination": error,
        "method_evidence": method,
        "text_method_markers": sorted({match.group(0) for match in _METHOD_RE.finditer(text)}),
    }


def _source_archive_identity(cache_root: Path) -> dict[str, Any]:
    archive = cache_root / SOURCE_ARCHIVE_NAME
    if not archive.is_file():
        raise AuditError(f"missing source archive: {SOURCE_ARCHIVE_NAME}")
    observed_md5 = hash_file(archive, algorithm="md5")
    observed_sha256 = hash_file(archive)
    return {
        "name": SOURCE_ARCHIVE_NAME,
        "size_bytes": archive.stat().st_size,
        "expected_size_bytes": SOURCE_ARCHIVE_SIZE_BYTES,
        "md5": observed_md5,
        "expected_md5": SOURCE_ARCHIVE_MD5,
        "sha256": observed_sha256,
        "expected_sha256": SOURCE_ARCHIVE_SHA256,
        "matches_expected": (
            archive.stat().st_size == SOURCE_ARCHIVE_SIZE_BYTES
            and observed_md5 == SOURCE_ARCHIVE_MD5
            and observed_sha256 == SOURCE_ARCHIVE_SHA256
        ),
    }


def _reference_archive_identity(cache_root: Path) -> dict[str, Any] | None:
    archive = cache_root / DFT_REFERENCE_ARCHIVE_NAME
    if not archive.is_file():
        return None
    observed_md5 = hash_file(archive, algorithm="md5")
    observed_sha256 = hash_file(archive)
    return {
        "name": DFT_REFERENCE_ARCHIVE_NAME,
        "size_bytes": archive.stat().st_size,
        "expected_size_bytes": DFT_REFERENCE_ARCHIVE_SIZE_BYTES,
        "md5": observed_md5,
        "expected_md5": DFT_REFERENCE_ARCHIVE_MD5,
        "sha256": observed_sha256,
        "expected_sha256": DFT_REFERENCE_ARCHIVE_SHA256,
        "matches_expected": (
            archive.stat().st_size == DFT_REFERENCE_ARCHIVE_SIZE_BYTES
            and observed_md5 == DFT_REFERENCE_ARCHIVE_MD5
            and observed_sha256 == DFT_REFERENCE_ARCHIVE_SHA256
        ),
    }


def _reference_file_path(reference_root: Path, rid: str, role: str) -> Path:
    suffix = {"product": "product_dft", "ts": "ts_dft"}.get(role, role)
    return reference_root / f"reaction_{rid}" / f"{rid}_{suffix}.xyz"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise AuditError(f"missing source table: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AuditError("source table is empty")
    return rows


def _energy_audit(
    source_rows: list[dict[str, str]],
    dft_rows: list[dict[str, str]] | None,
) -> dict[str, Any]:
    required = ("DG_act", "DrG", "DG_act_xtb", "DrG_xtb")
    missing_columns = sorted(set(required) - set(source_rows[0]))
    if missing_columns:
        raise AuditError(f"source energy columns missing: {missing_columns}")
    numeric_failures: Counter[str] = Counter()
    for row in source_rows:
        for column in required:
            try:
                value = float(row[column])
                if not math.isfinite(value):
                    raise ValueError
            except (TypeError, ValueError):
                numeric_failures[column] += 1
    dft_by_r = {row.get("R"): row for row in (dft_rows or []) if row.get("R")}
    dft_matches = 0
    dft_mismatches: list[str] = []
    if dft_rows is not None:
        for row in source_rows:
            reference = dft_by_r.get(row.get("R"))
            if reference is None:
                dft_mismatches.append(f"{row.get('R', 'unknown')}:missing_reference_row")
                continue
            try:
                if abs(float(row["DG_act"]) - float(reference["G(TS)"])) > 1e-5:
                    raise ValueError
                if abs(float(row["DrG"]) - float(reference["DrG"])) > 1e-5:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                dft_mismatches.append(f"{row.get('R', 'unknown')}:value_mismatch")
            else:
                dft_matches += 1
    return {
        "source_columns": {
            "DG_act": {
                "declared_level": "M06-2X/def2-TZVP DFT free-energy label",
                "unit": "kcal/mol (source article description)",
            },
            "DrG": {
                "declared_level": "M06-2X/def2-TZVP DFT free-energy label",
                "unit": "kcal/mol (source article description)",
            },
            "DG_act_xtb": {
                "declared_level": "GFN2-xTB free-energy label",
                "unit": "kcal/mol (source paper/source table context)",
            },
            "DrG_xtb": {
                "declared_level": "GFN2-xTB free-energy label",
                "unit": "kcal/mol (source paper/source table context)",
            },
        },
        "numeric_failures": dict(sorted(numeric_failures.items())),
        "reference_archive_supplied": dft_rows is not None,
        "dft_reference_rows_matching": dft_matches,
        "dft_reference_rows_mismatching": dft_mismatches,
        "dft_energy_claim_status": (
            "cross_archive_values_match"
            if dft_rows is not None and not dft_mismatches
            else "declared_by_source_only" if dft_rows is None else "cross_archive_mismatch"
        ),
        "geometry_level_warning": (
            "energy-level evidence does not establish XYZ geometry level"
        ),
    }


def _manifest_audit(manifest: Path | None, source_rows: list[dict[str, str]]) -> dict[str, Any]:
    if manifest is None:
        return {"supplied": False, "rows_checked": 0, "flags": []}
    if not manifest.is_file():
        raise AuditError(f"manifest does not exist: {manifest}")
    source_by_id = {row.get("R"): row for row in source_rows}
    flags: list[dict[str, str]] = []
    checked = 0
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"manifest line {line_number} is not JSON") from exc
            checked += 1
            source_id = row.get("source_record_id")
            if source_id not in source_by_id:
                flags.append({"source_record_id": str(source_id), "reason": "source_record_not_in_csv"})
                continue
            protocol = row.get("reference_protocol") or {}
            method = str(protocol.get("method", "")).lower()
            ts = row.get("ts_geometry") or {}
            if "m06" in method or "def2" in method:
                flags.append({
                    "source_record_id": str(source_id),
                    "reason": "manifest_reference_protocol_can_be_read_as_DFT_geometry",
                })
            if ts.get("evidence") == "derived_under_contract":
                flags.append({
                    "source_record_id": str(source_id),
                    "reason": "TS_mapping_is_contract_derived_without_source_sidecar",
                })
    return {
        "supplied": True,
        "relative_path": manifest.name,
        "sha256": hash_file(manifest),
        "rows_checked": checked,
        "flags": flags,
        "claim_limit": "manifest protocol fields are not accepted as file-level geometry evidence",
    }


def build_report(
    cache_root: Path,
    *,
    dft_cache_root: Path | None = None,
    manifest: Path | None = None,
    repository_base_commit: str = "unknown",
    audit_date: str = "unknown",
) -> dict[str, Any]:
    """Build a portable, per-file provenance report for the full archive."""

    cache_root = cache_root.expanduser().resolve()
    source_identity = _source_archive_identity(cache_root)
    source_root = cache_root / SOURCE_EXTRACTED_RELATIVE
    rows = _read_csv(cache_root / SOURCE_CSV_RELATIVE)
    if len({row.get("R") for row in rows}) != len(rows):
        raise AuditError("source table has duplicate R identifiers")

    reference_identity: dict[str, Any] | None = None
    reference_root: Path | None = None
    reference_rows: list[dict[str, str]] | None = None
    if dft_cache_root is not None:
        dft_cache_root = dft_cache_root.expanduser().resolve()
        reference_identity = _reference_archive_identity(dft_cache_root)
        reference_root = dft_cache_root / DFT_EXTRACTED_RELATIVE
        reference_csv = dft_cache_root / DFT_CSV_RELATIVE
        if reference_csv.is_file():
            reference_rows = _read_csv(reference_csv)

    file_roles = ("reactant", "product", "ts", "monomer_1", "monomer_2")
    row_reports: list[dict[str, Any]] = []
    hard_failures: Counter[str] = Counter()
    summary = {
        "rows": len(rows),
        "rows_with_all_xyz": 0,
        "rows_with_both_monomer_logs": 0,
        "rows_with_complete_gfn2_log_evidence": 0,
        "rows_with_neutral_singlet_state": 0,
        "rows_with_ts_element_order_match": 0,
        "rows_with_explicit_ts_map": 0,
        "reference_rows_with_all_geometry_files": 0,
        "reference_geometry_distinct_rows": Counter(),
        "method_status_counts": Counter(),
        "mapping_status_counts": Counter(),
        "reactant_origin_status_counts": Counter(),
    }
    for row in rows:
        rid = row.get("R") or "unknown"
        row_report: dict[str, Any] = {
            "source_record_id": rid,
            "parent_reaction_id": row.get("ID"),
            "source_directory": row.get("R_dir") or f"reaction_{rid}",
            "source_row_sha256": hashlib.sha256(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "energy_fields_present": all(row.get(key) not in (None, "") for key in ("DG_act", "DrG", "DG_act_xtb", "DrG_xtb")),
            "files": {},
            "hard_failures": [],
        }
        try:
            reaction = parse_reaction_smiles(row.get("smiles") or row.get("reaction") or "")
        except (AuditError, ValueError) as exc:
            row_report["mapped_smiles_status"] = "invalid"
            row_report["mapped_smiles_error"] = str(exc)
            hard_failures["mapped_smiles_invalid"] += 1
            reaction = None
        else:
            row_report["mapped_smiles_status"] = "mapped_atom_sets_match"
            row_report["reactant_map_ids"] = list(reaction["reactant_map_ids"])

        reaction_dir = source_root / (row.get("R_dir") or f"reaction_{rid}")
        file_names = {
            "reactant": f"{rid}_reactant.xyz",
            "product": f"{rid}_product.xyz",
            "ts": f"{rid}_ts.xyz",
            "monomer_1": f"{rid}_monomer_1.xyz",
            "monomer_2": f"{rid}_monomer_2.xyz",
        }
        parsed: dict[str, XYZ] = {}
        for role in file_roles:
            path = reaction_dir / file_names[role]
            try:
                value = read_xyz(path, source_root)
            except (OSError, UnicodeError, ValueError) as exc:
                row_report["files"][role] = {
                    "status": "invalid_or_missing",
                    "relative_path": _relative(path, source_root),
                    "error": type(exc).__name__,
                }
                row_report["hard_failures"].append(f"{role}_xyz_invalid")
                hard_failures[f"{role}_xyz_invalid"] += 1
            else:
                parsed[role] = value
                row_report["files"][role] = {
                    "status": "present",
                    "relative_path": value.relative_path,
                    "sha256": value.sha256,
                    "size_bytes": value.size_bytes,
                    "atom_count": value.atom_count,
                    "symbols": list(value.symbols),
                    "comment": value.comment,
                    "embedded_method_evidence": "none" if not _METHOD_RE.search(value.comment) else "comment_marker",
                }
        if len(parsed) == len(file_roles):
            summary["rows_with_all_xyz"] += 1

        if reaction is not None:
            expected = {
                "reactant": reaction["reactant_symbols"],
                "product": reaction["product_symbols_in_reactant_order"],
                "ts": reaction["reactant_symbols"],
            }
            for role in ("reactant", "product", "ts"):
                current = parsed.get(role)
                if current is None:
                    continue
                match = current.symbols == expected[role]
                row_report["files"][role]["mapped_atom_order_match"] = match
                if not match:
                    row_report["hard_failures"].append(f"{role}_atom_order_mismatch")
                    hard_failures[f"{role}_atom_order_mismatch"] += 1
            ts_match = bool(parsed.get("ts") and parsed["ts"].symbols == expected["ts"])
            if ts_match:
                summary["rows_with_ts_element_order_match"] += 1
            explicit_ts_map = bool(re.search(r"(?:map|atom)[-_ ]?ids?\s*[:=]", parsed.get("ts", XYZ("", 0, (), (), "", "", 0)).comment, re.I))
            if explicit_ts_map:
                summary["rows_with_explicit_ts_map"] += 1
            row_report["ts_mapping"] = {
                "status": "explicit_map_evidence" if explicit_ts_map else "element_order_only",
                "reactant_element_order_matches_ts": ts_match,
                "map_ids_embedded_in_xyz": explicit_ts_map,
                "claim_limit": "XYZ has no explicit atom-map sidecar; sequence consistency is not map proof",
            }
            row_report["mapping_status"] = "order_consistent_but_unmapped" if ts_match and not explicit_ts_map else "unresolved"
            summary["mapping_status_counts"][row_report["mapping_status"]] += 1

        logs = [_log_evidence(reaction_dir / "monomer_1.log"), _log_evidence(reaction_dir / "monomer_2.log")]
        row_report["monomer_logs"] = logs
        both_logs = all(log["status"] == "present" for log in logs)
        complete_logs = all(log["method_evidence"] == "GFN2-xTB_external_supported" for log in logs)
        if both_logs:
            summary["rows_with_both_monomer_logs"] += 1
        if complete_logs:
            summary["rows_with_complete_gfn2_log_evidence"] += 1
        neutral_singlet = complete_logs and all(
            log["charge_multiplicity"] == [[0, 1]]
            and log["xtb_states"] == [[0, 0]]
            for log in logs
        )
        if neutral_singlet:
            summary["rows_with_neutral_singlet_state"] += 1
        row_report["monomer_geometry_method_evidence"] = (
            "GFN2-xTB_external_supported" if complete_logs else "partial_or_missing"
        )
        row_report["electronic_state_status"] = (
            "neutral_singlet_logs_supported" if neutral_singlet else "unknown_or_incomplete"
        )

        # Aggregate reactant/product/TS files carry no method metadata.  The
        # published workflow identifies these as xTB profile geometries, while
        # the logs independently prove GFN2-xTB only for monomer optimization.
        method_status = (
            "source_declared_xTB_log_supported_for_monomers"
            if complete_logs
            else "source_declared_xTB_file_level_method_unresolved"
        )
        row_report["geometry_method_status"] = method_status
        row_report["reactant_origin_status"] = (
            "declared_reactant_endpoint_independent_origin_unproven"
        )
        row_report["provenance_gate"] = {
            "geometry_method": (
                "declared_xTB_unknown_file_level"
                if method_status == "source_declared_xTB_file_level_method_unresolved"
                else "declared_xTB_monomer_log_supported"
            ),
            "ts_mapping": "quarantine_no_explicit_map_sidecar",
            "reactant_origin": "unknown_independent_origin",
            "dft_geometry_claim": "quarantine",
        }
        summary["method_status_counts"][method_status] += 1
        summary["reactant_origin_status_counts"][row_report["reactant_origin_status"]] += 1

        if reference_root is not None:
            reference_files: dict[str, Any] = {}
            all_reference = True
            for role in ("monomer_1", "monomer_2", "product", "ts"):
                path = _reference_file_path(reference_root, rid, role)
                if not path.is_file():
                    all_reference = False
                    reference_files[role] = {"status": "missing", "relative_path": _relative(path, reference_root)}
                    continue
                try:
                    value = read_xyz(path, reference_root)
                except (OSError, UnicodeError, ValueError) as exc:
                    all_reference = False
                    reference_files[role] = {
                        "status": "invalid",
                        "relative_path": _relative(path, reference_root),
                        "error": type(exc).__name__,
                    }
                    continue
                source_value = parsed.get(role)
                comparison = geometry_difference(source_value, value) if source_value is not None else {"status": "source_missing"}
                reference_files[role] = {
                    "status": "present",
                    "relative_path": value.relative_path,
                    "sha256": value.sha256,
                    "atom_count": value.atom_count,
                    "symbols": list(value.symbols),
                    "comparison_to_main_archive": comparison,
                }
                if comparison.get("status") == "distinct_geometry":
                    summary["reference_geometry_distinct_rows"][role] += 1
            if all_reference:
                summary["reference_rows_with_all_geometry_files"] += 1
            row_report["dft_reference_files"] = reference_files
            row_report["dft_reference_geometry_status"] = (
                "all_present_and_distinct" if all_reference and all(
                    item.get("comparison_to_main_archive", {}).get("status") == "distinct_geometry"
                    for item in reference_files.values()
                ) else "partial_or_not_distinct"
            )
        row_reports.append(row_report)

    energy = _energy_audit(rows, reference_rows)
    claim_limits = [
        "DFT energy columns do not prove that the main archive XYZ geometries are DFT geometries.",
        "Main archive XYZ comments do not contain a method tag; aggregate reactant/product/TS file-level method evidence is unresolved without the source workflow declaration.",
        "TS element-order agreement is not explicit atom-map evidence; no TS map sidecar is present in the archive.",
        "The reactant endpoint is source-declared and xTB-workflow-derived; independent reactant preparation is not demonstrated.",
        "This audit does not verify TS stationary-point frequencies, IRC connectivity, or DFT optimization convergence.",
    ]
    report = {
        "schema": "xtbflow-dft-da-provenance-audit/v1",
        "audit_date": audit_date,
        "repository_base_commit": repository_base_commit,
        "source": {
            "url": SOURCE_URL,
            "doi": SOURCE_DOI,
            "archive": source_identity,
            "source_table_relative_path": SOURCE_CSV_RELATIVE.as_posix(),
            "declared_geometry_level": "GFN2-xTB reaction-profile geometries (source paper/workflow)",
            "geometry_file_level_metadata": "absent_in_main_archive",
            "geometry_protocol": SOURCE_GEOMETRY_PROTOCOL,
            "geometry_protocol_sha256": canonical_hash(SOURCE_GEOMETRY_PROTOCOL),
            "declared_energy_levels": {
                "DG_act": "M06-2X/def2-TZVP DFT",
                "DrG": "M06-2X/def2-TZVP DFT",
                "DG_act_xtb": "GFN2-xTB",
                "DrG_xtb": "GFN2-xTB",
            },
        },
        "dft_reference_source": {
            "url": DFT_REFERENCE_URL,
            "doi": DFT_REFERENCE_DOI,
            "archive": reference_identity,
            "supplied": reference_root is not None,
            "archive_relative_root": DFT_EXTRACTED_RELATIVE.as_posix() if reference_root is not None else None,
        },
        "audit_script_sha256": hash_file(Path(__file__)),
        "manifest_audit": _manifest_audit(manifest, rows),
        "source_row_count": len(rows),
        "summary": {
            **{key: (dict(value) if isinstance(value, Counter) else value) for key, value in summary.items()},
        },
        "energy_audit": energy,
        "rows": row_reports,
        "hard_failure_counts": dict(sorted(hard_failures.items())),
        "hard_failure_count": sum(hard_failures.values()),
        "provenance_gate": {
            "dft_geometry_claim": "quarantine",
            "ts_mapping_claim": "quarantine",
            "independent_reactant_claim": "unknown",
            "development_xTB_geometry_use": "allowed_only_with_source_declared_xTB_claim_limit",
        },
        "status": "completed_with_claim_limits",
        "scientific_claim_allowed": False,
        "claim_limits": claim_limits,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--dft-cache-root",
        type=Path,
        help="optional cache containing DFT_DA_DATASET.tar.gz and extracted/DFT_DA_DATASET",
    )
    parser.add_argument("--manifest", type=Path, help="optional Track-B JSONL manifest to claim-audit")
    parser.add_argument("--repository-base-commit", default="unknown")
    parser.add_argument("--audit-date", default="unknown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-hard-failures", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = build_report(
            args.cache_root,
            dft_cache_root=args.dft_cache_root,
            manifest=args.manifest,
            repository_base_commit=args.repository_base_commit,
            audit_date=args.audit_date,
        )
    except (AuditError, OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            parser.error(f"refusing to overwrite existing output: {args.output}")
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if not report["hard_failure_count"] or args.allow_hard_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
