#!/usr/bin/env python3
"""Run a bounded, provenance-checked tblite GFN2 diagnostic.

This is an evidence runner, not a qualification or data-admission command.  It
reads an immutable source and its audited input manifest, verifies both byte
hashes and each selected molecular input, then delegates calculations to the
existing GFN2 oracle adapter.  No labels, units, charge, or multiplicity are
inferred; every input or backend failure remains in the report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from numbers import Integral
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


MAX_COUNT = 64
DEFAULT_COUNT = 32
HISTORICAL_EVIDENCE = "docs/evidence/gfn2_units_diagnostic_20260927.json"
ATOMIC_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La "
    "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi "
    "Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt "
    "Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_MISSING = object()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be an explicit 64-character SHA-256 digest")
    value = value.lower()
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a hexadecimal SHA-256 digest")
    return value


def _digest_argument(value: str, name: str) -> str:
    try:
        return _digest(value, name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _count_argument(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("count must be an integer") from exc
    if not 1 <= count <= MAX_COUNT:
        raise argparse.ArgumentTypeError(f"count must be between 1 and {MAX_COUNT}")
    return count


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    if isinstance(value, Sequence):
        result = tuple(value)
    else:
        tolist = getattr(value, "tolist", None)
        if not callable(tolist):
            raise ValueError(f"{name} must be a sequence")
        listed = tolist()
        if not isinstance(listed, list):
            raise ValueError(f"{name} must be a sequence")
        result = tuple(listed)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _symbols(value: Any) -> tuple[str, ...]:
    values = _sequence(value, "atoms/symbols")
    if all(isinstance(item, Integral) and not isinstance(item, bool) for item in values):
        numbers = tuple(int(item) for item in values)
        if any(number < 1 or number > len(ATOMIC_SYMBOLS) for number in numbers):
            raise ValueError("atomic number is outside the supported range")
        return tuple(ATOMIC_SYMBOLS[number - 1] for number in numbers)
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError("symbols must contain nonempty element names")
    return tuple(values)


def _field(row: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    try:
        return row[name]
    except (KeyError, IndexError, TypeError):
        return default


def _manifest(path: Path, source_sha256: str) -> dict[int, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value.get("records") if isinstance(value, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("input manifest must contain a records list")
    declared = value.get("source_sha256")
    if declared is None and isinstance(value.get("source"), Mapping):
        declared = value["source"].get("sha256")
    if declared is not None and _digest(declared, "manifest source_sha256") != source_sha256:
        raise ValueError("input manifest source hash does not match --source-sha256")
    output: dict[int, dict[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"manifest records[{position}] must be objects")
        index = record.get("record_index", position)
        if type(index) is not int or index < 0 or index in output:
            raise ValueError("manifest record_index values must be unique nonnegative integers")
        output[index] = dict(record)
    return output


def _system(raw: Any, audited: Mapping[str, Any], index: int):
    """Build one contract input and verify the manifest's exact input hash."""

    from xtbflow.calculators import MolecularSystem

    symbols = _symbols(_field(raw, "atoms", _field(raw, "symbols", _MISSING)))
    coordinates_value = _field(raw, "inpt_orien", _field(raw, "coordinates", _MISSING))
    coordinates = tuple(tuple(float(component) for component in _sequence(row, "coordinate row")) for row in _sequence(coordinates_value, "coordinates"))
    charge = audited.get("charge", _MISSING)
    multiplicity = audited.get("multiplicity", _MISSING)
    if type(charge) is not int:
        raise ValueError("charge must be explicitly present as an integer in the input manifest")
    if type(multiplicity) is not int or multiplicity < 1:
        raise ValueError("multiplicity must be explicitly present as a positive integer in the input manifest")
    raw_charge = _field(raw, "chrg")
    raw_multiplicity = _field(raw, "mult")
    if raw_charge is not _MISSING and (type(raw_charge) is not int or raw_charge != charge):
        raise ValueError("source charge disagrees with the explicit input manifest")
    if raw_multiplicity is not _MISSING and (type(raw_multiplicity) is not int or raw_multiplicity != multiplicity):
        raise ValueError("source multiplicity disagrees with the explicit input manifest")
    input_hash = _digest(audited.get("input_hash"), "input_hash")
    system = MolecularSystem(
        symbols,
        coordinates,
        charge,
        multiplicity,
        dict(audited.get("environment", {})),
        str(audited.get("system_id", audited.get("record_id", f"units:{index}"))),
    )
    if system.input_hash != input_hash:
        raise ValueError("input_hash does not match the exact source geometry and electronic state")
    return system


def _input_failure(index: int, audited: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    charge = audited.get("charge")
    multiplicity = audited.get("multiplicity")
    input_hash = audited.get("input_hash")
    return {
        "record_index": index,
        "system_id": str(audited.get("system_id", audited.get("record_id", f"units:{index}"))),
        "input_hash": input_hash if isinstance(input_hash, str) else None,
        "charge": charge if type(charge) is int else None,
        "multiplicity": multiplicity if type(multiplicity) is int and multiplicity > 0 else None,
        "backend_status": "failure",
        "converged": False,
        "calculator_calls": 0,
        "error_category": "input",
        "error_message": f"{type(error).__name__}: {error}",
    }


def _result_row(index: int, system: Any, result: Any) -> dict[str, Any]:
    return {
        "record_index": index,
        "system_id": system.system_id,
        "input_hash": system.input_hash,
        "charge": system.charge,
        "multiplicity": system.multiplicity,
        "atom_count": len(system.symbols),
        "backend_status": result.status,
        "converged": result.converged,
        "calculator_calls": result.calculator_calls,
        "backend_energy": result.energy,
        "backend_forces_finite": result.forces is not None,
        "error_category": result.error_category,
        "error_message": result.error_message,
        "metadata": dict(result.metadata),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="immutable UniTS-style .npy object array")
    parser.add_argument("--source-sha256", required=True, type=lambda value: _digest_argument(value, "source-sha256"), help="verified SHA-256 of --source")
    parser.add_argument("--input-manifest", required=True, type=Path, help="audited JSON manifest containing per-record input_hash, charge, and multiplicity")
    parser.add_argument("--input-manifest-sha256", "--input-sha256", required=True, dest="input_manifest_sha256", type=lambda value: _digest_argument(value, "input-manifest-sha256"), help="verified SHA-256 of --input-manifest")
    parser.add_argument("--output", required=True, type=Path, help="new JSON evidence file")
    parser.add_argument("--start", type=int, default=0, help="first source record index")
    parser.add_argument("--count", type=_count_argument, default=DEFAULT_COUNT, help=f"records to inspect (1-{MAX_COUNT})")
    parser.add_argument("--implementation", choices=("tblite", "xtb"), default="tblite")
    parser.add_argument("--protocol-id", default="tblite-gfn2-units-diagnostic-v1")
    parser.add_argument("--accuracy", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument("--electronic-temperature", type=float, default=300.0)
    parser.add_argument("--superseded-evidence", default=HISTORICAL_EVIDENCE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not 1 <= args.count <= MAX_COUNT:
        parser.error(f"--count must be between 1 and {MAX_COUNT}")
    if args.start < 0:
        parser.error("--start must be nonnegative")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    for path, label in ((args.source, "source"), (args.input_manifest, "input manifest")):
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    source_sha256 = _sha256(args.source)
    if source_sha256 != args.source_sha256:
        parser.error("--source-sha256 does not match --source")
    manifest_sha256 = _sha256(args.input_manifest)
    if manifest_sha256 != args.input_manifest_sha256:
        parser.error("--input-manifest-sha256 does not match --input-manifest")
    try:
        audited = _manifest(args.input_manifest, source_sha256)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"invalid input manifest: {exc}")

    try:
        import numpy as np
        dataset = np.load(args.source, allow_pickle=True)
    except (ImportError, OSError, ValueError) as exc:
        parser.error(f"cannot read source object array: {exc}")
    total = len(dataset)
    indices = list(range(args.start, min(total, args.start + args.count)))
    if not indices:
        parser.error("selected source slice is empty")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from xtbflow.calculators import CalculatorProtocol, XTBOracleAdapter

    protocol = CalculatorProtocol(
        protocol_id=args.protocol_id,
        calculator="xtb_oracle",
        method="GFN2-xTB",
        backend="cpu",
        parameters={
            "accuracy": args.accuracy,
            "max_iterations": args.max_iterations,
            "electronic_temperature": args.electronic_temperature,
            "implementation": args.implementation,
            "solvent": None,
        },
    )
    adapter = XTBOracleAdapter(protocol=protocol, implementation=args.implementation)
    prepared: list[tuple[int, Any]] = []
    report_rows: list[dict[str, Any]] = []
    for index in indices:
        metadata = audited.get(index, {})
        try:
            prepared.append((index, _system(dataset[index], metadata, index)))
        except (TypeError, ValueError, KeyError, IndexError, OverflowError) as exc:
            report_rows.append(_input_failure(index, metadata, exc))
    results = adapter.evaluate_batch((system for _, system in prepared), operation="energy_forces")
    report_rows.extend(_result_row(index, system, result) for (index, system), result in zip(prepared, results, strict=True))
    report_rows.sort(key=lambda row: row["record_index"])
    categories = Counter(row["error_category"] for row in report_rows if row.get("error_category"))
    capabilities = adapter.capabilities
    report = {
        "schema": "xtbflow-gfn2-diagnostic/v1",
        "status": "diagnostic_quarantine",
        "admission": "diagnostic_quarantine",
        "source_path": args.source.name,
        "source_sha256": source_sha256,
        "source_record_count": total,
        "input_manifest_path": args.input_manifest.name,
        "input_manifest_sha256": manifest_sha256,
        "slice": {"start": args.start, "count_requested": args.count, "count_observed": len(indices)},
        "protocol": {"protocol_id": protocol.protocol_id, "calculator": protocol.calculator, "method": protocol.method, "parameters": dict(protocol.parameters), "identity": protocol.identity},
        "backend_capabilities": {"calculator": capabilities.calculator, "status": capabilities.status, "detail": capabilities.detail, "version": capabilities.version, "build_hash": capabilities.build_hash, "operations": list(capabilities.operations), "backends": list(capabilities.backends), "qualification": capabilities.qualification},
        "raw_units_status": "unknown; diagnostic output must not be interpreted as qualified error",
        "failure_categories": dict(sorted(categories.items())),
        "success_count": sum(row["backend_status"] == "success" for row in report_rows),
        "failure_count": sum(row["backend_status"] != "success" for row in report_rows),
        "records": report_rows,
        "superseded_historical_evidence": args.superseded_evidence,
        "limits": [
            f"Hard record limit: --count <= {MAX_COUNT}.",
            "Source and input-manifest hashes are verified before any calculator call.",
            "Missing charge or multiplicity is retained as an input failure; no state is inferred.",
            "This report remains diagnostic_quarantine and cannot qualify units or admit labels.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "observed": len(indices), "failure_categories": dict(sorted(categories.items()))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
