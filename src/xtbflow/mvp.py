"""Bounded, replayable end-to-end MVP for xtbflow.

The MVP deliberately has one serial path:

``fixed reactant input -> deterministic candidate geometries -> calculator ->
geometry/QC filters -> ledger + manifest + reports``.

The mock calculator is a software proxy.  The optional xTB path uses the
existing :class:`~xtbflow.calculators.XTBOracleAdapter` and fails closed when
the native module is not available.  Neither path makes a transition-state or
reaction-mechanism claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import struct
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import zlib

from xtbflow.calculators import (
    CalculatorBackend,
    CalculatorCapabilities,
    CalculatorConvergenceError,
    CalculatorProtocol,
    CalculatorProtocolError,
    CalculatorUnavailable,
    MolecularSystem,
    MockCalculator,
    XTBOracleAdapter,
)
from xtbflow.runtime.ledger import (
    BudgetExceeded,
    LedgerEvent,
    RunLedger,
    StageBudget,
)


MVP_SCHEMA = "xtbflow-mvp-manifest/v1"
INPUT_SCHEMA = "xtbflow-mvp-input/v1"
CANDIDATE_SCHEMA = "xtbflow-mvp-candidate/v1"
CONFIG_SCHEMA = "xtbflow-mvp-config/v1"
PROTOCOL_CONTRACT = "xtbflow.calculators.CalculatorProtocol/v1"
REASON_CODES = (
    "QC_MIN_DISTANCE",
    "QC_MAX_DISPLACEMENT",
    "QC_NONFINITE_ENERGY",
    "QC_FORCE_THRESHOLD",
    "BLOCKED_BACKEND",
    "CALCULATOR_NOT_CONVERGED",
    "CALCULATOR_PROTOCOL",
    "CALCULATOR_EXECUTION",
    "BUDGET_EXCEEDED",
    "CONFIG_INVALID",
    "INTERNAL_ERROR",
)
FORBIDDEN_INPUT_FIELDS = {
    "product",
    "product_coordinates",
    "transition_state",
    "ts_coordinates",
    "reference_energy",
    "reference_forces",
    "sealed_test_label",
    "target_derived_active_water",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
    temporary.replace(path)


def _git_identity(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, check=False, capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
    }


def _safe_locator(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        try:
            import yaml  # type: ignore
        except ImportError as yaml_exc:
            raise ValueError(
                "MVP config must be JSON-compatible YAML; install PyYAML for non-JSON YAML syntax"
            ) from yaml_exc
        value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError("MVP config must contain a mapping")
    return value, raw


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _as_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be an integer{suffix}")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"unsupported MVP config schema: {config.get('schema')!r}")
    _as_int(config.get("seed"), "seed", minimum=0)
    backend = config.get("backend", "mock")
    if backend not in {"mock", "xtb"}:
        raise ValueError("backend must be 'mock' or 'xtb'")
    fixture = config.get("fixture")
    if not isinstance(fixture, Mapping):
        raise ValueError("fixture must be a mapping")
    if FORBIDDEN_INPUT_FIELDS.intersection(fixture):
        raise ValueError("reactant fixture contains forbidden target/reference fields")
    for name in ("input_id", "parent_reaction_id", "reaction_family_id"):
        if not isinstance(fixture.get(name), str) or not fixture[name].strip():
            raise ValueError(f"fixture.{name} must be a non-empty string")
    symbols = fixture.get("symbols")
    coordinates = fixture.get("coordinates_angstrom")
    if not isinstance(symbols, list) or not symbols or any(not isinstance(item, str) for item in symbols):
        raise ValueError("fixture.symbols must be a nonempty list of strings")
    if not isinstance(coordinates, list) or len(coordinates) != len(symbols):
        raise ValueError("fixture.coordinates_angstrom must match symbols")
    for row in coordinates:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("fixture coordinates must be an N x 3 list")
        for value in row:
            _as_float(value, "fixture coordinate")
    _as_int(fixture.get("charge"), "fixture.charge")
    _as_int(fixture.get("multiplicity"), "fixture.multiplicity", minimum=1)
    environment = fixture.get("environment", {})
    if not isinstance(environment, Mapping):
        raise ValueError("fixture.environment must be a mapping")

    generation = config.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("generation must be a mapping")
    _as_float(generation.get("noise_scale_angstrom", 0.0), "generation.noise_scale_angstrom")
    candidates = generation.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("generation.candidates must be a nonempty list")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"generation.candidates[{index}] must be a mapping")
        if not isinstance(candidate.get("event_id"), str) or not candidate["event_id"].strip():
            raise ValueError(f"generation.candidates[{index}].event_id is required")
        atoms = candidate.get("atom_indices", [])
        if not isinstance(atoms, list) or any(type(atom) is not int for atom in atoms):
            raise ValueError(f"generation.candidates[{index}].atom_indices must be integer indices")
        if any(atom < 0 or atom >= len(symbols) for atom in atoms):
            raise ValueError(f"generation.candidates[{index}] references an atom outside the input")
        delta = candidate.get("coordinate_delta_angstrom", [0.0, 0.0, 0.0])
        if not isinstance(delta, list) or len(delta) != 3:
            raise ValueError(f"generation.candidates[{index}].coordinate_delta_angstrom must have length 3")
        for value in delta:
            _as_float(value, "candidate coordinate delta")

    qc = config.get("qc")
    if not isinstance(qc, Mapping):
        raise ValueError("qc must be a mapping")
    for name in ("min_distance_angstrom", "max_displacement_angstrom", "max_force_norm_hartree_per_angstrom"):
        if _as_float(qc.get(name), f"qc.{name}") <= 0:
            raise ValueError(f"qc.{name} must be positive")

    budget = config.get("budget")
    if not isinstance(budget, Mapping):
        raise ValueError("budget must be a mapping")
    for name in ("max_system_evaluations", "max_calculator_calls"):
        _as_int(budget.get(name), f"budget.{name}", minimum=1)
    for name in ("max_cpu_core_hours", "max_gpu_hours"):
        if _as_float(budget.get(name), f"budget.{name}") < 0:
            raise ValueError(f"budget.{name} must be nonnegative")
    if len(candidates) > int(budget["max_system_evaluations"]):
        raise ValueError("fixture candidate count exceeds max_system_evaluations")


def _fixture_system(config: Mapping[str, Any]) -> MolecularSystem:
    fixture = config["fixture"]
    return MolecularSystem(
        symbols=tuple(fixture["symbols"]),
        coordinates=tuple(tuple(float(value) for value in row) for row in fixture["coordinates_angstrom"]),
        charge=int(fixture["charge"]),
        multiplicity=int(fixture["multiplicity"]),
        environment=dict(fixture.get("environment", {})),
        system_id=str(fixture["input_id"]),
    )


def _input_payload(config: Mapping[str, Any], system: MolecularSystem) -> dict[str, Any]:
    fixture = config["fixture"]
    return {
        "schema": INPUT_SCHEMA,
        "input_id": fixture["input_id"],
        "parent_reaction_id": fixture["parent_reaction_id"],
        "reaction_family_id": fixture["reaction_family_id"],
        "symbols": list(system.symbols),
        "coordinates_angstrom": [list(row) for row in system.coordinates],
        "charge": system.charge,
        "multiplicity": system.multiplicity,
        "environment": dict(system.environment),
        "input_hash": system.input_hash,
        "target_fields_read": False,
        "claim_limit": "fixed reactant fixture for software MVP; no product or TS inference",
    }


def _build_candidates(config: Mapping[str, Any], system: MolecularSystem, seed: int) -> list[dict[str, Any]]:
    generation = config["generation"]
    noise_scale = _as_float(generation.get("noise_scale_angstrom", 0.0), "generation.noise_scale_angstrom")
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    for candidate_index, spec in enumerate(generation["candidates"]):
        atoms = tuple(int(atom) for atom in spec.get("atom_indices", []))
        delta = tuple(float(value) for value in spec.get("coordinate_delta_angstrom", [0.0, 0.0, 0.0]))
        coordinates = [list(row) for row in system.coordinates]
        noise: dict[str, list[float]] = {}
        for atom in atoms:
            jitter = [rng.uniform(-noise_scale, noise_scale) for _ in range(3)]
            noise[str(atom)] = jitter
            for axis in range(3):
                coordinates[atom][axis] += delta[axis] + jitter[axis]
        event = {
            "event_id": str(spec["event_id"]),
            "intent": str(spec.get("intent", "serial_fixture_perturbation")),
            "atom_indices": list(atoms),
            "coordinate_delta_angstrom": list(delta),
            "seed_noise_angstrom": noise,
            "event_origin": "fixed_fixture",
            "generation_method": "serial_fixture_rule_v1",
        }
        geometry_system = system.with_coordinates(coordinates)
        fingerprint_payload = {
            "input_hash": system.input_hash,
            "event": event,
            "coordinates_angstrom": [list(row) for row in geometry_system.coordinates],
        }
        fingerprint = _sha256_text(_canonical(fingerprint_payload))
        candidates.append(
            {
                "schema": CANDIDATE_SCHEMA,
                "candidate_index": candidate_index,
                "candidate_fingerprint": fingerprint,
                "input_id": system.system_id,
                "input_hash": system.input_hash,
                "event": event,
                "coordinates_angstrom": [list(row) for row in geometry_system.coordinates],
                "geometry_input_hash": geometry_system.input_hash,
            }
        )
    return candidates


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _geometry_reason(candidate: Mapping[str, Any], system: MolecularSystem, qc: Mapping[str, Any]) -> str | None:
    coordinates = candidate["coordinates_angstrom"]
    max_displacement = max(
        _distance(row, reference)
        for row, reference in zip(coordinates, system.coordinates)
    )
    if max_displacement > float(qc["max_displacement_angstrom"]):
        return "QC_MAX_DISPLACEMENT"
    minimum_distance = min(
        _distance(coordinates[i], coordinates[j])
        for i in range(len(coordinates))
        for j in range(i + 1, len(coordinates))
    )
    if minimum_distance < float(qc["min_distance_angstrom"]):
        return "QC_MIN_DISTANCE"
    return None


def _mock_protocol(config: Mapping[str, Any], system: MolecularSystem) -> CalculatorProtocol:
    settings = config["calculator"]["mock"]
    force_constant = float(settings.get("force_constant", 1.0))
    return CalculatorProtocol(
        protocol_id=str(settings.get("protocol_id", "mvp-mock-harmonic-v1")),
        calculator="mvp_mock",
        method="deterministic_harmonic",
        backend="cpu",
        parameters={
            "force_constant": force_constant,
            "reference_coordinates_angstrom": [list(row) for row in system.coordinates],
        },
    )


def _xtb_protocol(config: Mapping[str, Any]) -> CalculatorProtocol:
    settings = config["calculator"]["xtb"]
    return CalculatorProtocol(
        protocol_id=str(settings.get("protocol_id", "mvp-gfn2-xtb-v1")),
        calculator="xtb_oracle",
        method="GFN2-xTB",
        backend="cpu",
        parameters={
            "accuracy": float(settings.get("accuracy", 1.0)),
            "max_iterations": int(settings.get("max_iterations", 100)),
            "electronic_temperature": int(settings.get("electronic_temperature", 300)),
            "solvent": settings.get("solvent"),
        },
    )


def _build_backend(
    backend_name: str,
    config: Mapping[str, Any],
    system: MolecularSystem,
) -> tuple[CalculatorBackend | None, CalculatorProtocol, CalculatorCapabilities, str | None]:
    if backend_name == "mock":
        protocol = _mock_protocol(config, system)
        adapter = MockCalculator(
            reference_coordinates=system.coordinates,
            force_constant=float(protocol.parameters["force_constant"]),
            protocol=protocol,
        )
        return adapter, protocol, adapter.capabilities, None

    protocol = _xtb_protocol(config)
    settings = config["calculator"]["xtb"]
    implementation = str(settings.get("implementation", "xtb"))
    if implementation not in {"xtb", "tblite"}:
        raise ValueError("calculator.xtb.implementation must be 'xtb' or 'tblite'")
    if XTBOracleAdapter is None:
        capabilities = CalculatorCapabilities(
            calculator="xtb_oracle",
            status="unavailable",
            detail="xTB oracle adapter dependencies are not importable in the active environment",
            operations=(),
            backends=(protocol.backend,),
            qualification="unavailable",
        )
        return None, protocol, capabilities, "BLOCKED_BACKEND"
    try:
        available = importlib.util.find_spec(implementation) is not None
    except (ImportError, AttributeError, ValueError):
        available = False
    if not available:
        capabilities = CalculatorCapabilities(
            calculator="xtb_oracle",
            status="unavailable",
            detail=f"{implementation} module is not importable in the active environment",
            operations=(),
            backends=(protocol.backend,),
            qualification="unavailable",
        )
        return None, protocol, capabilities, "BLOCKED_BACKEND"
    adapter = XTBOracleAdapter(protocol=protocol, implementation=implementation)
    capabilities = adapter.capabilities
    if capabilities.status == "unavailable":
        return None, protocol, capabilities, "BLOCKED_BACKEND"
    return adapter, protocol, capabilities, None


def _force_norm(forces: Sequence[Sequence[float]] | None) -> float | None:
    if forces is None:
        return None
    value = math.sqrt(sum(float(component) ** 2 for row in forces for component in row))
    return value if math.isfinite(value) else None


def _reason_for_exception(exc: BaseException) -> str:
    if isinstance(exc, (CalculatorUnavailable, ModuleNotFoundError, ImportError)):
        return "BLOCKED_BACKEND"
    if isinstance(exc, CalculatorConvergenceError):
        return "CALCULATOR_NOT_CONVERGED"
    if isinstance(exc, CalculatorProtocolError):
        return "CALCULATOR_PROTOCOL"
    if isinstance(exc, BudgetExceeded):
        return "BUDGET_EXCEEDED"
    return "CALCULATOR_EXECUTION"


def _base_candidate_row(
    candidate: Mapping[str, Any],
    *,
    run_id: str,
    backend_name: str,
    protocol: CalculatorProtocol,
    build_hash: str,
    evidence_level: str,
) -> dict[str, Any]:
    return {
        "schema": CANDIDATE_SCHEMA,
        "run_id": run_id,
        "backend": backend_name,
        "evidence_level": evidence_level,
        "candidate_index": candidate["candidate_index"],
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "input_id": candidate["input_id"],
        "input_hash": candidate["input_hash"],
        "geometry_input_hash": candidate["geometry_input_hash"],
        "event": candidate["event"],
        "coordinates_angstrom": candidate["coordinates_angstrom"],
        "calculator": protocol.calculator,
        "protocol_id": protocol.protocol_id,
        "calculator_build_hash": build_hash,
        "status": "failure",
        "reason_code": "INTERNAL_ERROR",
        "attempt": 1,
        "wall_seconds": 0.0,
        "cpu_hours": 0.0,
        "gpu_hours": 0.0,
        "calculator_status": "not_requested",
        "calculator_calls": 0,
        "energy_hartree": None,
        "forces_hartree_per_angstrom": None,
        "force_norm_hartree_per_angstrom": None,
        "calculator_metadata": {},
    }


def _ledger_event(
    candidate: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    run_id: str,
    phase: str,
    protocol: CalculatorProtocol,
    build_hash: str,
) -> LedgerEvent:
    ledger_status = "success" if row["calculator_status"] == "success" else "failure"
    return LedgerEvent(
        run_id=run_id,
        phase=phase,
        calculator=protocol.calculator,
        protocol_id=protocol.protocol_id,
        calculator_build_hash=build_hash,
        system_id=f"{candidate['input_id']}:{candidate['event']['event_id']}",
        input_hash=str(candidate["geometry_input_hash"]),
        operation="energy_forces",
        status=ledger_status,
        attempt=int(row["attempt"]),
        calculator_calls=int(row["calculator_calls"]),
        cpu_seconds=float(row["cpu_hours"]) * 3600.0,
        gpu_seconds=float(row["gpu_hours"]) * 3600.0,
        error=None if ledger_status == "success" else str(row["reason_code"]),
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _write_summary_png(path: Path, counts: Mapping[str, int]) -> None:
    """Write a dependency-free RGB bar chart for the run summary."""

    width, height = 640, 360
    pixels = bytearray([255, 255, 255] * width * height)

    def pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(color)

    for x in range(70, 600):
        pixel(x, 310, (40, 40, 40))
    for y in range(40, 311):
        pixel(70, y, (40, 40, 40))
    labels = (("accepted", (46, 125, 50)), ("rejected", (230, 126, 34)), ("failure", (192, 57, 43)))
    maximum = max(1, *(int(counts.get(label, 0)) for label, _ in labels))
    bar_width = 100
    for index, (label, color) in enumerate(labels):
        value = int(counts.get(label, 0))
        bar_height = int(240 * value / maximum)
        left = 125 + index * 150
        for x in range(left, left + bar_width):
            for y in range(310 - bar_height, 310):
                pixel(x, y, color)
    raw = bytearray()
    for row in range(height):
        raw.append(0)
        raw.extend(pixels[row * width * 3 : (row + 1) * width * 3])
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    png.extend(_png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
    png.extend(_png_chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _summary_markdown(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    counts = manifest["counts"]
    lines = [
        "# xtbflow MVP summary",
        "",
        f"- status: `{manifest['status']}`",
        f"- backend: `{manifest['backend']}`",
        f"- evidence_level: `{manifest['evidence_level']}`",
        f"- seed: `{manifest['seed']}`",
        f"- input_hash: `{manifest['input_hash']}`",
        f"- config_sha256: `{manifest['config_sha256']}`",
        f"- git_commit: `{manifest['git']['commit']}`",
        f"- deterministic_summary_sha256: `{manifest['replay']['deterministic_summary_sha256']}`",
        "",
        "## Counts",
        "",
        f"candidates={counts['candidate_count']} accepted={counts['accepted']} rejected={counts['rejected']} failure={counts['failure']}",
        "",
        "## Candidate outcomes",
        "",
        "| index | event | status | reason_code | calculator_calls | force_norm |",
        "| ---: | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        force_norm = row.get("force_norm_hartree_per_angstrom")
        force_text = "" if force_norm is None else f"{float(force_norm):.6g}"
        lines.append(
            f"| {row['candidate_index']} | `{row['event']['event_id']}` | `{row['status']}` | "
            f"`{row['reason_code'] or ''}` | {row['calculator_calls']} | {force_text} |"
        )
    lines.extend(
        [
            "",
            "## Budget",
            "",
            "```json",
            json.dumps(manifest["budget"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Claim boundary",
            "",
            "- `software_demo`: the fixed serial workflow, failure handling and replay contract.",
            "- `proxy_metric`: calculator energy/force values and the configured QC acceptance counts.",
            "- `strict_validation`: required before any transition-state, barrier, mechanism or chemical-accuracy claim; this MVP does not make those claims.",
            "",
            f"Artifacts are written under `{output_path}`.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _deterministic_summary(
    *,
    backend: str,
    seed: int,
    config_sha256: str | None,
    input_hash: str | None,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "backend": backend,
        "seed": seed,
        "config_sha256": config_sha256,
        "input_hash": input_hash,
        "candidate_order": [
            {
                "candidate_index": row.get("candidate_index"),
                "candidate_fingerprint": row.get("candidate_fingerprint"),
                "status": row.get("status"),
                "reason_code": row.get("reason_code"),
            }
            for row in rows
        ],
        "counts": {
            "candidate_count": len(rows),
            "accepted": sum(row.get("status") == "accepted" for row in rows),
            "rejected": sum(row.get("status") == "rejected" for row in rows),
            "failure": sum(row.get("status") == "failure" for row in rows),
        },
    }


def _failure_bundle(
    output_dir: Path,
    *,
    config_path: Path,
    backend: str,
    error: BaseException,
    rows: Sequence[Mapping[str, Any]] = (),
    config_sha256: str | None = None,
    input_hash: str | None = None,
    seed: int | None = None,
    ledger: RunLedger | None = None,
    root: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if seed is None:
        seed = 0
    deterministic = _deterministic_summary(
        backend=backend,
        seed=seed,
        config_sha256=config_sha256,
        input_hash=input_hash,
        rows=rows,
    )
    deterministic_hash = _sha256_text(_canonical(deterministic))
    ledger_summary = ledger.summary() if ledger is not None else {"totals": {}}
    failure_code = "CONFIG_INVALID" if config_sha256 is None or input_hash is None else "INTERNAL_ERROR"
    manifest = {
        "schema": MVP_SCHEMA,
        "status": "failed",
        "failure_stage": "run",
        "failure_reason": f"{type(error).__name__}: {error}",
        "backend": backend,
        "evidence_level": "software_demo",
        "seed": seed,
        "config_locator": _safe_locator(config_path, root),
        "config_sha256": config_sha256,
        "input_hash": input_hash,
        "git": _git_identity(root),
        "execution_host": platform.node(),
        "python": sys.version.split()[0],
        "counts": deterministic["counts"],
        "failure_reason_counts": {failure_code: 1},
        "budget": ledger_summary,
        "replay": {"deterministic_summary_sha256": deterministic_hash, "deterministic_summary": deterministic},
        "output_paths": {},
        "claims": [
            {"name": "MVP execution", "level": "software_demo", "status": "failed"},
            {"name": "QC acceptance metric", "level": "proxy_metric", "status": "not_available"},
            {"name": "chemical interpretation", "level": "strict_validation", "status": "not_claimed"},
        ],
    }
    paths = {
        "manifest": output_dir / "manifest.json",
        "candidates_jsonl": output_dir / "candidates.jsonl",
        "summary_md": output_dir / "summary.md",
        "summary_png": output_dir / "summary.png",
        "ledger": output_dir / "ledger.json",
    }
    _write_jsonl(paths["candidates_jsonl"], rows)
    if ledger is not None:
        ledger.save(paths["ledger"])
    for name, path in paths.items():
        manifest["output_paths"][name] = str(path.resolve())
    _write_summary_png(paths["summary_png"], {"accepted": 0, "rejected": 0, "failure": len(rows) or 1})
    _summary_markdown(manifest, rows, output_dir / "summary.md")
    _write_json(paths["manifest"], manifest)
    return manifest


def run_mvp(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    backend_override: str | None = None,
) -> dict[str, Any]:
    """Run the fixed MVP and return the written manifest payload."""

    config_path = Path(config_path).resolve()
    root = Path(__file__).resolve().parents[2]
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = root / "artifacts" / f"mvp_run_{stamp}"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    ledger: RunLedger | None = None
    config_sha256: str | None = None
    input_hash: str | None = None
    seed: int | None = None
    backend_name = backend_override or "mock"

    try:
        config, raw_config = _load_config(config_path)
        config_sha256 = _sha256_text(raw_config)
        _validate_config(config)
        seed = int(config["seed"])
        backend_name = backend_override or str(config.get("backend", "mock"))
        if backend_name not in {"mock", "xtb"}:
            raise ValueError("backend override must be 'mock' or 'xtb'")
        system = _fixture_system(config)
        input_payload = _input_payload(config, system)
        input_hash = system.input_hash
        candidates = _build_candidates(config, system, seed)
        adapter, protocol, capabilities, blocked_reason = _build_backend(backend_name, config, system)
        build_hash = str(getattr(adapter, "build_hash", None) or capabilities.build_hash or "unavailable")
        evidence_level = "software_demo" if backend_name == "mock" else "proxy_metric"
        budget_cfg = config["budget"]
        budget = StageBudget(
            phase="mvp",
            max_gpu_hours=float(budget_cfg["max_gpu_hours"]),
            max_cpu_core_hours=float(budget_cfg["max_cpu_core_hours"]),
            max_system_evaluations=int(budget_cfg["max_system_evaluations"]),
            max_calculator_calls=int(budget_cfg["max_calculator_calls"]),
            max_retries_per_job=int(budget_cfg.get("max_retries_per_job", 0)),
            max_wall_minutes_per_job=float(budget_cfg.get("max_wall_minutes_per_job", 5.0)),
        )
        run_key = _sha256_text(
            _canonical(
                {
                    "config_sha256": config_sha256,
                    "input_hash": input_hash,
                    "seed": seed,
                    "backend": backend_name,
                }
            )
        )
        run_id = f"mvp-{run_key[:16]}"
        ledger = RunLedger(budget, ledger_id=f"ledger-{run_key[:16]}")
        paths = {
            "manifest": output_dir / "manifest.json",
            "candidates_jsonl": output_dir / "candidates.jsonl",
            "summary_md": output_dir / "summary.md",
            "summary_png": output_dir / "summary.png",
            "ledger": output_dir / "ledger.json",
            "input": output_dir / "input.json",
        }
        _write_json(paths["input"], input_payload)
        initial_manifest = {
            "schema": MVP_SCHEMA,
            "status": "running",
            "run_id": run_id,
            "backend": backend_name,
            "evidence_level": evidence_level,
            "seed": seed,
            "config_locator": _safe_locator(config_path, root),
            "config_sha256": config_sha256,
            "input_hash": input_hash,
            "input_schema": INPUT_SCHEMA,
            "git": _git_identity(root),
            "execution_host": platform.node(),
            "python": sys.version.split()[0],
            "calculator": {
                "protocol_contract": PROTOCOL_CONTRACT,
                "capabilities": asdict(capabilities),
                "protocol": asdict(protocol) | {"identity": protocol.identity},
                "blocked_reason": blocked_reason,
            },
            "generation": {
                "mode": "serial_fixture_rule_v1",
                "candidate_count": len(candidates),
                "candidate_order": [candidate["candidate_fingerprint"] for candidate in candidates],
            },
            "budget": asdict(budget),
            "counts": {"candidate_count": len(candidates), "accepted": 0, "rejected": 0, "failure": 0},
            "failure_reason_counts": {},
            "claims": [
                {"name": "MVP execution", "level": "software_demo", "status": "running"},
                {"name": "QC acceptance metric", "level": "proxy_metric", "status": "running"},
                {"name": "chemical interpretation", "level": "strict_validation", "status": "not_claimed"},
            ],
            "output_paths": {name: str(path.resolve()) for name, path in paths.items()},
        }
        _write_json(paths["manifest"], initial_manifest)

        qc = config["qc"]
        for candidate in candidates:
            started_wall = time.perf_counter()
            started_cpu = time.process_time()
            row = _base_candidate_row(
                candidate,
                run_id=run_id,
                backend_name=backend_name,
                protocol=protocol,
                build_hash=build_hash,
                evidence_level=evidence_level,
            )
            geometry_reason = _geometry_reason(candidate, system, qc)
            if geometry_reason is not None:
                row["status"] = "rejected"
                row["reason_code"] = geometry_reason
                row["calculator_status"] = "not_requested"
            elif blocked_reason is not None or adapter is None:
                row["status"] = "failure"
                row["reason_code"] = blocked_reason or "BLOCKED_BACKEND"
                row["calculator_status"] = "unavailable"
            else:
                candidate_system = system.with_coordinates(candidate["coordinates_angstrom"])
                try:
                    result = adapter.evaluate(candidate_system, operation="energy_forces")
                    row["calculator_status"] = result.status
                    row["calculator_calls"] = result.calculator_calls
                    row["energy_hartree"] = result.energy
                    row["forces_hartree_per_angstrom"] = (
                        [list(force) for force in result.forces] if result.forces is not None else None
                    )
                    row["force_norm_hartree_per_angstrom"] = _force_norm(result.forces)
                    row["calculator_metadata"] = dict(result.metadata)
                    if result.status != "success":
                        row["status"] = "failure"
                        row["reason_code"] = (
                            "CALCULATOR_NOT_CONVERGED" if result.status == "not_converged" else "CALCULATOR_EXECUTION"
                        )
                    elif result.energy is None or not math.isfinite(float(result.energy)):
                        row["status"] = "rejected"
                        row["reason_code"] = "QC_NONFINITE_ENERGY"
                    elif row["force_norm_hartree_per_angstrom"] is None:
                        row["status"] = "rejected"
                        row["reason_code"] = "QC_NONFINITE_ENERGY"
                    elif float(row["force_norm_hartree_per_angstrom"]) > float(qc["max_force_norm_hartree_per_angstrom"]):
                        row["status"] = "rejected"
                        row["reason_code"] = "QC_FORCE_THRESHOLD"
                    else:
                        row["status"] = "accepted"
                        row["reason_code"] = None
                except Exception as exc:  # every backend failure becomes a row, never a traceback-only result
                    row["status"] = "failure"
                    row["reason_code"] = _reason_for_exception(exc)
                    row["calculator_status"] = "failure"
                    row["calculator_calls"] = 0 if row["reason_code"] in {"BLOCKED_BACKEND", "CALCULATOR_PROTOCOL"} else 1
                    row["calculator_metadata"] = {"error_type": type(exc).__name__, "error": str(exc)}

            row["wall_seconds"] = max(0.0, time.perf_counter() - started_wall)
            row["cpu_hours"] = max(0.0, (time.process_time() - started_cpu) / 3600.0)
            row["gpu_hours"] = 0.0
            event = _ledger_event(
                candidate,
                row,
                run_id=run_id,
                phase=budget.phase,
                protocol=protocol,
                build_hash=build_hash,
            )
            try:
                ledger.append(event, persist_path=paths["ledger"])
            except BudgetExceeded:
                row["status"] = "failure"
                row["reason_code"] = "BUDGET_EXCEEDED"
                row["calculator_status"] = "not_run"
                row["calculator_calls"] = 0
            rows.append(row)

        counts = {
            "candidate_count": len(rows),
            "accepted": sum(row["status"] == "accepted" for row in rows),
            "rejected": sum(row["status"] == "rejected" for row in rows),
            "failure": sum(row["status"] == "failure" for row in rows),
        }
        failure_reason_counts: dict[str, int] = {}
        for row in rows:
            reason = row.get("reason_code")
            if reason:
                failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
        deterministic = _deterministic_summary(
            backend=backend_name,
            seed=seed,
            config_sha256=config_sha256,
            input_hash=input_hash,
            rows=rows,
        )
        deterministic_hash = _sha256_text(_canonical(deterministic))
        status = "blocked_backend" if blocked_reason == "BLOCKED_BACKEND" and backend_name == "xtb" else "completed"
        manifest = {
            "schema": MVP_SCHEMA,
            "status": status,
            "run_id": run_id,
            "backend": backend_name,
            "evidence_level": evidence_level,
            "seed": seed,
            "config_locator": _safe_locator(config_path, root),
            "config_sha256": config_sha256,
            "input": input_payload,
            "input_hash": input_hash,
            "git": _git_identity(root),
            "execution_host": platform.node(),
            "python": sys.version.split()[0],
            "calculator": {
                "protocol_contract": PROTOCOL_CONTRACT,
                "capabilities": asdict(capabilities),
                "protocol": asdict(protocol) | {"identity": protocol.identity},
                "build_hash": build_hash,
                "blocked_reason": blocked_reason,
            },
            "generation": {
                "mode": "serial_fixture_rule_v1",
                "candidate_count": len(candidates),
                "candidate_order": [candidate["candidate_fingerprint"] for candidate in candidates],
            },
            "qc": dict(qc),
            "counts": counts,
            "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
            "budget": ledger.summary(),
            "replay": {
                "deterministic_summary_sha256": deterministic_hash,
                "deterministic_summary": deterministic,
                "environment_variance_note": "wall_seconds, CPU timing, git dirty state and native backend build identity may vary across environments",
            },
            "claims": [
                {"name": "MVP execution", "level": "software_demo", "status": "observed"},
                {"name": "QC acceptance metric", "level": "proxy_metric", "status": "observed"},
                {"name": "chemical interpretation", "level": "strict_validation", "status": "not_claimed"},
            ],
            "output_paths": {name: str(path.resolve()) for name, path in paths.items()},
        }
        _write_jsonl(paths["candidates_jsonl"], rows)
        ledger.save(paths["ledger"])
        _write_summary_png(paths["summary_png"], counts)
        _summary_markdown(manifest, rows, paths["summary_md"])
        _write_json(paths["manifest"], manifest)
        return manifest
    except Exception as exc:
        return _failure_bundle(
            output_dir,
            config_path=config_path,
            backend=backend_name,
            error=exc,
            rows=rows,
            config_sha256=config_sha256,
            input_hash=input_hash,
            seed=seed,
            ledger=ledger,
            root=root,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the fixed serial MVP fixture")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--backend", choices=("mock", "xtb"), default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "run":
        return 2
    manifest = run_mvp(args.config, output_dir=args.output_dir, backend_override=args.backend)
    print(
        json.dumps(
            {
                "status": manifest.get("status"),
                "backend": manifest.get("backend"),
                "accepted": manifest.get("counts", {}).get("accepted", 0),
                "rejected": manifest.get("counts", {}).get("rejected", 0),
                "failure": manifest.get("counts", {}).get("failure", 0),
                "manifest": manifest.get("output_paths", {}).get("manifest"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if manifest.get("status") in {"completed", "blocked_backend"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
