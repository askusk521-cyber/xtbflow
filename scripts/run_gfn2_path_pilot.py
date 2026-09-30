#!/usr/bin/env python3
"""Run one metered GFN2 internal-Hessian and bidirectional endpoint pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ase.units import Hartree  # noqa: E402

from xtbflow.calculators import (  # noqa: E402
    CalculatorProtocol,
    MolecularSystem,
    XTBOracleAdapter,
)
from xtbflow.physics.curvature import projected_hessian  # noqa: E402
from xtbflow.runtime import RunLedger, StageBudget  # noqa: E402
from xtbflow.validation.ase_dimer import (  # noqa: E402
    project_rigid_body_components,
)
from xtbflow.validation.ase_path import (  # noqa: E402
    ASERelaxConfig,
    internal_cartesian_basis,
    reconstruct_projected_mode,
    run_ase_minimum_relaxation,
    signed_atom_plane_distance,
)
from xtbflow.validation.connectivity import (  # noqa: E402
    ConnectivityEvidence,
    endpoint_connectivity_gate_pass,
    infer_binary_connectivity,
    observed_event,
)
from xtbflow.validation.evidence import (  # noqa: E402
    index_artifacts,
    sanitize_error_message,
    sanitize_public_value,
    sha256_file,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _protocol(payload: Mapping[str, Any]) -> CalculatorProtocol:
    return CalculatorProtocol(
        protocol_id=str(payload["protocol_id"]),
        calculator=str(payload["calculator"]),
        method=str(payload["method"]),
        backend=str(payload["backend"]),
        parameters=dict(payload["parameters"]),
    )


def _system(payload: Mapping[str, Any]) -> MolecularSystem:
    return MolecularSystem(
        tuple(str(item) for item in payload["symbols"]),
        tuple(
            tuple(float(value) for value in row)
            for row in payload["coordinates_angstrom"]
        ),
        int(payload["charge"]),
        int(payload["multiplicity"]),
        dict(payload["environment"]),
        str(payload["system_id"]),
    )


def _relax_config(payload: Mapping[str, Any]) -> ASERelaxConfig:
    return ASERelaxConfig(
        fmax_eV_per_angstrom=float(payload["fmax_eV_per_angstrom"]),
        max_steps=int(payload["max_steps"]),
        maximum_step_angstrom=float(payload["maximum_step_angstrom"]),
        remove_rigid_body_modes=payload["remove_rigid_body_modes"],
        rigid_body_tolerance=float(payload["rigid_body_tolerance"]),
    )


def _stage_budget(payload: Mapping[str, Any], *, phase: str | None = None) -> StageBudget:
    return StageBudget(
        phase or str(payload["phase"]),
        max_calculator_calls=int(payload["max_calculator_calls"]),
        max_concurrent_jobs=int(payload["max_concurrent_jobs"]),
        max_retries_per_job=int(payload["max_retries_per_job"]),
    )


def _normalized_internal_mode(
    mode: Any,
    coordinates: Any,
    *,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    projected, rank = project_rigid_body_components(
        coordinates, mode, tolerance=tolerance
    )
    norm = float(np.linalg.norm(projected))
    if not math.isfinite(norm) or norm <= tolerance:
        raise ValueError("declared dimer mode has no internal component")
    return projected / norm, rank


def _verify_parent(document: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    provenance = document["provenance"]
    relative = Path(str(provenance["parent_evidence"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("parent_evidence must be a repository-relative path")
    path = ROOT / relative
    if not path.is_file():
        raise ValueError(f"parent evidence does not exist: {relative}")
    observed_hash = sha256_file(path)
    if observed_hash != str(provenance["parent_evidence_sha256"]):
        raise ValueError("parent evidence hash does not match frozen provenance")
    parent = json.loads(path.read_text(encoding="utf-8"))
    transition = document["transition_state"]
    if parent.get("source_commit") != provenance["parent_source_commit"]:
        raise ValueError("parent source commit does not match frozen provenance")
    if parent.get("status") != "pass":
        raise ValueError("parent dimer evidence did not pass its frozen driver gate")
    driver = parent.get("driver_result") or {}
    if driver.get("coordinates_angstrom") != transition["system"]["coordinates_angstrom"]:
        raise ValueError("transition-state coordinates differ from parent evidence")
    if driver.get("eigenmode") != transition["dimer_mode"]:
        raise ValueError("declared dimer mode differs from parent evidence")
    if driver.get("energy_hartree") != transition["energy_hartree"]:
        raise ValueError("declared transition-state energy differs from parent evidence")
    return path, parent


def _settled_accounting(ledger: RunLedger, ledger_path: Path) -> dict[str, Any]:
    payload = ledger.to_dict()
    return {
        "ledger_sha256": sha256_file(ledger_path),
        "committed_calculator_calls": payload["committed_calculator_calls"],
        "pending_reservations": payload["reservations"],
        "committed_tokens": payload["committed_tokens"],
        "settled": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/validation/gfn2_nh3_path_v0.1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    document = json.loads(args.config.read_text(encoding="utf-8"))
    if document.get("schema") != "xtbflow-gfn2-path-pilot/v1":
        parser.error("unsupported path pilot config schema")
    if args.output.exists():
        parser.error("refusing to overwrite an existing output")
    for directory, label in (
        (args.ledger_dir, "ledger"),
        (args.artifact_dir, "artifact"),
    ):
        if directory.exists() and any(directory.iterdir()):
            parser.error(f"refusing to mix with a nonempty {label} directory")
        directory.mkdir(parents=True, exist_ok=True)

    try:
        parent_path, _ = _verify_parent(document)
    except ValueError as error:
        parser.error(str(error))
    protocol = _protocol(document["protocol"])
    transition = document["transition_state"]
    system = _system(transition["system"])
    adapter = XTBOracleAdapter(
        protocol=protocol,
        implementation="tblite",
        require_budget_token=True,
    )
    source_commit = _source_commit()
    report: dict[str, Any] = {
        "schema": "xtbflow-gfn2-path-pilot-result/v1",
        "status": "initializing",
        "scientific_qualification": False,
        "reference_path_status": "not_validated",
        "development_path_status": "not_run",
        "source_commit": source_commit,
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        "ase_path_driver_sha256": sha256_file(
            ROOT / "src/xtbflow/validation/ase_path.py"
        ),
        "curvature_driver_sha256": sha256_file(
            ROOT / "src/xtbflow/physics/curvature.py"
        ),
        "xtb_oracle_adapter_sha256": sha256_file(
            ROOT / "src/xtbflow/calculators/xtb_oracle.py"
        ),
        "parent_evidence": {
            "relative_path": parent_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(parent_path),
            "source_commit": document["provenance"]["parent_source_commit"],
        },
        "protocol": {
            "protocol_id": protocol.protocol_id,
            "protocol_identity": protocol.identity,
            "method": protocol.method,
            "backend": protocol.backend,
            "parameters": dict(protocol.parameters),
        },
        "runtime": {
            "implementation": "tblite",
            "version": adapter.version,
            "build_hash": adapter.build_hash,
            "capabilities": asdict(adapter.capabilities),
        },
        "transition_state": transition,
        "hessian": None,
        "endpoints": {},
        "observed_bond_event": [],
        "artifact_index": [],
        "calculator_accounting": None,
        "claim_limits": list(document["claim_limits"]),
    }
    # Keep execution identities explicit.  A later evidence publication step
    # must not replace these with hashes from its own checkout.
    report["execution_source_commit"] = source_commit
    report["execution_script_sha256"] = report["script_sha256"]
    report["execution_ase_path_driver_sha256"] = report[
        "ase_path_driver_sha256"
    ]
    report["execution_curvature_driver_sha256"] = report[
        "curvature_driver_sha256"
    ]
    report["execution_xtb_oracle_adapter_sha256"] = report[
        "xtb_oracle_adapter_sha256"
    ]

    hessian_config = document["hessian"]
    internal_basis, rigid_rank = internal_cartesian_basis(
        system.coordinates,
        tolerance=float(
            document["endpoints"]["relaxation"]["rigid_body_tolerance"]
        ),
    )
    expected_hessian_calls = 1 + 2 * len(internal_basis)
    hessian_budget = hessian_config["budget"]
    if int(hessian_budget["max_calculator_calls"]) != expected_hessian_calls:
        parser.error(
            "hessian budget must equal one base E/F plus two calls per internal basis direction"
        )
    hessian_ledger_path = args.ledger_dir / "hessian-ledger.json"
    hessian_ledger = RunLedger(_stage_budget(hessian_budget))
    hessian_token = hessian_ledger.issue_calculator_token(
        "gfn2-internal-hessian",
        expected_hessian_calls,
        metadata={
            "protocol_id": protocol.protocol_id,
            "system_id": system.system_id,
            "internal_dimension": len(internal_basis),
            "rigid_body_rank": rigid_rank,
            "finite_difference_step_angstrom": float(
                hessian_config["step_angstrom"]
            ),
        },
        persist_path=hessian_ledger_path,
    )

    base_result = None
    hessian_result = None
    hessian_error = None
    try:
        base_result = adapter.evaluate(
            system,
            operation="energy_forces",
            budget_token=hessian_token,
        )
        if (
            base_result.status == "success"
            and base_result.energy is not None
            and base_result.forces is not None
        ):
            hessian_result = projected_hessian(
                adapter,
                system,
                internal_basis,
                step=float(hessian_config["step_angstrom"]),
                orthonormal_tolerance=float(
                    hessian_config["orthonormal_tolerance"]
                ),
                budget_token=hessian_token,
            )
        else:
            hessian_error = (
                base_result.error_message
                or base_result.error_category
                or "base E/F evaluation did not return complete evidence"
            )
    except Exception as error:
        hessian_error = sanitize_error_message(f"{type(error).__name__}: {error}")

    base_gradient = None
    base_energy = None
    if (
        base_result is not None
        and base_result.status == "success"
        and base_result.energy is not None
        and base_result.forces is not None
    ):
        base_energy = float(base_result.energy)
        base_gradient = float(np.linalg.norm(np.asarray(base_result.forces, dtype=float)))

    eigenvalues_hartree: list[float] | None = None
    eigenvalues_ev: list[float] | None = None
    negative_indices: list[int] = []
    hessian_mode: np.ndarray | None = None
    absolute_overlap = None
    signed_overlap = None
    if (
        hessian_result is not None
        and hessian_result.status == "success"
        and hessian_result.eigenvalues is not None
        and hessian_result.eigenvectors is not None
    ):
        eigenvalues_hartree = [float(value) for value in hessian_result.eigenvalues]
        eigenvalues_ev = [value * Hartree for value in eigenvalues_hartree]
        threshold = float(
            hessian_config["negative_threshold_eV_per_angstrom2"]
        )
        negative_indices = [
            index for index, value in enumerate(eigenvalues_ev) if value < -threshold
        ]
        if negative_indices:
            hessian_mode = reconstruct_projected_mode(
                hessian_result.basis,
                hessian_result.eigenvectors,
                negative_indices[0],
            )
            declared_mode, declared_rank = _normalized_internal_mode(
                transition["dimer_mode"],
                system.coordinates,
                tolerance=float(
                    document["endpoints"]["relaxation"][
                        "rigid_body_tolerance"
                    ]
                ),
            )
            if declared_rank != rigid_rank:
                raise RuntimeError("inconsistent rigid-body rank for declared mode")
            signed_overlap = float(np.sum(hessian_mode * declared_mode))
            if signed_overlap < 0.0:
                hessian_mode = -hessian_mode
                signed_overlap = -signed_overlap
            absolute_overlap = abs(signed_overlap)

    expected_negative = int(hessian_config["expected_negative_modes"])
    maximum_gradient = float(
        hessian_config["max_gradient_norm_hartree_per_angstrom"]
    )
    minimum_overlap = float(
        hessian_config["minimum_absolute_dimer_mode_overlap"]
    )
    hessian_gate_pass = (
        base_gradient is not None
        and base_gradient <= maximum_gradient
        and hessian_result is not None
        and hessian_result.status == "success"
        and len(negative_indices) == expected_negative
        and absolute_overlap is not None
        and absolute_overlap >= minimum_overlap
    )
    hessian_record: dict[str, Any] = {
        "status": "pending_settlement",
        "base_evaluation": {
            "status": None if base_result is None else base_result.status,
            "energy_hartree": base_energy,
            "gradient_norm_hartree_per_angstrom": base_gradient,
            "error": hessian_error,
        },
        "rigid_body_rank": rigid_rank,
        "internal_dimension": len(internal_basis),
        "finite_difference_result": (
            None if hessian_result is None else asdict(hessian_result)
        ),
        "eigenvalues_hartree_per_angstrom2": eigenvalues_hartree,
        "eigenvalues_eV_per_angstrom2": eigenvalues_ev,
        "negative_mode_indices": negative_indices,
        "selected_negative_mode": (
            None if hessian_mode is None else hessian_mode.tolist()
        ),
        "dimer_mode_signed_overlap": signed_overlap,
        "dimer_mode_absolute_overlap": absolute_overlap,
        "gate_evaluation": {
            "max_gradient_norm_hartree_per_angstrom": maximum_gradient,
            "gradient_within_limit": (
                base_gradient is not None and base_gradient <= maximum_gradient
            ),
            "negative_threshold_eV_per_angstrom2": float(
                hessian_config["negative_threshold_eV_per_angstrom2"]
            ),
            "expected_negative_modes": expected_negative,
            "observed_negative_modes": len(negative_indices),
            "minimum_absolute_dimer_mode_overlap": minimum_overlap,
            "mode_overlap_pass": (
                absolute_overlap is not None and absolute_overlap >= minimum_overlap
            ),
            "hessian_gate_pass": hessian_gate_pass,
        },
        "calculator_accounting": {
            "reserved_calls": hessian_token.reserved_calls,
            "consumed_calls": hessian_token.consumed_calls,
            "settled": False,
        },
    }
    hessian_artifact = args.artifact_dir / "hessian.json"
    _atomic_json(hessian_artifact, hessian_record)
    hessian_token.settle(
        recorded_calls=hessian_token.consumed_calls,
        persist_path=hessian_ledger_path,
    )
    hessian_record["status"] = "pass" if hessian_gate_pass else "fail"
    hessian_record["calculator_accounting"] = _settled_accounting(
        hessian_ledger, hessian_ledger_path
    )
    hessian_record = sanitize_public_value(hessian_record)
    _atomic_json(hessian_artifact, hessian_record)
    report["hessian"] = hessian_record

    if not hessian_gate_pass or hessian_mode is None or base_energy is None:
        report["status"] = "fail"
        report["development_path_status"] = "not_validated"
        report["artifact_index"] = index_artifacts(args.artifact_dir)
        report["calculator_accounting"] = {
            "hessian_calls": hessian_ledger.to_dict()[
                "committed_calculator_calls"
            ],
            "endpoint_calls": 0,
            "total_calls": hessian_ledger.to_dict()[
                "committed_calculator_calls"
            ],
        }
        _atomic_json(args.output, sanitize_public_value(report))
        print(
            json.dumps(
                {
                    "status": "fail",
                    "stage": "hessian",
                    "gradient_norm_hartree_per_angstrom": base_gradient,
                    "negative_modes": len(negative_indices),
                    "mode_overlap": absolute_overlap,
                    "calculator_calls": report["calculator_accounting"][
                        "total_calls"
                    ],
                },
                sort_keys=True,
            )
        )
        return 2

    endpoint_config = document["endpoints"]
    relax_config = _relax_config(endpoint_config["relaxation"])
    displacement = float(endpoint_config["displacement_angstrom"])
    if not math.isfinite(displacement) or displacement <= 0.0:
        parser.error("endpoint displacement_angstrom must be finite and positive")
    order_parameter = endpoint_config["order_parameter"]
    if order_parameter.get("kind") != "signed_atom_plane_distance":
        parser.error("unsupported endpoint order parameter")
    endpoint_budget = endpoint_config["budget_per_endpoint"]
    endpoint_records: dict[str, Any] = {}
    endpoint_total_calls = 0

    transition_coordinates = np.asarray(system.coordinates, dtype=float)
    for label, sign in (("plus", 1.0), ("minus", -1.0)):
        endpoint_coordinates = transition_coordinates + sign * displacement * hessian_mode
        endpoint_system = system.with_coordinates(endpoint_coordinates.tolist())
        ledger_path = args.ledger_dir / f"endpoint-{label}-ledger.json"
        phase = f"{endpoint_budget['phase_prefix']}-{label}"
        ledger = RunLedger(_stage_budget(endpoint_budget, phase=phase))
        token = ledger.issue_calculator_token(
            f"gfn2-endpoint-{label}",
            int(endpoint_budget["max_calculator_calls"]),
            metadata={
                "protocol_id": protocol.protocol_id,
                "system_id": endpoint_system.system_id,
                "direction": label,
                "displacement_angstrom": displacement,
                "mode_source": "independent_internal_hessian",
            },
            persist_path=ledger_path,
        )
        endpoint_artifact_dir = args.artifact_dir / "endpoints" / label
        result = run_ase_minimum_relaxation(
            adapter,
            endpoint_system,
            budget_token=token,
            artifact_dir=endpoint_artifact_dir,
            config=relax_config,
            require_durable_token=True,
        )
        signed_distance = None
        order_error = None
        try:
            signed_distance = signed_atom_plane_distance(
                result.coordinates_angstrom,
                atom_index=int(order_parameter["atom_index"]),
                plane_indices=order_parameter["plane_indices"],
            )
        except Exception as error:
            order_error = sanitize_error_message(f"{type(error).__name__}: {error}")
        energy_delta = (
            None
            if result.energy_hartree is None
            else float(result.energy_hartree) - base_energy
        )
        endpoint_gate = endpoint_config["gate"]
        endpoint_gradient_limit = float(
            endpoint_gate["max_gradient_norm_hartree_per_angstrom"]
        )
        minimum_distance = float(
            order_parameter["minimum_absolute_distance_angstrom"]
        )
        require_lower_energy = bool(
            endpoint_gate["require_energy_below_transition_state"]
        )
        endpoint_gate_pass = (
            result.converged
            and result.gradient_norm_hartree_per_angstrom is not None
            and result.gradient_norm_hartree_per_angstrom
            <= endpoint_gradient_limit
            and signed_distance is not None
            and abs(signed_distance) >= minimum_distance
            and (
                not require_lower_energy
                or (energy_delta is not None and energy_delta < 0.0)
            )
        )
        endpoint_record: dict[str, Any] = {
            "status": "pending_settlement",
            "direction": label,
            "initial_coordinates_angstrom": endpoint_coordinates.tolist(),
            "relaxation_result": asdict(result),
            "order_parameter": {
                "kind": order_parameter["kind"],
                "signed_distance_angstrom": signed_distance,
                "error": order_error,
            },
            "energy_delta_from_transition_state_hartree": energy_delta,
            "gate_evaluation": {
                "optimizer_converged": result.converged,
                "gradient_norm_hartree_per_angstrom": (
                    result.gradient_norm_hartree_per_angstrom
                ),
                "max_gradient_norm_hartree_per_angstrom": (
                    endpoint_gradient_limit
                ),
                "minimum_absolute_distance_angstrom": minimum_distance,
                "distance_gate_pass": (
                    signed_distance is not None
                    and abs(signed_distance) >= minimum_distance
                ),
                "require_energy_below_transition_state": require_lower_energy,
                "energy_below_transition_state": (
                    energy_delta is not None and energy_delta < 0.0
                ),
                "endpoint_gate_pass": endpoint_gate_pass,
            },
            "calculator_accounting": {
                "reserved_calls": token.reserved_calls,
                "consumed_calls": token.consumed_calls,
                "settled": False,
            },
        }
        endpoint_record = sanitize_public_value(endpoint_record)
        endpoint_summary = endpoint_artifact_dir / "summary.json"
        _atomic_json(endpoint_summary, endpoint_record)
        token.settle(
            recorded_calls=token.consumed_calls,
            persist_path=ledger_path,
        )
        endpoint_record["status"] = "pass" if endpoint_gate_pass else "fail"
        endpoint_record["calculator_accounting"] = _settled_accounting(
            ledger, ledger_path
        )
        endpoint_total_calls += int(
            endpoint_record["calculator_accounting"][
                "committed_calculator_calls"
            ]
        )
        _atomic_json(endpoint_summary, endpoint_record)
        endpoint_records[label] = endpoint_record

    plus_distance = endpoint_records["plus"]["order_parameter"][
        "signed_distance_angstrom"
    ]
    minus_distance = endpoint_records["minus"]["order_parameter"][
        "signed_distance_angstrom"
    ]
    require_opposite = bool(order_parameter["require_opposite_signs"])
    opposite_signs = (
        plus_distance is not None
        and minus_distance is not None
        and float(plus_distance) * float(minus_distance) < 0.0
    )
    endpoint_bonds = {
        label: infer_binary_connectivity(
            system.symbols,
            endpoint_records[label]["relaxation_result"]["coordinates_angstrom"],
        )
        for label in ("minus", "plus")
    }
    connectivity = ConnectivityEvidence(
        reactant_bonds=endpoint_bonds["minus"],
        product_bonds=endpoint_bonds["plus"],
    )
    observed_bond_event = [list(edit) for edit in observed_event(connectivity)]
    same_bond_connectivity = not observed_bond_event
    endpoint_gate = endpoint_config["gate"]
    require_same_bond = bool(
        endpoint_gate.get("require_same_bond_connectivity", True)
    )
    connectivity_gate_pass = endpoint_connectivity_gate_pass(
        connectivity, require_same_bonds=require_same_bond
    )
    endpoint_pair_pass = (
        endpoint_records["plus"]["gate_evaluation"]["endpoint_gate_pass"]
        and endpoint_records["minus"]["gate_evaluation"]["endpoint_gate_pass"]
        and (opposite_signs or not require_opposite)
        and connectivity_gate_pass
    )
    report["endpoints"] = endpoint_records
    report["endpoint_connectivity"] = {
        "representation": "covalent-radius-binary-endpoint-comparison-v1",
        "bond_cutoff_scale": 1.25,
        "minus_bond_matrix": [list(row) for row in endpoint_bonds["minus"]],
        "plus_bond_matrix": [list(row) for row in endpoint_bonds["plus"]],
        "observed_event_from_minus_to_plus": observed_bond_event,
        "same_bond_connectivity": same_bond_connectivity,
        "require_same_bond_connectivity": require_same_bond,
        "connectivity_gate_pass": connectivity_gate_pass,
        "source": "calculated_relaxed_endpoint_coordinates",
    }
    report["endpoint_pair_gate"] = {
        "require_opposite_signs": require_opposite,
        "opposite_signs": opposite_signs,
        "plus_signed_distance_angstrom": plus_distance,
        "minus_signed_distance_angstrom": minus_distance,
        "same_bond_connectivity": same_bond_connectivity,
        "require_same_bond_connectivity": require_same_bond,
        "connectivity_gate_pass": connectivity_gate_pass,
        "observed_bond_event": observed_bond_event,
        "endpoint_pair_gate_pass": endpoint_pair_pass,
    }
    report["status"] = "pass" if endpoint_pair_pass else "fail"
    report["development_path_status"] = (
        "validated" if endpoint_pair_pass else "not_validated"
    )
    hessian_calls = int(
        hessian_record["calculator_accounting"]["committed_calculator_calls"]
    )
    report["calculator_accounting"] = {
        "hessian_calls": hessian_calls,
        "endpoint_calls": endpoint_total_calls,
        "total_calls": hessian_calls + endpoint_total_calls,
    }
    report["artifact_index"] = index_artifacts(args.artifact_dir)
    _atomic_json(args.output, sanitize_public_value(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "stage": "complete",
                "development_path_status": report["development_path_status"],
                "reference_path_status": report["reference_path_status"],
                "negative_modes": len(negative_indices),
                "mode_overlap": absolute_overlap,
                "plus_signed_distance_angstrom": plus_distance,
                "minus_signed_distance_angstrom": minus_distance,
                "calculator_calls": report["calculator_accounting"][
                    "total_calls"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if endpoint_pair_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
