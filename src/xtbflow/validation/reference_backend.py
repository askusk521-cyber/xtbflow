"""Versioned PySCF adapter for the durable certification service."""
from __future__ import annotations

import importlib.metadata
import hashlib
from pathlib import Path

from .certification import CallMeter, CertificationCandidate, file_hash


class ReferenceBackend:
    def __init__(self, *, device: str, execution_build: dict | None = None) -> None:
        if device not in {"cpu", "gpu"}:
            raise ValueError("device must be cpu or gpu")
        self.device = device
        if device == "gpu" and (not execution_build or not all(execution_build.get(k) for k in ("gpu", "driver", "runtime"))):
            raise ValueError("GPU identity requires observed device, CUDA driver and runtime versions")
        self.execution_build = execution_build or {}

    @property
    def identity(self) -> dict:
        from . import reference_protocol as protocol
        packages = ["numpy", "scipy", "pyscf", "geometric", "rdkit"]
        if self.device == "gpu":
            packages += ["gpu4pyscf-cuda12x", "cupy-cuda12x"]
        versions = {name: importlib.metadata.version(name) for name in packages}
        builds = {}
        for name in packages:
            manifest = importlib.metadata.distribution(name).read_text("RECORD")
            if not manifest:
                raise ValueError(f"missing installed build manifest: {name}")
            builds[name] = hashlib.sha256(manifest.encode()).hexdigest()
        method = protocol.Method(charge=0, spin=0, device=self.device)
        return dict(engine="gpu4pyscf" if self.device == "gpu" else "pyscf", device=self.device,
                    version=protocol.PROTOCOL_VERSION, method=method.as_dict(), packages=versions,
                    installed_record_sha256=builds,
                    execution_build=self.execution_build,
                    protocol_source_sha256=file_hash(Path(protocol.__file__)),
                    adapter_source_sha256=file_hash(Path(__file__)),
                    thresholds=dict(tight=protocol.TIGHT, very_tight=protocol.VERY_TIGHT, irc=protocol.IRC,
                                    negative_frequency_cm1=protocol.CLEAR_NEGATIVE_CM1,
                                    allow_endpoint_gray_zone=False, opt_maxiter=200,
                                    min_retry_displacement_angstrom=protocol.MIN_RETRY_DISPLACEMENT),
                    units=dict(coordinates="Angstrom", energy="Hartree", gradient="Hartree/bohr",
                               hessian="Hartree/bohr^2", frequencies="cm^-1"),
                    evidence_scope="gas_phase_neutral_singlet_CHNO_reference_protocol")

    def run(self, candidate: CertificationCandidate, meter: CallMeter, work: Path) -> dict:
        from . import reference_protocol as protocol
        method = protocol.Method(charge=candidate.charge, spin=candidate.multiplicity - 1, device=self.device)
        return protocol.certification_chain(candidate.atomic_numbers, candidate.coordinates_angstrom,
                                            candidate.reactant_be, candidate.reactant_automorphisms,
                                            candidate.planned_product_be, method, work.resolve(), meter)
