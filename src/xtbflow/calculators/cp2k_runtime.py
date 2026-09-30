"""Resolve a frozen CP2K protocol against an observed local runtime."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from .cp2k import CP2KProtocol


@dataclass(frozen=True)
class CP2KRuntimeIdentity:
    """Observed executable identity without publishing a private host path."""

    executable: str
    version: str
    source_revision: str
    build_hash: str
    executable_sha256: str
    data_dir: str

    def public_record(self) -> dict[str, str]:
        return {
            "executable_name": Path(self.executable).name,
            "cp2k_version": self.version,
            "source_revision": self.source_revision,
            "build_hash": self.build_hash,
            "executable_sha256": self.executable_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cp2k_protocol_document(path: str | Path) -> dict[str, Any]:
    """Load the JSON-compatible YAML protocol document and validate its shell."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid CP2K protocol document {source}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("protocol"), dict):
        raise ValueError("CP2K protocol document must contain a protocol object")
    return payload


def inspect_cp2k_runtime(
    executable: str | Path,
    *,
    data_dir: str | Path | None = None,
) -> CP2KRuntimeIdentity:
    resolved = Path(executable).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"CP2K executable does not exist: {resolved}")
    completed = subprocess.run(
        [str(resolved), "-v"],
        capture_output=True,
        text=True,
        check=False,
    )
    text = f"{completed.stdout}\n{completed.stderr}"
    version_match = re.search(r"CP2K version\s+([0-9][^\s]*)", text)
    revision_match = re.search(
        r"Source code revision\s+([0-9a-f]+)", text, re.IGNORECASE
    )
    if completed.returncode != 0 or version_match is None or revision_match is None:
        raise ValueError("CP2K executable did not expose a usable version identity")
    version = version_match.group(1)
    revision = revision_match.group(1)
    resolved_data = (
        Path(data_dir).expanduser().resolve()
        if data_dir is not None
        else resolved.parent.parent / "share" / "cp2k" / "data"
    )
    if not resolved_data.is_dir():
        raise ValueError(f"CP2K data directory does not exist: {resolved_data}")
    return CP2KRuntimeIdentity(
        executable=str(resolved),
        version=version,
        source_revision=revision,
        build_hash=f"cp2k-{version}-{revision}",
        executable_sha256=_sha256(resolved),
        data_dir=str(resolved_data),
    )


def _resolved_parameter_files(
    parameters: Mapping[str, Any], data_dir: Path
) -> dict[str, Any]:
    resolved = dict(parameters)
    for name in (
        "basis_set_file",
        "pseudopotential_file",
        "dispersion_parameter_file",
    ):
        value = resolved.get(name)
        if value is None:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = data_dir / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"CP2K protocol file is missing: {path}")
        resolved[name] = str(path)
    return resolved


def build_cp2k_protocol(
    document: Mapping[str, Any],
    runtime: CP2KRuntimeIdentity,
    *,
    charge: int,
    multiplicity: int,
    protocol_id: str | None = None,
) -> CP2KProtocol:
    """Bind one molecular state to a frozen, runtime-matched CP2K protocol."""

    if not isinstance(document, Mapping) or not isinstance(
        document.get("protocol"), Mapping
    ):
        raise ValueError("CP2K protocol document must contain a protocol mapping")
    values = dict(document["protocol"])
    expected_version = str(values.get("cp2k_version", ""))
    expected_build = str(values.get("build_hash", ""))
    if expected_version != runtime.version:
        raise ValueError(
            f"CP2K version mismatch: protocol={expected_version}, runtime={runtime.version}"
        )
    if expected_build != runtime.build_hash:
        raise ValueError(
            f"CP2K build mismatch: protocol={expected_build}, runtime={runtime.build_hash}"
        )
    parameters = values.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("CP2K protocol parameters must be a mapping")
    values["parameters"] = _resolved_parameter_files(
        parameters, Path(runtime.data_dir)
    )
    base_id = str(values["protocol_id"])
    values.update(
        protocol_id=protocol_id or f"{base_id}:q{charge}:m{multiplicity}",
        charge=charge,
        multiplicity=multiplicity,
        build_hash=runtime.build_hash,
        cp2k_version=runtime.version,
    )
    return CP2KProtocol(**values)


def physical_protocol_record(
    document: Mapping[str, Any],
    runtime: CP2KRuntimeIdentity,
    *,
    charge: int,
    multiplicity: int,
    protocol_id: str | None = None,
) -> dict[str, Any]:
    """Return a path-independent record of the frozen physical protocol."""

    if not isinstance(document, Mapping) or not isinstance(
        document.get("protocol"), Mapping
    ):
        raise ValueError("CP2K protocol document must contain a protocol mapping")
    values = json.loads(json.dumps(document["protocol"], allow_nan=False))
    if str(values.get("cp2k_version", "")) != runtime.version:
        raise ValueError("CP2K version mismatch while building physical identity")
    if str(values.get("build_hash", "")) != runtime.build_hash:
        raise ValueError("CP2K build mismatch while building physical identity")
    parameters = values.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("CP2K protocol parameters must be a mapping")
    data_dir = Path(runtime.data_dir)
    for name in (
        "basis_set_file",
        "pseudopotential_file",
        "dispersion_parameter_file",
    ):
        value = parameters.get(name)
        if value is None:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = data_dir / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"CP2K protocol file is missing: {path}")
        parameters[name] = {
            "name": path.name,
            "sha256": _sha256(path),
        }
    base_id = str(values["protocol_id"])
    values["protocol_id"] = protocol_id or f"{base_id}:q{charge}:m{multiplicity}"
    values["charge"] = charge
    values["multiplicity"] = multiplicity
    values["source_revision"] = runtime.source_revision
    return values


def physical_protocol_identity(
    document: Mapping[str, Any],
    runtime: CP2KRuntimeIdentity,
    *,
    charge: int,
    multiplicity: int,
    protocol_id: str | None = None,
) -> str:
    """Hash the path-independent physical CP2K protocol record."""

    record = physical_protocol_record(
        document,
        runtime,
        charge=charge,
        multiplicity=multiplicity,
        protocol_id=protocol_id,
    )
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def protocol_document_sha256(path: str | Path) -> str:
    return _sha256(Path(path).expanduser().resolve())
