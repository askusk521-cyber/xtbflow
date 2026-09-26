"""Read-only runtime capability probes used before scheduling a run."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from typing import Any, Iterable


@dataclass(frozen=True)
class Capability:
    """One observed capability; unknown is kept distinct from failure."""

    name: str
    status: str
    detail: str
    version: str | None = None


def _command(name: str) -> Capability:
    path = shutil.which(name)
    if path is None:
        return Capability(name, "unavailable", "command not found")
    try:
        output = subprocess.run([path, "--version"], check=False, capture_output=True, text=True, timeout=5)
        detail = (output.stdout or output.stderr).splitlines()[0] if (output.stdout or output.stderr) else path
    except (OSError, subprocess.SubprocessError) as exc:
        return Capability(name, "unknown", str(exc), path)
    return Capability(name, "pass", detail, path)


def _module(name: str) -> Capability:
    spec = importlib.util.find_spec(name)
    return Capability(f"python:{name}", "pass" if spec else "unavailable", str(spec.origin) if spec else "module not found")


def collect_capabilities(commands: Iterable[str] = ("srun", "nvidia-smi", "cmake", "ninja"), modules: Iterable[str] = ("numpy", "torch")) -> list[Capability]:
    """Collect only facts observed locally; do not infer missing calculators."""

    return [_command(name) for name in commands] + [_module(name) for name in modules]


def probe_environment(*, commands: Iterable[str] = ("srun", "nvidia-smi", "cmake", "ninja"), modules: Iterable[str] = ("numpy", "torch")) -> dict[str, Any]:
    """Return a serializable, read-only host report suitable for a run manifest."""

    return {
        "schema": "xtbflow-runtime-probe/v1",
        "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "capabilities": [asdict(item) for item in collect_capabilities(commands, modules)],
    }


def write_probe_report(path: Path, report: dict[str, Any]) -> None:
    """Write a probe report without embedding credentials or private paths."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(__import__("json").dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
