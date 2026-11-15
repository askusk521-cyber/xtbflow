#!/usr/bin/env bash
# Read-only metadata probe. No native import, GPU request, QC or dataset access.
# Remote cwd is the login home; no remote project checkout is executed.
# Source identity is this script's SHA-256, plus the declared GQC source/library.
# Limits: 75 s outer wall, 60 s Python, 2 GiB library input, no retry.
# Redirect stdout to a NEW local JSON file with shell noclobber enabled.
set -euo pipefail
timeout 75s ssh -o BatchMode=yes -o ConnectTimeout=15 n2 '
  set -e
  cd "$HOME"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate xtbflow-qc
  source ~/.config/n2-qc/runtime.env
  export GQC_SOURCE_SHA
  timeout 60s "$GQC_PYTHON" -
' <<'PY'
import datetime
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
import time

started = time.monotonic()
cpu = time.process_time()
package = Path(sysconfig.get_paths()["purelib"]) / "generativeqc"
libraries = list(package.rglob("libgenerativeqc.so"))
if not libraries:
    raise RuntimeError("No expected native library; no repair attempted")
if sum(p.stat().st_size for p in libraries) > 2 * 1024**3:
    raise RuntimeError("Read budget exceeded")
versions = {}
for name in ("generativeqc", "numpy", "scipy", "pyscf"):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = "not_installed_in_this_interpreter"
identities = []
for path in libraries:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    identities.append({"basename": path.name, "bytes": path.stat().st_size, "sha256": digest})
print(json.dumps({
    "schema": "xtbflow-runtime-metadata-recheck/v1",
    "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "host": platform.node(), "remote_cwd": "login_home",
    "remote_project_checkout_executed": False,
    "python": sys.version.split()[0], "versions": versions,
    "declared_source_sha": os.environ["GQC_SOURCE_SHA"], "libraries": identities,
    "status": "read_only_metadata_no_native_import",
    "resources": {"qc_calls": 0, "training_runs": 0, "gpu_jobs_submitted": 0,
                  "input_bytes": sum(x["bytes"] for x in identities),
                  "elapsed_seconds": time.monotonic() - started,
                  "process_cpu_seconds": time.process_time() - cpu},
    "claim_limits": ["No new numerical qualification.",
                     "Configured source SHA is declared, not independently rebuilt."]
}, indent=2))
PY
