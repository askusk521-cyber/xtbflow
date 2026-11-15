#!/usr/bin/env python3
"""Bounded, standard-library-only audit of the pinned related-work sources.

Input: code-only clones, not molecular data. Output: a new JSON receipt.
No source import, dataset loading, training, calculator call or network access.
Maximum source file size: 4 MiB; maximum total inspected bytes: 32 MiB.
This verifies source observations, not paper reproduction or scientific efficacy.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import time


FILES = {
    "geom": ["README.md", "tutorials/02_loading_rdkit_mols.ipynb", "tutorials/03_loading_molecule_net.ipynb"],
    "hippynn": ["examples/multifidelity_ani1x_training.py", "hippynn/databases/database.py", "hippynn/databases/h5_pyanitools.py", "LICENSE.txt"],
    "PICPS-ML4Sci": ["preprocessing/ani-1x_new.py", "mdsim/trainers/trainer.py", "configs/ani-1x/base_piwsl.yml", "README.md"],
    "EnFlow": ["scripts/prepare_data.py", "enflow/data/dataset.py", "enflow/models/model.py", "env.yml", "README.md", "LICENSE"],
    "ETFlow": ["scripts/prepare_data.py", "etflow/data/dataset.py", "README.md", "LICENSE"],
    "qml": ["setup.py", "test/test_fchl_force.py", "LICENSE"],
    "ANI1x_datasets": ["dataloader.py", "README.md", "LICENSE"],
}


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, timeout=15).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite a prior receipt")
    started = time.monotonic()
    cpu_start = time.process_time()
    manifest = json.loads(args.manifest.read_text())
    pins = {r["repository"].split("/")[-1]: r["sha"] for r in manifest if r["status"] == "cloned"}
    texts: dict[str, str] = {}
    files = []
    repositories = []
    byte_count = 0
    for name, paths in FILES.items():
        repo = args.repos / name
        sha = git(repo, "rev-parse", "HEAD")
        if sha != pins[name] or git(repo, "status", "--porcelain"):
            raise ValueError(f"Changed source checkout: {name}")
        repositories.append({"name": name, "sha": sha, "clean": True})
        for relative in paths:
            path = repo / relative
            if path.stat().st_size > 4 * 1024**2:
                raise ValueError("Per-file input budget exceeded")
            raw = path.read_bytes()
            byte_count += len(raw)
            if byte_count > 32 * 1024**2:
                raise ValueError("Total input budget exceeded")
            key = f"{name}/{relative}"
            texts[key] = raw.decode("utf-8")
            files.append({"path": key, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})

    # Bind an AST-derived signature without importing or executing upstream code.
    db = ast.parse(texts["hippynn/hippynn/databases/database.py"])
    method = next(n for n in ast.walk(db) if isinstance(n, ast.FunctionDef) and n.name == "make_trainvalidtest_split")
    script = ast.parse(texts["hippynn/examples/multifidelity_ani1x_training.py"])
    call = next(n for n in ast.walk(script) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == method.name)
    parameters = [inspect.Parameter(a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for a in method.args.args]
    parameters += [inspect.Parameter(a.arg, inspect.Parameter.KEYWORD_ONLY) for a in method.args.kwonlyargs]
    signature = inspect.Signature(parameters)
    try:
        signature.bind(object(), *[ast.literal_eval(a) for a in call.args], **{k.arg: ast.literal_eval(k.value) for k in call.keywords})
        binding = "compatible"
    except TypeError as exc:
        binding = str(exc)

    prep = ast.parse(texts["PICPS-ML4Sci/preprocessing/ani-1x_new.py"])
    sizes = next(ast.literal_eval(n.value) for n in prep.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "Data_Size" for t in n.targets))
    option = next(n for n in ast.walk(prep) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--size")
    default = next(ast.literal_eval(k.value) for k in option.keywords if k.arg == "default")

    def locate(key: str, needle: str) -> list[int]:
        return [i for i, line in enumerate(texts[key].splitlines(), 1) if needle in line]

    observations = [
        {"id": "hippynn_split_api", "status": "confirmed_static", "caller_line": call.lineno, "definition_line": method.lineno, "signature": str(signature), "binding_result": binding},
        {"id": "hippynn_complete_case_filter", "status": "confirmed_static", "lines": locate("hippynn/examples/multifidelity_ani1x_training.py", "both_found"), "meaning": "Example retains intersection including CC energy; not a general partial-label loader."},
        {"id": "picps_default_size", "status": "confirmed_static", "line": option.lineno, "default": default, "allowed": list(sizes), "default_exists": default in sizes},
        {"id": "picps_perturbation_scale", "status": "source_observation_intent_unverified", "first_scale_lines": locate("PICPS-ML4Sci/mdsim/trainers/trainer.py", "dirc1 = mean_bond"), "meaning": "Random dirc1 is scaled before and after centering; dirc2 only after centering. Check intended amplitudes before reproduction."},
        {"id": "picps_variable_atoms", "status": "adaptation_required", "lines": locate("PICPS-ML4Sci/mdsim/trainers/trainer.py", "reshape(nb, -1, 3)"), "meaning": "Audit molecule segmentation before applying this loss to variable-size batches."},
        {"id": "enflow_energy_normalization", "status": "confirmed_static", "lines": locate("EnFlow/enflow/data/dataset.py", "energies = (energies - mean)"), "meaning": "Per-molecule standardization; score gradients are not automatically physical forces."},
        {"id": "enflow_gradient_target", "status": "confirmed_static", "lines": locate("EnFlow/enflow/models/model.py", "l_em = batchwise_l2_loss(-dEdx, s_t"), "meaning": "Gradient matching uses flow displacement target s_t, not reference QC forces."},
        {"id": "etflow_training_energy_route", "status": "confirmed_static", "get_reads_data_energy": "data.energy" in texts["ETFlow/etflow/data/dataset.py"], "meaning": "Preprocessor retains energy; this dataset getter reconstructs a training graph without that field."},
        {"id": "ani_complete_case_filter", "status": "confirmed_static", "lines": locate("ANI1x_datasets/dataloader.py", "mask = mask &"), "meaning": "Requesting extra sparse properties can drop otherwise useful configurations."},
        {"id": "qml_build_dependency", "status": "compatibility_unexecuted", "lines": locate("qml/setup.py", "numpy.distutils"), "meaning": "Legacy compiled Fortran build; install in a separate compatible environment if used."},
    ]
    report = {
        "schema": "xtbflow-related-source-audit/v1", "status": "completed_static_audit",
        "environment": {"python": platform.python_version(), "system": platform.system(), "location": "local_development_host"},
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repositories": repositories, "files": files, "observations": observations,
        "resources": {"input_bytes": byte_count, "elapsed_seconds": time.monotonic() - started, "process_cpu_seconds": time.process_time() - cpu_start, "qc_calls": 0, "training_runs": 0, "dataset_reads": 0, "network_calls": 0},
        "claim_limits": ["No external package imported or tested.", "No public molecular dataset downloaded or parsed.", "No physical accuracy, learning benefit or paper reproduction established.", "Findings apply only to the recorded source commits."],
    }
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"repositories": len(repositories), "files": len(files), "api_binding": binding, "picps_size_exists": default in sizes, "status": report["status"]}))


if __name__ == "__main__":
    main()
