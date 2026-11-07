"""Serialized, bounded generation actions with durable pre-forward accounting."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from xtbflow.governance.generation_bridge import validate_generation_config, validate_journal, validate_query
from xtbflow.runtime import BudgetExceeded, LedgerEvent, RunLedger, StageBudget
from xtbflow.evaluation.discovery import COST_AXES, check_hash
from xtbflow.validation.certification import CertificationCandidate, _directory_bytes, file_hash, identity, write_json


def artifact_index(folder: Path) -> list[dict]:
    paths = [p for p in folder.rglob("*") if p.is_file() and p.name not in {"action-result.json", "action-settlement.json"}]
    if any(p.is_symlink() for p in paths):
        raise ValueError("generation artifact symlink")
    return [dict(path=str(p.relative_to(folder)), bytes=p.stat().st_size, sha256=file_hash(p)) for p in sorted(paths)]


class GenerationActions:
    def __init__(self, root: Path, budget: StageBudget, *, source: Path, research_source: Path,
                 model_root: Path, template: dict):
        self.root, self.budget = Path(root).resolve(), budget
        self.source, self.research_source, self.model_root = Path(source).resolve(), Path(research_source).resolve(), Path(model_root).resolve()
        self.template = deepcopy(template)
        validate_generation_config(template)
        if (any(v is None for v in (budget.max_gpu_hours, budget.max_cpu_core_hours, budget.max_system_evaluations,
                                    budget.max_calculator_calls, budget.max_storage_gb))
                or budget.max_concurrent_jobs != 1 or budget.max_retries_per_job != 0 or budget.max_wall_minutes_per_job > 12):
            raise ValueError("finite generation stage, <=12 minutes, one worker and no automatic retry required")
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.root / "ledger.json"

    @property
    def identity(self) -> dict:
        files = ["scripts/research_generation_worker.py", "scripts/run_research_generation.py",
                 "src/xtbflow/governance/generation_bridge.py", "src/xtbflow/runtime/generation_actions.py",
                 "src/xtbflow/runtime/ledger.py"]
        packages = {}
        for name in ("torch", "numpy"):
            dist = importlib.metadata.distribution(name)
            records = [p for p in (dist.files or []) if str(p).endswith(".dist-info/RECORD")]
            if len(records) != 1:
                raise ValueError("generation package build identity unavailable")
            packages[name] = dict(version=dist.version, record_sha256=file_hash(Path(dist.locate_file(records[0]))))
        return dict(schema="xtbflow-generation-action-service/v1", template_sha256=identity(self.template), packages=packages,
                    sources={p: file_hash(self.source / p) for p in files}, stage_budget=asdict(self.budget))

    def _ledger(self) -> RunLedger:
        ledger = RunLedger.load(self.ledger_path) if self.ledger_path.exists() else RunLedger(self.budget)
        if ledger.budget != self.budget:
            raise ValueError("generation stage budget changed")
        return ledger

    def _configuration(self, action: dict) -> dict:
        if action["name"] not in {"propose", "resample_event", "resample_geometry"}:
            raise ValueError("unavailable generator action")
        params = action["parameters"]
        if set(params) != {"sampler", "sampling_seed"} or params["sampler"] not in {"joint_event_lead2", "independent_serial"}:
            raise ValueError("unregistered sampler parameters")
        cfg = deepcopy(self.template)
        method = "conditioned_geometry" if action["name"] == "resample_geometry" else (
            "joint_generation" if params["sampler"] == "joint_event_lead2" else "independent_event_then_geometry")
        cfg.update(methods=[method], query_count=1, proposals_per_query=1, sampling_seed=params["sampling_seed"],
                   candidate_namespace=action["action_id"])
        cfg["generation"].update(max_forward_evaluations=100 if method == "independent_event_then_geometry" else 50,
                                 max_wall_seconds=120)
        validate_generation_config(cfg)
        return cfg

    def _check_envelope(self, ledger: RunLedger, cfg: dict) -> None:
        g, totals = cfg["generation"], ledger.summary()["totals"]
        wall = g["max_wall_seconds"] + 10
        if wall > self.budget.max_wall_minutes_per_job * 60:
            raise BudgetExceeded("generation envelope exceeds per-job time cap")
        requirements = dict(gpu_hours=wall / 3600, cpu_core_hours=wall * (g["cpu_cores"] + 1) / 3600,
                            system_evaluations=g["max_forward_evaluations"],
                            storage_gb=(2 * g["max_output_bytes"] + 1024**2) / 1024**3)
        for name, value in requirements.items():
            if totals[name] + value > getattr(self.budget, "max_" + name):
                raise BudgetExceeded(f"insufficient complete generation envelope: {name}")

    def _execute(self, folder: Path, cfg: dict, conditioned: bool) -> None:
        cmd = [sys.executable, str(self.source / "scripts/run_research_generation.py"),
               "--config", str(folder / "config.json"), "--reactants", str(folder / "reactants.json"),
               "--research-source", str(self.research_source), "--model-root", str(self.model_root),
               "--out", str(folder / "generation")]
        if conditioned:
            cmd += ["--conditioned-events", str(folder / "conditioned-events.json")]
        env = dict(os.environ, XTBFlow_PARENT_PID=str(os.getpid()), PYTHONDONTWRITEBYTECODE="1")
        started = time.monotonic()
        with (folder / "runner.log").open("wb") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                while process.poll() is None:
                    if time.monotonic() - started > cfg["generation"]["max_wall_seconds"] + 10:
                        raise TimeoutError("generation action wall exhausted")
                    if _directory_bytes(folder) > 2 * cfg["generation"]["max_output_bytes"] + 1024**2:
                        raise BudgetExceeded("generation action storage exhausted")
                    time.sleep(.05)
            finally:
                if process.poll() is None:
                    process.kill()  # Runner/worker parent-death signals terminate descendants.
                process.wait()
            if process.returncode:
                raise RuntimeError("generation runner failed; retained report/journal are authoritative")

    @staticmethod
    def _cpu() -> float:
        parent, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        return parent.ru_utime + parent.ru_stime + children.ru_utime + children.ru_stime

    def _record(self, folder: Path, request: dict, status: str, wall: float, cpu: float, basis: str) -> dict:
        generation, cfg = folder / "generation", request["config"]
        journal_path = generation / "raw/calls.json"
        counter_basis = "durable_forward_journal"
        try:
            journal = json.loads(journal_path.read_text())
            calls = validate_journal(journal, cap=cfg["generation"]["max_forward_evaluations"])
            if any(r["query_id"] != request["query"]["query_id"] or r["method"] != cfg["methods"][0] or r["proposal"] != 0 for r in journal["calls"]):
                raise ValueError("generation journal context differs from the action")
        except (OSError, ValueError, KeyError, TypeError):
            # Never turn a missing/corrupt pre-call journal into free work, and
            # never manufacture individual journal rows. Retain a labeled cap.
            counter_basis = "conservative_unresolved_forward_envelope"
            journal = {"calls": []}
            roles = (["event", "geometry"] if cfg["methods"] == ["independent_event_then_geometry"] else
                     ["geometry"] if cfg["methods"] == ["conditioned_geometry"] else ["joint"])
            calls = dict(forward_evaluations=cfg["generation"]["max_forward_evaluations"],
                kernel_invocations=None, by_role={r: cfg["steps_per_branch"] for r in roles}, unfinished_calls=None)
            if status == "generation_completed":
                status = "accounting_unresolved"
        candidates, report_hash, plan_hash = [], None, None
        if status == "generation_completed":
            report = json.loads((generation / "report.json").read_text())
            if report["status"] != status or report["settlement_status"] != "settled" or report["config_sha256"] != file_hash(folder / "config.json"):
                raise ValueError("generation report incomplete or config changed")
            if report["call_summary"] != calls or calls["unfinished_calls"]:
                raise ValueError("generation primitive accounting differs")
            for artifact in report["artifacts"]:
                p = generation / artifact["path"]
                if not p.resolve().is_relative_to(generation.resolve()) or p.is_symlink() or p.stat().st_size != artifact["bytes"] or file_hash(p) != artifact["sha256"]:
                    raise ValueError("generation artifact changed")
            inner = RunLedger.load(generation / "ledger.json")
            if inner.summary() != report["ledger"] or inner.reservations:
                raise ValueError("generation inner ledger does not reconcile")
            plan_hash = file_hash(generation / "raw/certification-plan.json")
            plan = json.loads((generation / "raw/certification-plan.json").read_text())
            if plan_hash != report["certification_plan_sha256"] or plan["generator_journal_sha256"] != file_hash(journal_path):
                raise ValueError("generation handoff changed")
            expected = [r for r in report["candidates"] if r["status"] == "generated"]
            if len(expected) != len(plan["candidates"]) or len(expected) != report["certification_candidates"]:
                raise ValueError("generation handoff dropped candidates")
            for row, item in zip(expected, plan["candidates"]):
                c = CertificationCandidate(**item["candidate"])
                if identity(c.content()) != row["candidate_content_sha256"] or c.provenance["generation_protocol_sha256"] != identity(plan["generation_protocol"]):
                    raise ValueError("generation candidate identity changed")
                if request["conditioned_be"] is not None and c.planned_product_be != request["conditioned_be"]:
                    raise ValueError("geometry resampling changed the generated event")
                candidates.append(asdict(c))
            report_hash = file_hash(generation / "report.json")
        artifacts = artifact_index(folder)
        cost = dict.fromkeys(COST_AXES, 0)
        cost.update(raw_proposals=(1 if counter_basis != "durable_forward_journal" else
                    len({(r["query_id"], r["method"], r["proposal"]) for r in journal["calls"]})),
                    wall_seconds=wall, cpu_seconds=cpu, gpu_seconds=wall,
                    storage_bytes=sum(r["bytes"] for r in artifacts))
        cost.update({role + "_forwards": calls["by_role"].get(role, 0) for role in ("joint", "event", "geometry")})
        record = dict(schema="xtbflow-generation-action-result/v1", action_id=request["action"]["action_id"],
            request_sha256=file_hash(folder / "request.json"), status=status, cost=cost, cost_basis=basis, counter_basis=counter_basis,
            calls=calls, candidates=candidates, artifacts=artifacts,
            generation_report_sha256=report_hash, certification_plan_sha256=plan_hash,
            generation_directory="generation", generation_config="config.json")
        event_id = record["action_id"]
        record["ledger_event"] = asdict(LedgerEvent(run_id=event_id, phase=self.budget.phase, calculator="trained_generator",
            protocol_id="xtbflow-generation-action/v1", calculator_build_hash=identity(self.identity),
            system_id=event_id, input_hash=record["request_sha256"], operation="other",
            status="success" if status == "generation_completed" else "failure", event_id=event_id,
            calculator_calls=0, system_evaluations=calls["forward_evaluations"], cpu_seconds=cpu, cpu_cores=1,
            gpu_seconds=wall, storage_bytes=cost["storage_bytes"]))
        record["content_sha256"] = identity(record)
        return record

    def _settle(self, folder: Path, record: dict, ledger: RunLedger) -> None:
        event_id = record["action_id"]
        existing = [e for e in ledger.events if e.event_id == event_id]
        if not existing:
            ledger.append(LedgerEvent(**record["ledger_event"]), persist_path=self.ledger_path)
        elif len(existing) != 1 or asdict(existing[0]) != record["ledger_event"]:
            raise ValueError("generation settlement identity changed")
        write_json(folder / "action-settlement.json", dict(status="settled", result_sha256=file_hash(folder / "action-result.json")))

    def _load(self, folder: Path) -> dict:
        record = json.loads((folder / "action-result.json").read_text())
        if (identity({k: v for k, v in record.items() if k != "content_sha256"}) != record["content_sha256"]
                or file_hash(folder / "request.json") != record["request_sha256"] or artifact_index(folder) != record["artifacts"]):
            raise ValueError("generation cache identity changed")
        return record

    def run(self, action: dict, query: dict, conditioned_be: list | None = None) -> dict:
        check_hash(action["action_id"])
        if identity({k: v for k, v in action.items() if k != "action_id"}) != action["action_id"]:
            raise ValueError("generation action hash changed")
        validate_query(query)
        cfg = self._configuration(action)
        conditioned = action["name"] == "resample_geometry"
        if conditioned != (conditioned_be is not None):
            raise ValueError("geometry resampling requires an explicit generated event")
        if conditioned:
            payload = asdict(validate_query(query))
            payload["planned_product_be"] = conditioned_be
            CertificationCandidate(**payload)
        request = dict(action=action, query=query, conditioned_be=conditioned_be, config=cfg, service=self.identity)
        folder = self.root / action["action_id"]
        with (self.root / "service.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            ledger = self._ledger()
            for prior in sorted(self.root.glob("*/request.json")):
                work = prior.parent
                saved = json.loads(prior.read_text())
                if saved["service"] != self.identity:
                    raise ValueError("pending generation service identity changed")
                if not (work / "action-result.json").exists():
                    limit = saved["config"]["generation"]
                    wall = limit["max_wall_seconds"] + 10
                    record = self._record(work, saved, "execution_interrupted", wall,
                                          wall * (limit["cpu_cores"] + 1), "conservative_interrupted_envelope")
                    write_json(work / "action-result.json", record)
                self._settle(work, self._load(work), ledger)
            if (folder / "request.json").exists():
                if json.loads((folder / "request.json").read_text()) != request:
                    raise ValueError("action ID reused with different generation inputs")
                return self._load(folder)
            self._check_envelope(ledger, cfg)
            folder.mkdir()
            write_json(folder / "request.json", request)
            write_json(folder / "config.json", cfg)
            write_json(folder / "reactants.json", {"queries": [query]})
            if conditioned:
                write_json(folder / "conditioned-events.json", {query["query_id"]: conditioned_be})
            started, cpu_started = time.monotonic(), self._cpu()
            status = "generation_completed"
            try:
                self._execute(folder, cfg, conditioned)
            except TimeoutError:
                status = "execution_timeout"
            except BudgetExceeded:
                status = "budget_exhausted"
            except Exception:
                status = "execution_failed"
            wall, cpu = time.monotonic() - started, self._cpu() - cpu_started
            try:
                record = self._record(folder, request, status, wall, cpu, "measured_action_including_runner")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                write_json(folder / "action-validation-failure.json", dict(status="report_validation_failed", error_type=type(exc).__name__))
                record = self._record(folder, request, "report_validation_failed", wall, cpu, "measured_action_including_runner")
            try:
                write_json(folder / "action-result.json", record)
            except OSError as exc:
                write_json(folder / "action-report-failure.json", dict(status="report_write_failed", error_type=type(exc).__name__))
                raise
            try:
                self._settle(folder, record, ledger)
            except Exception as exc:
                write_json(folder / "action-settlement.json", dict(status="settlement_failed", error_type=type(exc).__name__))
                raise
            return record
