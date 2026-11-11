"""Durable candidate certification with shared hard call budgets and feedback.

One service directory is serialized with an OS lock. Results are durable before
token settlement; recovery never silently re-executes an interrupted attempt.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import stat
import tempfile
import time
from typing import Any, Protocol

from xtbflow.runtime import BudgetExceeded, CalculatorCallToken, LedgerEvent, RunLedger, StageBudget


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomic durable replacement; the caller owns the directory lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class CertificationCandidate:
    """Mapped fixed atom pool. Unknown planned BE is explicit None, not a no-edit prediction."""

    candidate_id: str
    parent_id: str
    atomic_numbers: tuple[int, ...]
    coordinates_angstrom: tuple[tuple[float, float, float], ...]
    atom_ids: tuple[str, ...]
    reactant_be: tuple[tuple[int, ...], ...]
    planned_product_be: tuple[tuple[int, ...], ...] | None
    reactant_automorphisms: tuple[tuple[int, ...], ...]
    charge: int
    multiplicity: int
    environment: dict
    provenance: dict

    def __post_init__(self) -> None:
        n = len(self.atomic_numbers)
        if not 2 <= n <= 64 or any(type(z) is not int or z not in {1, 6, 7, 8} for z in self.atomic_numbers):
            raise ValueError("reference service scope is 2..64 explicit CHNO atoms")
        if type(self.charge) is not int or type(self.multiplicity) is not int or (self.charge, self.multiplicity) != (0, 1):
            raise ValueError("reference service requires explicit neutral singlet state")
        if self.environment != {"kind": "gas", "periodic": False}:
            raise ValueError("unsupported or unknown environment")
        if not self.candidate_id or not self.parent_id or len(self.atom_ids) != n or len(set(self.atom_ids)) != n:
            raise ValueError("unique atom identities and candidate/parent identity are required")
        if any(not isinstance(atom, str) or not atom for atom in self.atom_ids):
            raise ValueError("atom identities must be nonempty strings")
        if len(self.coordinates_angstrom) != n or any(len(row) != 3 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row)
                for row in self.coordinates_angstrom):
            raise ValueError("finite N x 3 Angstrom coordinates required")
        if self.reactant_be is None:
            raise ValueError("mapped reactant BE is required even without an event proposal")
        matrices = (self.reactant_be,) if self.planned_product_be is None else (self.reactant_be, self.planned_product_be)
        for matrix in matrices:
            if len(matrix) != n or any(len(row) != n for row in matrix):
                raise ValueError("BE shape must match the fixed atom pool")
            if any(type(v) is not int or v < 0 for row in matrix for v in row):
                raise ValueError("BE entries must be nonnegative integers")
            if any(matrix[i][j] != matrix[j][i] for i in range(n) for j in range(n)):
                raise ValueError("BE must be symmetric")
        valence = {1: 1, 6: 4, 7: 5, 8: 6}
        electrons = sum(valence[z] for z in self.atomic_numbers)
        if any(sum(map(sum, b)) != electrons for b in matrices):
            raise ValueError("BE electron inventory must match the declared atom pool and charge")
        if not 1 <= len(self.reactant_automorphisms) <= 2048:
            raise ValueError("explicit bounded reactant automorphisms required")
        for permutation in self.reactant_automorphisms:
            if any(type(i) is not int for i in permutation) or sorted(permutation) != list(range(n)):
                raise ValueError("invalid atom permutation")
            if any(self.atomic_numbers[i] != self.atomic_numbers[permutation[i]] for i in range(n)) or any(
                    self.reactant_be[i][j] != self.reactant_be[permutation[i]][permutation[j]]
                    for i in range(n) for j in range(n)):
                raise ValueError("permutation does not preserve the reactant")
        for key in ("input_sha256", "atom_mapping_evidence", "electronic_state_evidence"):
            if not self.provenance.get(key):
                raise ValueError("explicit input, mapping and electronic-state provenance required")
        if len(self.provenance["input_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in self.provenance["input_sha256"]):
            raise ValueError("input_sha256 must be a SHA256")
        identity(asdict(self))

    def content(self) -> dict:
        # Cosmetic IDs do not cause duplicate work; parent and mapped content do.
        result = asdict(self)
        result.pop("candidate_id")
        return result


class Backend(Protocol):
    @property
    def identity(self) -> dict: ...

    def run(self, candidate: CertificationCandidate, meter: "CallMeter", work: Path) -> dict: ...


class CallMeter:
    """Charge before each SCF, gradient or Hessian, including failed kernels."""

    def __init__(self, token: CalculatorCallToken, work: Path, max_wall_seconds: float) -> None:
        self.token, self.work = token, work
        self.started = time.monotonic()
        self.max_wall_seconds = max_wall_seconds
        self.rows: list[dict] = []
        self.attempts: list[dict] = []

    def before(self, stage: str, kind: str) -> None:
        if kind not in {"scf", "gradient", "hessian"}:
            raise ValueError("unsupported primitive")
        if time.monotonic() - self.started >= self.max_wall_seconds:
            raise BudgetExceeded("candidate_wall_budget_exhausted")
        self.token.consume()
        self.attempts.append(dict(stage=stage, kind=kind, status="started", sequence=self.token.consumed_calls))
        self.flush()

    def record(self, stage: str, kind: str, seconds: float, **extra: Any) -> None:
        if kind != "geometric":
            if not self.attempts or (self.attempts[-1]["stage"], self.attempts[-1]["kind"], self.attempts[-1]["status"]) != (stage, kind, "started"):
                raise ValueError("primitive completion has no matching pre-call charge")
            self.attempts[-1]["status"] = "completed"
        self.rows.append(dict(stage=stage, kind=kind, wall_s=seconds, **extra))
        self.flush()

    def flush(self) -> None:
        write_json(self.work / "calls.json", dict(attempts=self.attempts, timings=self.rows,
                   consumed_calls=self.token.consumed_calls, elapsed_wall_s=time.monotonic() - self.started))

    def checkpoint(self, evidence: dict) -> None:
        write_json(self.work / "partial.json", evidence)


def stage_diagnostics(raw: dict) -> dict:
    """Versioned read-only projection; never changes the physical verdict.

    Status strings are observed protocol facts. Missing stages stay unknown;
    a later graph summary cannot erase an earlier explicit numerical failure.
    """
    ts, irc, ends, graph = (raw.get('ts', {}), raw.get('irc', {}),
                            raw.get('endpoints', []), raw.get('identity', {}))
    stages = [dict(stage='ts', status=ts.get('status'), reason=ts.get('reason')),
              dict(stage='irc', status=irc.get('status'), reason=irc.get('reason'))]
    stages.extend(dict(stage=f'endpoint_{i}', status=e.get('status'), reason=e.get('reason'))
                  for i, e in enumerate(ends))
    stages.append(dict(stage='graph_identity', status=graph.get('status'), reason=graph.get('reason')))
    terminal, status, reason = 'unknown', None, None
    for row in stages:
        if row['status'] is None:
            continue
        terminal, status, reason = row['stage'], row['status'], row['reason'] or row['status']
        expected = 'TS_OPTFREQ_PASS' if terminal == 'ts' else 'IRC_COMPLETE' if terminal == 'irc' else 'MINIMUM_PASS'
        if terminal == 'graph_identity' or status != expected:
            break
    if terminal == 'ts':
        if any(term in status for term in ('MODE', 'FREQUENCY', 'SADDLE')):
            terminal = 'ts_modes'
        elif 'GRADIENT' in status:
            terminal = 'ts_gradient'
        elif 'OPT' in status and status != 'TS_OPTFREQ_PASS':
            terminal = 'ts_optimization'
    if raw.get('failure') and (terminal == 'unknown' or status in {'TS_OPTFREQ_PASS', 'IRC_COMPLETE', 'MINIMUM_PASS'}):
        terminal, reason = 'execution', raw['failure']
    return dict(projection_schema='xtbflow-stage-projection/v2', terminal_stage=terminal,
                terminal_stage_status=status, stage_reason=reason, stages=stages,
                execution_failure=raw.get('failure'), raw_evidence_sha256=identity(raw))


def feedback(raw: dict) -> dict:
    """Layered observed evidence; neither predictions nor TS distances define endpoints."""
    ts, irc, ends = raw.get("ts", {}), raw.get("irc", {}), raw.get("endpoints", [])
    ident = raw.get("identity", {})
    gradient = [v for row in ts.get("gradient_hartree_bohr", []) for v in row]
    stationary = bool(gradient) and all(math.isfinite(v) for v in gradient) and max(map(abs, gradient)) <= 1e-4 and math.sqrt(sum(v*v for v in gradient) / len(gradient)) <= 3e-5
    freq = ts.get("frequencies_cm1", [])
    mode = [v for row in ts.get("reaction_mode", []) for v in row]
    mode_valid = len(mode) == len(gradient) and bool(mode) and all(math.isfinite(v) for v in mode) and sum(v*v for v in mode) > 0
    saddle = stationary and mode_valid and bool(freq) and all(math.isfinite(v) for v in freq) and sum(v < -30 for v in freq) == 1 and sum(v < 0 for v in freq) == 1
    connected = saddle and irc.get("status") == "IRC_COMPLETE" and len(ends) == 2 and all(
        _minimum_passes(e, len(gradient)) for e in ends)
    actual = ident.get("actual_event_id")
    matched = ident.get("status") == "STRICT_JOINT_GRAPH_VALID"
    unproposed = ident.get("status") == "REFERENCE_CONNECTED_WITHOUT_PROPOSAL" and ident.get("predicted_event_id") is None
    same = ident.get("reason") == "both_ends_reactant"
    if connected and (matched or unproposed or ident.get("status") == "STRICT_EVENT_MISMATCH") and actual and ident.get("product_be") is not None and ident.get("reactant_end") in (0, 1):
        status = "reference_ts_connected" if matched or unproposed else "valid_alternative"
    elif connected and same:
        status = "same_endpoint"
    elif connected and ident.get("status") == "IRC_REACTANT_MISMATCH":
        status = "reactant_mismatch"
    else:
        status = "attempted_unresolved"
    diagnostics = stage_diagnostics(raw)
    return dict(evidence_status=status, stationary_point=stationary,
                first_order_saddle=saddle, bidirectional_minima=connected,
                planned_event_realized=matched if connected and actual and ident.get("predicted_event_id") is not None else None,
                planned_event_id=ident.get("predicted_event_id"), observed_event_id=actual,
                endpoint_be=ident.get("endpoint_be"), observed_product_be=ident.get("product_be"),
                raw_status=raw.get("strict_status"),
                reason=ident.get("reason") or raw.get("failure") or diagnostics['stage_reason'],
                **diagnostics)


def _minimum_passes(endpoint: dict, size: int) -> bool:
    g = [v for row in endpoint.get("gradient_hartree_bohr", []) for v in row]
    f = endpoint.get("frequencies_cm1", [])
    return (endpoint.get("status") == "MINIMUM_PASS" and len(g) == size and bool(g)
            and all(math.isfinite(v) for v in g) and max(map(abs, g)) <= 1e-4
            and math.sqrt(sum(v*v for v in g) / len(g)) <= 3e-5
            and bool(f) and all(math.isfinite(v) and v >= 0 for v in f))


def _artifact_index(work: Path) -> list[dict]:
    return [dict(path=str(p.relative_to(work)), sha256=file_hash(p), bytes=p.stat().st_size)
            for p in sorted((work / "raw").rglob("*")) if p.is_file() and not p.is_symlink()]


def _directory_bytes(folder: Path) -> int:
    total = 0
    for path in folder.rglob("*"):
        # Atomic journal replacement can remove a temporary file between the
        # directory listing and lstat. A vanished entry occupies no space now.
        # Use one metadata read and do not follow symlinks or hide other errors.
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


def _worker(backend: Backend, candidate: CertificationCandidate, request: dict,
            ledger_path: Path, raw_dir: Path) -> None:
    """Fresh interpreter: CUDA is never initialized in the service parent."""
    cores = request["cpu_cores"]
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[:cores])
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(cores)
    # A fresh worker spends bounded CPU on interpreter/import startup before
    # the backend's wall clock begins. Reserve the same five-second envelope
    # used by the parent admission check so a valid low-core fixture is not
    # killed by RLIMIT_CPU before its first durable primitive charge.
    cpu_limit = math.ceil((request["max_wall_seconds"] + 5) * cores)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
    size_limit = request["max_raw_bytes"]
    resource.setrlimit(resource.RLIMIT_FSIZE, (size_limit, size_limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with (raw_dir / "process.log").open("a") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    ledger = RunLedger.load(ledger_path)
    token = CalculatorCallToken(ledger, request["token_id"], persist_path=ledger_path)
    meter = CallMeter(token, raw_dir, request["max_wall_seconds"])
    try:
        raw = backend.run(candidate, meter, raw_dir)
        status = "completed"
    except BudgetExceeded as exc:
        raw = CertificationService._partial(raw_dir)
        raw["failure"] = str(exc)
        status = "budget_exhausted"
    except Exception as exc:
        raw = CertificationService._partial(raw_dir)
        raw["failure"] = f"{type(exc).__name__}: {exc}"
        status = "execution_failed"
    write_json(raw_dir / "backend-outcome.json", dict(evidence=raw, status=status))


class CertificationService:
    def __init__(self, root: Path, budget: StageBudget) -> None:
        self.root, self.budget = Path(root).resolve(), budget
        if any(v is None for v in (budget.max_calculator_calls, budget.max_gpu_hours,
                                  budget.max_cpu_core_hours, budget.max_storage_gb)) or budget.max_wall_minutes_per_job > 120 or budget.max_concurrent_jobs != 1:
            raise ValueError("finite calls/time/storage, <=120 minute jobs and one serialized worker required")
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.root / "ledger.json"

    @contextmanager
    def _locked(self):
        with (self.root / "service.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _ledger(self) -> RunLedger:
        ledger = RunLedger.load(self.ledger_path) if self.ledger_path.exists() else RunLedger(self.budget)
        if ledger.budget != self.budget:
            raise ValueError("existing stage budget mismatch")
        return ledger

    def _settle(self, ledger: RunLedger, record: dict, work: Path) -> bool:
        token_id = record["token_id"]
        if token_id in ledger.reservations:
            CalculatorCallToken(ledger, token_id, persist_path=self.ledger_path).commit(recorded_calls=record["charged_calls"])
        elif ledger.committed_tokens.get(token_id, {}).get("consumed_calls") != record["charged_calls"]:
            if record["charged_calls"] or record["execution_status"] not in {"not_started_budget_exhausted", "not_started_error", "execution_interrupted"}:
                raise ValueError("result has no matching durable token accounting")
        if not any(event.event_id == token_id for event in ledger.events):
            ledger.append(LedgerEvent(**record["ledger_event"]), persist_path=self.ledger_path)
        write_json(work / "settlement.json", {"status": "settled", "result_sha256": file_hash(work / "result.json")})
        index_path = self.root / "channels.json"
        channels = json.loads(index_path.read_text()) if index_path.exists() else {}
        channel = record["channel_id"]
        fresh = bool(channel and channel not in channels)
        if fresh:
            channels[channel] = {"cache_key": record["cache_key"], "attempt": record["attempt"],
                                 "evidence_status": record["feedback"]["evidence_status"]}
            write_json(index_path, channels)
        return fresh

    def _finish(self, ledger: RunLedger, record: dict, work: Path) -> bool:
        try:
            return self._settle(ledger, record, work)
        except Exception as exc:
            write_json(work / "settlement.json", dict(status="settlement_failed", error_type=type(exc).__name__,
                                                       result_sha256=file_hash(work / "result.json")))
            raise

    def run(self, candidate: CertificationCandidate, backend: Backend, *, max_calls: int,
            max_wall_seconds: float, cpu_cores: int, max_raw_bytes: int,
            retry: bool = False) -> dict:
        if type(max_calls) is not int or max_calls < 1 or not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= self.budget.max_wall_minutes_per_job * 60:
            raise ValueError("invalid per-attempt resource limits")
        if type(cpu_cores) is not int or not 1 <= cpu_cores <= len(os.sched_getaffinity(0)) or type(max_raw_bytes) is not int or max_raw_bytes < 65536:
            raise ValueError("explicit available CPU cores and >=64 KiB raw artifact cap required")
        # Snapshot nested values so another caller cannot change cached content.
        candidate = CertificationCandidate(**json.loads(json.dumps(asdict(candidate))))
        backend_id = json.loads(json.dumps(backend.identity, allow_nan=False))
        key = identity(dict(candidate=candidate.content(), protocol=backend_id,
                            service_source_sha256=file_hash(Path(__file__)),
                            ledger_source_sha256=file_hash(Path(__file__).parents[1] / "runtime/ledger.py")))
        with self._locked():
            ledger = self._ledger()
            # Other interrupted candidates must be reconciled first. A new key
            # must never bypass an outstanding resource reservation.
            for other in sorted(self.root.glob("*/attempt-*")):
                if other.parent.name == key:
                    continue
                if not (other / "result.json").exists():
                    self._finish(ledger, self._interrupted(other, ledger), other)
                elif not (other / "settlement.json").exists() or json.loads((other / "settlement.json").read_text()).get("status") != "settled":
                    self._finish(ledger, self._load_record(other, other.parent.name), other)
            folder = self.root / key
            folder.mkdir(exist_ok=True)
            attempts = sorted(folder.glob("attempt-*"))
            previous = None
            if attempts:
                work = attempts[-1]
                if (work / "result.json").exists():
                    previous = self._load_record(work, key)
                else:
                    previous = self._interrupted(work, ledger)
                fresh = self._finish(ledger, previous, work)
                if not retry or previous["feedback"]["evidence_status"] in {"reference_ts_connected", "valid_alternative"}:
                    return dict(record=previous, cache_hit=True, new_calculator_calls=0, new_channel=fresh,
                                requested_candidate_id=candidate.candidate_id)
            attempt = len(attempts) + 1
            if attempt > self.budget.max_retries_per_job + 1:
                raise BudgetExceeded("candidate_retry_budget_exhausted")
            work = folder / f"attempt-{attempt:04d}"
            work.mkdir()
            token_id = f"{key}:{attempt}"
            request = dict(cache_key=key, candidate=asdict(candidate), protocol=backend_id, token_id=token_id,
                           attempt=attempt, max_calls=max_calls, max_wall_seconds=max_wall_seconds,
                           started_unix=time.time(), cpu_cores=cpu_cores, max_raw_bytes=max_raw_bytes)
            write_json(work / "request.json", request)
            started = time.monotonic()
            cpu_started = self._child_cpu()
            raw_dir = work / "raw"
            raw_dir.mkdir()
            raw, status, token = {}, "completed", None
            try:
                self._check_envelope(ledger, request)
                token = ledger.issue_calculator_token(token_id, max_calls,
                    metadata={"cache_key": key, "request_sha256": file_hash(work / "request.json")},
                    persist_path=self.ledger_path)
                raw, status = self._execute(backend, candidate, request, raw_dir)
            except BudgetExceeded as exc:
                status = "not_started_budget_exhausted" if token is None else "budget_exhausted"
                raw = self._partial(raw_dir)
                raw["failure"] = str(exc)
            except Exception as exc:
                status = "execution_failed" if token is not None else "not_started_error"
                raw = self._partial(raw_dir)
                raw["failure"] = f"{type(exc).__name__}: {exc}"
            wall, cpu = time.monotonic() - started, self._child_cpu() - cpu_started
            # The worker owns call consumption, so reload its durable accounting.
            if token is not None:
                ledger = self._ledger()
                token = CalculatorCallToken(ledger, token_id, persist_path=self.ledger_path)
            else:
                wall = cpu = 0.0
            record = self._record(request, raw, status, token.consumed_calls if token else 0, wall, cpu, work)
            # A report-write failure deliberately leaves the consumed reservation
            # intact. A later call sees interruption and settles it without rerun.
            try:
                write_json(work / "result.json", record)
            except OSError as exc:
                # Best effort; the request and charged token remain recoverable
                # even if a full filesystem also prevents this status write.
                try:
                    write_json(work / "report-failure.json", dict(status="report_write_failed", error_type=type(exc).__name__))
                except OSError:
                    pass
                raise
            fresh = self._finish(ledger, record, work)
            return dict(record=record, cache_hit=False, new_calculator_calls=record["charged_calls"], new_channel=fresh,
                        requested_candidate_id=candidate.candidate_id)

    @staticmethod
    def _load_record(work: Path, key: str) -> dict:
        record = json.loads((work / "result.json").read_text())
        payload = {k: v for k, v in record.items() if k != "content_sha256"}
        if record["content_sha256"] != identity(payload) or record["cache_key"] != key or record["artifacts"] != _artifact_index(work) or record["request_sha256"] != file_hash(work / "request.json"):
            raise ValueError("cached result identity or artifact mismatch")
        return record

    @staticmethod
    def _child_cpu() -> float:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return usage.ru_utime + usage.ru_stime

    def _check_envelope(self, ledger: RunLedger, request: dict) -> None:
        totals = ledger.summary()["totals"]
        # Five seconds cover worker termination/serialization. No other process
        # can spend this envelope while the service OS lock is held.
        wall = request["max_wall_seconds"] + 5
        requirements = {
            "gpu_hours": wall / 3600 if request["protocol"].get("device") == "gpu" else 0,
            "cpu_core_hours": wall * request["cpu_cores"] / 3600,
            "system_evaluations": 1,
            "storage_gb": (2 * request["max_raw_bytes"] + 1024**2) / 1024**3,
        }
        for name, amount in requirements.items():
            cap = getattr(self.budget, f"max_{name}")
            if cap is not None and totals[name] + amount > cap:
                raise BudgetExceeded(f"insufficient {name} for complete attempt envelope")

    def _execute(self, backend: Backend, candidate: CertificationCandidate,
                 request: dict, raw_dir: Path) -> tuple[dict, str]:
        # ``spawn`` imports the backend/test module before entering ``_worker``.
        # Set bounded BLAS/OpenMP thread counts in the parent for that startup
        # interval too; setting them only inside the child is too late for a
        # low-core worker whose import can otherwise exhaust RLIMIT_CPU.
        thread_env = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        for name in thread_env:
            os.environ[name] = str(request["cpu_cores"])
        process = multiprocessing.get_context("spawn").Process(
            target=_worker, args=(backend, candidate, request, self.ledger_path, raw_dir))
        started = time.monotonic()
        try:
            process.start()
        finally:
            for name, value in thread_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        stopped = None
        try:
            while process.is_alive():
                if time.monotonic() - started >= request["max_wall_seconds"]:
                    stopped = "wall_timeout"
                elif _directory_bytes(raw_dir) > request["max_raw_bytes"]:
                    stopped = "storage_budget_exhausted"
                if stopped:
                    process.terminate()
                    process.join(1)
                    if process.is_alive():
                        process.kill()
                    break
                process.join(0.1)
        finally:
            if process.is_alive():
                process.kill()
            process.join()
        outcome = raw_dir / "backend-outcome.json"
        if _directory_bytes(raw_dir) > request["max_raw_bytes"]:
            stopped = "storage_budget_exhausted"
        if stopped is None and process.exitcode == 0 and outcome.exists():
            value = json.loads(outcome.read_text())
            return value["evidence"], value["status"]
        raw = self._partial(raw_dir)
        raw["failure"] = stopped or f"worker_exit_{process.exitcode}"
        return raw, "execution_timeout" if stopped == "wall_timeout" else "budget_exhausted" if stopped else "execution_failed"

    @staticmethod
    def _partial(raw_dir: Path) -> dict:
        p = raw_dir / "partial.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _record(self, request: dict, raw: dict, status: str, calls: int, wall: float,
                cpu: float, work: Path, *, resource_basis: str = "measured") -> dict:
        info = feedback(raw)
        if status != "completed":
            info["evidence_status"] = "attempted_unresolved"
        channel = identity({"parent_id": request["candidate"]["parent_id"], "event": info["observed_event_id"],
                            "protocol": request["protocol"]}) if info["evidence_status"] in {"reference_ts_connected", "valid_alternative"} else None
        artifacts = _artifact_index(work)
        event = LedgerEvent(run_id=request["cache_key"], phase=self.budget.phase, calculator=request["protocol"]["engine"],
                            protocol_id=request["protocol"]["version"], calculator_build_hash=identity(request["protocol"]),
                            system_id=request["cache_key"], input_hash=identity(request["candidate"]), operation="other",
                            status="success" if status == "completed" and raw.get("strict_status") != "PROTOCOL_FAILURE" else "failure", attempt=request["attempt"],
                            system_evaluations=0 if status.startswith("not_started") else 1,
                            calculator_calls=0, cpu_seconds=cpu, cpu_cores=1,
                            gpu_seconds=wall if request["protocol"].get("device") == "gpu" else 0,
                            storage_bytes=sum(a["bytes"] for a in artifacts), event_id=request["token_id"])
        record = dict(schema="xtbflow-certification/v1", cache_key=request["cache_key"], token_id=request["token_id"],
                    request_sha256=file_hash(work / "request.json"),
                    attempt=request["attempt"], candidate_id=request["candidate"]["candidate_id"],
                    protocol=request["protocol"], execution_status=status, feedback=info, channel_id=channel,
                    charged_calls=calls, evidence=raw, artifacts=artifacts, ledger_event=asdict(event),
                    resources=dict(wall_seconds=wall, cpu_seconds=cpu, basis=resource_basis))
        record["content_sha256"] = identity(record)
        return record

    def _interrupted(self, work: Path, ledger: RunLedger) -> dict:
        request = json.loads((work / "request.json").read_text())
        token = ledger.reservations.get(request["token_id"])
        if token is not None and token["request_sha256"] != file_hash(work / "request.json"):
            raise ValueError("interrupted request identity mismatch")
        if token is None and request["token_id"] in ledger.committed_tokens:
            raise ValueError("settled attempt is missing its durable result")
        consumed = token.get("consumed_calls", 0) if token else 0
        # Unknown in-flight duration is charged at its declared envelope. Never
        # call this measured consumption or release it as though nothing ran.
        wall = request["max_wall_seconds"] + 5 if token is not None else 0
        raw = self._partial(work / "raw")
        raw["failure"] = "report_write_failed" if (work / "report-failure.json").exists() else "interrupted_without_durable_result"
        record = self._record(request, raw, "execution_interrupted", consumed, wall,
                              wall * request["cpu_cores"], work, resource_basis="conservative_upper_bound")
        write_json(work / "result.json", record)
        return record
