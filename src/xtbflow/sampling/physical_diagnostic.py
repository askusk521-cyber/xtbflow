"""Finite CPU physical-direction diagnostics with durable started-work receipts."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import torch

from xtbflow.sampling.integrated_guidance import FlowState, centered, model_values
from xtbflow.validation.certification import file_hash, identity, write_json


def cpu_seconds() -> float:
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time()+children.ru_utime+children.ru_stime


class StageMeter:
    def __init__(self, out: Path, caps: dict):
        self.out, self.caps = out, caps
        self.start, self.cpu_start = time.monotonic(), cpu_seconds()
        self.rows: list[dict] = []
        self.reverse_reserved = 0
        self.reverse_completed = 0
        self.flush()

    def check(self) -> None:
        if time.monotonic()-self.start >= self.caps['wall_seconds'] or cpu_seconds()-self.cpu_start >= self.caps['cpu_seconds']:
            raise RuntimeError('stage_time_budget_exhausted')
        if sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) > self.caps['output_bytes']:
            raise RuntimeError('stage_storage_budget_exhausted')

    def begin(self, kind: str, context: str) -> dict:
        self.check()
        if sum(r['kind'] == kind for r in self.rows) >= self.caps[kind]:
            raise RuntimeError(kind+'_budget_exhausted')
        if kind == 'jvp':
            if self.reverse_reserved+2 > self.caps['reverse']:
                raise RuntimeError('reverse_budget_exhausted')
            self.reverse_reserved += 2
        row = dict(sequence=len(self.rows)+1, kind=kind, context=context, status='started',
                   elapsed_start=time.monotonic()-self.start)
        self.rows.append(row)
        self.flush()
        return row

    def end(self, row: dict, status: str = 'completed', **extra) -> None:
        row.update(status=status, elapsed_end=time.monotonic()-self.start, **extra)
        if row['kind'] == 'jvp' and status == 'completed':
            self.reverse_completed += 2
        self.flush()

    def flush(self) -> None:
        write_json(self.out/'work.json', self.summary())

    def summary(self) -> dict:
        return dict(rows=self.rows, counts={k:sum(r['kind'] == k for r in self.rows) for k in ('forward','ef','jvp')},
                    reverse_reserved=self.reverse_reserved, reverse_completed=self.reverse_completed,
                    wall_seconds=time.monotonic()-self.start, process_cpu_including_children_seconds=cpu_seconds()-self.cpu_start,
                    gpu_seconds=0, training_steps=0, reference_primitives=0,
                    unresolved_started=sum(r['status']=='started' for r in self.rows))

    def install_model(self, model):
        def before(module, args, kwargs):
            self.begin('forward', 'trained_joint')
        def after(module, args, output):
            row = next((r for r in reversed(self.rows) if r['kind']=='forward' and r['status']=='started'), None)
            if row is None:  # Pre-hook rejected a call before it started.
                return
            if output is None:
                self.end(row, 'execution_failed')
                return
            if not all(torch.isfinite(v).all() for v in output.values()):
                self.end(row, 'failed_nonfinite')
                raise ValueError('nonfinite_model_output')
            self.end(row)
        model.register_forward_pre_hook(before, with_kwargs=True)
        model.register_forward_hook(after, always_call=True)


def rms_atom(value, mask):
    return ((value.square().sum(-1)*mask).sum(1)/mask.sum(1).clamp_min(1)).sqrt()


def displacement_metrics(value, mask) -> dict:
    return dict(l2_angstrom=float(value.norm()), rms_atom_angstrom=float(rms_atom(value, mask)[0]),
                max_atom_angstrom=float(value.norm(dim=-1).max()))


def event_direction(b_hat, b_r, x, mask, eps=1e-8):
    """Unchanged ffca40f predicted-ΔBE construction; not a verified negative mode."""
    pair=(mask[:,:,None]&mask[:,None,:]).to(x.dtype)
    eye=torch.eye(x.shape[1],device=x.device,dtype=x.dtype)[None]
    w=(b_hat-b_r)*pair*(1-eye);w=.5*(w+w.transpose(1,2))
    difference=x[:,None,:,:]-x[:,:,None,:]
    unit=difference/difference.norm(dim=-1,keepdim=True).clamp_min(eps)
    u=centered((w[...,None]*unit).sum(2),mask)
    size=u.flatten(1).norm(dim=1)
    return u/size.clamp_min(eps)[:,None,None],size>eps


def physical_fields(force, predicted_b, reactant_b, predicted_x, mask):
    force = centered(force, mask)
    u, valid = event_direction(predicted_b, reactant_b, predicted_x, mask)
    reflected = force-2*(force*u).sum((1,2),keepdim=True)*u
    return force, reflected, u, bool(valid.all())


def measure_direction(model, query, base: FlowState, direction, meter: StageMeter,
                      *, context: str, scales: list[float]) -> tuple[dict, dict]:
    norm = float(direction.norm())
    if not math.isfinite(norm) or norm <= 1e-10:
        return dict(status='unavailable_zero_direction', qualified=False), {}
    unit = direction/direction.norm()
    def values(x):
        y, vb, _ = model_values(model, query, replace(base, x=x))
        return y, vb
    row = meter.begin('jvp', context)
    try:
        values0, derivative = torch.autograd.functional.jvp(values, base.x, unit, strict=True)
    except Exception as exc:
        meter.end(row, 'failed', failure_type=type(exc).__name__)
        raise
    meter.end(row)
    y0, vb0 = values0
    jx, jb = derivative
    floor = 100*torch.finfo(base.x.dtype).eps*(float(y0.norm()+vb0.norm())+1)/min(scales)
    measurements = []
    for epsilon in scales:
        with torch.no_grad():
            plus, minus = values(base.x+epsilon*unit), values(base.x-epsilon*unit)
        fd = tuple((p-m)/(2*epsilon) for p,m in zip(plus,minus))
        errors = [float((f-j).norm()) for f,j in zip(fd,derivative)]
        norms = [float(j.norm()) for j in derivative]
        qualified = all(e <= (1e-3*n if n>floor else floor) for e,n in zip(errors,norms))
        measurements.append(dict(epsilon_l2_angstrom=epsilon, endpoint_absolute_error=errors[0], event_absolute_error=errors[1],
                                 endpoint_relative_error=errors[0]/max(norms[0],1e-30),
                                 event_relative_error=errors[1]/max(norms[1],1e-30), qualified=qualified))
    qualified = any(a['qualified'] and b['qualified'] for a,b in zip(measurements,measurements[1:]))
    finite = all(torch.isfinite(v).all() for v in derivative)
    qualified = bool(qualified and finite and float(centered(jx,query.atom_mask).sub(jx).abs().max())<=1e-7)
    projection = float((jx*unit).sum())
    report = dict(status='qualified' if qualified else 'derivative_unqualified', qualified=qualified,
                  direction_norm_hartree_per_angstrom=norm, endpoint_gain=float(jx.norm()),
                  endpoint_projection=projection, endpoint_cosine=projection/max(float(jx.norm()),1e-30),
                  event_velocity_response_norm=float(jb.norm()), numerical_floor_absolute=floor,
                  event_response_resolved=bool(jb.norm()>floor), endpoint_response_resolved=bool(jx.norm()>floor),
                  event_symmetry_error=float((jb-jb.transpose(1,2)).abs().max()),
                  event_electron_sum=float(jb.sum()), fd=measurements,
                  l2_unit_rms=float(rms_atom(unit,query.atom_mask)[0]))
    cache = dict(unit=unit.detach(), jx=jx.detach(), jb=jb.detach(), y0=y0.detach(), vb0=vb0.detach())
    return report, cache


def evaluate_physical(coordinates, numbers, cfg: dict, meter: StageMeter, context: str) -> dict:
    """One process per E/F, killed and reaped before a timeout can be settled."""
    row = meter.begin('ef', context)
    stem = f"ef-{row['sequence']:03d}"
    request, result, log = (meter.out/(stem+suffix) for suffix in ('.request.json','.result.json','.log'))
    write_json(request, dict(coordinates=coordinates.detach()[0].tolist(), atomic_numbers=numbers[0].tolist(), calculator=cfg))
    timeout = min(60., meter.caps['wall_seconds']-(time.monotonic()-meter.start))
    with log.open('wb') as stream:
        process = subprocess.Popen([sys.executable, '-m', 'xtbflow.sampling.physical_diagnostic', str(request), str(result)],
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                meter.end(row, 'termination_unknown', termination='unknown', settlement='unsettled')
                raise RuntimeError('backend_termination_unknown')
            meter.end(row, 'timeout', termination='confirmed', settlement='settled', reserved_ef=1)
            return dict(status='timeout', force=None)
    if not result.exists():
        meter.end(row, 'report_missing', termination='confirmed', settlement='unknown', returncode=process.returncode)
        raise RuntimeError('backend_report_missing_cost_unknown')
    payload = json.loads(result.read_text())
    meter.end(row, payload['status'], termination='confirmed', settlement='settled',
              worker=payload, result_sha256=file_hash(result))
    return payload


def ef_worker(request: Path, result: Path) -> None:
    from xtbflow.calculators import CalculatorProtocol, MolecularSystem, XTBOracleAdapter
    from xtbflow.runtime import RunLedger, StageBudget
    resource.setrlimit(resource.RLIMIT_CPU, (60,60))
    cfg = json.loads(request.read_text())
    spec = cfg['calculator']
    protocol = CalculatorProtocol(**spec['protocol'])
    adapter = XTBOracleAdapter(protocol, implementation='tblite', require_budget_token=True)
    if adapter.version != spec['version'] or adapter.build_hash != spec['build_sha256']:
        raise ValueError('backend_identity_mismatch')
    symbols={1:'H',6:'C',7:'N',8:'O'}
    system=MolecularSystem(tuple(symbols[z] for z in cfg['atomic_numbers']), tuple(map(tuple,cfg['coordinates'])),
                           0,1,{'kind':'gas','periodic':False},'ID63623')
    ledger=RunLedger(StageBudget('one-ef',max_calculator_calls=1))
    token=ledger.issue_calculator_token('one-ef',1,persist_path=result.with_suffix('.ledger.json'))
    start,cpu=time.monotonic(),time.process_time()
    try:
        value=adapter.evaluate(system,budget_token=token)
        payload=dict(status='completed',energy=value.energy,force=value.forces,input_sha256=system.input_hash,
                     build_sha256=adapter.build_hash,protocol_sha256=protocol.identity)
    except Exception as exc:
        payload=dict(status='calculation_failed',force=None,failure_type=type(exc).__name__)
    token.commit(recorded_calls=token.consumed_calls)
    payload.update(attempted_calls=token.consumed_calls,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu)
    write_json(result,payload)


if __name__ == '__main__':
    ef_worker(Path(sys.argv[1]),Path(sys.argv[2]))
