"""GFN2-xTB scores in kcal/mol; saddle relaxation displacements in Angstrom."""
import time

import numpy as np
import torch

from xtbflow.calculators.xtb_oracle import BOHR_IN_ANGSTROM
from .explore_rollout import event_direction, saddle_reflect
from .geometry_information import HARTREE_TO_KCAL, min_distance, xtb_energy_kcal
from .guidance import centered


def gradient_to_force(gradient):
    return -np.asarray(gradient) * HARTREE_TO_KCAL / BOHR_IN_ANGSTROM


def score_candidate(z, x, b_hat, b_r, anchor_energy, cfg, *, relaxed=True, gradient_job=None):
    """Return failures explicitly, including invalid coordinates at displaced probes.

    gradient_job is the existing explore_geometry_information._force_job. Injection
    avoids making the library depend on a scripts directory and enables unit tests.
    """
    if relaxed and gradient_job is None:
        raise ValueError('relaxed scoring requires the existing gradient job')
    started = time.monotonic()
    calls = 0
    result = dict(status='ok', barrier=None, curvature=None, raw_barrier=None,
                  energy_drop=None, calls=0, elapsed_s=0.)
    x = torch.as_tensor(np.asarray(x), dtype=torch.float64).clone()[None]
    bh = torch.as_tensor(np.asarray(b_hat), dtype=torch.float64)[None]
    br = torch.as_tensor(np.asarray(b_r), dtype=torch.float64)[None]
    mask = torch.ones(x.shape[:2], dtype=torch.bool)

    def coords(value):
        arr = value[0].numpy()
        if not np.isfinite(arr).all() or min_distance(arr) < .5:
            raise ValueError('invalid_geometry')
        return arr

    def energy(value):
        nonlocal calls
        arr = coords(value)
        calls += 1
        e, status = xtb_energy_kcal(z, arr, cfg)
        if status != 'ok' or not np.isfinite(e):
            raise ValueError(status)
        return e

    def force(value):
        nonlocal calls
        arr = coords(value)
        calls += 1
        g, status = gradient_job((z, arr, cfg))
        if status != 'ok' or g is None:
            raise ValueError(status)
        return torch.as_tensor(gradient_to_force(g), dtype=torch.float64)[None]

    def direction(value):
        u, valid = event_direction(bh, br, value, mask)
        if not bool(valid[0]) or not torch.isfinite(u).all():
            raise ValueError('invalid_direction')
        return u

    try:
        if not np.isfinite(anchor_energy):
            raise ValueError('invalid_anchor')
        raw = energy(x)
        result['raw_barrier'] = raw - anchor_energy
        result['barrier'] = result['raw_barrier']
        if relaxed:
            for _ in range(20):
                u = direction(x)
                delta = .001 * saddle_reflect(force(x), u)
                largest = float(delta.norm(dim=-1).max())
                delta *= min(1., .05 / largest) if largest else 1.
                x = centered(x + delta, mask)
                coords(x)
            er = energy(x)
            u = direction(x)
            fp, fm = force(x + .005 * u), force(x - .005 * u)
            result.update(barrier=er - anchor_energy,
                          curvature=float(-((fp - fm) * u).sum() / .01), energy_drop=raw - er)
    except ValueError as exc:
        result.update(status=str(exc), barrier=None, curvature=None)
    result.update(calls=calls, elapsed_s=time.monotonic() - started)
    return result
