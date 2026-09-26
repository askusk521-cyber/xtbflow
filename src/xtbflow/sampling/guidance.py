"""Strength schedules shared by compatibility and physical guidance ablations."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GuidanceSchedule:
    start: float = 0.0
    end: float = 1.0
    warmup_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name in ("start", "end", "warmup_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.start < 0 or self.end < 0 or not 0 <= self.warmup_fraction <= 1:
            raise ValueError("guidance schedule values are outside their bounds")


def schedule_strength(schedule: GuidanceSchedule, tau: float) -> float:
    if not isinstance(tau, (int, float)) or not math.isfinite(float(tau)) or not 0 <= tau <= 1:
        raise ValueError("tau must be a finite value in [0,1]")
    if schedule.warmup_fraction and tau < schedule.warmup_fraction:
        return schedule.start
    fraction = 1.0 if schedule.warmup_fraction == 1 else max(0.0, (tau - schedule.warmup_fraction) / (1.0 - schedule.warmup_fraction))
    return (1.0 - fraction) * schedule.start + fraction * schedule.end
