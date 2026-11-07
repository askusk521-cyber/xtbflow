"""Frozen monotone generation schedules; these coordinates are not physical time."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DualClockSchedule:
    event_power: float = 1.0
    geometry_power: float = 1.0
    event_finish: float = 1.0
    coupling_start: float = 0.0
    coupling_end: float = 1.0

    def __post_init__(self) -> None:
        for name in ("event_power", "geometry_power", "event_finish", "coupling_start", "coupling_end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"finite {name} required")
        if not (0 < self.event_power <= 8 and 0 < self.geometry_power <= 8 and 0 < self.event_finish <= 1):
            raise ValueError("powers must be in (0,8] and event_finish in (0,1]")
        if not 0 <= self.coupling_start < self.coupling_end <= 1:
            raise ValueError("coupling window must be a nonempty interval in [0,1]")

    def at(self, progress: float) -> tuple[float, float]:
        if isinstance(progress, bool) or not isinstance(progress, (int, float)) or not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("generation progress must be in [0,1]")
        return min(progress / self.event_finish, 1.0) ** self.event_power, progress ** self.geometry_power

    def grid(self, steps: int) -> list[float]:
        if type(steps) is not int or not 1 <= steps <= 10000:
            raise ValueError("steps must be an integer in [1,10000]")
        # Include every discontinuity explicitly: a step never straddles an
        # event-freeze or coupling-window boundary. Report actual grid length.
        return sorted({i / steps for i in range(steps + 1)} | {self.event_finish, self.coupling_start, self.coupling_end})

    def coupled(self, progress: float) -> bool:
        self.at(progress)
        return self.coupling_start <= progress < self.coupling_end
