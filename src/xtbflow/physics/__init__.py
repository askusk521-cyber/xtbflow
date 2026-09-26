"""Physical guidance and curvature contracts."""

from .curvature import HVPResult, finite_difference_hvp
from .saddle import (
    FailurePolicy,
    GuidanceConfig,
    GuidanceMode,
    GuidanceResult,
    PhysicalPostProcessor,
    apply_guidance,
    energy_descent_guidance,
    normalize_direction,
    saddle_guidance,
)

__all__ = [
    "HVPResult", "finite_difference_hvp", "FailurePolicy", "GuidanceConfig", "GuidanceMode",
    "GuidanceResult", "PhysicalPostProcessor", "apply_guidance", "energy_descent_guidance",
    "normalize_direction", "saddle_guidance",
]
