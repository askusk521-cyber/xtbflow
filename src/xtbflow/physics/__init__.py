"""Physical guidance and curvature contracts."""

from .curvature import HVPResult, ProjectedHessianResult, finite_difference_hvp, projected_hessian
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
    "HVPResult", "ProjectedHessianResult", "finite_difference_hvp", "projected_hessian", "FailurePolicy", "GuidanceConfig", "GuidanceMode",
    "GuidanceResult", "PhysicalPostProcessor", "apply_guidance", "energy_descent_guidance",
    "normalize_direction", "saddle_guidance",
]
