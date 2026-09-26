"""Public package entry point for the xtbflow research adapters."""
from __future__ import annotations

from . import legacy
from .legacy import *  # noqa: F403,F401

__version__ = "0.1.0"

__all__ = ["__version__", "legacy", *legacy.__all__]
