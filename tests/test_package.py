from __future__ import annotations

from pathlib import Path

import xtbflow
from xtbflow import legacy


def test_package_metadata_and_adapter_exports() -> None:
    assert xtbflow.__version__ == "0.1.0"
    assert xtbflow.LewisState is legacy.LewisState
    assert xtbflow.EventGeometryCompatibility is legacy.EventGeometryCompatibility


def test_adapter_resolves_the_packaged_legacy_namespace() -> None:
    events = legacy.module("mechai.data.events")
    source = legacy.source_root()
    assert events.LewisState is legacy.LewisState
    assert (source / "data" / "events.py").is_file()
    assert Path(events.__file__).name == "events.py"


def test_adapter_rejects_non_legacy_modules() -> None:
    try:
        legacy.module("xtbflow.internal")
    except ValueError as exc:
        assert "mechai" in str(exc)
    else:
        raise AssertionError("non-legacy imports must stay outside the adapter")
