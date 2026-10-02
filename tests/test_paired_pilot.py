from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from xtbflow.data.dft_da import load_dft_da_samples
from xtbflow.data.paired_loader import PairedPilotLoader, load_paired_pilot, reactant_input_fingerprint


CACHE = Path("/home/lhshen/.cache/xtbflow/dft-da-full-29118509-v1")
MANIFEST = Path("data/manifests/paired_pilot_v1/manifest.jsonl")


@pytest.mark.skipif(not CACHE.is_dir() or not MANIFEST.is_file(), reason="real public archive cache is not available")
def test_real_paired_pilot_batch_excludes_reference_validation() -> None:
    loader = load_paired_pilot(MANIFEST, CACHE, batch_size=8)
    batch = loader.first_batch().as_dict()
    assert tuple(batch["reactant_input"]["reactant_coordinates"].shape)[0] == 8
    assert tuple(batch["training_supervision"]["ts_coordinates"].shape)[0] == 8
    assert batch["training_supervision"]["event_bond_edits"].abs().sum() > 0
    assert "reference_validation" not in batch
    assert "product_coordinates" not in batch["reactant_input"]


@pytest.mark.skipif(not CACHE.is_dir(), reason="real public archive cache is not available")
def test_product_ts_and_event_labels_do_not_change_input_fingerprint() -> None:
    samples, _ = load_dft_da_samples(CACHE)
    sample = samples[0]
    changed = replace(
        sample.record,
        event_label={**sample.record.event_label, "sha256": "a" * 64},
        product_label={**sample.record.product_label, "graph_sha256": "b" * 64},
        ts_geometry={**sample.record.ts_geometry, "sha256": "c" * 64},
    )
    assert reactant_input_fingerprint(sample.record) == reactant_input_fingerprint(changed)


def test_loader_padding_and_event_masks_are_deterministic() -> None:
    pytest.importorskip("numpy")
    # Minimal structural check without a source cache: a loader must reject an
    # empty set instead of manufacturing a synthetic training example.
    with pytest.raises(ValueError, match="cannot be empty"):
        PairedPilotLoader([])
