from __future__ import annotations

from scripts.audit_reaction_qm_coordinate_mapping import _expected_map_order, _parse_mapped_side


def test_map_number_order_is_independent_of_smiles_traversal_order() -> None:
    # Textual traversal is O then C, while map-id order is map 1 (C), map 2 (O).
    expected, failures = _expected_map_order("[O:2][C:1]", reaction_id="RXN_TEST", species="R0")
    assert failures == []
    assert expected == [6, 8]
    assert expected != [8, 6]


def test_duplicate_map_ids_are_rejected() -> None:
    _, failures = _parse_mapped_side("[C:1][O:1]", reaction_id="RXN_TEST", species="R0")
    assert "duplicate_map_id_in_target_side" in failures


def test_missing_or_noncontiguous_map_ids_are_rejected() -> None:
    _, failures = _parse_mapped_side("[C:1][O:3]", reaction_id="RXN_TEST", species="R0")
    assert "map_id_not_contiguous_from_one" in failures
