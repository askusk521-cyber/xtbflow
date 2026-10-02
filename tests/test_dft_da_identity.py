from __future__ import annotations

from xtbflow.data.dft_da import _canonical_component_smiles


def test_family_key_ignores_atom_map_numbers() -> None:
    first = _canonical_component_smiles("[CH3:1][OH:2].[CH3:3][OH:4]")
    renumbered = _canonical_component_smiles("[CH3:101][OH:202].[CH3:303][OH:404]")
    reordered = _canonical_component_smiles("[CH3:404][OH:303].[CH3:202][OH:101]")
    assert first == renumbered == reordered


def test_family_key_preserves_connectivity_and_charge() -> None:
    neutral = _canonical_component_smiles("[CH3:1][OH:2]")
    disconnected = _canonical_component_smiles("[CH3:1].[OH:2]")
    charged = _canonical_component_smiles("[CH3:1][O-:2]")
    assert neutral != disconnected
    assert neutral != charged
