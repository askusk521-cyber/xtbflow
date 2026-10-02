"""Answer-blind input validation for reactant-side generation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from xtbflow.data.records import FORBIDDEN_INPUT_KEYS, canonical_hash


class InputFirewallError(ValueError):
    """Raised when a target-derived field enters a blind inference view."""


_EXTRA_FORBIDDEN = frozenset({
    "product", "mapped_product", "product_coordinates", "ts_geometry", "reference_ts",
    "reference_mode", "active_water_from_ts", "target_derived_solvent", "sealed_test_label",
})
FORBIDDEN_INFERENCE_FIELDS = frozenset(FORBIDDEN_INPUT_KEYS) | _EXTRA_FORBIDDEN


@dataclass(frozen=True)
class BlindInput:
    """Immutable-by-convention reactant input plus a reproducible fingerprint."""

    payload: Mapping[str, Any]
    fingerprint: str

    def __post_init__(self) -> None:
        if self.fingerprint != canonical_hash(self.payload):
            raise InputFirewallError("blind input fingerprint does not match payload")
        forbidden = _find_forbidden_keys(self.payload)
        if forbidden:
            raise InputFirewallError(
                "target-derived fields cannot enter blind inference: " + ", ".join(forbidden)
            )


def _find_forbidden_keys(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_INFERENCE_FIELDS:
                found.append(".".join((*path, key_text)))
            found.extend(_find_forbidden_keys(nested, (*path, key_text)))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_find_forbidden_keys(nested, (*path, str(index))))
    return sorted(set(found))


def make_blind_input(payload: Mapping[str, Any]) -> BlindInput:
    """Validate and snapshot a reactant-only input mapping."""

    if not isinstance(payload, Mapping):
        raise InputFirewallError("blind input must be a mapping")
    copied = deepcopy(dict(payload))
    forbidden = _find_forbidden_keys(copied)
    if forbidden:
        raise InputFirewallError(
            "target-derived fields cannot enter blind inference: " + ", ".join(forbidden)
        )
    try:
        fingerprint = canonical_hash(copied)
    except (TypeError, ValueError) as exc:
        raise InputFirewallError("blind input must be JSON-compatible") from exc
    return BlindInput(payload=copied, fingerprint=fingerprint)


def assert_same_blind_input(left: BlindInput, right: BlindInput) -> None:
    """Fail if two evaluations did not use exactly the same reactant view."""

    if left.fingerprint != right.fingerprint:
        raise InputFirewallError("blind evaluation inputs differ")
