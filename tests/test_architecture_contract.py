from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/models/xtbflow_v0.1.yaml"
SCHEMA = ROOT / "schemas/model_contract.schema.json"


def load_config():
    # The YAML file is deliberately JSON-compatible so the contract can be
    # parsed in minimal CPU environments without silently adding a YAML parser.
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_canonical_contract_files_are_parseable():
    config = load_config()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert config["schema"] == schema["properties"]["schema"]["const"]
    for key in schema["required"]:
        assert key in config


def test_scope_and_input_firewall_are_explicit():
    config = load_config()
    assert config["scope"]["elements"] == ["H", "C", "N", "O", "S"]
    assert config["scope"]["fixed_atom_pool"] is True
    forbidden = set(config["inputs"]["forbidden_at_inference"])
    assert {"true_product", "reference_ts", "target_derived_active_water", "sealed_test_label"} <= forbidden
    assert config["training_label_views"]["input_supervision_isolation"] is True


def test_conservation_and_physical_zero_controls_are_registered():
    config = load_config()
    conservation = config["conservation"]
    assert conservation["symmetric_be_upper_triangle_weights"] == {"diagonal": 1, "off_diagonal": 2}
    assert set(("constraint_projection", "integerization", "legal_state_check", "rejection_count")) <= set(conservation["discrete_decoder"])
    guidance = config["physical_guidance"]
    assert guidance["zero_strength_behavior"] == "exact_downstream_baseline"
    assert guidance["trust_region"] == "required"
    assert guidance["differentiable_qc"] is False


def test_starting_capacity_is_not_claimed_optimal_and_open_items_remain():
    config = load_config()
    rep = config["representation"]
    assert (rep["message_passing_blocks"], rep["scalar_channels"], rep["vector_channels"], rep["pair_feature_channels"]) == (6, 128, 32, 128)
    assert config["open_items"]
    assert "mixture_of_experts" in config["out_of_scope"]
    assert "serial_event_then_geometry" in config["baselines"]


def test_exploratory_design_points_to_the_canonical_contract():
    text = (ROOT / "docs/model_design.md").read_text(encoding="utf-8")
    assert "architecture/architecture_v0.1.md" in text
