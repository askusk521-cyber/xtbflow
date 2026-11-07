"""Validate semantic controls and expose their real dispatch readiness."""
from pathlib import Path


def validate_matrix(matrix: dict, root: Path) -> dict:
    methods = {m["id"]: m for m in matrix["methods"]}
    if len(methods) != len(matrix["methods"]):
        raise ValueError("duplicate semantic method")
    known = {"functional_batch_available", "adapter_required", "implementation_required",
             "historical_proxy_only", "separate_information_condition"}
    for method in methods.values():
        if method["status"] not in known or method["information"] not in {"reactant_only", "known_product", "declared_partial_condition"}:
            raise ValueError("invalid method readiness or information condition")
        path = method["entrypoint"]
        if path and (not (root / path).resolve().is_relative_to(root.resolve()) or not (root / path).is_file()):
            raise ValueError("missing or outside dispatch entrypoint")
        if method["status"] in {"functional_batch_available", "historical_proxy_only"} and not path:
            raise ValueError("available method requires an executable entrypoint")
    for question in matrix["questions"].values():
        pair = [methods[m] for m in question["primary"]]
        if len(pair) != 2 or pair[0]["information"] != pair[1]["information"]:
            raise ValueError("primary comparison must have matched input permissions")
    return dict(schema="xtbflow-comparison-readiness/v1", methods=list(methods.values()),
                functional_methods=[m["id"] for m in methods.values() if m["status"] == "functional_batch_available"],
                unavailable_primary_controls=[m["id"] for m in methods.values() if m["status"] != "functional_batch_available"],
                full_experiment_ready=False)


def functional_dispatch(matrix: dict, root: Path, generation_profile: dict, requested: list[str]) -> dict:
    """Dispatch the actual integrated batch; do not emulate unavailable arms."""
    available = validate_matrix(matrix, root)["functional_methods"]
    if len(requested) != len(set(requested)) or set(requested) != set(generation_profile["methods"]):
        raise ValueError("request must match the complete frozen functional batch")
    if any(m not in available for m in requested):
        raise ValueError("method implementation or completion adapter is unavailable")
    return dict(script="scripts/run_research_generation.py", methods=requested,
                configuration="configs/research/generation_bridge_20261107.json",
                reference_script="scripts/certify_generated_candidates.py",
                scope="functional_batch_only", equal_total_budget_comparison=False)
