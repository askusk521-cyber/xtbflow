"""A single deterministic decisioner over verified, sequential tool observations.

This first tool profile replays existing reference attempts. It does not run a
generator or a calculator. Attributed historical costs and current replay costs
are deliberately distinct; replay is not a new adaptive discovery experiment.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
from typing import Protocol

from xtbflow.evaluation.certified_trace import read_reference_attempt
from xtbflow.evaluation.discovery import CONNECTED, COST_AXES, add_costs, check_cost, check_hash
from xtbflow.validation.certification import CertificationCandidate, file_hash, identity, write_json


ACTIONS = ("propose", "resample_event", "resample_geometry", "select_sampling_policy",
           "physical_probe", "reference_label", "refine_TS", "validate_connectivity",
           "expand_intermediate", "finish")
SUPPORTED = frozenset({"select_sampling_policy", "validate_connectivity", "finish"})
GENERATION_ACTIONS = frozenset({"propose", "resample_event", "resample_geometry"})
SAMPLERS = {
    "joint_event_lead2": {"method": "joint_generation", "clock": "event_lead2", "steps": 50},
    "independent_serial": {"method": "independent_event_then_geometry", "clock": "serial", "steps": 50},
}


def zero_cost() -> dict:
    return dict.fromkeys(COST_AXES, 0)


def validate_config(config: dict) -> None:
    live = config.get("tool_profile") == "reference_feedback_generation"
    fields = {"schema", "policy", "seed", "max_actions", "budget", "reference_envelope",
              "tool_profile", "initial_sampler", "samplers"}
    if live:
        fields |= {"generation_budget", "generation_envelope", "generation_rounds", "prior_generation_cost"}
    if set(config) != fields:
        raise ValueError("unexpected decision configuration fields")
    if config["schema"] != ("xtbflow-feedback-policy/v2" if live else "xtbflow-feedback-policy/v1") or config["policy"] not in {"fixed", "evidence_heuristic_v1"}:
        raise ValueError("unknown decision policy")
    if config["tool_profile"] not in {"verified_reference_replay", "reference_feedback_generation"} or config["samplers"] != SAMPLERS:
        raise ValueError("unsupported tools or sampler catalogue")
    if type(config["seed"]) is not int or config["seed"] < 0 or type(config["max_actions"]) is not int or not 2 <= config["max_actions"] <= 128:
        raise ValueError("finite nonnegative seed and 2..128 actions required")
    if config["initial_sampler"] not in SAMPLERS:
        raise ValueError("unknown initial sampler")
    if live and (type(config["generation_rounds"]) is not int or not 1 <= config["generation_rounds"] <= 4):
        raise ValueError("bounded generation rounds required")
    for name in (("budget", "reference_envelope", "generation_budget", "generation_envelope", "prior_generation_cost") if live else ("budget", "reference_envelope")):
        check_cost(config[name])
        if any(v is None for v in config[name].values()):
            raise ValueError("decision budgets cannot contain unknown costs")


def candidate_view(candidate: CertificationCandidate, protocol_sha256: str) -> dict:
    """Only input hypotheses; never import a future result into initial state."""
    check_hash(protocol_sha256)
    pool = {k: asdict(candidate)[k] for k in ("parent_id", "atomic_numbers", "atom_ids", "charge", "multiplicity", "environment")}
    pool.update(input_sha256=candidate.provenance["input_sha256"], reactant_be_sha256=identity(candidate.reactant_be))
    return json.loads(json.dumps(dict(candidate_id=candidate.candidate_id, pool=pool, pool_sha256=identity(pool),
                geometry_sha256=identity(candidate.coordinates_angstrom),
                planned_product_be_sha256=identity(candidate.planned_product_be),
                candidate_sha256=identity(candidate.content()), protocol_sha256=protocol_sha256,
                event_family=None, generation_protocol=None)))


def initial_state(candidates: list[dict], config: dict) -> dict:
    validate_config(config)
    if not candidates or len(candidates) > 64 or len({c["candidate_id"] for c in candidates}) != len(candidates):
        raise ValueError("1..64 uniquely identified candidates required")
    fields = {"candidate_id", "pool", "pool_sha256", "geometry_sha256", "planned_product_be_sha256",
              "candidate_sha256", "protocol_sha256", "event_family", "generation_protocol"}
    for c in candidates:
        if set(c) != fields or c["pool_sha256"] != identity(c["pool"]):
            raise ValueError("unexpected candidate view or atom-pool identity")
        if set(c["pool"]) != {"parent_id", "atomic_numbers", "atom_ids", "charge", "multiplicity",
                             "environment", "input_sha256", "reactant_be_sha256"}:
            raise ValueError("unexpected reactant input fields")
        if (c["pool"]["charge"], c["pool"]["multiplicity"]) != (0, 1) or c["pool"]["environment"] != {"kind": "gas", "periodic": False}:
            raise ValueError("unsupported electronic state or environment")
        for k in ("geometry_sha256", "planned_product_be_sha256", "candidate_sha256", "protocol_sha256"):
            check_hash(c[k])
        if c["event_family"] is not None or c["generation_protocol"] is not None:
            raise ValueError("replay input does not establish family or generation protocol")
    if len({c["pool_sha256"] for c in candidates}) != 1 or len({c["protocol_sha256"] for c in candidates}) != 1:
        raise ValueError("one episode requires one mapped atom pool and reference protocol")
    ordered = sorted(candidates, key=lambda c: identity([config["seed"], c["candidate_id"]]))
    state = dict(schema="xtbflow-feedback-state/v1", candidates=deepcopy(ordered), observations={},
                evidence={}, channels={}, duplicate_channels=0, reused_evidence=0,
                attributed_reference_cost=zero_cost(), remaining=deepcopy(config["budget"]),
                sampler=config["initial_sampler"], sampler_changes=[], step=0, finished=False,
                stop_reason=None, configuration_sha256=identity(config))
    if config["tool_profile"] == "reference_feedback_generation":
        state.update(schema="xtbflow-feedback-state/v2", generation=[], generation_spent=zero_cost(),
                     generation_remaining=deepcopy(config["generation_budget"]),
                     prior_generation_cost=deepcopy(config["prior_generation_cost"]), pending_certification=[])
    return state


def choose(state: dict, config: dict) -> dict:
    """No filesystem/backend access and no unobserved results in policy input."""
    if state["finished"] or state["configuration_sha256"] != identity(config):
        raise ValueError("finished or mismatched decision state")
    action = dict(schema="xtbflow-feedback-action/v1", sequence=state["step"],
                  state_sha256=identity(state), policy=config["policy"], seed=config["seed"],
                  name="finish", candidate_id=None, parameters={}, expected_cost=zero_cost(),
                  rationale="reference_batch_complete", mechanistic_question="What evidence is established?",
                  expected_evidence="Retained endpoints, unresolved attempts and explicit unfinished tools")
    if state["step"] >= config["max_actions"] - 1:
        action["rationale"] = "action_budget_exhausted"
    elif any(v < 0 for v in state["remaining"].values()):
        action["rationale"] = "observed_reference_cost_exceeded_budget"
    elif any(v < 0 for v in state.get("generation_remaining", {}).values()):
        action["rationale"] = "observed_generation_cost_exceeded_budget"
    elif state.get("generation") and state["generation"][-1]["status"] != "generation_completed":
        action["rationale"] = "generation_failed_no_automatic_retry"
    elif config["policy"] == "evidence_heuristic_v1" and state["sampler"] != "independent_serial" and (
            state["duplicate_channels"] or any(e["feedback"]["evidence_status"] == "valid_alternative" for e in state["evidence"].values())):
        action.update(name="select_sampling_policy", parameters={"sampler": "independent_serial"},
                      rationale="repeat_channel" if state["duplicate_channels"] else "planned_observed_event_disagreement",
                      mechanistic_question="Should the next proposal change event selection after observed endpoint evidence?",
                      expected_evidence="Recorded sampler selection only; no new proposal or physical calculation")
    else:
        pending = [c for c in state["candidates"] if c["candidate_id"] not in state["observations"]]
        if pending:
            if any(config["reference_envelope"][k] > state["remaining"][k] for k in COST_AXES):
                action["rationale"] = "insufficient_complete_reference_envelope"
            else:
                action.update(name="validate_connectivity", candidate_id=pending[0]["candidate_id"],
                              parameters={"profile": config["tool_profile"]}, expected_cost=deepcopy(config["reference_envelope"]),
                              rationale="unobserved_candidate", mechanistic_question="Which endpoints did this attempted candidate actually connect?",
                              expected_evidence="Artifact-verified reference result, layered feedback and original charged cost")
        elif config["tool_profile"] == "reference_feedback_generation":
            if len(state["generation"]) >= config["generation_rounds"]:
                action["rationale"] = "generation_complete_pending_independent_certification"
            elif any(config["generation_envelope"][k] > state["generation_remaining"][k] for k in COST_AXES):
                action["rationale"] = "insufficient_complete_generation_envelope"
            else:
                targets = [c for c in state["candidates"] if state["evidence"][state["observations"][c["candidate_id"]]]["feedback"]["first_order_saddle"]]
                geometry = config["policy"] == "evidence_heuristic_v1" and targets and not state["channels"]
                name = "resample_geometry" if geometry else ("resample_event" if config["policy"] == "evidence_heuristic_v1" else "propose")
                action.update(name=name, candidate_id=targets[0]["candidate_id"] if geometry else None,
                    parameters=dict(sampler=state["sampler"], sampling_seed=config["seed"] + len(state["generation"])),
                    expected_cost=deepcopy(config["generation_envelope"]),
                    rationale="saddle_without_connected_channel" if geometry else "explore_after_observed_batch",
                    mechanistic_question="Can a new generated event or geometry address the observed unresolved or repeated outcome?",
                    expected_evidence="Metered fresh model proposal and a hashed independent-certification handoff")
    action["action_id"] = identity(action)
    return action


class DeterministicTools(Protocol):
    """Execution must be idempotent for an identical action_id after a crash."""
    @property
    def identity(self) -> dict: ...

    def execute(self, action: dict) -> dict: ...


class CachedReferenceTools:
    """Read-only verified tool; opaque paths/results never reach the decisioner."""
    def __init__(self, attempts: dict[str, Path], pinned_results: dict[str, str]) -> None:
        if not attempts or set(attempts) != set(pinned_results):
            raise ValueError("pin every tool result before replay")
        for digest in pinned_results.values():
            check_hash(digest)
        self.attempts = {k: Path(v) for k, v in attempts.items()}
        self.pins = deepcopy(pinned_results)

    @property
    def identity(self) -> dict:
        return dict(profile="verified_reference_replay", result_sha256=self.pins,
                    implementation_sha256=file_hash(Path(__file__)),
                    verifier_sha256=file_hash(Path(__file__).parents[1] / "evaluation/certified_trace.py"))

    def execute(self, action: dict) -> dict:
        if action["name"] not in SUPPORTED:
            raise ValueError("tool adapter unavailable")
        result = dict(schema="xtbflow-feedback-tool-result/v1", action_id=action["action_id"],
                      tool_profile="verified_reference_replay", new_calculator_calls=0,
                      status="completed", observation=None)
        if action["name"] != "validate_connectivity":
            return result
        work = self.attempts[action["candidate_id"]]
        if file_hash(work / "result.json") != self.pins[action["candidate_id"]]:
            raise ValueError("pinned reference result changed")
        attempt, provenance = read_reference_attempt(work)
        request = json.loads((work / "request.json").read_text())
        record = json.loads((work / "result.json").read_text())
        view = candidate_view(CertificationCandidate(**request["candidate"]), identity(record["protocol"]))
        # Error text may contain private paths. Keep its hash and raw artifact locator.
        observed = {k: record["feedback"][k] for k in ("evidence_status", "stationary_point", "first_order_saddle",
                    "bidirectional_minima", "planned_event_realized", "planned_event_id", "observed_event_id")}
        observed.update(reason_sha256=identity(record["feedback"].get("reason")),
                        endpoint_be_sha256=identity(record["feedback"]["endpoint_be"]) if record["feedback"]["endpoint_be"] is not None else None,
                        observed_product_be_sha256=identity(record["feedback"]["observed_product_be"]) if record["feedback"]["observed_product_be"] is not None else None)
        ts = record["evidence"].get("ts", {})
        result["observation"] = dict(evidence_id=provenance["result_sha256"], candidate=view,
            execution_status=attempt.status, feedback=observed, original_cost=dict(attempt.cost),
            provenance=provenance, physical=dict(energy_hartree=ts.get("energy_hartree"),
                gradient_hartree_bohr_sha256=identity(ts["gradient_hartree_bohr"]) if "gradient_hartree_bohr" in ts else None,
                frequencies_cm1=ts.get("frequencies_cm1"), uncertainty=None),
            evidence_locator=dict(cache_key=work.parent.name, attempt=work.name,
                                  result_sha256=provenance["result_sha256"], request_sha256=provenance["request_sha256"]))
        return result


def reduce_outcome(state: dict, action: dict, outcome: dict, config: dict) -> dict:
    if choose(state, config) != action or outcome["action_id"] != action["action_id"]:
        raise ValueError("outcome is not bound to the exact visible state and deterministic action")
    if outcome["schema"] != "xtbflow-feedback-tool-result/v1" or outcome["status"] != "completed" or outcome["tool_profile"] != config["tool_profile"] or outcome["new_calculator_calls"] != 0:
        raise ValueError("unsupported or incomplete replay tool result")
    after = deepcopy(state)
    after["step"] += 1
    if action["name"] == "finish":
        after.update(finished=True, stop_reason=action["rationale"])
    elif action["name"] in GENERATION_ACTIONS:
        generated = outcome["generation"]
        check_cost(generated["cost"])
        if any(v is None for v in generated["cost"].values()) or generated["action_id"] != action["action_id"]:
            raise ValueError("generation cost or action identity is unresolved")
        for c in generated["candidates"]:
            if c["pool_sha256"] != state["candidates"][0]["pool_sha256"] or c["protocol_sha256"] != state["candidates"][0]["protocol_sha256"]:
                raise ValueError("new generation changed fixed atom pool or protocol")
            after["pending_certification"].append(c)
        after["generation"].append(deepcopy(generated))
        after["generation_spent"] = add_costs([state["generation_spent"], generated["cost"]])
        after["generation_remaining"] = {k: config["generation_budget"][k] - after["generation_spent"][k] for k in COST_AXES}
    elif action["name"] == "select_sampling_policy":
        after["sampler"] = action["parameters"]["sampler"]
        after["sampler_changes"].append(dict(action_id=action["action_id"], sampler=after["sampler"], executed_generation=False))
    else:
        observation = outcome["observation"]
        candidate = next(c for c in state["candidates"] if c["candidate_id"] == action["candidate_id"])
        # Cosmetic aliases can reuse an identical stored geometry without new evidence.
        if {k: v for k, v in candidate.items() if k != "candidate_id"} != {k: v for k, v in observation["candidate"].items() if k != "candidate_id"}:
            raise ValueError("tool result differs from candidate, atom pool or protocol")
        key = observation["evidence_id"]
        check_hash(key)
        check_cost(observation["original_cost"])
        if any(v is None for v in observation["original_cost"].values()):
            raise ValueError("unknown reference cost cannot enter budget decisions")
        after["observations"][candidate["candidate_id"]] = key
        if key in after["evidence"]:
            if after["evidence"][key] != observation:
                raise ValueError("evidence identity changed")
            after["reused_evidence"] += 1
        else:
            after["evidence"][key] = deepcopy(observation)
            after["attributed_reference_cost"] = add_costs([state["attributed_reference_cost"], observation["original_cost"]])
            f = observation["feedback"]
            if f["evidence_status"] in CONNECTED:
                if observation["execution_status"] != "completed" or not f["observed_event_id"] or not f["endpoint_be_sha256"] or not f["observed_product_be_sha256"]:
                    raise ValueError("missing actual endpoint evidence")
                channel = identity([candidate["pool_sha256"], candidate["protocol_sha256"], f["observed_event_id"]])
                after["duplicate_channels"] += int(channel in after["channels"])
                after["channels"].setdefault(channel, []).append(key)
        after["remaining"] = {k: config["budget"][k] - after["attributed_reference_cost"][k] for k in COST_AXES}
    return after


class FeedbackLoop:
    """Hash-linked, single-writer actions; results precede state settlement.

Recovery rechecks persisted outcomes against the pinned, read-only source tool.
No unavailable action, automatic retry or silent mutation of the run is allowed.
"""
    def __init__(self, root: Path, candidates: list[dict], config: dict, tools: DeterministicTools):
        self.root, self.config, self.tools = Path(root), deepcopy(config), tools
        self.initial = initial_state(candidates, config)
        self.root.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        with (self.root / "loop.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = dict(schema="xtbflow-feedback-run/v1", initial=self.initial,
                            config=self.config, tools=self.tools.identity)
            path = self.root / "manifest.json"
            if path.exists() and json.loads(path.read_text()) != manifest:
                raise ValueError("resume manifest changed")
            if not path.exists():
                write_json(path, manifest)
            state = deepcopy(self.initial)
            while not state["finished"]:
                action = choose(state, self.config)
                directory = self.root / f"step-{state['step']:04d}"
                directory.mkdir(exist_ok=True)
                action_path, outcome_path = directory / "action.json", directory / "outcome.json"
                if action_path.exists() and json.loads(action_path.read_text()) != action:
                    raise ValueError("stored action differs from deterministic replay")
                if not action_path.exists():
                    write_json(action_path, action)
                phase = "tool_execution_failed"
                try:
                    outcome = self.tools.execute(action)
                    if outcome_path.exists() and json.loads(outcome_path.read_text()) != outcome:
                        raise ValueError("stored outcome differs from verified tool evidence")
                    if not outcome_path.exists():
                        phase = "report_write_failed"
                        write_json(outcome_path, outcome)
                    phase = "state_settlement_failed"
                    after = reduce_outcome(state, action, outcome, self.config)
                    transition = dict(before_sha256=identity(state), action_sha256=file_hash(action_path),
                                      outcome_sha256=file_hash(outcome_path), after_sha256=identity(after))
                    settled = directory / "transition.json"
                    if settled.exists() and json.loads(settled.read_text()) != transition:
                        raise ValueError("stored state transition changed")
                    write_json(settled, transition)
                    state = after
                    write_json(self.root / "state.json", state)
                except Exception as exc:
                    failure = directory / f"failure-{len(list(directory.glob('failure-*.json'))):04d}.json"
                    write_json(failure, dict(status=phase,
                        error_type=type(exc).__name__, action_id=action["action_id"],
                        new_calculator_calls=0, original_evidence_preserved=True))
                    raise
            if sorted(p.name for p in self.root.glob("step-*")) != [f"step-{i:04d}" for i in range(state["step"])]:
                raise ValueError("unexpected trailing decision records")
            return state
