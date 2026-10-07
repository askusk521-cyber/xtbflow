"""The single pre-registered V1a v2.1 gate."""
def decide_v1a(integrity_ok, complete, power_plan_frozen,
               primary, mech_vs_none, mech_vs_random,
               window_stop_supported=False):
    import math
    if not integrity_ok:
        return {"decision": "HOLD_IMPLEMENTATION", "reason": "invalid_comparison"}
    if not complete:
        return {"decision": "HOLD_INCOMPLETE", "reason": "restore_paired_units"}
    if not power_plan_frozen:
        return {"decision": "HOLD_POWER", "reason": "no_pre_screen_power_plan"}

    needed = ("lower_one95", "upper_one95")
    for summary in (primary, mech_vs_none, mech_vs_random):
        if any(k not in summary for k in needed):
            raise ValueError("missing one-sided confidence limits")
        if any(not math.isfinite(summary[k]) for k in needed):
            raise ValueError("nonfinite confidence limits")
        if summary["lower_one95"] > summary["upper_one95"]:
            raise ValueError("inverted confidence limits")

    if primary["upper_one95"] < 0.05:
        return {"decision": "NO_GO_RESOURCE_SMALL", "reason": "target_gain_excluded"}

    if mech_vs_none["upper_one95"] < 0.02 or mech_vs_random["upper_one95"] < 0.02:
        return {"decision": "NO_GO_MECHANISM_SMALL", "reason": "useful_mechanism_gain_excluded"}

    if (primary["lower_one95"] > 0 and mech_vs_none["lower_one95"] > 0
            and mech_vs_random["lower_one95"] > 0):
        return {"decision": "GO_V1b", "reason": "positive_efficiency_and_mechanism"}

    if window_stop_supported:
        return {"decision": "NO_GO_WINDOW_THIS_CONFIG",
                "reason": "no_supported_response_in_tested_window"}

    return {"decision": "HOLD_INCONCLUSIVE", "reason": "fixed_sample_not_decisive"}

