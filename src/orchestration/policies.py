ACTIONS = {
    "REQUEST FIELDS": "Request missing triage fields before escalation beyond routine monitoring.",
    "ESCALATE-24H": "Schedule a counsellor check-in call within 24 hours.",
    "VERIFY-HUMAN-24H": "Schedule a human check-in call within 24 hours to verify before escalation.",
    "HUMAN-72H": "Assign a human check-in call within 72 hours to disambiguate.",
    "HUMAN-48H": "Schedule a human check-in within 48 hours and screen for intimidation per protocol.",
    "MONITOR-CLOSELY": "Keep the case on the counsellor watchlist and re-evaluate after the next check-in.",
    "PROACTIVE-24H": "Schedule a proactive check-in within 24 hours.",
    "ROUTINE": "Continue routine scheduled check-ins.",
}


def summary_label(facts: dict, cfg: dict) -> tuple[str, str]:
    """Single source of threshold logic -- consumed by evaluate() and the
    deterministic summary fallback. Prevents policy drift."""
    P = cfg.get("policies", {})
    hi, mod = P.get("escalate_score", 0.75), 0.8 * P.get("escalate_score", 0.75)
    floor, fthr = P.get("escalate_confidence_floor", 0.45), P.get("forecast_escalation_prob", 0.60)
    s, c = facts.get("composite_score"), facts.get("confidence")
    threat = "threat_language_detected" in (facts.get("top_signals") or [])
    if s is None:
        return "INSUFFICIENT DATA", ACTIONS["REQUEST FIELDS"]
    if s >= hi:
        if c is not None and c < floor:
            return "HIGH (LOW CONFIDENCE)", ACTIONS["VERIFY-HUMAN-24H"]
        return "HIGH", ACTIONS["ESCALATE-24H"]
    if s >= mod:
        if c is not None and c < floor:
            return "MODERATE (LOW CONFIDENCE)", ACTIONS["HUMAN-72H"]
        if threat:
            return "MODERATE (THREAT SIGNAL)", ACTIONS["HUMAN-48H"]
        return "MODERATE", ACTIONS["MONITOR-CLOSELY"]
    if threat:
        return "LOW (THREAT SIGNAL)", ACTIONS["HUMAN-72H"]
    if facts.get("p_escalation") is not None and facts["p_escalation"] >= fthr:
        return "LOW (FORECAST ALERT)", ACTIONS["PROACTIVE-24H"]
    return "LOW", ACTIONS["ROUTINE"]
