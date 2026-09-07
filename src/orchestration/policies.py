"""Policy engine -- single source of truth for escalation thresholds.
Consumed by: graph.py (decide node), agents.py (summary fallback), tests."""

from src.orchestration.state import (
    CrisisAssessment,
    Decision,
    ForecastResult,
    FusionResult,
)

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


def _hard_high(v) -> bool:
    try:
        return float(v.get("prob", v.get("score", 0))) >= 0.9
    except Exception:
        return False


def evaluate(
    crisis: CrisisAssessment | None,
    fusion: FusionResult | None,
    forecast: ForecastResult | None,
    signals: dict,
    cfg: dict,
) -> Decision:
    """Pure, unit-tested, explainable. Two INDEPENDENT human-review triggers:
    T1: high score OR (moderate + low confidence); T2: forecast probability.
    Crisis preempts all."""
    P, ver = cfg["policies"], cfg.get("policy_version", "?")
    if crisis and crisis.is_crisis:
        return Decision(route="crisis", reasons=[f"crisis precheck: {crisis.source}"], policy_version=ver)

    reasons: list[str] = []
    if fusion is None:
        hard = [k for k, v in signals.items() if v and _hard_high(v)]
        if hard:
            return Decision(route="escalate", reasons=[f"degraded mode, hard signal high: {hard}"], policy_version=ver)
        return Decision(route="routine", reasons=["degraded mode, no hard signals"], policy_version=ver)

    s, c = fusion.composite_score, fusion.confidence
    if s >= P["escalate_score"]:
        reasons.append(f"composite {s:.2f} >= {P['escalate_score']}")
    elif s >= 0.8 * P["escalate_score"] and c < P["escalate_confidence_floor"]:
        reasons.append(f"moderate score {s:.2f} with low confidence {c:.2f} -> human review")
    if forecast and forecast.p_escalation >= P["forecast_escalation_prob"]:
        reasons.append(f"forecast p(escalation)={forecast.p_escalation:.2f} >= {P['forecast_escalation_prob']}")

    return Decision(route="escalate" if reasons else "routine", reasons=reasons, policy_version=ver)


def summary_label(facts: dict, cfg: dict) -> tuple[str, str]:
    """Single source of threshold logic -- consumed by agents.py SummaryAgent
    fallback. Prevents policy drift between escalation and summary generation."""
    P = cfg.get("policies", {})
    hi = P.get("escalate_score", 0.75)
    mod = 0.8 * hi
    floor = P.get("escalate_confidence_floor", 0.45)
    fthr = P.get("forecast_escalation_prob", 0.60)
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
