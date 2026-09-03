from src.orchestration.state import CrisisAssessment, Decision, FusionResult, ForecastResult


def _hard_high(v) -> bool:
    try:
        return float(v.get("prob", v.get("score", 0))) >= 0.9
    except Exception:
        return False


def evaluate(crisis: CrisisAssessment | None, fusion: FusionResult | None,
             forecast: ForecastResult | None, signals: dict, cfg: dict) -> Decision:
    """Pure, unit-tested, explainable. Two INDEPENDENT human-review triggers:
    T1 high score OR (moderate + low confidence); T2 forecast probability. Crisis preempts all."""
    P, ver = cfg["policies"], cfg.get("policy_version", "?")
    if crisis and crisis.is_crisis:
        return Decision(route="crisis", reasons=[f"crisis precheck: {crisis.source}"], policy_version=ver)

    reasons: list[str] = []
    if fusion is None:  # degraded mode: downstream services unreachable
        hard = [k for k, v in signals.items() if v and _hard_high(v)]
        if hard:
            return Decision(route="escalate", reasons=[f"degraded mode, hard signal high: {hard}"],
                            policy_version=ver)
        return Decision(route="routine", reasons=["degraded mode, no hard signals"], policy_version=ver)

    s, c = fusion.composite_score, fusion.confidence
    if s >= P["escalate_score"]:
        reasons.append(f"composite {s:.2f} >= {P['escalate_score']}")
    elif s >= 0.8 * P["escalate_score"] and c < P["escalate_confidence_floor"]:
        reasons.append(f"moderate score {s:.2f} with low confidence {c:.2f} -> human review")
    if forecast and forecast.p_escalation >= P["forecast_escalation_prob"]:
        reasons.append(f"forecast p(escalation)={forecast.p_escalation:.2f} >= {P['forecast_escalation_prob']}")

    return Decision(route="escalate" if reasons else "routine", reasons=reasons, policy_version=ver)