# tests/test_policies.py
from src.orchestration.policies import evaluate
from src.orchestration.state import CrisisAssessment, ForecastResult, FusionResult

CFG = {
    "policies": {"escalate_score": 0.75, "escalate_confidence_floor": 0.45, "forecast_escalation_prob": 0.60},
    "policy_version": "test",
}


def test_crisis_overrides():
    d = evaluate(CrisisAssessment(is_crisis=True), FusionResult(composite_score=0.1, confidence=0.9), None, {}, CFG)
    assert d.route == "crisis"


def test_high_score_escalates():
    d = evaluate(None, FusionResult(composite_score=0.85, confidence=0.9), None, {}, CFG)
    assert d.route == "escalate"
    assert "composite" in d.reasons[0]


def test_moderate_low_conf():
    d = evaluate(None, FusionResult(composite_score=0.65, confidence=0.3), None, {}, CFG)
    assert d.route == "escalate"


def test_forecast_alone_escalates():
    d = evaluate(None, FusionResult(composite_score=0.4, confidence=0.9), ForecastResult(p_escalation=0.8), {}, CFG)
    assert d.route == "escalate"


def test_degraded_hard_signal():
    assert evaluate(None, None, None, {"voice": {"score": 0.95}}, CFG).route == "escalate"


def test_degraded_routine():
    assert evaluate(None, None, None, {"voice": None}, CFG).route == "routine"


def test_routine():
    d = evaluate(None, FusionResult(composite_score=0.3, confidence=0.9), ForecastResult(p_escalation=0.1), {}, CFG)
    assert d.route == "routine"
