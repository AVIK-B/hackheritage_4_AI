"""Agent layer unit tests: bulkhead, retry, fallback ladder, validation,
quarantine, cache, idempotency. All dependencies faked; no network."""

import threading

import pytest

from src.orchestration.agents import (
    AlertDispatchAgent,
    ErrorCode,
    FusionAgent,
    Priority,
    Quarantine,
    SentimentAgent,
    SummaryAgent,
    ThreatAgent,
    TTLCache,
)
from src.orchestration.state import Decision, FusionResult


class FakeHTTP:
    """Scripted httpx.Client double. Matches URLs by substring."""

    def __init__(self, behavior: dict[str, list] | None = None):
        self.calls: list[tuple] = []
        self._behavior = behavior or {}

    def queue(self, url_part: str, outcomes: list) -> None:
        self._behavior[url_part] = outcomes

    def _next(self, url: str):
        for key, seq in self._behavior.items():
            if key in url:
                return seq.pop(0) if len(seq) > 1 else seq[0]
        return None

    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append(("POST", url, json, headers))
        r = self._next(url)
        if isinstance(r, Exception):
            raise r
        if r is None:
            raise ConnectionError(f"no route for {url}")
        return _Resp(r)

    def get(self, url, timeout=None):
        self.calls.append(("GET", url, None, None))
        r = self._next(url)
        if isinstance(r, Exception):
            raise r
        if r is None:
            raise ConnectionError(f"no route for {url}")
        return _Resp(r)


class _Resp:
    def __init__(self, payload):
        self._p = payload
        self.status_code = 200 if not isinstance(payload, dict) or "status" not in payload else payload["status"]
        self.text = str(payload)

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


@pytest.fixture
def cfg():
    return {
        "services": {
            "scoring": "http://scoring",
            "fusion": "http://fusion",
            "forecast": "http://forecast",
            "backend": "http://backend",
        },
        "policies": {"escalate_score": 0.75},
    }


@pytest.fixture(autouse=True)
def _reset_breakers():
    from src.common.resilience import breakers

    breakers.reset()
    yield
    breakers.reset()


# ── TTLCache ──────────────────────────────────────────────────────────────────
def test_ttl_cache_roundtrip_and_expiry():
    fake = {"t": 0.0}
    c = TTLCache(ttl_s=0.05, clock=lambda: fake["t"])
    c.put(["k"], {"v": 1})
    assert c.get(["k"]) == {"v": 1}
    fake["t"] += 1.0
    assert c.get(["k"]) is None


# ── retry + fallback ladder ──────────────────────────────────────────────────
def test_sentiment_retry_then_success(cfg):
    http = FakeHTTP()
    http.queue(
        "/v1/signals/text",
        [ConnectionError("boom1"), {"sentiment": {"label": "negative", "score": 0.8}}],
    )
    agent = SentimentAgent(cfg, http)
    res = agent.run({"message": "x", "language": "en"})
    assert res.ok, f"res.ok={res.ok}, err={res.error}"
    assert res.attempts == 2
    assert res.data["sentiment"]["score"] == 0.8


def test_sentiment_fallback_on_persistent_failure(cfg):
    http = FakeHTTP()
    http.queue("/v1/signals/text", [ConnectionError("down")])
    agent = SentimentAgent(cfg, http)
    res = agent.run({"message": "x", "language": "en"})
    assert res.degraded and res.data["sentiment"]["score"] == 0.5


# ── validation → quarantine ──────────────────────────────────────────────────
def test_bad_payload_goes_to_quarantine(cfg, tmp_path):
    q = Quarantine(str(tmp_path / "q.jsonl"))
    http = FakeHTTP()
    http.queue("/v1/signals/text", [{"sentiment": {"label": "neg", "score": 9.9}}])
    agent = SentimentAgent(cfg, http)
    agent.quarantine = q
    res = agent.run({"message": "x", "language": "en"})
    assert res.degraded
    assert q.path.exists(), f"quarantine file missing: {q.path}"
    assert "VALIDATION_FAILED" in q.path.read_text()


# ── conservative threat fallback ─────────────────────────────────────────────
def test_threat_fallback_is_conservative(cfg):
    agent = ThreatAgent(cfg, FakeHTTP())
    res = agent.run({"message": "x", "language": "en"})
    assert res.data["threat"]["prob"] == 0.55 and res.data["threat"]["_assumed"]


# ── fusion local approximation ───────────────────────────────────────────────
def test_fusion_fallback_agreement_vs_disagreement(cfg):
    agent = FusionAgent(cfg, FakeHTTP())
    agree = agent._fallback({"signals": {"sentiment": {"score": 0.8}, "voice": {"score": 0.82}}}, None).data
    disagree = agent._fallback({"signals": {"sentiment": {"score": 0.9}, "voice": {"score": 0.2}}}, None).data
    assert disagree["confidence"] < agree["confidence"]


# ── bulkhead saturation ──────────────────────────────────────────────────────
def test_bulkhead_saturates(cfg):
    import time as _t

    agent = SentimentAgent(cfg, FakeHTTP())
    agent.bulkhead = type(agent.bulkhead)("t", 1)
    with agent.bulkhead.sem:
        results = []

        def hit():
            results.append(agent.run({"message": "x", "language": "en"}))

        t = threading.Thread(target=hit)
        t.start()
        _t.sleep(0.1)
        assert results[0].degraded and results[0].error == ErrorCode.BULKHEAD_SATURATED
        t.join()


# ── idempotency key stability ────────────────────────────────────────────────
def test_dispatch_idempotency_key_stable(cfg):
    http = FakeHTTP()
    http.queue("/v1/alerts", [{"alert_id": "A1"}, {"alert_id": "A1"}])  # TWO responses
    agent = AlertDispatchAgent(cfg, http)
    state = {
        "case_id": "CASE-1",
        "thread_id": "t",
        "summary_text": "",
        "decision": Decision(route="escalate", reasons=["r1"]),
    }
    agent.run(state)
    agent.run(state)
    _, _, _, h1 = http.calls[0]
    _, _, _, h2 = http.calls[1]
    assert h1["Idempotency-Key"] == h2["Idempotency-Key"]


# ── summary grounding regen + deterministic fallback ─────────────────────────
class FakeLLM:
    SYSTEM_A = SYSTEM_B = "sys"

    def __init__(self, outs):
        self.model_map = {"dialogue": "sahayak", "summary": "casewriter"}
        self.outs = list(outs)
        self.calls = []

    def chat(self, model, msgs, **kw):
        self.calls.append((model, msgs))
        return self.outs.pop(0)


def _summary_state():
    return {
        "case_id": "CASE-1",
        "thread_id": "t",
        "case_ctx": {},
        "fusion": FusionResult(composite_score=0.90, confidence=0.85, top_signals=["voice_stress_high"]),
        "forecast": None,
        "errors": [],
        "decision": None,
    }


def test_summary_grounded_ok(cfg):
    http = FakeHTTP()
    llm = FakeLLM(
        [
            (
                "Risk: HIGH (composite 0.90, confidence 0.85). Stage: trial. "
                "Key signals: voice_stress_high. Recommended action: schedule a call."
            )
        ]
    )
    agent = SummaryAgent(cfg, http, llm)
    res = agent.run(_summary_state())
    assert res.ok and res.data["grounded"] is True


def test_summary_ungrounded_falls_back_to_deterministic(cfg):
    http = FakeHTTP()
    llm = FakeLLM(
        [
            "Risk: HIGH (composite 0.99, confidence 0.99). Invented.",
            "Risk: HIGH (composite 0.99, confidence 0.99). Invented again.",
        ]
    )
    agent = SummaryAgent(cfg, http, llm)
    res = agent.run(_summary_state())
    assert res.ok, f"res.ok={res.ok}, err={res.error}"
    assert res.data.get("grounded") is False
    assert "0.99" not in res.data["text"]


# ── priority/deadline skip ───────────────────────────────────────────────────
def test_best_effort_skipped_when_budget_spent(cfg):
    from src.orchestration.agents import Deadline

    agent = SentimentAgent(cfg, FakeHTTP())
    agent.priority = Priority.BEST_EFFORT
    state = {"message": "x", "language": "en", "_deadline": Deadline(budget_ms=0.0001)}
    res = agent.run(state)
    assert res.skipped and res.error == ErrorCode.BUDGET_EXCEEDED
