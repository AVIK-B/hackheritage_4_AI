"""PS-26094 Agent Layer v2.1 -- production specialist agents.

Per-agent armor (outermost first): bulkhead -> deadline-aware skip ->
circuit breaker -> jittered retry -> schema validation -> grounding guard
-> TTL cache -> metrics -> fallback ladder -> quarantine.

Invariants:
  Crisis precheck: deterministic lexicon, never an LLM, cannot fail.
  Escalation: pure functions in policies.py. Agents produce EVIDENCE.
  Safety replies: templated or enforced; a model failure can never
  suppress a crisis handoff (enforce_crisis_handoff).
"""

from __future__ import annotations

import contextlib
import enum
import hashlib
import json
import pathlib
import random
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from src.common.resilience import breakers
from src.common.safety import (
    FALLBACK_REPLY,
    assess_crisis,
    scrub_pii,
)
from src.orchestration.policies import summary_label
from src.orchestration.state import (
    CrisisAssessment,
    Decision,
    ForecastResult,
    FusionResult,
)

log = structlog.get_logger()

# ═══════════════════════════ metrics ═══════════════════════════════════════
try:
    from prometheus_client import Counter, Gauge, Histogram

    AGENT_CALLS = Counter("agent_calls_total", "agent calls", ["agent", "outcome"])
    AGENT_LAT = Histogram(
        "agent_latency_seconds",
        "agent latency",
        ["agent"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
    )
    AGENT_HEALTH = Gauge("agent_health_score", "rolling success 0-1", ["agent"])
    AGENT_INFLIGHT = Gauge("agent_inflight", "bulkhead occupancy", ["agent"])
    DEGRADATION = Gauge("system_degradation_level", "0=L0 full..3=L3 minimal")
except Exception:  # pragma: no cover

    class _N:
        def labels(self, *a, **k):
            return self

        def inc(self, *a, **k):
            pass

        def observe(self, *a, **k):
            pass

        def set(self, *a, **k):
            pass

    AGENT_CALLS = _N()
    AGENT_LAT = _N()
    AGENT_HEALTH = _N()
    AGENT_INFLIGHT = _N()
    DEGRADATION = _N()


# ═══════════════════════════ error taxonomy ════════════════════════════════
class ErrorCode(enum.StrEnum):
    BREAKER_OPEN = "BREAKER_OPEN"
    TIMEOUT = "TIMEOUT"
    UNREACHABLE = "UNREACHABLE"
    BAD_PAYLOAD = "BAD_PAYLOAD"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    BULKHEAD_SATURATED = "BULKHEAD_SATURATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INTERNAL = "INTERNAL"


class AgentError(Exception):
    def __init__(self, code: ErrorCode, detail: str = ""):
        self.code, self.detail = code, detail
        super().__init__(f"{code.value}: {detail}")


@dataclass
class AgentResult:
    agent: str
    ok: bool
    data: Any = None
    error: ErrorCode | None = None
    detail: str | None = None
    latency_ms: int = 0
    attempts: int = 1
    degraded: bool = False
    cached: bool = False
    skipped: bool = False
    raw: dict | None = None


# ═══════════════════════════ infrastructure ════════════════════════════════
class Deadline:
    def __init__(self, budget_ms: float):
        self.t0 = time.perf_counter()
        self.budget_ms = budget_ms

    def fraction_left(self) -> float:
        used = (time.perf_counter() - self.t0) * 1000
        return max(0.0, (self.budget_ms - used)) / max(1.0, self.budget_ms)


class Priority(enum.Enum):
    CRITICAL = 0
    HIGH = 1
    BEST_EFFORT = 2


class TTLCache:
    def __init__(self, ttl_s: float, maxsize: int = 512, clock=time.monotonic):
        self.ttl, self.maxsize, self._clock = ttl_s, maxsize, clock
        self._d: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def _key(self, parts: list) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, parts: list):
        k = self._key(parts)
        with self._lock:
            hit = self._d.get(k)
            if hit and self._clock() - hit[0] <= self.ttl:
                return hit[1]
            self._d.pop(k, None)
            return None

    def put(self, parts: list, val: Any) -> None:
        with self._lock:
            if len(self._d) >= self.maxsize:
                self._d.pop(next(iter(self._d)), None)
            self._d[self._key(parts)] = (self._clock(), val)


class Bulkhead:
    def __init__(self, name: str, size: int = 4):
        self.name, self.sem = name, threading.BoundedSemaphore(size)

    def acquire_or_raise(self):
        if not self.sem.acquire(timeout=0.05):
            raise AgentError(ErrorCode.BULKHEAD_SATURATED, self.name)

    def release(self):
        self.sem.release()


class HealthWindow:
    def __init__(self, alpha: float = 0.3):
        self.alpha, self._ema = alpha, 1.0

    def record(self, ok: bool):
        self._ema = self.alpha * (1.0 if ok else 0.0) + (1 - self.alpha) * self._ema

    def score(self) -> float:
        return round(self._ema, 3)

    def level(self) -> int:
        s = self.score()
        return 0 if s >= 0.9 else 1 if s >= 0.7 else 2 if s >= 0.4 else 3


class Quarantine:
    def __init__(self, path: str = "./data/quarantine.jsonl"):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def put(self, agent: str, payload: Any, reason: str):
        rec = {
            "ts": time.time(),
            "agent": agent,
            "reason": reason,
            # AFTER:
            "payload": payload if isinstance(payload, dict | list | str | int | float) else str(payload)[:2000],
        }
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ═══════════════════════════ base agent ════════════════════════════════════
class BaseAgent:
    name = "base"
    timeout_s: float = 3.0
    max_attempts: int = 2
    priority: Priority = Priority.HIGH
    cache_ttl_s: float = 0.0
    bulkhead_size: int = 4
    backoff_base_s: float = 0.25

    def __init__(self, cfg: dict, http: httpx.Client, llm=None):
        self.cfg, self.http, self.llm = cfg, http, llm
        self.breaker = breakers.get(self.name)
        self.bulkhead = Bulkhead(self.name, self.bulkhead_size)
        self.health = HealthWindow()
        self.cache = None
        self.quarantine = None

    def run(self, state: dict) -> AgentResult:
        t0 = time.perf_counter()
        deadline: Deadline | None = state.get("_deadline")

        if deadline and self.priority != Priority.CRITICAL:
            frac = deadline.fraction_left()
            limit = 0.6 if self.priority == Priority.BEST_EFFORT else 0.2
            if frac < limit:
                res = AgentResult(
                    self.name, ok=False, skipped=True, error=ErrorCode.BUDGET_EXCEEDED, detail=f"budget {frac:.0%}"
                )
                self._emit(res, t0, ok=False)
                return res

        try:
            self.bulkhead.acquire_or_raise()
        except AgentError as e:
            res = self._fallback(state, e)
            res.error, res.degraded = e.code, True
            self._emit(res, t0, ok=False)
            return res
        AGENT_INFLIGHT.labels(self.name).inc()
        try:
            return self._guarded(state, t0)
        finally:
            self.bulkhead.release()
            AGENT_INFLIGHT.labels(self.name).dec()

    def _guarded(self, state: dict, t0: float) -> AgentResult:
        ck = self._cache_key(state) if (self.cache_ttl_s > 0 and self.cache) else None
        if ck is not None:
            hit = self.cache.get(ck)
            if hit is not None:
                res = AgentResult(self.name, True, data=hit, cached=True)
                self._emit(res, t0, ok=True)
                return res
        if not self.breaker.allow():
            res = self._fallback(state, AgentError(ErrorCode.BREAKER_OPEN, self.name))
            res.error, res.degraded = ErrorCode.BREAKER_OPEN, True
            self._emit(res, t0, ok=False)
            return res
        last: Exception | None = None
        last_payload: Any = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw_result = self._run(state)
                last_payload = raw_result
                data = self._validate(raw_result)
                res = AgentResult(
                    self.name,
                    True,
                    data=data,
                    attempts=attempt,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    raw=data if isinstance(data, dict) else None,
                )
                if ck is not None and self.cache:
                    self.cache.put(ck, data)
                self._emit(res, t0, ok=True)
                return res
            except AgentError as e:
                last = e
                self.breaker.record(False)
                self.health.record(False)
            except AssertionError as e:
                last = AgentError(ErrorCode.VALIDATION_FAILED, str(e))
                self.breaker.record(False)
                self.health.record(False)
            log.warning("agent_attempt_failed", agent=self.name, attempt=attempt, err=str(last))
            if attempt < self.max_attempts:
                time.sleep(self.backoff_base_s * (2 ** (attempt - 1)) * (0.5 + random.random()))

        if isinstance(last, AgentError) and last.code == ErrorCode.VALIDATION_FAILED:
            with contextlib.suppress(Exception):
                self.quarantine.put(self.name, last_payload, f"VALIDATION_FAILED: {last.detail}")
        res = self._fallback(state, last)
        res.degraded = True

        self._emit(res, t0, ok=False)

        return res

    def _run(self, state) -> Any:
        raise NotImplementedError

    def _fallback(self, state, err) -> AgentResult:
        return AgentResult(self.name, False, error=getattr(err, "code", ErrorCode.INTERNAL), detail=str(err))

    def _validate(self, data):
        s = (data or {}).get("sentiment") or {}
        assert isinstance(s.get("score"), (int | float)) and 0 <= s["score"] <= 1, (
            f"sentiment score out of range: {s.get('score')}"
        )
        return data

    def _cache_key(self, state):
        return None

    def _post(self, url: str, payload: dict) -> dict:
        try:
            r = self.http.post(url, json=payload, timeout=self.timeout_s)
            if r.status_code >= 500:
                raise AgentError(ErrorCode.UNREACHABLE, f"5xx {r.status_code}")
            if r.status_code >= 400:
                raise AgentError(ErrorCode.BAD_PAYLOAD, f"4xx {r.status_code}")
            return r.json()
        except AgentError:
            raise
        except Exception as e:
            raise AgentError(ErrorCode.UNREACHABLE, str(e)) from e

    def _get(self, url: str) -> dict:
        try:
            r = self.http.get(url, timeout=self.timeout_s)
            if r.status_code >= 500:
                raise AgentError(ErrorCode.UNREACHABLE, f"5xx {r.status_code}")
            if r.status_code >= 400:
                raise AgentError(ErrorCode.BAD_PAYLOAD, f"4xx {r.status_code}")
            return r.json()
        except AgentError:
            raise
        except Exception as e:
            raise AgentError(ErrorCode.UNREACHABLE, str(e)) from e

    def _emit(self, res, t0, ok):
        outcome = (
            "cached"
            if res.cached
            else "skipped"
            if res.skipped
            else "ok"
            if ok and not res.degraded
            else "fallback"
            if res.degraded
            else "error"
        )
        AGENT_CALLS.labels(self.name, outcome).inc()
        AGENT_LAT.labels(self.name).observe(time.perf_counter() - t0)
        AGENT_HEALTH.labels(self.name).set(self.health.score())
        log.info(
            "agent",
            agent=self.name,
            outcome=outcome,
            ms=res.latency_ms,
            attempts=res.attempts,
            err=(res.error.value if res.error else res.detail),
        )


# ═══════════════════════════ specialists ═══════════════════════════════════
class CrisisPrecheckAgent(BaseAgent):
    name = "crisis_precheck"
    timeout_s = 2.0
    priority = Priority.CRITICAL

    def _run(self, state) -> CrisisAssessment:
        text = scrub_pii(state.get("message", ""))
        lang = state.get("language", "en")
        a = assess_crisis(text, lang)
        try:
            r = self.http.post(
                f"{self.cfg['services']['scoring']}/v1/signals/threat",
                json={"text": text, "lang": lang},
                timeout=self.timeout_s,
            )
            if r.ok:
                prob = (r.json() or {}).get("threat", {}).get("prob", 0)
                if prob and prob >= 0.9:
                    a["hits"] = [*list(a["hits"]), f"threat_svc:{prob:.2f}"]
        except Exception as enrichment_err:
            log.debug("threat_enrichment_skipped", err=str(enrichment_err))
        return CrisisAssessment(**a)

    def _fallback(self, state, err):
        a = assess_crisis(scrub_pii(state.get("message", "")), state.get("language", "en"))
        return AgentResult(self.name, True, data=CrisisAssessment(**a), degraded=True)

    def _validate(self, data):
        assert isinstance(data, CrisisAssessment)
        return data


class SentimentAgent(BaseAgent):
    name = "sentiment"
    timeout_s = 2.5
    cache_ttl_s = 30.0

    def _cache_key(self, state):
        return [state.get("message", ""), state.get("language", "en")]

    def _run(self, state):
        return self._post(
            f"{self.cfg['services']['scoring']}/v1/signals/text",
            {"text": state.get("message", ""), "lang": state.get("language", "en")},
        )

    def _validate(self, data):
        s = (data or {}).get("sentiment") or {}
        assert isinstance(s.get("score"), (int | float)) and 0 <= s["score"] <= 1, (
            f"sentiment score out of range: {s.get('score')}"
        )
        return data

    def _fallback(self, state, err):
        return AgentResult(
            self.name,
            False,
            data={"sentiment": {"label": "unknown", "score": 0.5}, "_degraded": True},
            error=getattr(err, "code", ErrorCode.INTERNAL),
            degraded=True,
        )


class ThreatAgent(BaseAgent):
    name = "threat"
    timeout_s = 2.5
    cache_ttl_s = 30.0

    def _cache_key(self, state):
        return [state.get("message", ""), state.get("language", "en")]

    def _run(self, state):
        return self._post(
            f"{self.cfg['services']['scoring']}/v1/signals/threat",
            {"text": state.get("message", ""), "lang": state.get("language", "en")},
        )

    def _validate(self, data):
        t = (data or {}).get("threat") or {}
        assert isinstance(t.get("prob"), (int | float)) and 0 <= t["prob"] <= 1
        return data

    def _fallback(self, state, err):
        return AgentResult(
            self.name,
            False,
            data={"threat": {"flag": False, "prob": 0.55, "_assumed": True}, "_degraded": True},
            error=getattr(err, "code", ErrorCode.INTERNAL),
            degraded=True,
        )


class VoiceAgent(BaseAgent):
    name = "voice"
    timeout_s = 4.0
    priority = Priority.BEST_EFFORT

    def _run(self, state):
        if not state.get("audio_ref"):
            raise AgentError(ErrorCode.NOT_APPLICABLE, "no audio_ref")
        return self._post(f"{self.cfg['services']['scoring']}/v1/signals/voice", {"audio_ref": state["audio_ref"]})

    def _validate(self, data):
        assert isinstance(((data or {}).get("voice_stress") or {}).get("score"), (int | float))
        return data

    def _fallback(self, state, err):
        return AgentResult(self.name, False, data=None, error=getattr(err, "code", ErrorCode.INTERNAL))


class ContextAgent(BaseAgent):
    name = "case_context"
    timeout_s = 3.0
    cache_ttl_s = 45.0

    def _cache_key(self, state):
        return ["ctx", state.get("case_id", "")]

    def _run(self, state):
        return self._get(f"{self.cfg['services']['backend']}/v1/case/{state['case_id']}/context")

    def _validate(self, data):
        assert {"case_stage", "language"} <= set(data or {})
        return data

    def _fallback(self, state, err):
        return AgentResult(
            self.name,
            False,
            data={
                "case_stage": "unknown",
                "days_to_next_hearing": None,
                "language": state.get("language", "en"),
                "consent_scopes": [],
                "recent_checkins": [],
                "recent_messages": [],
                "score_history": [],
                "_degraded": True,
            },
            error=getattr(err, "code", ErrorCode.INTERNAL),
            degraded=True,
        )


class FusionAgent(BaseAgent):
    name = "fusion"
    timeout_s = 3.0
    priority = Priority.CRITICAL
    cache_ttl_s = 15.0

    def _cache_key(self, state):
        sig = state.get("signals") or {}
        return [
            "fus",
            sig.get("sentiment"),
            sig.get("threat"),
            sig.get("voice"),
            (state.get("case_ctx") or {}).get("case_stage"),
        ]

    def _run(self, state):
        sig = state.get("signals") or {}
        return self._post(
            f"{self.cfg['services']['fusion']}/v1/fusion",
            {
                "sentiment": sig.get("sentiment"),
                "threat": sig.get("threat"),
                "voice_stress": sig.get("voice"),
                "engagement": (state.get("case_ctx") or {}).get("recent_checkins"),
                "case_stage": (state.get("case_ctx") or {}).get("case_stage"),
            },
        )

    def _validate(self, data):
        f = FusionResult(**data)
        assert 0 <= f.composite_score <= 1 and 0 <= f.confidence <= 1
        return f.model_dump()

    def _fallback(self, state, err):
        sig = state.get("signals") or {}

        def sc(d, k="score"):
            try:
                return float((d or {}).get(k) or 0)
            except Exception:
                return 0.0

        sent, thr, voi = sc(sig.get("sentiment")), sc(sig.get("threat"), "prob"), sc(sig.get("voice"))
        score = round(0.45 * sent + 0.35 * voi + 0.20 * thr, 3)
        vals = [v for v in (sent, voi, thr) if v > 0]
        spread = (max(vals) - min(vals)) if vals else 0.5
        conf = round(max(0.10, min(0.9, 0.85 - spread)), 2)
        top = [n for n, v in (("sentiment", sent), ("voice", voi), ("threat", thr)) if v > 0.5][:3]
        return AgentResult(
            self.name,
            False,
            data=FusionResult(composite_score=score, confidence=conf, top_signals=top).model_dump(),
            error=getattr(err, "code", ErrorCode.INTERNAL),
            degraded=True,
        )


class ForecastAgent(BaseAgent):
    name = "forecast"
    timeout_s = 3.0
    priority = Priority.BEST_EFFORT
    cache_ttl_s = 60.0

    def _cache_key(self, state):
        return ["fc", state.get("case_id", ""), (state.get("case_ctx") or {}).get("score_history", [])]

    def _run(self, state):
        hist = (state.get("case_ctx") or {}).get("score_history", [])
        return self._post(
            f"{self.cfg['services']['forecast']}/v1/forecast", {"case_id": state["case_id"], "score_history": hist}
        )

    def _validate(self, data):
        f = ForecastResult(**data)
        assert 0 <= f.p_escalation <= 1 and f.horizon_days >= 1
        return f.model_dump()

    def _fallback(self, state, err):
        hist = (state.get("case_ctx") or {}).get("score_history", [])
        trend = (hist[-1] - sum(hist[:-1]) / (len(hist) - 1)) if len(hist) >= 2 else 0
        p = min(0.9, max(0.05, 0.15 + trend * 0.8)) if hist else 0.10
        return AgentResult(
            self.name,
            False,
            data=ForecastResult(p_escalation=round(p, 3), horizon_days=7).model_dump(),
            error=getattr(err, "code", ErrorCode.INTERNAL),
            degraded=True,
        )


class AlertDispatchAgent(BaseAgent):
    name = "alert_dispatch"
    timeout_s = 5.0
    max_attempts = 3
    priority = Priority.CRITICAL
    backoff_base_s = 0.5

    def _key(self, state):
        dec: Decision = state["decision"]
        return hashlib.sha256(
            json.dumps([state["case_id"], dec.route, dec.reasons], sort_keys=True).encode()
        ).hexdigest()

    def _run(self, state):
        dec: Decision = state["decision"]
        r = self.http.post(
            f"{self.cfg['services']['backend']}/v1/alerts",
            json={
                "case_id": state["case_id"],
                "severity": dec.route,
                "reasons": dec.reasons,
                "summary_text": state.get("summary_text", ""),
            },
            headers={"Idempotency-Key": self._key(state)},
            timeout=self.timeout_s,
        )
        if r.status_code >= 500:
            raise AgentError(ErrorCode.UNREACHABLE, f"5xx {r.status_code}") from None
        if r.status_code >= 400:
            raise AgentError(ErrorCode.BAD_PAYLOAD, f"{r.status_code}") from None
        return r.json()

    def _validate(self, data):
        assert (data or {}).get("alert_id")
        return data

    def _fallback(self, state, err):
        code = getattr(err, "code", ErrorCode.INTERNAL)
        log.error(
            "alert_dispatch_deadletter",
            case_id=state.get("case_id"),
            route=state.get("decision").route if state.get("decision") else "?",
            code=code.value if isinstance(code, ErrorCode) else str(code),
        )
        return AgentResult(self.name, False, error=code if isinstance(code, ErrorCode) else ErrorCode.INTERNAL)


class ConversationAgent(BaseAgent):
    name = "conversation"
    timeout_s = 25.0
    max_attempts = 2
    priority = Priority.CRITICAL

    def _history(self, state):
        out = []
        for item in (state.get("case_ctx") or {}).get("recent_messages", [])[-6:]:
            if not isinstance(item, dict):
                continue
            role, content = item.get("role"), scrub_pii(str(item.get("content", ""))).strip()
            if role in ("user", "assistant") and content:
                out.append({"role": role, "content": content[:1000]})
        return out

    def _context_msg(self, state):
        ctx = state.get("case_ctx") or {}
        parts = [f"User language: {state.get('language', 'en')}."]
        if ctx.get("case_stage"):
            parts.append(f"Case stage: {ctx['case_stage']}.")
        if ctx.get("days_to_next_hearing") is not None:
            parts.append(f"Days to next hearing: {ctx['days_to_next_hearing']}.")
        if len(parts) == 1:
            return None
        parts.append("Use only to tune supportiveness; never reveal scores, monitoring, or internal analysis.")
        return {"role": "system", "content": " ".join(parts)}

    def _run(self, state):
        msgs = [{"role": "system", "content": self.llm.SYSTEM_A}]
        ctx = self._context_msg(state)
        if ctx:
            msgs.append(ctx)
        msgs += [*self._history(state), {"role": "user", "content": state["message"]}]
        try:
            return self.llm.chat(self.llm.model_map["dialogue"], msgs, temperature=0.6, max_tokens=220)
        except Exception:
            return self.llm.chat(self.llm.model_map["dialogue"], msgs, temperature=0.3, max_tokens=220)

    def _validate(self, data):
        assert isinstance(data, str) and len(data.strip()) >= 2
        return data

    def _fallback(self, state, err):
        lang = state.get("language", "en")
        return AgentResult(
            self.name,
            False,
            data=FALLBACK_REPLY.get(lang, FALLBACK_REPLY["en"]),
            error=getattr(err, "code", ErrorCode.INTERNAL),
            degraded=True,
        )


class SummaryAgent(BaseAgent):
    name = "summary"
    timeout_s = 20.0
    max_attempts = 2
    priority = Priority.HIGH

    def _facts(self, state) -> dict:
        f, fc = state.get("fusion"), state.get("forecast")
        return {
            "case_id": state["case_id"],
            "case_stage": (state.get("case_ctx") or {}).get("case_stage", "data unavailable"),
            "days_to_next_hearing": (state.get("case_ctx") or {}).get("days_to_next_hearing"),
            "composite_score": f.composite_score if f else None,
            "confidence": f.confidence if f else None,
            "top_signals": f.top_signals if f else [],
            "p_escalation": fc.p_escalation if fc else None,
            "errors": state.get("errors", []),
        }

    def _gen(self, facts, t):
        return self.llm.chat(
            self.llm.model_map["summary"],
            [
                {"role": "system", "content": self.llm.SYSTEM_B},
                {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
            ],
            temperature=t,
            max_tokens=300,
        )

    def _grounded(self, out: str, facts: dict) -> bool:
        if not out.startswith("Risk: ") or "Recommended action:" not in out:
            return False
        known = {
            f"{facts[k]:.2f}" for k in ("composite_score", "confidence", "p_escalation") if facts.get(k) is not None
        }
        if facts.get("days_to_next_hearing") is not None:
            known.add(str(facts["days_to_next_hearing"]))
        bad = {n for n in re.findall(r"\b\d+(?:\.\d+)?\b", out) if n not in known and n != "7"}
        return not bad

    def _run(self, state):
        facts = self._facts(state)
        out = self._gen(facts, 0.2)
        if not self._grounded(out, facts):
            log.warning("summary_ungrounded_regen", case=state["case_id"])
            out = self._gen(facts, 0.1)
        if not self._grounded(out, facts):
            label, action = summary_label(facts, self.cfg)

            def fm(x):
                return "data unavailable" if x is None else f"{x:.2f}"

            out = (
                f"Risk: {label} (composite {fm(facts['composite_score'])}, "
                f"confidence {fm(facts['confidence'])}). Stage: {facts['case_stage']}. "
                f"Recommended action: {action}"
            )
            return {"text": out, "facts": facts, "grounded": False}
        return {"text": out, "facts": facts, "grounded": True}

    def _validate(self, data):
        assert isinstance(data.get("text"), str) and data["text"].strip()
        return data

    def _fallback(self, state, err):
        facts = self._facts(state)
        state.get("decision")
        label, action = summary_label(facts, self.cfg)

        def fm(x):
            return "data unavailable" if x is None else f"{x:.2f}"

        parts = [
            f"Risk: {label} (composite {fm(facts['composite_score'])}, confidence {fm(facts['confidence'])}).",
            f"Stage: {facts['case_stage']}.",
            f"Automated summary generation unavailable; review raw signals. Recommended action: {action}",
        ]
        return AgentResult(
            self.name,
            False,
            data={"text": " ".join(parts), "facts": facts},
            error=getattr(err, "code", ErrorCode.INTERNAL),
            degraded=True,
        )


# ═══════════════════════════ registry ══════════════════════════════════════
class AgentRegistry:
    _LLM_AGENTS = (ConversationAgent, SummaryAgent)
    _CACHED = (SentimentAgent, ThreatAgent, ContextAgent, FusionAgent, ForecastAgent)

    def __init__(self, cfg: dict, http: httpx.Client, llm):
        self.cfg, self.http, self.llm = cfg, http, llm
        self.cache = TTLCache(ttl_s=45)
        self.quarantine = Quarantine()
        self._agents: dict[str, BaseAgent] = {}
        for cls in self._CACHED:
            cls.cache, cls.quarantine = self.cache, self.quarantine
        SummaryAgent.quarantine = self.quarantine

    def get(self, cls: type[BaseAgent]) -> BaseAgent:
        if cls.name not in self._agents:
            self._agents[cls.name] = (
                cls(self.cfg, self.http, self.llm) if cls in self._LLM_AGENTS else cls(self.cfg, self.http)
            )
        return self._agents[cls.name]

    def parallel(self, classes: list[type[BaseAgent]], state: dict, max_workers: int = 4) -> dict[str, AgentResult]:
        out: dict[str, AgentResult] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self.get(c).run, state): c for c in classes}
            for fut, c in futs.items():
                out[c.name] = fut.result()
        return out

    def degradation_level(self) -> int:
        lvl = max((a.health.level() for n, a in self._agents.items() if a.priority != Priority.BEST_EFFORT), default=0)
        DEGRADATION.set(lvl)
        return lvl

    def health_report(self) -> dict:
        return {
            n: {
                "state": a.breaker.state,
                "score": a.health.score(),
                "level": a.health.level(),
                "priority": a.priority.name,
            }
            for n, a in sorted(self._agents.items())
        }


from concurrent.futures import ThreadPoolExecutor  # noqa: E402
