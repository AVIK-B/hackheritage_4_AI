import time

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_fixed

from src.common.resilience import breakers
from src.common.safety import assess_crisis
from src.orchestration.audit import AuditLedger
from src.orchestration.policies import evaluate
from src.orchestration.state import CrisisAssessment, ForecastResult, FusionResult

log = structlog.get_logger()
_ledger: AuditLedger | None = None


def init_ledger(path: str) -> AuditLedger:
    global _ledger
    _ledger = AuditLedger(path)
    return _ledger


def _post(http: httpx.Client, url: str, payload: dict, timeout: float, breaker_name: str) -> dict:
    br = breakers.get(breaker_name)
    if not br.allow():
        raise ConnectionError(f"breaker_open:{breaker_name}")
    try:
        r = http.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        br.record(True)
        return r.json()
    except Exception:
        br.record(False)
        raise


def _get(http: httpx.Client, url: str, timeout: float, breaker_name: str) -> dict:
    br = breakers.get(breaker_name)
    if not br.allow():
        raise ConnectionError(f"breaker_open:{breaker_name}")
    try:
        r = http.get(url, timeout=timeout)
        r.raise_for_status()
        br.record(True)
        return r.json()
    except Exception:
        br.record(False)
        raise


# ---------------- nodes ----------------
def ingest(state, *, cfg, http) -> dict:
    from src.common.safety import scrub_pii

    try:
        ctx = _get(
            http,
            f"{cfg['services']['backend']}/v1/case/{state['case_id']}/context",
            cfg["service_timeout_s"],
            "backend",
        )
    except Exception as e:
        ctx, err = {}, [f"context:unavailable:{type(e).__name__}"]
    else:
        err = []
    return {
        "case_ctx": ctx,
        "language": ctx.get("language") or state.get("language") or "en",
        "message": scrub_pii(state["message"]),
        "errors": err,
        "t_start_ms": time.time() * 1000,
    }


def crisis_precheck(state, *, cfg) -> dict:
    return {"crisis": CrisisAssessment(**assess_crisis(state["message"], state.get("language", "en")))}


def make_signal_node(name: str, path: str):
    def node(state, *, cfg, http) -> dict:
        try:
            r = _post(
                http,
                f"{cfg['services']['scoring']}{path}",
                {
                    "text": state.get("message", ""),
                    "audio_ref": state.get("audio_ref"),
                    "lang": state.get("language", "en"),
                },
                cfg["service_timeout_s"],
                f"scoring:{name}",
            )
            return {"signals": {name: r}}
        except Exception as e:
            log.warning("signal_failed", signal=name, err=str(e))
            return {"signals": {name: None}, "errors": [f"signal:{name}:unavailable"]}

    node.__name__ = f"signal_{name}"
    return node


def fuse(state, *, cfg, http) -> dict:
    try:
        r = _post(
            http,
            f"{cfg['services']['fusion']}/v1/fusion",
            {
                "sentiment": state["signals"].get("sentiment"),
                "threat": state["signals"].get("threat"),
                "voice_stress": state["signals"].get("voice"),
                "engagement": state["case_ctx"].get("recent_checkins"),
                "case_stage": state["case_ctx"].get("case_stage"),
            },
            cfg["service_timeout_s"],
            "fusion",
        )
        return {"fusion": FusionResult(**r)}
    except Exception as e:
        return {"fusion": None, "errors": [f"fusion:unavailable:{type(e).__name__}"]}


def forecast(state, *, cfg, http) -> dict:
    try:
        r = _post(
            http,
            f"{cfg['services']['forecast']}/v1/forecast",
            {"case_id": state["case_id"], "score_history": state["case_ctx"].get("score_history", [])},
            cfg["service_timeout_s"],
            "forecast",
        )
        return {"forecast": ForecastResult(**r)}
    except Exception as e:
        return {"forecast": None, "errors": [f"forecast:unavailable:{type(e).__name__}"]}


def decide(state, *, cfg) -> dict:
    d = evaluate(
        state.get("crisis"),
        state.get("fusion"),
        state.get("forecast"),
        {k: v for k, v in state.get("signals", {}).items() if v},
        cfg,
    )
    log.info("decision", thread=state["thread_id"], route=d.route, reasons=d.reasons)
    return {"decision": d}


def victim_handoff(state, *, cfg) -> dict:
    """Safety-critical copy is TEMPLATED, never generated. Also writes the crisis
    Decision so downstream consumers (dispatch, audit) see a uniform state —
    the crisis path skips `decide` by design, so this node owns it here."""
    from src.common.safety import CRISIS_COPY
    from src.orchestration.state import Decision

    crisis = state.get("crisis")
    hits = ", ".join(crisis.hits) if crisis and crisis.hits else "pattern match"
    decision = Decision(
        route="crisis",
        reasons=[f"crisis precheck: {crisis.source} ({hits})"] if crisis else ["crisis precheck"],
        policy_version=cfg.get("policy_version", "?"),
    )
    return {"reply": CRISIS_COPY.get(state.get("language", "en"), CRISIS_COPY["en"]), "decision": decision}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
def _dispatch(http, cfg, state):
    decision = state.get("decision")
    return _post(
        http,
        f"{cfg['services']['backend']}/v1/alerts",
        {
            "case_id": state["case_id"],
            "severity": state["_severity"],
            "reasons": decision.reasons if decision else [],
            "summary_text": state.get("summary_text", ""),
        },
        5.0,
        "backend:alerts",
    )


def dispatch_alert(severity: str):
    def node(state, *, cfg, http) -> dict:
        if state.get("counsellor_action") and not state["counsellor_action"].get("approved"):
            log.info("escalation_rejected_by_counsellor", thread=state["thread_id"])
            return {"dispatched": True, "alert_id": None}
        state["_severity"] = severity
        try:
            r = _dispatch(http, cfg, state)
            return {"dispatched": True, "alert_id": r["alert_id"]}
        except Exception as e:
            # dead-letter: never lose an escalation — record it in the tamper-evident ledger
            decision = state.get("decision")
            _ledger.append(
                {
                    "kind": "alert_dispatch_failed",
                    "thread_id": state["thread_id"],
                    "case_id": state["case_id"],
                    "severity": severity,
                    "reasons": decision.reasons if decision else [],
                    "error": str(e),
                }
            )
            return {"dispatched": False, "errors": [*state.get("errors", []), "alert:dispatch_failed"]}

    node.__name__ = f"dispatch_{severity}"
    return node


def audit_node(state, *, cfg) -> dict:
    rec = _ledger.append(
        {
            "kind": "interaction",
            "thread_id": state["thread_id"],
            "case_id": state["case_id"],
            "channel": state["channel"],
            "decision": state["decision"].model_dump(),
            "fusion": state["fusion"].model_dump() if state.get("fusion") else None,
            "forecast": state["forecast"].model_dump() if state.get("forecast") else None,
            "errors": state.get("errors", []),
            "alert_id": state.get("alert_id"),
            "latency_ms": int(time.time() * 1000 - state["t_start_ms"]),
        }
    )
    return {"audit_ref": rec["record_hash"]}
