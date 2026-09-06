import json
import queue
import re

import structlog

from src.common.safety import FALLBACK_REPLY, guard_reply, scrub_pii

log = structlog.get_logger()
_ALLOWED_HISTORY_ROLES = {"user", "assistant"}
_SUMMARY_ACTIONS = {
    "ESCALATE-24H": "Schedule a counsellor check-in call within 24 hours.",
    "VERIFY-HUMAN-24H": "Schedule a human check-in call within 24 hours to verify before escalation.",
    "HUMAN-72H": "Assign a human check-in call within 72 hours to disambiguate.",
    "HUMAN-48H": "Schedule a human check-in within 48 hours and screen for intimidation per protocol.",
    "MONITOR-CLOSELY": "Keep the case on the counsellor watchlist and re-evaluate after the next check-in.",
    "ROUTINE": "Continue routine scheduled check-ins.",
    "PROACTIVE-24H": "Schedule a proactive check-in within 24 hours.",
    "REQUEST FIELDS": "Request missing triage fields before escalation beyond routine monitoring.",
}


def _recent_messages(case_ctx: dict, limit: int = 6) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in case_ctx.get("recent_messages", [])[-limit:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = scrub_pii(str(item.get("content", ""))).strip()
        if role in _ALLOWED_HISTORY_ROLES and content:
            messages.append({"role": role, "content": content[:1000]})
    return messages


def _dialogue_context(state: dict) -> dict[str, str] | None:
    ctx = state.get("case_ctx", {})
    parts = [f"User language: {state.get('language', 'en')}."]
    if ctx.get("case_stage"):
        parts.append(f"Case stage: {ctx['case_stage']}.")
    if ctx.get("days_to_next_hearing") is not None:
        parts.append(f"Days to next hearing: {ctx['days_to_next_hearing']}.")
    if len(parts) == 1:
        return None
    parts.append("Use this only to tune supportiveness; never reveal scores, monitoring, or internal analysis.")
    return {"role": "system", "content": " ".join(parts)}


def respond(state, *, cfg, llm) -> dict:
    """Victim-facing reply (adapter A). Streams into state['stream_q'] when present."""
    history = _recent_messages(state.get("case_ctx", {}))
    context = _dialogue_context(state)
    msgs = [{"role": "system", "content": llm.SYSTEM_A}]
    if context:
        msgs.append(context)
    msgs.extend([*history, {"role": "user", "content": state["message"]}])
    q: queue.Queue | None = state.get("stream_q")
    lang = state.get("language", "en")
    try:
        if q is not None:
            chunks = []
            for chunk in llm.stream("dialogue", msgs, temperature=0.6, max_tokens=220):
                chunks.append(chunk)
                q.put(chunk)
            text = "".join(chunks)
        else:
            text = llm.chat("dialogue", msgs, temperature=0.6, max_tokens=220)
        text = guard_reply(text, lang)
    except Exception as e:
        log.warning("llm_degraded", thread=state["thread_id"], err=str(e))
        text = FALLBACK_REPLY.get(lang, FALLBACK_REPLY["en"])
        return {"reply": text, "errors": [*state.get("errors", []), "llm:fallback_reply"]}
    if not text:
        text = FALLBACK_REPLY.get(lang, FALLBACK_REPLY["en"])
        return {"reply": text, "errors": [*state.get("errors", []), "llm:empty_reply"]}
    return {"reply": text}


def _fmt_score(value: float | None) -> str:
    return "data unavailable" if value is None else f"{value:.2f}"


def _summary_label_action(facts: dict, cfg: dict) -> tuple[str, str]:
    policies = cfg.get("policies", {})
    escalate_score = policies.get("escalate_score", 0.75)
    moderate_score = 0.8 * escalate_score
    confidence_floor = policies.get("escalate_confidence_floor", 0.45)
    forecast_threshold = policies.get("forecast_escalation_prob", 0.60)

    score = facts["composite_score"]
    confidence = facts["confidence"]
    top_signals = facts.get("top_signals") or []
    threat = "threat_language_detected" in top_signals

    if score is None:
        return "INSUFFICIENT DATA", _SUMMARY_ACTIONS["REQUEST FIELDS"]
    if score >= escalate_score:
        if confidence is not None and confidence < confidence_floor:
            return "HIGH (LOW CONFIDENCE)", _SUMMARY_ACTIONS["VERIFY-HUMAN-24H"]
        return "HIGH", _SUMMARY_ACTIONS["ESCALATE-24H"]
    if score >= moderate_score:
        if confidence is not None and confidence < confidence_floor:
            return "MODERATE (LOW CONFIDENCE)", _SUMMARY_ACTIONS["HUMAN-72H"]
        if threat:
            return "MODERATE (THREAT SIGNAL)", _SUMMARY_ACTIONS["HUMAN-48H"]
        return "MODERATE", _SUMMARY_ACTIONS["MONITOR-CLOSELY"]
    if threat:
        return "LOW (THREAT SIGNAL)", _SUMMARY_ACTIONS["HUMAN-72H"]
    if facts["p_escalation"] is not None and facts["p_escalation"] >= forecast_threshold:
        return "LOW (FORECAST ALERT)", _SUMMARY_ACTIONS["PROACTIVE-24H"]
    return "LOW", _SUMMARY_ACTIONS["ROUTINE"]


def _render_summary(facts: dict, cfg: dict) -> str:
    label, action = _summary_label_action(facts, cfg)
    signals = ", ".join(facts["top_signals"]) or "no distinguishing signals"
    score = _fmt_score(facts["composite_score"])
    confidence = _fmt_score(facts["confidence"])
    parts = [
        f"Risk: {label} (composite {score}, confidence {confidence}).",
        f"Stage: {facts['case_stage']}.",
        f"Key signals: {signals}.",
    ]
    if facts["days_to_next_hearing"] is not None:
        days = facts["days_to_next_hearing"]
        parts.append(f"Hearing in {days} day(s) - expect elevated stress around the date.")
    if facts["p_escalation"] is not None:
        parts.append(f"Forecast: escalation probability {facts['p_escalation']:.2f} over the next 7 days.")
    if facts["errors"]:
        joined_errors = "; ".join(facts["errors"])
        parts.append(f"Note: some signals were unavailable at scoring time ({joined_errors}) - interpret with caution.")
    if "threat_language_detected" in facts["top_signals"]:
        action = f"{action} Notify the district protection officer per protocol."
    parts.append(f"Recommended action: {action}")
    return " ".join(parts)


def _is_grounded_summary(text: str, facts: dict) -> bool:
    if not text.startswith("Risk: ") or "Recommended action:" not in text:
        return False
    known = {_fmt_score(facts[k]) for k in ("composite_score", "confidence", "p_escalation")}
    known.add(str(facts["days_to_next_hearing"]))
    known.discard("data unavailable")
    unsupported = {num for num in re.findall(r"\b\d+(?:\.\d+)?\b", text) if num not in known and num != "7"}
    return not unsupported


def summarize(state, *, cfg, llm) -> dict:
    """Counsellor-facing summary (adapter B) over ONLY structured facts — RAG-grounded shape."""
    f = state.get("fusion")
    fc = state.get("forecast")
    facts = {
        "case_id": state["case_id"],
        "case_stage": state["case_ctx"].get("case_stage", "data unavailable"),
        "days_to_next_hearing": state["case_ctx"].get("days_to_next_hearing"),
        "composite_score": f.composite_score if f else None,
        "confidence": f.confidence if f else None,
        "top_signals": f.top_signals if f else [],
        "p_escalation": fc.p_escalation if fc else None,
        "errors": state.get("errors", []),
    }
    deterministic = _render_summary(facts, cfg)
    try:
        text = llm.chat(
            "summary",
            [
                {"role": "system", "content": llm.SYSTEM_B},
                {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        if not _is_grounded_summary(text, facts):
            log.warning("summary_repaired", thread=state["thread_id"])
            text = deterministic
    except Exception as e:
        log.warning("summary_degraded", thread=state["thread_id"], err=str(e))
        return {"summary_text": deterministic, "errors": [*state.get("errors", []), "summary:fallback"]}
    return {"summary_text": text}
