import json
import queue

import structlog

from src.common.safety import FALLBACK_REPLY, guard_reply

log = structlog.get_logger()


def respond(state, *, cfg, llm) -> dict:
    """Victim-facing reply (adapter A). Streams into state['stream_q'] when present."""
    history = state["case_ctx"].get("recent_messages", [])[-6:]
    msgs = [{"role": "system", "content": llm.SYSTEM_A}] + history + \
           [{"role": "user", "content": state["message"]}]
    q: queue.Queue | None = state.get("stream_q")
    lang = state.get("language", "en")
    try:
        if q is not None:
            chunks = []
            for chunk in llm.stream("dialogue", msgs, temperature=0.6, max_tokens=220):
                chunks.append(chunk); q.put(chunk)
            text = "".join(chunks)
        else:
            text = llm.chat("dialogue", msgs, temperature=0.6, max_tokens=220)
        text = guard_reply(text, lang)
    except Exception as e:
        log.warning("llm_degraded", thread=state["thread_id"], err=str(e))
        text = FALLBACK_REPLY.get(lang, FALLBACK_REPLY["en"])
        return {"reply": text, "errors": state.get("errors", []) + ["llm:fallback_reply"]}
    return {"reply": text}


def summarize(state, *, cfg, llm) -> dict:
    """Counsellor-facing summary (adapter B) over ONLY structured facts — RAG-grounded shape."""
    f = state.get("fusion"); fc = state.get("forecast")
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
    try:
        text = llm.chat("summary", [
            {"role": "system", "content": llm.SYSTEM_B},
            {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}],
            temperature=0.2, max_tokens=300)
    except Exception as e:
        log.warning("summary_degraded", thread=state["thread_id"], err=str(e))
        text = (f"Automated summary unavailable ({type(e).__name__}). "
                f"Decision reasons: {'; '.join(state['decision'].reasons)}.")
    return {"summary_text": text}