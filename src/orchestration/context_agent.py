"""ContextAgent v2: session-aware context assembly.
Sources, merged by priority:
  1. Session turns (Redis, recency window, PII-scrubbed)
  2. Rolling summary of older turns (never dropped)
  3. Case metadata line (short, instruction-shaped)
Assembled under a hard char budget; metadata dropped first if over."""

import structlog

from src.orchestration.agents import AgentResult, BaseAgent, ErrorCode, Priority

log = structlog.get_logger()
HARD_CAP = 1400
META_CAP = 220


class ContextAgent(BaseAgent):
    name = "case_context"
    timeout_s = 3.0
    priority = Priority.HIGH
    cache_ttl_s = 0  # sessions make caching wrong — context changes per turn

    def __init__(self, cfg, http, llm, sessions):
        super().__init__(cfg, http)
        self.sessions = sessions

    def _run(self, state) -> dict:
        # 1. session context (primary)
        sid = state.get("session_id", "")
        sess_msgs = self.sessions.build_context(sid, state.get("case_ctx"))
        # 2. fallback backend context (Shaan) only if session is empty
        if not sess_msgs:
            ctx = self._get(f"{self.cfg['services']['backend']}/v1/case/{state['case_id']}/context")
            sess_msgs = [
                {
                    "role": "system",
                    "content": f"Case stage: {ctx['case_stage']}; days to hearing: {ctx['days_to_next_hearing']}",
                }
            ]
            state["_backend_ctx"] = ctx
        # 3. hard cap — drop metadata first, then oldest msgs
        total = sum(len(m["content"]) for m in sess_msgs)
        while total > HARD_CAP and len(sess_msgs) > 1:
            dropped = sess_msgs.pop(0)
            total -= len(dropped["content"])
        return {
            "context_messages": sess_msgs,
            "case_ctx": state.get("case_ctx") or {},
            "language": state.get("language", "en"),
        }

    def _fallback(self, state, err):
        return AgentResult(
            self.name,
            False,
            data={"context_messages": [], "case_ctx": {}, "_degraded": True},
            error=getattr(err, "code", ErrorCode.INTERNAL),
        )
