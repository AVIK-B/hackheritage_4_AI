"""Conversation session store + context engineering (enterprise pattern).

Browser holds ONLY an opaque session id in an httpOnly cookie.
Conversation turns live server-side (Redis; in-memory fallback for dev).

Context engineering (OpenAI/Anthropic-style hybrid):
  1. RECENCY WINDOW   : last N turns verbatim (N=6)
  2. ROLLING SUMMARY  : older turns compressed to <=60 tokens (LLM or extractive)
  3. METADATA LINE    : case stage / hearing (<=40 tokens, non-sensitive)
  4. TOKEN BUDGET     : hard cap on assembled context chars; oldest turns
                        dropped first, summary never dropped
  5. PII HYGIENE      : every stored+injected turn passes scrub_pii
Eviction: 7-day TTL, sliding on activity. Revocation: DELETE /session → cookie
orphaned instantly (pointer-only design benefit).
"""

from __future__ import annotations

import json
import secrets
import time
import uuid

import structlog

from src.common.safety import scrub_pii

log = structlog.get_logger()

MAX_STORED_TURNS = 40
WINDOW_TURNS = 6
SUMMARY_MAX_CHARS = 240
TTL_S = 7 * 24 * 3600


class SessionStore:
    """Redis-backed with in-memory fallback (dev / no-redis)."""

    def __init__(self, redis_url: str | None, ttl_s: int = TTL_S):
        self.ttl = ttl_s
        self._mem: dict[str, dict] = {}
        self._r = None
        if redis_url:
            try:
                import redis

                self._r = redis.Redis.from_url(redis_url, decode_responses=True)
                self._r.ping()
                log.info("session_store", backend="redis")
            except Exception as e:
                log.warning("session_store_redis_unavailable", err=str(e))
        if self._r is None:
            log.warning("session_store", backend="in_memory (dev only)")

    # ── raw io ────────────────────────────────────────────────────────────
    def _get(self, sid: str) -> dict | None:
        if self._r:
            raw = self._r.get(f"sessions:{sid}")
            return json.loads(raw) if raw else None
        return self._mem.get(sid)

    def _put(self, sid: str, data: dict) -> None:
        if self._r:
            self._r.setex(f"sessions:{sid}", self.ttl, json.dumps(data, ensure_ascii=False))
        else:
            self._mem[sid] = data

    # ── public API ────────────────────────────────────────────────────────
    def new_session(self) -> str:
        sid = uuid.uuid4().hex
        self._put(sid, {"created": time.time(), "turns": [], "summary": None})
        return sid

    def exists(self, sid: str) -> bool:
        return bool(sid) and self._get(sid) is not None

    def append_turn(self, sid: str, user: str, assistant: str) -> None:
        data = self._get(sid)
        if data is None:  # expired mid-session → recreate
            sid_new = self.new_session()
            data = self._get(sid_new)
            sid = sid_new
        data["turns"].append(
            {
                "u": scrub_pii(user)[:1000],
                "a": scrub_pii(assistant)[:1000],
                "t": time.time(),
            }
        )
        data["turns"] = data["turns"][-MAX_STORED_TURNS:]
        data["last_active"] = time.time()
        self._put(sid, data)

    def turns(self, sid: str) -> list[dict]:
        return (self._get(sid) or {}).get("turns", [])

    def summary(self, sid: str) -> str | None:
        return (self._get(sid) or {}).get("summary")

    def set_summary(self, sid: str, summary: str) -> None:
        data = self._get(sid)
        if data is not None:
            data["summary"] = summary[:SUMMARY_MAX_CHARS]
            self._put(sid, data)

    def revoke(self, sid: str) -> None:
        if self._r:
            self._r.delete(f"sessions:{sid}")
        self._mem.pop(sid, None)

    # ── context engineering ───────────────────────────────────────────────
    def build_context(self, sid: str, case_ctx: dict | None, budget_chars: int = 2400) -> list[dict]:
        """Assemble LLM-ready context under a char budget.
        Order: [summary] + [metadata] + [recent turns]. Drop oldest turns first."""
        msgs: list[dict] = []
        data = self._get(sid) or {}

        if data.get("summary"):
            msgs.append({"role": "system", "content": f"Conversation so far: {data['summary']}"})

        if case_ctx:
            parts = []
            if case_ctx.get("case_stage") and case_ctx["case_stage"] != "unknown":
                parts.append(f"Case stage: {case_ctx['case_stage']}")
            if case_ctx.get("days_to_next_hearing") is not None:
                parts.append(f"Days to next hearing: {case_ctx['days_to_next_hearing']}")
            if parts:
                msgs.append(
                    {
                        "role": "system",
                        "content": "; ".join(parts) + ". Use only to tune supportiveness; never mention this.",
                    }
                )

        turns = data.get("turns", [])[-6:]
        used = sum(len(m["content"]) for m in msgs)
        for t in reversed(turns):
            cost = len(t["u"]) + len(t["a"])
            if used + cost > budget_chars:
                break
            msgs.insert(2, {"role": "user", "content": t["u"]})
            msgs.insert(3, {"role": "assistant", "content": t["a"]})
            used += cost
        return msgs


def new_sid() -> str:
    return secrets.token_urlsafe(24)  # opaque, unguessable
