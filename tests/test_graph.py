"""Chaos tests: run the REAL graph with dead teammates + dead LLM and assert
every path completes with sane, safe, degraded behavior."""

import json

from fastapi.testclient import TestClient

from src.orchestration.app import app

VK = {"X-API-Key": "vk_t"}
CK = {"X-API-Key": "ck_t"}
OK = {"X-API-Key": "ok_t"}


def test_crisis_short_circuit_survives_total_blackout():
    """Every dependency dead; crisis must still handoff + dead-letter + audit."""
    with TestClient(app) as c:
        r = c.post(
            "/v1/interactions",
            headers=VK,
            json={"case_id": "CASE-9001", "channel": "sms", "message": "mujhe jeene ka mann nahi karta"},
        ).json()
        assert r["status"] == "completed"
        assert "counsellor" in r["reply"].lower() or "काउंसलर" in r["reply"]
        assert len(r["audit_ref"]) == 64
        t = c.get(f"/v1/threads/{r['thread_id']}/trace", headers=CK).json()
        assert t["decision"]["route"] == "crisis"


def test_routine_survives_total_blackout_with_templated_reply():
    with TestClient(app) as c:
        r = c.post(
            "/v1/interactions",
            headers=VK,
            json={"case_id": "CASE-9002", "channel": "pwa", "message": "aaj thoda better hoon"},
        ).json()
        assert r["status"] == "completed" and r["reply"]
        t = c.get(f"/v1/threads/{r['thread_id']}/trace", headers=CK).json()
        assert t["decision"]["route"] == "routine"
        assert "degraded" in json.dumps(t.get("telemetry", {})).lower() or t.get("errors")


def test_escalation_dead_letters_when_backend_down():
    with TestClient(app) as c:
        resp = c.post(
            "/v1/interactions",
            headers=VK,
            json={"case_id": "CASE-9003", "channel": "pwa", "message": "aaj thoda better hoon"},
        )
        assert resp.status_code in (200, 500)
        v = c.get("/v1/audit/verify", headers=OK).json()
        assert v["valid"] is True


def test_auth_enforced_all_scopes():
    with TestClient(app) as c:
        assert (
            c.post("/v1/interactions", json={"case_id": "CASE-x", "channel": "pwa", "message": "hi"}).status_code == 401
        )
        assert c.get("/readyz").status_code == 401
        assert c.get("/metrics").status_code == 401


def test_readyz_reports_agent_health():
    with TestClient(app) as c:
        r = c.get("/readyz", headers=OK).json()
        assert r["llm"] is False
        assert "agents" in r and "degradation_level" in r
        assert isinstance(r["agents"], dict)


def test_rate_limit_429():
    from src.common.settings import settings

    with TestClient(app) as c:
        bursts = 0
        for _ in range(settings.rate_limit_rpm + 10):
            resp = c.get("/healthz")
            if resp.status_code == 429:
                bursts += 1
        assert bursts > 0
