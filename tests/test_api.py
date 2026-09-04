from fastapi.testclient import TestClient

from src.orchestration.app import app

H = {"X-API-Key": "vk_t"}


def test_crisis_short_circuit_no_llm():
    with TestClient(app) as c:
        r = c.post(
            "/v1/interactions",
            headers=H,
            json={"case_id": "CASE-1001", "channel": "sms", "message": "mujhe jeene ka mann nahi karta"},
        ).json()
        assert r["status"] == "completed"
        assert "counsellor" in r["reply"].lower()
        assert len(r["audit_ref"]) == 64
    with TestClient(app) as c:
        r = c.post(
            "/v1/interactions",
            headers=H,
            json={"case_id": "CASE-1001", "channel": "sms", "message": "mujhe jeene ka mann nahi karta"},
        ).json()
        assert r["status"] == "completed" and "counsellor" in r["reply"].lower()
        assert len(r["audit_ref"]) == 64


def test_degraded_routine_with_fallback_reply():
    with TestClient(app) as c:
        r = c.post(
            "/v1/interactions",
            headers=H,
            json={"case_id": "CASE-1002", "channel": "pwa", "message": "aaj thoda better hoon"},
        ).json()
        assert r["status"] == "completed" and r["reply"]  # templated fallback, not empty
        t = c.get(f"/v1/threads/{r['thread_id']}/trace", headers={"X-API-Key": "ck_t"}).json()
        assert t["decision"]["route"] == "routine"


def test_auth_enforced():
    with TestClient(app) as c:
        r = c.post("/v1/interactions", json={"case_id": "CASE-1003", "channel": "pwa", "message": "hi"})
        assert r.status_code == 401
