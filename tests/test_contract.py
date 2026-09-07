"""Contract tests: response shapes exactly match contracts/api-contracts.md.
These break loudly if anyone changes a public payload."""

import os

os.environ.setdefault("ORCH_API_KEYS", '{"victim":"vk_t","counsellor":"ck_t","ops":"ok_t"}')

CONTRACT_KEYS = {
    "thread_id": str,
    "reply": str,
    "status": str,
    "audit_ref": str,
}
STATUS_VALUES = {"awaiting_counsellor", "completed"}


def test_interaction_contract_shape():
    from fastapi.testclient import TestClient

    from src.orchestration.app import app

    with TestClient(app) as c:
        r = c.post(
            "/v1/interactions",
            headers={"X-API-Key": "vk_t"},
            json={"case_id": "CASE-777", "channel": "pwa", "message": "hello there"},
        ).json()
        for k, typ in CONTRACT_KEYS.items():
            assert k in r and isinstance(r[k], typ), f"contract break: {k}"
        assert r["status"] in STATUS_VALUES
        assert len(r["audit_ref"]) == 64
