"""Judge-facing demo: runs three scripted scenarios against a LIVE orchestrator
and prints the full decision trail.

Usage:
  python scripts/demo_trace.py                          # http://localhost:8500
  python scripts/demo_trace.py --base-url http://<vm>:8500

Scenarios:
  1. routine check-in      -> routine path (LLM reply or templated fallback)
  2. ambiguous message     -> degraded-mode or fusion-driven escalation -> counsellor resume
  3. crisis message        -> zero-LLM short-circuit, templated handoff, dispatch
Then: trace inspection + tamper-evident audit verification.
"""

import argparse
import json
import sys

import httpx

sys.path.insert(0, ".")  # run from repo root so settings/.env resolve
from src.common.settings import settings  # noqa: E402

SCENARIOS = [
    (
        "ROUTINE",
        {
            "case_id": "CASE-1001",
            "channel": "pwa",
            "message": "aaj thoda better feel kar raha hoon, neend theek ho gayi",
        },
    ),
    (
        "AMBIGUOUS",
        {"case_id": "CASE-1002", "channel": "pwa", "message": "pata nahi kya karu, sab mix ho gaya hai dimaag mein"},
    ),
    ("CRISIS", {"case_id": "CASE-1003", "channel": "sms", "message": "mujhe jeene ka mann nahi karta"}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8500")
    ap.add_argument(
        "--skip-resume", action="store_true", help="don't auto-approve escalations (inspect externally instead)"
    )
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    vk = settings.api_keys["victim"]
    ck = settings.api_keys["counsellor"]
    ok = settings.api_keys["ops"]
    c = httpx.Client(timeout=60)

    print(f"orchestrator: {base}")
    health = c.get(f"{base}/healthz")
    print(f"healthz     : {health.json()} (HTTP {health.status_code})\n")

    for name, payload in SCENARIOS:
        print(f"=== {name} " + "=" * 60)
        r = c.post(f"{base}/v1/interactions", headers={"X-API-Key": vk}, json=payload)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}: {r.text}\n")
            continue
        body = r.json()
        print(f"thread_id : {body['thread_id']}")
        print(f"status    : {body['status']}")
        print(f"reply     : {body['reply'][:180]}{'…' if len(body['reply']) > 180 else ''}")
        print(f"audit_ref : {body['audit_ref'][:24]}…")

        t = c.get(f"{base}/v1/threads/{body['thread_id']}/trace", headers={"X-API-Key": ck})
        if t.status_code == 200:
            print("trace     :")
            print(json.dumps(t.json(), indent=2, default=str)[:1200])

        if body["status"] == "awaiting_counsellor" and not a.skip_resume:
            res = c.post(
                f"{base}/v1/counsellor/{body['thread_id']}/resume",
                headers={"X-API-Key": ck},
                json={"approved": True, "notes": "demo: counsellor approved escalation"},
            )
            print(f"resume    : {json.dumps(res.json())}")
        print()

    print("=== AUDIT LEDGER " + "=" * 52)
    v = c.get(f"{base}/v1/audit/verify", headers={"X-API-Key": ok})
    print(json.dumps(v.json(), indent=2))
    print("\nDEMO COMPLETE")


if __name__ == "__main__":
    main()
