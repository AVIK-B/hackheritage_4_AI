"""Verify the tamper-evident audit ledger — two modes:
  python scripts/verify_ledger.py            # verify the local ledger file directly
  python scripts/verify_ledger.py --api http://localhost:8500   # verify via live API (ops key)
Exit 0 = chain valid, exit 1 = tampered/missing."""

import argparse
import json
import sys

from src.orchestration.audit import AuditLedger
from src.common.settings import settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=None, help="base URL of a running orchestrator")
    ap.add_argument("--path", default=settings.audit_path)
    a = ap.parse_args()

    if a.api:
        import httpx

        r = httpx.get(
            f"{a.api.rstrip('/')}/v1/audit/verify", headers={"X-API-Key": settings.api_keys["ops"]}, timeout=10
        )
        result = r.json()
    else:
        result = AuditLedger(a.path).verify()

    print(json.dumps(result, indent=2))
    if result.get("valid"):
        print(f"LEDGER OK — {result.get('n', 0)} records, last hash {str(result.get('last_hash'))[:24]}…")
        sys.exit(0)
    print("LEDGER TAMPERED — chain broken; investigate immediately.")
    sys.exit(1)


if __name__ == "__main__":
    main()
