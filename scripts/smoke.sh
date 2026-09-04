#!/usr/bin/env bash
# Automated acceptance test. Run from repo root (Git Bash on Windows):
#   BASE=http://localhost:8500 bash scripts/smoke.sh
set -euo pipefail

BASE="${BASE:-http://localhost:8500}"
ENVF="${ENVF:-.env}"

# Pull keys from .env (single source of truth)
read_keys() {
  python - "$ENVF" "$1" <<'PY'
import json, sys
for line in open(sys.argv[1], encoding="utf-8"):
    if line.startswith("ORCH_API_KEYS="):
        print(json.loads(line.split("=", 1)[1])[sys.argv[2]])
        break
PY
}
VK="${VK:-$(read_keys "$ENVF" victim)}"
CK="${CK:-$(read_keys "$ENVF" counsellor)}"
OK="${OK:-$(read_keys "$ENVF" ops)}"

say() { printf "\n== %s\n" "$1"; }

say "healthz";              curl -sf "$BASE/healthz"; echo
say "readyz (ops)";         curl -sf -H "X-API-Key: $OK" "$BASE/readyz"; echo
say "unauthorized (expect 401)"
curl -s -o /dev/null -w "%{http_code}\n" -X POST "$BASE/v1/interactions" \
  -H "Content-Type: application/json" \
  -d '{"case_id":"CASE-x","channel":"pwa","message":"hi"}'

say "crisis short-circuit"
curl -sf -X POST "$BASE/v1/interactions" -H "X-API-Key: $VK" \
  -H "Content-Type: application/json" \
  -d '{"case_id":"CASE-1001","channel":"sms","message":"mujhe jeene ka mann nahi karta"}'; echo

say "routine check-in"
R=$(curl -sf -X POST "$BASE/v1/interactions" -H "X-API-Key: $VK" \
  -H "Content-Type: application/json" \
  -d '{"case_id":"CASE-1002","channel":"pwa","message":"aaj thoda better feel kar raha hoon"}')
echo "$R"
TID=$(echo "$R" | python -c "import sys,json;print(json.load(sys.stdin)['thread_id'])")

say "trace (counsellor)";   curl -sf -H "X-API-Key: $CK" "$BASE/v1/threads/$TID/trace"; echo
say "audit verify (ops)";   curl -sf -H "X-API-Key: $OK" "$BASE/v1/audit/verify"; echo

echo "\nSMOKE OK"