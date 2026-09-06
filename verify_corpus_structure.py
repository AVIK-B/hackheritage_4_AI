# verify_corpus_structure.py — run from repo root
import json, yaml
from pathlib import Path

def check_file(path: str, fmt: str):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    problems = []
    for i, r in enumerate(rows):
        if fmt == "prefix_completion":                      # adapter A
            if set(r.keys()) != {"prefix", "target", "source"}:
                problems.append((i, f"keys={set(r.keys())}"))
            if not r["prefix"] or r["prefix"][0]["role"] != "system":
                problems.append((i, "prefix missing system role"))
            if r["prefix"][-1]["role"] != "user":
                problems.append((i, f"prefix ends on {r['prefix'][-1]['role']} (must be user)"))
            if not r["target"].strip():
                problems.append((i, "empty target"))
            roles = [m["role"] for m in r["prefix"][1:]]
            if roles != sorted(roles, key=lambda x: 0) and any(
                    a == b for a, b in zip(roles, roles[1:])):
                problems.append((i, f"consecutive same roles: {roles}"))
        else:                                               # adapter B (messages)
            if [m["role"] for m in r["messages"]] != ["system", "user", "assistant"]:
                problems.append((i, "message shape"))
    return rows, problems

train, p1 = check_file("data/processed-smoke/dialogue_train.jsonl", "prefix_completion")
val,   p2 = check_file("data/processed-smoke/dialogue_val.jsonl",   "prefix_completion")
print(f"train: {len(train)} rows, {len(p1)} structural problems")
print(f"val:   {len(val)} rows, {len(p2)} structural problems")
for p in (p1 + p2)[:10]:
    print("  ", p)

# leakage check: val and train must share NO user utterances
train_users = {r["prefix"][-1]["content"] for r in train}
val_users   = {r["prefix"][-1]["content"] for r in val}
overlap = train_users & val_users
print(f"train/val user-overlap: {len(overlap)}", "— LEAK" if overlap else "— clean")
assert not p1 and not p2 and not overlap, "structural verification FAILED"
print("STRUCTURE: PASS ✅")