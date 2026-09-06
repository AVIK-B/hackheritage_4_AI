# verify_corpus_provenance.py
import json, re
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, ".")
from src.data.build_dialogue_corpus import CRISIS_KEYS, HANDOFF, INTENTS

rows = [json.loads(l) for l in Path("data/processed-smoke/dialogue_train.jsonl")
        .read_text(encoding="utf-8").splitlines()]

# 1. provenance split
src = Counter(r.get("source", "MISSING") for r in rows)
print("sources:", dict(src))
assert "MISSING" not in src, "untagged rows — provenance broken"

# 2. ED must contain ZERO crisis-lexicon content (contamination, defense-in-depth)
ed = [r for r in rows if r["source"] == "ed"]
ed_crisis = [r for r in ed if any(k in (r["prefix"][-1]["content"] + r["target"]).lower()
                                  for k in CRISIS_KEYS)]
print(f"ED pairs: {len(ed)} | with crisis lexicon: {len(ed_crisis)} (must be 0)")

# 3. ED must contain NO handoff phrases (handoff is synthetic-only by design)
htoks = {v.lower() for v in HANDOFF.values()}
ed_handoff = [r for r in ed if any(t in r["target"].lower() for t in htoks)]
print(f"ED pairs with handoff text: {len(ed_handoff)} (should be ~0)")

# 4. synthetic crisis coverage — every crisis-lexicon line got a handoff?
syn_crisis = [r for r in rows if r["source"] == "synthetic"
              and any(k in r["prefix"][-1]["content"].lower() for k in CRISIS_KEYS)]
missing_handoff = [r for r in syn_crisis
                   if not any(t in r["target"].lower()
                              for t in {v.lower() for v in HANDOFF.values()})]
print(f"synthetic crisis pairs: {len(syn_crisis)} | without handoff: {len(missing_handoff)} (must be 0)")

# 5. artifact scan — ED encoding junk must not survive
artifacts = [r for r in rows if "_comma_" in json.dumps(r) or "_period_" in json.dumps(r)]
print(f"rows containing _comma_/_period_ artifacts: {len(artifacts)} (must be 0)")

# 6. near-duplicate density (Jaccard dedup actually ran?)
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
targets = [norm(r["target"]) for r in rows]
exact_dupes = len(targets) - len(set(targets))
print(f"exact duplicate targets: {exact_dupes} (should be ~0)")

assert not ed_crisis and not missing_handoff and not artifacts, "PROVENANCE FAILED"
print("PROVENANCE: PASS ✅")