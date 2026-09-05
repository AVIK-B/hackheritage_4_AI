import json, re, yaml

rows = [json.loads(l) for l in open("data/processed/case_summary_val.jsonl", encoding="utf-8")]
sys_p = open("prompts/casewriter_system.txt", encoding="utf-8").read()
cfg = yaml.safe_load(open("configs/adapter_b.yaml"))

# 1. SCHEMA MATCH — facts keys == exactly what nodes_llm.summarize() sends at runtime
RUNTIME_KEYS = {"case_id", "case_stage", "days_to_next_hearing", "composite_score",
                "confidence", "top_signals", "p_escalation", "errors"}
bad_schema = [i for i, r in enumerate(rows)
              if set(json.loads(r["messages"][1]["content"])) != RUNTIME_KEYS]
print(f"1. schema match: {'PASS' if not bad_schema else f'FAIL rows {bad_schema[:5]}'}")

# 2. PROMPT CONSISTENCY — every training example embeds the EXACT serving prompt
mismatch = [i for i, r in enumerate(rows) if r["messages"][0]["content"] != sys_p]
print(f"2. prompt consistency: {'PASS' if not mismatch else f'FAIL rows {mismatch[:5]}'}")

# 3. COVERAGE — all 9 decision categories present, none starved
from collections import Counter
import importlib.util
spec = importlib.util.spec_from_file_location("b", "src/data/build_case_summary_corpus.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cats = Counter(mod.decide(json.loads(r["messages"][1]["content"]))[0] for r in rows)
print(f"3. coverage: {dict(cats)}")
assert len(cats) >= 9 and min(cats.values()) > len(rows) * 0.02, "starved category"

# 4. GROUNDEDNESS BY CONSTRUCTION — every number in target exists in input
def fmt(x): return "data unavailable" if x is None else f"{x:.2f}"
ungrounded = []
for i, r in enumerate(rows):
    f = json.loads(r["messages"][1]["content"])
    known = {fmt(f[k]) for k in ("composite_score", "confidence", "p_escalation")} | {str(f["days_to_next_hearing"])}
    for num in re.findall(r"\d+\.\d+", r["messages"][2]["content"]):
        if num not in known: ungrounded.append((i, num))
print(f"4. groundedness: {'PASS' if not ungrounded else f'FAIL {ungrounded[:5]}'}")