"""Crisis-patch set: crisis pairs x8 + refusal x2 + support anchor ~30% (anti-forgetting).
Reads data/processed/dialogue_train.jsonl. Output: data/processed/crisis_patch.jsonl"""
import json, random, pathlib
rng = random.Random(99)
rows = [json.loads(l) for l in pathlib.Path("data/processed/dialogue_train.jsonl")
        .read_text(encoding="utf-8").splitlines()]
crisis, refusal, support = [], [], []
for r in rows:
    t = r["target"].lower()
    if any(h in t for h in ("counsellor", "काउंसलर")):
        crisis.append(r)
    elif any(k in r["prefix"][-1]["content"].lower()
             for k in ("jeetne ka", "tareeka", "medicine", "promise", "guarantee",
                       "score", "monitoring", "jeet jau")):
        refusal.append(r)
    else:
        support.append(r)
mix = crisis * 8 + refusal * 2 + rng.sample(support, min(len(support), int(len(crisis) * 3)))
rng.shuffle(mix)
out = pathlib.Path("data/processed/crisis_patch.jsonl")
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in mix), encoding="utf-8")
print(f"crisis {len(crisis)}x8 + refusal {len(refusal)}x2 + support {len(mix)-len(crisis)*8-len(refusal)*2} = {len(mix)} pairs")