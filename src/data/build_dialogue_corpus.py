"""(prefix, target) pairs for adapter A: EmpatheticDialogues (+optional translation)
+ synthetic bilingual check-ins. Exploded per assistant turn, deduped, 95/5 split."""
import argparse
import hashlib
import json
import pathlib
import random
import re

SYSTEM = pathlib.Path("prompts/sahayak_system.txt").read_text().strip()

CRISIS_TARGETS = [
    ("I'm really glad you told me. You shouldn't have to face this alone. Would you like me to "
    "connect you with a counsellor right now? They can help, and I'll stay here with you.")]
USER_LINES = [
    "aaj court ka notice aaya, bahut darr lag raha hai",
    "I couldn't sleep last night, hearing is in 3 days",
    "sab log pooch rahe hai but main kisi se baat nahi karna chahta",
    "I feel like giving up on the case",
    "mujhe lagta hai mujhe akela chhod diya gaya hai"]
GOOD_REPLIES = [
    ("That sounds really heavy. It makes complete sense to feel scared before a hearing date. "
    "What's weighing on you most right now?"),
    ("Not sleeping before something this important is exhausting. Would it help to talk through "
    "what's worrying you, or would you prefer something calming instead?"),
    "Wanting space is okay — you don't owe anyone your story. I'm here whenever you feel like talking.",
    ("Feeling like giving up doesn't mean you are giving up. That feeling is a signal you're carrying "
    "a lot. Would it help to have a counsellor walk through the next steps with you?")]


def norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def to_pair(messages: list) -> list[dict]:
    pairs = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            prefix = messages[:i]
            if prefix and prefix[0]["role"] != "system":
                prefix = [{"role": "system", "content": SYSTEM}, *prefix]
            pairs.append({"prefix": prefix, "target": m["content"]})
    return pairs


def load_public() -> list[dict]:
    try:
        from datasets import load_dataset
        ds, by_conv = load_dataset("empathetic_dialogues", split="train"), {}
        for r in ds.select(range(20000)):
            by_conv.setdefault(r["conv_id"], []).append((r["utterance_idx"], r["utterance"]))
        out = []
        for turns in by_conv.values():
            msgs = [{"role": "system", "content": SYSTEM}]
            for _, u in sorted(turns):
                msgs.append({"role": "user" if len(msgs) % 2 else "assistant", "content": u})
            out += to_pair(msgs)
        return out
    except Exception as e:
        print(f"[corpus] public datasets unavailable, synthetic-only: {e}")
        return []


def synthetic(rng: random.Random, n: int) -> list[dict]:
    pairs = []
    for _ in range(n):
        u, a1 = rng.choice(USER_LINES), rng.choice(GOOD_REPLIES)
        u2 = rng.choice(USER_LINES)
        a2 = rng.choice(CRISIS_TARGETS) if any(w in u2 for w in ["give up", "akela", "darr"]) \
            else rng.choice(GOOD_REPLIES)
        pairs += to_pair([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": u}, {"role": "assistant", "content": a1},
                          {"role": "user", "content": u2}, {"role": "assistant", "content": a2}])
    return pairs


def main(n_synth: int, seed: int, out: str) -> None:
    rng = random.Random(seed)
    pairs = load_public() + synthetic(rng, n_synth)
    seen, dedup = set(), []
    for p in pairs:
        h = hashlib.md5(norm(p["prefix"][-1]["content"] + p["target"]).encode()).hexdigest()
        if h not in seen:
            seen.add(h); dedup.append(p)
    rng.shuffle(dedup)
    root = pathlib.Path(out); root.mkdir(parents=True, exist_ok=True)
    k = max(1, int(0.05 * len(dedup)))
    for split, rows in (("dialogue_val.jsonl", dedup[:k]), ("dialogue_train.jsonl", dedup[k:])):
        (root / split).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(f"{len(dedup)} pairs (val={k}) -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-synth", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/processed")
    main(**vars(p.parse_args()))