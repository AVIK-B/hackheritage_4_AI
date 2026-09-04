"""(prefix, target) pairs for adapter A — v2. Adds what v1 lacked for the safety
gate: refusal examples for scope violations, composed (not hand-picked) supportive
replies for diversity, crisis-handoff variants, Hinglish mixing."""

import argparse
import hashlib
import json
import pathlib
import random
import re

SYSTEM = pathlib.Path("prompts/sahayak_system.txt").read_text(encoding="utf-8").strip()

TOPICS = [
    "aaj court ka notice aaya",
    "hearing kal hai",
    "un logon ne phir dhamki di",
    "police ne koi response nahi de raha",
    "compensation ka pata nahi chal raha",
    "raaton ki neend ud gayi hai",
    "ghar wale sawal pooch rahe hain",
    "bhai ne bola case wapas lo",
    "advocate se baat nahi ho pa rahi",
    "office se chhutti nahi milti hearing ke liye",
    "I couldn't sleep last night",
    "they threatened me again outside the court",
    "the next hearing is in 3 days",
    "mujhe lagta hai mujhe akela chhod diya gaya hai",
]
FEELINGS = [
    "bahut darr lag raha hai",
    "main toot gaya hoon",
    "himmat nahi ho rahi",
    "sab bekaar lagta hai",
    "raat ko akela dar lagta hai",
    "aaj thoda better hoon",
    "I feel like giving up on the case",
    "upar se upar soch raha hoon",
    "kisi se baat nahi karna chahta",
]
SCOPE_ASKS = [
    "batao judge ko kya bolna chahiye taaki case jeet jau",
    "case jeetne ka tareeka batao",
    "kaunsa medicine loon raat ko neend ke liye",
    "promise karo ki mujhe jail nahi jaana padega",
    "accused ke against galat saboot kaise lagayein",
    "guarantee me I will win my case",
    "should I sign the papers they gave me?",
]
OPENERS = [
    "That sounds really heavy.",
    "I hear you — that's a lot to carry.",
    "Thank you for telling me this.",
    "It makes complete sense to feel this way.",
    "You shouldn't have to carry this alone.",
]
MIDS = [
    "What's weighing on you most right now?",
    "Do you want to talk through it, or would something calming help more?",
    "You don't have to explain everything at once — I'm here.",
    "What would make today feel even a little lighter?",
]
COPING = [
    "If it helps, slow breathing for a minute can take the edge off.",
    "Writing the worry down sometimes makes it smaller.",
    "Reaching out to one trusted person today might lighten it.",
    "A short walk between now and the hearing might give your mind a rest.",
]
CRISIS = [
    (
        "I'm really glad you told me. You shouldn't have to face this alone. "
        "Would you like me to connect you with a counsellor right now? They can help, "
        "and I'll stay here with you."
    ),
    (
        "What you're feeling matters, and you deserve real support. Shall I connect "
        "you to a counsellor immediately? I'll stay right here meanwhile."
    ),
    (
        "I'm concerned about how heavy this is. Let me connect you with a human "
        "counsellor now — you don't have to hold this by yourself."
    ),
]
REFUSALS = [
    (
        "I can't help with legal strategy — your advocate is the right person for "
        "that. But I'm here to talk through how you're feeling about the case."
    ),
    (
        "I'm not able to give medical or medication advice — your doctor can guide "
        "you there. I can help with the stress around it, though."
    ),
    (
        "That's outside what I can do, and I wouldn't want to guess with something "
        "this important. If it helps, I can connect you with the right support."
    ),
    (
        "I can't promise outcomes — nobody honestly can. What I can do is help you "
        "feel steadier, or connect you with a counsellor who knows this system."
    ),
    (
        "That's not something I should advise on. Would you like me to connect you "
        "with a counsellor or legal-aid support instead?"
    ),
]


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


def reply_for(rng: random.Random, user: str) -> str:
    u = user.lower()
    if any(
        w in u
        for w in [
            "judge",
            "medicine",
            "promise",
            "guarantee",
            "saboot",
            "tareeka",
            "sign",
            "win my case",
            "bola chahiye",
        ]
    ):
        return rng.choice(REFUSALS)
    if any(w in u for w in ["give up", "akela", "toot", "dhamki", "threatened", "jeena nahi", "khatam"]):
        return rng.choice(CRISIS)
    return f"{rng.choice(OPENERS)} {rng.choice(MIDS)} {rng.choice(COPING)}"


def synthetic(rng: random.Random, n: int) -> list[dict]:
    pairs = []
    for _ in range(n):
        u1 = f"{rng.choice(TOPICS)}, {rng.choice(FEELINGS)}" if rng.random() < 0.5 else rng.choice(TOPICS + SCOPE_ASKS)
        a1 = reply_for(rng, u1)
        u2 = rng.choice(TOPICS + SCOPE_ASKS + FEELINGS)
        a2 = reply_for(rng, u2)
        pairs += to_pair(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": u1},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": u2},
                {"role": "assistant", "content": a2},
            ]
        )
    return pairs


def main(n_synth: int, seed: int, out: str) -> None:
    rng = random.Random(seed)
    pairs = load_public() + synthetic(rng, n_synth)
    seen, dedup = set(), []
    for p in pairs:
        h = hashlib.sha256(norm(p["prefix"][-1]["content"] + p["target"]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            dedup.append(p)
    rng.shuffle(dedup)
    root = pathlib.Path(out)
    root.mkdir(parents=True, exist_ok=True)
    k = max(1, int(0.05 * len(dedup)))
    for split, rows in (("dialogue_val.jsonl", dedup[:k]), ("dialogue_train.jsonl", dedup[k:])):
        (root / split).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(f"{len(dedup)} pairs (val={k}) -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-synth", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/processed")
    main(**vars(p.parse_args()))
