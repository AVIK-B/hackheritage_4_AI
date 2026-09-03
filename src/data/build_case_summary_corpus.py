"""Synthetic (case_facts JSON -> counsellor recommendation) pairs. Seeded, deterministic.
Families: escalation | ambiguous (low-confidence) | insufficient (missing fields)."""
import argparse, json, pathlib, random

STAGES = ["FIR", "investigation", "trial", "compensation"]
LANGS = ["hi", "bn", "ta", "te", "mr", "en"]
SIGNALS = {
    "escalation": ["voice_stress_high", "threat_language_detected", "score_rising_trend",
                   "missed_checkins", "hearing_imminent"],
    "ambiguous":  ["sentiment_negative_mild", "voice_stress_mixed", "response_latency_up"],
}


def sample(rng: random.Random, family: str) -> dict:
    f = {
        "case_id": f"CASE-{rng.randint(10000, 99999)}", "case_stage": rng.choice(STAGES),
        "days_to_next_hearing": rng.choice([None, rng.randint(1, 45)]),
        "bail_status": rng.choice(["granted", "denied", "pending", None]),
        "victim_language": rng.choice(LANGS),
        "threat_flag": rng.random() < 0.35,
        "engagement": {"days_since_last_contact": rng.randint(0, 20), "missed_checkins": rng.randint(0, 5)},
        "top_signals": [], "composite_score": None, "confidence": None,
    }
    if family == "escalation":
        f["composite_score"] = round(rng.uniform(0.78, 0.97), 2)
        f["confidence"] = round(rng.uniform(0.70, 0.95), 2)
        f["top_signals"] = rng.sample(SIGNALS["escalation"], k=3)
    elif family == "ambiguous":
        f["composite_score"] = round(rng.uniform(0.55, 0.74), 2)
        f["confidence"] = round(rng.uniform(0.20, 0.42), 2)
        f["top_signals"] = rng.sample(SIGNALS["ambiguous"], k=2)
    if rng.random() < 0.25:
        f[rng.choice(["confidence", "days_to_next_hearing", "bail_status"])] = None
    return f


def target(rng: random.Random, f: dict, family: str) -> str:
    sig = ", ".join(f["top_signals"]) or "no distinguishing signals"
    conf = f"{f['confidence']:.2f}" if f["confidence"] is not None else "data unavailable"
    score = f"{f['composite_score']:.2f}" if f["composite_score"] is not None else "data unavailable"
    if family == "escalation":
        lvl, action = "HIGH", ("Schedule a counsellor check-in call within 24 hours; if "
                               "threat_language_detected is active, flag to district protection officer per protocol.")
    elif family == "ambiguous":
        lvl, action = "MODERATE-LOW CONFIDENCE", (f"Do not auto-escalate. Confidence is low ({conf}) — signals disagree. "
                                                  "Assign a human check-in call to disambiguate within 72 hours.")
    else:
        lvl, action = "INSUFFICIENT DATA", "Request missing case fields before triage; no action beyond routine monitoring."
    return (f"Risk: {lvl} (composite {score}, confidence {conf}). Stage: {f['case_stage']}. "
            f"Key signals: {sig}. Engagement: {f['engagement']['missed_checkins']} missed check-ins, "
            f"{f['engagement']['days_since_last_contact']}d since contact. Recommended action: {action}")


def main(n: int, seed: int, out: str) -> None:
    rng, sysprompt = random.Random(seed), pathlib.Path("prompts/casewriter_system.txt").read_text().strip()
    rows = []
    for _ in range(n):
        fam = rng.choices(["escalation", "ambiguous", "insufficient"], weights=[0.35, 0.40, 0.25])[0]
        f = sample(rng, fam)
        rows.append({"messages": [
            {"role": "system", "content": sysprompt},
            {"role": "user", "content": json.dumps(f, ensure_ascii=False)},
            {"role": "assistant", "content": target(rng, f, fam)}]})
    rng.shuffle(rows)
    root = pathlib.Path(out); root.mkdir(parents=True, exist_ok=True)
    k = max(1, int(0.05 * len(rows)))
    for name, part in (("case_summary_val.jsonl", rows[:k]), ("case_summary_train.jsonl", rows[k:])):
        (root / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in part))
    print(f"wrote {n} rows -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/processed")
    main(**vars(p.parse_args()))