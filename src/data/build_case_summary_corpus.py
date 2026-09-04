"""Synthetic (case_facts JSON -> counsellor recommendation) pairs — v2.

v1 defects fixed here:
  1. facts schema EXACTLY matches nodes_llm.summarize() (adds p_escalation/errors;
     drops bail_status/victim_language/engagement that never arrive at runtime).
  2. `decide()` is a PURE field->decision function, aligned with
     src/orchestration/policies.py defaults (0.75 / 0.45 / 0.60) — the writer's
     prose can never contradict the orchestrator's routing.
  3. Scores sampled CONTINUOUSLY across [0,1] with boundary density near the
     policy thresholds, so the step boundaries the model must express are trained.
  4. Self-audit: every generated row is re-checked — target labels re-derived and
     every numeric token in the target verified present in the input JSON
     (groundedness by construction)."""

import argparse
import json
import pathlib
import random
import re

T_ESCALATE, T_MODERATE, T_CONF, T_FORECAST = 0.75, 0.60, 0.45, 0.60  # == policies.py defaults
STAGES = ["FIR", "investigation", "trial", "compensation"]
HARD = ["voice_stress_high", "threat_language_detected", "score_rising_trend", "hearing_imminent"]
SOFT = ["sentiment_negative_mild", "voice_stress_mixed", "response_latency_up", "missed_checkins"]
ACTIONS = {
    "ESCALATE-24H": (
        "Schedule a counsellor check-in call within 24 hours; if "
        "threat_language_detected is active, notify the district protection "
        "officer per witness-protection protocol."
    ),
    "VERIFY-HUMAN-24H": (
        "Score is high but confidence is low — schedule a human check-in "
        "call within 24 hours to verify before escalation."
    ),
    "HUMAN-72H": (
        "Do not auto-escalate on this score alone. Assign a human check-in call within 72 hours to disambiguate."
    ),
    "HUMAN-48H": (
        "A threat/intimidation signal is present despite a moderate score — human "
        "check-in within 48 hours; screen for intimidation per protocol."
    ),
    "MONITOR-CLOSELY": (
        "Keep the case on the counsellor watchlist; no immediate escalation; "
        "re-evaluate after the next scheduled check-in."
    ),
    "ROUTINE": ("No escalation. Continue routine scheduled check-ins."),
    "PROACTIVE-24H": (
        "The score alone is not elevated, but forecasted escalation probability "
        "crosses the alert threshold — schedule a proactive check-in within "
        "24 hours."
    ),
    "REQUEST FIELDS": (
        "Required triage fields are missing (composite score unavailable). "
        "Request them before triage; no escalation beyond routine monitoring."
    ),
}


def decide(f: dict) -> tuple[str, str]:
    score, conf = f["composite_score"], f["confidence"]
    threat = "threat_language_detected" in (f["top_signals"] or [])
    if score is None:
        return "INSUFFICIENT DATA", "REQUEST FIELDS"
    if score >= T_ESCALATE:
        if conf is not None and conf < T_CONF:
            return "HIGH (LOW CONFIDENCE)", "VERIFY-HUMAN-24H"
        return "HIGH", "ESCALATE-24H"
    if score >= T_MODERATE:
        if conf is not None and conf < T_CONF:
            return "MODERATE (LOW CONFIDENCE)", "HUMAN-72H"
        if threat:
            return "MODERATE (THREAT SIGNAL)", "HUMAN-48H"
        return "MODERATE", "MONITOR-CLOSELY"
    if threat:
        return "LOW (THREAT SIGNAL)", "HUMAN-72H"
    if f["p_escalation"] is not None and f["p_escalation"] >= T_FORECAST:
        return "LOW (FORECAST ALERT)", "PROACTIVE-24H"
    return "LOW", "ROUTINE"


def _fmt(x: float | None) -> str:
    return "data unavailable" if x is None else f"{x:.2f}"


def target(f: dict) -> str:
    label, action = decide(f)
    sigs = ", ".join(f["top_signals"]) or "no distinguishing signals"
    parts = [
        (f"Risk: {label} (composite {_fmt(f['composite_score'])}, confidence {_fmt(f['confidence'])})."),
        f"Stage: {f['case_stage']}.",
        f"Key signals: {sigs}.",
    ]
    d = f["days_to_next_hearing"]
    if d is not None and d <= 7:
        parts.append(f"Hearing in {d} day(s) — expect elevated stress around the date.")
    if f["p_escalation"] is not None:
        parts.append(f"Forecast: escalation probability {f['p_escalation']:.2f} over the next 7 days.")
    if f["errors"]:
        parts.append(
            f"Note: some signals were unavailable at scoring time ({'; '.join(f['errors'])}) — interpret with caution."
        )
    parts.append(f"Recommended action: {ACTIONS[action]}")
    return " ".join(parts)


def sample_score(rng: random.Random, band: str) -> float | None:
    if band == "missing":
        return None
    if band == "high":
        s = rng.uniform(0.75, 0.99)
    elif band == "moderate":
        s = rng.uniform(0.60, 0.749)
    else:
        s = rng.uniform(0.03, 0.599)
    if rng.random() < 0.35:  # boundary density: straddle the threshold
        t = T_ESCALATE if band == "high" else T_MODERATE
        s = rng.uniform(t - 0.03, t + 0.03)
    return round(min(max(s, 0.0), 0.99), 3)


def sample_row(rng: random.Random) -> dict:
    band = rng.choices(["low", "moderate", "high", "missing"], weights=[0.30, 0.30, 0.25, 0.15])[0]
    if rng.random() < 0.08:  # full-range exploration
        band = rng.choice(["low", "moderate", "high"])
    f = {
        "case_id": f"CASE-{rng.randint(10000, 99999)}",
        "case_stage": rng.choice(STAGES),
        "days_to_next_hearing": None if rng.random() < 0.40 else rng.randint(1, 45),
        "composite_score": sample_score(rng, band),
        "confidence": (
            None
            if rng.random() < 0.15
            else round(rng.uniform(0.15, 0.44), 2)
            if rng.random() < 0.30
            else round(rng.uniform(0.45, 0.95), 2)
        ),
        "top_signals": [],
        "p_escalation": (round(rng.uniform(0.05, 0.95), 2) if rng.random() < 0.35 and band != "missing" else None),
        "errors": (rng.choice([["signal:voice:unavailable"], ["forecast:unavailable"]]) if rng.random() < 0.07 else []),
    }
    pool = HARD if band == "high" else (SOFT + HARD[:2] if band == "moderate" else SOFT)
    f["top_signals"] = rng.sample(pool, k=rng.randint(1, min(3, len(pool))))
    if "hearing_imminent" in f["top_signals"]:
        f["days_to_next_hearing"] = rng.randint(1, 5)
    if band != "missing" and rng.random() < 0.08:  # T2-trigger cases: low score, hot forecast
        f["p_escalation"] = round(rng.uniform(0.62, 0.95), 2)
    return f


def _audit(rows: list[dict]) -> None:
    cats: dict[str, int] = {}
    for r in rows:
        facts = json.loads(r["messages"][1]["content"])
        tgt = r["messages"][2]["content"]
        label, _ = decide(facts)
        if not tgt.startswith(f"Risk: {label}"):
            raise ValueError(f"label drift: expected {label!r}, got {tgt[:40]!r}")
        known = {_fmt(facts[k]) for k in ("composite_score", "confidence", "p_escalation")}
        known |= {str(facts["days_to_next_hearing"])}
        for num in re.findall(r"\d+\.\d+", tgt):
            if num not in known:
                raise ValueError(f"ungrounded number {num} in target: {tgt[:80]!r}")
        cats[label] = cats.get(label, 0) + 1
    print("coverage:", json.dumps(cats, sort_keys=True))


def _check_prompt_thresholds(sysprompt: str) -> None:
    """The prompt's stated bands must match the corpus constants exactly —
    drift here means the writer learns a policy the orchestrator doesn't run."""
    stated = set(re.findall(r"0\.\d{2}", sysprompt))
    for t in (T_ESCALATE, T_MODERATE, T_CONF, T_FORECAST):
        if f"{t:.2f}" not in stated:
            raise ValueError(
                f"prompt/corpus drift: threshold {t:.2f} not stated in "
                f"casewriter_system.txt — sync prompts and constants"
            )


def main(n: int, seed: int, out: str) -> None:
    rng = random.Random(seed)
    sysprompt = pathlib.Path("prompts/casewriter_system.txt").read_text(encoding="utf-8").strip()
    _check_prompt_thresholds(sysprompt)
    rows = [
        {
            "messages": [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": json.dumps(f := sample_row(rng), ensure_ascii=False)},
                {"role": "assistant", "content": target(f)},
            ]
        }
        for _ in range(n)
    ]
    _audit(rows)
    rng.shuffle(rows)
    root = pathlib.Path(out)
    root.mkdir(parents=True, exist_ok=True)
    k = max(1, int(0.05 * len(rows)))
    for name, part in (("case_summary_val.jsonl", rows[:k]), ("case_summary_train.jsonl", rows[k:])):
        (root / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in part))
    print(f"wrote {n} rows -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/processed")
    main(**vars(p.parse_args()))
