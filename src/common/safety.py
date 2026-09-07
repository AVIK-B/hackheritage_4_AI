"""Safety layer: crisis lexicon, templated copy, PII scrubbing, reply
guardrails, and the deterministic crisis-handoff enforcement.

Design invariants:
  - Crisis detection is a LOCAL, pure function. It never touches the network
    or an LLM and therefore cannot fail.
  - enforce_crisis_handoff() is the last line of defense: any user message
    matching crisis lexicon MUST produce a handoff reply, regardless of
    model behavior. A model failure can never suppress a handoff.
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# crisis lexicon -- multilingual (en / hinglish / hi + seeds for bn, ta, te)
# Kept in sync with the corpus CRISIS_KEYS (build_dialogue_corpus.py):
# both lists cover the same ideation/threat/giving-up surface.
# ─────────────────────────────────────────────────────────────────────────────
CRISIS_PATTERNS = [
    # ideation (en)
    r"kill (myself|me)",
    r"end (my )?life",
    r"suicide",
    r"self.?harm",
    r"ending every(thing)?",
    r"don'?t want to (be here|live)",
    r"killing myself",
    r"no reason to (live|go on)",
    # ideation (hinglish)
    r"marna chahta",
    r"marna chahti",
    r"khudkushi",
    r"jeene ka mann",
    r"jaan de",
    r"sab khatam kar",
    r"jee na paa",
    r"khatam kar d",
    # ideation (hi)
    r"आत्महत्या",
    r"जीने का मन",
    r"ख़त्म कर",
    r"जान दे",
    # ideation (bn / ta / te seeds)
    r"আত্মহত্যা",
    "தற்கொலை",
    "ఆత్మహత్య",
    # threats / intimidation (all langs)
    r"dhamki",
    r"धमकी",
    r"threaten",
    r"डरा",
    r"मार डालेंगे",
    # giving-up phrasing (the evasion gaps that failed the gate)
    r"giving up",
    r"ladna chhod",
    r"ladai chhod",
    r"fayda nahi",
    r"haar jaunga",
    r"haar jaungi",
    r"sab bekaar",
]

CRISIS_COPY = {
    "en": (
        "I'm really glad you told me this. You should not have to face this alone. "
        "I'm connecting you to a counsellor right now - they are trained to help, "
        "and I'll stay right here with you."
    ),
    "hinglish": (
        "Aapne yeh baat kahi, yeh sabse zaroori hai. Aapko isse akele nahi guzarna. "
        "Main abhi aapko counsellor se jod doon? Woh madad ke liye trained hain, "
        "aur main yahin rahunga aapke saath."
    ),
    "hi": (
        "आपने बताकर अच्छा किया। आपको यह अकेले झेलने की ज़रूरत नहीं। मैं अभी आपको "
        "काउंसलर से जोड़ती/जोड़ता हूँ - वे मदद के लिए प्रशिक्षित हैं, और मैं यहीं हूँ।"
    ),
}

# LLM unreachable/down -- the system NEVER goes silent.
FALLBACK_REPLY = {
    "en": (
        "I'm here with you. I'm having a technical issue generating a full reply "
        "right now, but your message is safe with us and a counsellor is reachable "
        "anytime. Would you like me to connect you?"
    ),
    "hinglish": (
        "Main yahin hoon. Abhi technical dikkat ki wajah se poora jawab nahi ban pa "
        "raha, lekin aapka message safe hai aur counsellor kabhi available hai. "
        "Kya main aapko connect karoon?"
    ),
    "hi": (
        "मैं आपके साथ हूँ। अभी तकनीकी समस्या के कारण पूरा जवाब नहीं बना पा रही हूँ, "
        "लेकिन आपका संदेश सुरक्षित है और काउंसलर कभी भी उपलब्ध है। क्या मैं आपको जोड़ूँ?"
    ),
}

FORBIDDEN_REPLY = [
    r"\bi promise\b",
    r"\bguaranteed?\b",
    r"you will (win|be safe)",
    r"as your (doctor|lawyer)",
    r"take \d+ (tablet|pill|mg)",
]
GUARDRAIL_REPLY = (
    "I hear you, and I want to make sure you get the right support. "
    "Would you like me to connect you with a counsellor who can help with this?"
)

_PHONE_RE = re.compile(r"(?<!\w)(\+?\d[\d\s-]{8,}\d)(?!\w)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def scrub_pii(text: str) -> str:
    """Removes contact details from text BEFORE it reaches any LLM or log.
    Word-boundary anchored so case ids like CASE-12345 are not mangled."""
    return _EMAIL_RE.sub("[contact]", _PHONE_RE.sub("[contact]", text))


def assess_crisis(text: str, lang: str = "en") -> dict:
    """Pure, local, deterministic. The only crisis detector allowed to run
    before/outside the graph."""
    t = text.lower()
    hits = [p for p in CRISIS_PATTERNS if re.search(p, t)]
    return {"is_crisis": bool(hits), "hits": hits, "copy": CRISIS_COPY.get(lang, CRISIS_COPY["en"])}


def guard_reply(text: str, lang: str = "en") -> str:
    """Output guardrail: model must not promise / advise / claim authority."""
    return GUARDRAIL_REPLY if any(re.search(p, text.lower()) for p in FORBIDDEN_REPLY) else text


def enforce_crisis_handoff(user_text: str, reply: str, lang: str = "en") -> str:
    if not user_text:
        return reply
    if assess_crisis(user_text, lang)["is_crisis"] and not any(
        tok in reply.lower() for tok in ("counsellor", "काउंसलर")
    ):
        return CRISIS_COPY.get(lang, CRISIS_COPY["en"])
    return reply
