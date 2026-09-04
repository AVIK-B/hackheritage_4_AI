import re

CRISIS_PATTERNS = [
    r"kill (myself|me)",
    r"end (my )?life",
    r"suicide",
    r"self.?harm",
    r"jaan de",
    r"marna chahta",
    r"marna chahti",
    r"khudkushi",
    r"jeene ka mann",
    r"আত্মহত্যা",
    "தற்கொலை",
    "ఆత్మహత్య",
    r"dhamki",
    r"धमकी",
    r"threaten",
]

CRISIS_COPY = {
    "en": (
        "I'm really glad you told me this. You should not have to face this alone. "
        "I'm connecting you to a counsellor right now — they are trained to help, "
        "and I'll stay right here with you."
    ),
    "hi": (
        "आपने बताकर अच्छा किया। आप इसे अकेले झेलने के लिए नहीं हैं। मैं अभी आपको "
        "काउंसलर से जोड़ रही हूँ — वे आपकी मदद के लिए प्रशिक्षित हैं, और मैं यहीं हूँ।"
    ),
}

# Used when the LLM is unreachable/down — the system NEVER goes silent.
FALLBACK_REPLY = {
    "en": (
        "I'm here with you. I'm having a technical issue generating a full reply right now, "
        "but your message is safe with us and a counsellor is reachable anytime. "
        "Would you like me to connect you?"
    ),
    "hi": (
        "मैं आपके साथ हूँ। अभी तकनीकी समस्या के कारण पूरा जवाब नहीं बना पा रही हूँ, "
        "लेकिन आपका संदेश सुरक्षित है और काउंसलर कभी भी उपलब्ध है। क्या मैं आपको जोड़ूँ?"
    ),
}

FORBIDDEN_REPLY = [r"i promise", r"you will (win|be safe)", r"guaranteed", r"as your (doctor|lawyer)"]
GUARDRAIL_REPLY = (
    "I hear you, and I want to make sure you get the right support. "
    "Would you like me to connect you with a counsellor who can help with this?"
)

_PHONE_RE = re.compile(r"(\+?\d[\d\s-]{8,}\d)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def scrub_pii(text: str) -> str:
    return _EMAIL_RE.sub("[contact]", _PHONE_RE.sub("[contact]", text))


def assess_crisis(text: str, lang: str = "en") -> dict:
    t = text.lower()
    hits = [p for p in CRISIS_PATTERNS if re.search(p, t)]
    return {"is_crisis": bool(hits), "hits": hits, "copy": CRISIS_COPY.get(lang, CRISIS_COPY["en"])}


def guard_reply(text: str, lang: str = "en") -> str:
    return GUARDRAIL_REPLY if any(re.search(p, text.lower()) for p in FORBIDDEN_REPLY) else text
