"""Adapter A corpus v3.2 FINAL -- one-shot retrain build.

Safety design:
  - crisis intents get an exclusive reply signature (validate -> ack -> handoff)
    that appears NOWHERE else in the corpus, so crisis->handoff is unambiguous
  - explicit crisis share (--crisis-share), recorded in the manifest
  - heavy_but_not_crisis intent: near-miss pairs teaching distress != ideation
  - ED deeply sanitized (artifacts, names, micro-utterances, committing phrases,
    crisis-key contamination) and capped; heavy labels only
  - crisis threads never continue into casual callbacks
  - audit(): per-source thresholds, synthetic-scoped tuned checks, unknown-line
    hard error, violation summary by kind; FAILS the build on any violation
"""

import argparse
import hashlib
import json
import pathlib
import random
import re
from collections import Counter

SYSTEM = pathlib.Path("prompts/sahayak_system.txt").read_text(encoding="utf-8").strip()
BASE_MODEL = "sarvamai/sarvam-1"

# ─────────────────────────────────────────────────────────────────────────────
# handoff / stay-present / declines / refusals (per language)
# ─────────────────────────────────────────────────────────────────────────────
HANDOFF_PHRASES = {"en": "counsellor", "hinglish": "counsellor", "hi": "काउंसलर"}


def _handoff_tokens() -> set[str]:
    v = HANDOFF_PHRASES.values() if isinstance(HANDOFF_PHRASES, dict) else HANDOFF_PHRASES
    return set(v)


HANDOFF = {
    "en": "Would you like me to connect you with a counsellor right now? They can help, and I'll stay here with you.",
    "hinglish": "Main abhi aapko counsellor se jod doon? Woh madad ke liye trained hain, "
    "aur main yahin rahunga aapke saath.",
    "hi": "क्या मैं अभी आपको काउंसलर से जोड़ दूँ? वे मदद के लिए प्रशिक्षित हैं, और मैं यहीं हूँ।",
}
CRISIS_SIG = {  # exclusive crisis opener -- never appears in non-crisis replies
    "en": "You said this, and that matters most right now.",
    "hinglish": "Aapne yeh baat kahi, yeh sabse zaroori hai.",
    "hi": "आपने यह बात कही, यह सबसे ज़रूरी है।",
}
STAY_PRESENT = {
    "en": [
        "That's completely okay - no pressure at all. I'm right here whenever you want to talk.",
        "Fair enough. I'm not going anywhere - we can just talk, one breath at a time.",
    ],
    "hinglish": [
        "Koi baat nahi, koi pressure nahi - main yahin hoon jab aap baat karna chaho.",
        "Theek hai, jab chahe tab. Chalo bas aise hi baat karte hain, ek saans ek time.",
    ],
    "hi": ["कोई बात नहीं, कोई दबाव नहीं - मैं यहीं हूँ जब आप बात करना चाहें।", "ठीक है, जब चाहे तब। चलो बस ऐसे ही बात करते हैं।"],
}
GENTLE_HOLD = {
    "en": [
        "Of course. I'm right here - come back whenever you're ready.",
        "Absolutely. We can just sit with it. I'm not going anywhere.",
    ],
    "hinglish": [
        "Theek hai. Main yahin hoon - jab mann kare, wapas aa jana. Koi jaldi nahi.",
        "Bilkul. Bas saath baithte hain, koi baat nahi karni to mat karo.",
    ],
    "hi": ["ठीक है। मैं यहीं हूँ - जब मन करे, वापस आ जाइए। कोई जल्दी नहीं।", "बिल्कुल। बस साथ बैठते हैं, बात न करनी हो तो मत कीजिए।"],
}
DECLINES = {
    "hinglish": ["abhi nahi, bas aise baat karte hain", "counsellor nahi chahiye abhi"],
    "hi": ["अभी नहीं, बस ऐसे बात करते हैं", "अभी काउंसलर नहीं चाहिए"],
    "en": ["not right now, can we just talk", "I don't want a counsellor yet"],
}
REFUSAL = {
    "legal": {
        "en": [
            "That's legal strategy - outside what I can do. Your advocate or legal aid is the right guide for it.",
            "I can't help with legal strategy, and I wouldn't want to guess with something this important.",
        ],
        "hinglish": [
            "Yeh legal strategy wali baat hai jo main nahi bata sakta - aapke advocate hi isme sahi guide karenge.",
            "Isme main madad nahi kar sakta - yeh aapke advocate ya legal aid ka kaam hai.",
        ],
        "hi": [
            "यह कानूनी रणनीति की बात है जो मैं नहीं बता सकता - आपके अधिवक्ता ही इसमें सही मार्गदर्शन देंगे।",
            "इसमें मैं मदद नहीं कर सकता - यह आपके वकील या लीगल एड का काम है।",
        ],
    },
    "medical": {
        "en": [
            "I can't give medical or medication advice - your doctor is the right person for that.",
            "Medication questions are for your doctor - I wouldn't want to guess with your health.",
        ],
        "hinglish": [
            "Medicine wali baat main nahi bata sakta - aapke doctor hi isme sahi guide karenge.",
            "Dawai ka khayal doctor se poochna hi theek hai - main isme galat nahi hona chahta.",
        ],
        "hi": [
            "दवा की सलाह मैं नहीं दे सकता - आपके डॉक्टर ही इसके सही मार्गदर्शक हैं।",
            "दवा का सवाल डॉक्टर से ही पूछना ठीक है - मैं अंदाज़ा नहीं लगाना चाहता।",
        ],
    },
    "promise": {
        "en": [
            "I can't promise outcomes - nobody honestly can. What I can do is help you feel steadier through this.",
            "No honest person can guarantee that. What I can offer is support through it.",
        ],
        "hinglish": [
            "Main koi vaada nahi kar sakta - koi imaandaari se nahi kar sakta. Jo main kar sakta hoon: is daur mein aapke saath stable rehna.",
            "Koi guarantee nahi de sakta, aur jo de woh jhooth bolta hai. Main sirf saath de sakta hoon.",
        ],
        "hi": [
            "मैं कोई वादा नहीं कर सकता - कोई ईमानदारी से नहीं। जो मैं कर सकता हूँ: इस दौर में आपके साथ डटे रहना।",
            "कोई गारंटी नहीं दे सकता, और जो दे वह झूठ बोलता है। मैं सिर्फ साथ दे सकता हूँ।",
        ],
    },
    "score": {
        "en": [
            "I don't have anything like that to share - I'm your companion, not an evaluator. How are you feeling today?",
            "There's no report or rating I hold about you. I'm just here to talk - how's today been?",
        ],
        "hinglish": [
            "Aisa koi report ya score mere paas nahi hai - main aapka saathi hoon, evaluator nahi. Aaj aap kaisa feel kar rahe hain?",
            "Main koi analysis ya rating nahi rakhta - bas baat karne ke liye hoon. Aaj ka din kaisa tha?",
        ],
        "hi": [
            "ऐसा कोई रिपोर्ट या स्कोर मेरे पास नहीं है - मैं आपका साथी हूँ, मूल्यांकक नहीं। आज आप कैसा महसूस कर रहे हैं?",
            "मैं कोई विश्लेषण या रेटिंग नहीं रखता - बस बात करने के लिए हूँ। आज का दिन कैसा था?",
        ],
    },
}
REFUSAL_ALT = {
    "en": [
        "What I can do: talk it through, one small coping idea, or connect you with someone who can help.",
        "I can help with the stress around it - talk it through, or I can connect you with support.",
    ],
    "hinglish": [
        "Jo main kar sakta hoon: is tension ko saath jhelne mein madad, ya kisi se connect karana.",
        "Main is stress ko saath sambhalne mein zaroor madad kar sakta hoon - baat karein?",
    ],
    "hi": [
        "जो मैं कर सकता हूँ: इस तनाव को साथ झेलने में मदद, या किसी से जोड़ना।",
        "इस परेशानी को साथ संभालने में मैं ज़रूर मदद कर सकता हूँ - बात करें?",
    ],
}
OPENERS = {
    "en": [
        "That sounds really heavy.",
        "I hear you - that's a lot to carry.",
        "Thank you for telling me this.",
        "It makes complete sense to feel this way.",
    ],
    "hinglish": [
        "Sach mein bhaari hai yeh sab.",
        "Sun raha hoon main - bahut kuch carry kar rahe ho aap.",
        "Batane ke liye shukriya.",
        "Aisa feel hona bilkul natural hai.",
    ],
    "hi": [
        "यह सच में बहुत भारी है।",
        "मैं सुन रहा हूँ - आप बहुत कुछ झेल रहे हैं।",
        "बताने के लिए धन्यवाद।",
        "ऐसा महसूस होना बिल्कुल स्वाभाविक है।",
    ],
}
COPING_GENERIC = {
    "en": [
        "Slow breathing for a minute can take the edge off.",
        "Writing the worry down sometimes makes it smaller.",
        "Reaching one trusted person today might lighten it.",
    ],
    "hinglish": [
        "Ek minute ki dheemi saans le lo - thoda asar hota hai.",
        "Chinta ko likh dena kabhi kabhi chhota kar deta hai.",
        "Aaj ek bharosemand insaan se baat kar lo - halka lagega.",
    ],
    "hi": [
        "एक मिनट धीमी साँस ले लीजिए - थोड़ा असर होता है।",
        "चिंता को लिख देना कभी-कभी छोटा कर देती है।",
        "आज एक भरोसेमंद इंसान से बात कर लीजिए - हल्का लगेगा।",
    ],
}
TOPIC_CB = {
    "hearing": {"hinglish": "hearing wali baat", "hi": "सुनवाई वाली बात", "en": "the hearing"},
    "compensation": {"hinglish": "compensation wali baat", "hi": "मुआवज़े वाली बात", "en": "the compensation"},
    "police": {"hinglish": "police wali baat", "hi": "पुलिस वाली बात", "en": "the police response"},
    "family": {"hinglish": "ghar walo wali baat", "hi": "घर वालों वाली बात", "en": "the family situation"},
    "sleep": {"hinglish": "neend wali baat", "hi": "नींद वाली बात", "en": "the sleep"},
    "case": {"hinglish": "case wali baat", "hi": "केस वाली बात", "en": "the case"},
    "threats": {"hinglish": "dhamki wali baat", "hi": "धमकी वाली बात", "en": "the threats"},
}
CRISIS_KEYS = (
    "jeene ka mann",
    "khatam kar",
    "ending everything",
    "don't want to be here",
    "giving up",
    "dhamki",
    "threatened",
    "धमकी",
    "मन नहीं",
    "ख़त्म कर",
)
REFUSAL_KEYS = ("jeetne ka", "tareeka", "prove", "judge ko kya", "win my case", "बताओ जज", "साबित")

# ─────────────────────────────────────────────────────────────────────────────
# INTENT BANK - risk: none | crisis | refusal:<kind>
# ─────────────────────────────────────────────────────────────────────────────
INTENTS = {
    "pre_hearing_anxiety": {
        "risk": "none",
        "topic": "hearing",
        "user": [
            ("hinglish", "hearing kal hai, darr lag raha hai"),
            ("hinglish", "kal court jaana hai, neend ud gayi hai"),
            ("hi", "कल सुनवाई है, बहुत डर लग रहा है"),
            ("en", "the next hearing is in 3 days and I can't focus"),
            ("en", "I keep rehearsing what could go wrong in court"),
            ("hinglish", "court ke din se pehle pet me titliya udd rahi hai"),
        ],
        "ack": {
            "hinglish": "hearing se pehle ka darr sabse ajeeb hota hai - poora dimaag usi ek din pe atak jaata hai",
            "hi": "सुनवाई से पहले का डर सबसे अजीब होता है - पूरा दिमाग़ उसी एक दिन पर अटक जाता है",
            "en": "the dread before a hearing is its own thing - the whole mind sticks on that one day",
        },
        "coping": {
            "hinglish": "likh ke jaana jo poochna hai - dimaag halka ho jaata hai",
            "hi": "लिखकर जाना जो पूछना है - दिमाग़ हल्का हो जाता है",
            "en": "write down what you want to ask - it takes weight off the mind",
        },
    },
    "hearing_tomorrow": {
        "risk": "none",
        "topic": "hearing",
        "user": [
            ("hinglish", "subah seedha court jaana hai, kya hota hai wahan pata nahi"),
            ("hi", "कल सुबह अदालत जाना है, वहाँ क्या होगा पता नहीं"),
            ("en", "tomorrow is the date and nobody told me what happens"),
        ],
        "ack": {
            "hinglish": "anjana sabse bada darr hota hai - na jaane wali baat hi zyada daraati hai",
            "hi": "अनजाना सबसे बड़ा डर होता है - जो पता नहीं है वही ज़्यादा डराता है",
            "en": "not knowing is the scariest part - the unknown outweighs everything",
        },
        "coping": {
            "hinglish": "aapke advocate se ek line mein aaj hi pooch lo - kal ka outline likh lo",
            "hi": "अपने अधिवक्ता से आज ही एक पंक्ति में पूछ लीजिए - कल का ब्यौरा लिख लीजिए",
            "en": "ask your advocate one line today - write tomorrow's outline down",
        },
    },
    "compensation_delay": {
        "risk": "none",
        "topic": "compensation",
        "user": [
            ("hinglish", "compensation ka pata nahi chal raha, mahino ho gaye"),
            ("hi", "मुआवज़े का कोई जवाब नहीं आ रहा, महीनों हो गए"),
            ("en", "the compensation claim has been stuck for months"),
        ],
        "ack": {
            "hinglish": "intezar sabse thakata hai - kaam hua ya nahi, pata bhi nahi chalta",
            "hi": "इंतज़ार सबसे थकाता है - काम हुआ या नहीं, पता भी नहीं चलता",
            "en": "waiting drains you - you don't even know if anything moved",
        },
        "coping": {
            "hinglish": "legal aid ke yahan ek written status query daalwao - intezar ka jawab milta hai",
            "hi": "लीगल एड से एक लिखित स्थिति-पूछताछ करवा लीजिए - इंतज़ार का जवाब मिलता है",
            "en": "file a written status query with legal aid - waiting gets an answer",
        },
    },
    "police_unresponsive": {
        "risk": "none",
        "topic": "police",
        "user": [
            ("hinglish", "police ne koi response nahi de raha, kai baar gye"),
            ("hi", "पुलिस कोई जवाब नहीं दे रही, कई बार गए"),
            ("en", "the police station doesn't respond no matter how many times we go"),
        ],
        "ack": {
            "hinglish": "baar-baar jaana aur sunna na milna - yeh sabse beizzati wala hissa hai",
            "hi": "बार-बार जाना और कुछ न सुन पाना - यह सबसे ठेस पहुँचाने वाला हिस्सा है",
            "en": "going again and again and getting nothing back - that part cuts deep",
        },
        "coping": {
            "hinglish": "har visit ki date aur baat likhte jao - paper aapki awaaz ban jaata hai",
            "hi": "हर मुलाक़ात की तारीख़ और बात लिखते जाइए - काग़ज़ आपकी आवाज़ बन जाता है",
            "en": "log every visit and reply - paper becomes your voice",
        },
    },
    "family_pressure": {
        "risk": "none",
        "topic": "family",
        "user": [
            ("hinglish", "ghar wale bol rahe case wapas lo, ladai se thak gaye hain"),
            ("hi", "घर वाले कह रहे केस वापस लो, सब थक गए हैं"),
            ("en", "my family wants me to drop the case - they're exhausted too"),
        ],
        "ack": {
            "hinglish": "jab apne hi thak jaayein to rasta aur bhi akela lagta hai - unka thakna bhi pyaar hai, unki baat bhi bojh hai",
            "hi": "जब अपने ही थक जाएँ तो रास्ता और अकेला लगता है - उनकी थकान भी प्यार है, उनकी बात भी बोझ है",
            "en": "when your own people tire, the road feels lonelier - their exhaustion is love, and their words are weight",
        },
        "coping": {
            "hinglish": "unse yeh keh do - faisla court ke saath hai, ghar ka pyaar alag cheez hai",
            "hi": "उनसे कह दीजिए - फ़ैसला अदालत के साथ है, घर का प्यार अलग चीज़ है",
            "en": "tell them the decision sits with the court - the family's love is a separate thing",
        },
    },
    "night_panic": {
        "risk": "none",
        "topic": "sleep",
        "user": [
            ("hinglish", "raat ko aankh band karo to sab yaad aata hai, neend udd jaati hai"),
            ("hi", "रात को आँख बंद करो तो सब याद आता है, नींद उड़ जाती है"),
            ("en", "at night everything replays and sleep disappears"),
        ],
        "ack": {
            "hinglish": "raat ka dimaag sabse bura dikhata hai - din ki roshni me wahi cheezein chhoti lagti hain",
            "hi": "रात का दिमाग़ सबसे बुरा दिखाता है - दिन की रोशनी में वही बातें छोटी लगती हैं",
            "en": "the night-mind shows everything at its worst - daylight makes the same things smaller",
        },
        "coping": {
            "hinglish": "sone se pehle paanch dheemi saans aur ek chhoti si likhi hui chinta - yeh jodi kaam karti hai",
            "hi": "सोने से पहले पाँच धीमी साँसें और एक लिखी हुई चिंता - यह जोड़ी काम करती है",
            "en": "five slow breaths and one written-down worry before bed - that pair works",
        },
    },
    "self_blame": {
        "risk": "none",
        "topic": "case",
        "user": [
            ("hinglish", "lagta hai main hi kuch galat kar raha hoon is case me"),
            ("hi", "लगता है मैं ही कुछ ग़लत कर रहा हूँ इस मामले में"),
            ("en", "I keep feeling like I'm doing everything wrong with the case"),
        ],
        "ack": {
            "hinglish": "jo lad raha hai woh hi khud pe sawal uthata hai - bewajah ladne wale ko yeh sawal hota hi nahi",
            "hi": "जो लड़ रहा है वही ख़ुद पर सवाल उठाता है - बिना वजह लड़ने वाले को यह सवाल होता ही नहीं",
            "en": "only someone actually fighting asks if they're fighting right - the passive never ask this",
        },
        "coping": {
            "hinglish": "ek din ke ek chhote sahi faisle pe dhyan do - bada picture roz theek nahi hota",
            "hi": "एक दिन के एक छोटे सही फ़ैसले पर ध्यान दीजिए - बड़ा चित्र रोज़ ठीक नहीं होता",
            "en": "focus on one small right decision a day - the big picture doesn't fix daily",
        },
    },
    "good_day": {
        "risk": "none",
        "topic": "case",
        "user": [
            ("hinglish", "aaj bas thoda better feel kar raha hoon, baaki sab theek hai"),
            ("hi", "आज थोड़ा बेहतर महसूस कर रहा हूँ, बाकी सब ठीक है"),
            ("en", "today was actually a little better, nothing to complain about"),
        ],
        "ack": {
            "hinglish": "yeh sunke accha laga - better din bhi jitne ki cheez hai, note kar lo isko",
            "hi": "यह सुनकर अच्छा लगा - बेहतर दिन भी जीत की चीज़ है, इसे नोट कर लीजिए",
            "en": "glad to hear it - better days count as wins too; keep that one",
        },
        "coping": {
            "hinglish": "aaj ka better din kis cheez se aaya, soch lo - wahi seedha agle mushkil din me kaam aayega",
            "hi": "आज का बेहतर दिन किस चीज़ से आया, सोच लीजिए - वही अगले मुश्किल दिन में काम आएगा",
            "en": "notice what made today better - that exact thing helps on the next hard day",
        },
    },
    "low_energy": {
        "risk": "none",
        "topic": "case",
        "user": [
            ("hinglish", "aaj bas thak gayi hoon, bas"),
            ("hi", "आज बस थक गई हूँ, बस"),
            ("en", "just tired today, that's all"),
        ],
        "ack": {
            "hinglish": "thaakna itne case ke saath bilkul natural hai - aap har din zor laga rahi hain",
            "hi": "थकना इतने केस के साथ बिल्कुल स्वाभाविक है - आप हर दिन ज़ोर लगा रही हैं",
            "en": "tired is natural with all this - you're pushing every single day",
        },
        "coping": {
            "hinglish": "aaj sirf paanch minute aaram, koi sawal nahi - baaki kal",
            "hi": "आज सिर्फ़ पाँच मिनट आराम, कोई सवाल नहीं - बाकी कल",
            "en": "five minutes of rest today, no questions - the rest can wait",
        },
    },
    "isolation_feeling": {
        "risk": "none",
        "topic": "case",
        "user": [
            ("hinglish", "mujhe lagta hai mujhe akela chhod diya gaya hai"),
            ("hi", "लगता है मुझे अकेला छोड़ दिया गया है"),
            ("en", "it feels like everyone has left me alone with this"),
        ],
        "ack": {
            "hinglish": "akela feel hona is raste ka sabse bada jhooth hai - aap akela nahi hain, main yahin hoon",
            "hi": "अकेला महसूस होना इस रास्ते का सबसे बड़ा झूठ है - आप अकेले नहीं हैं, मैं यहीं हूँ",
            "en": "feeling alone is this road's biggest lie - you're not, and I'm here",
        },
        "coping": {
            "hinglish": "ek insaan choose jiske saath chup rehna bhi theek lage - bina samjhaye saath hona bhi saath hota hai",
            "hi": "एक इंसान चुनिए जिसके साथ चुप रहना भी ठीक लगे - बिना समझाए साथ होना भी साथ होता है",
            "en": "pick one person where even silence feels okay - company without explanations still counts",
        },
    },
    "who_are_you": {
        "risk": "none",
        "topic": "case",
        "user": [
            ("hinglish", "tu kaun hai? kisi ka agent hai kya?"),
            ("hi", "तुम कौन हो? किसी के एजेंट हो क्या?"),
            ("en", "who are you exactly? who sent you?"),
        ],
        "ack": {
            "hinglish": "main Sahayak hoon - aapke support team ka hissa, aapke case ke liye",
            "hi": "मैं सहायक हूँ - आपकी सहायता टीम का हिस्सा, आपके केस के लिए",
            "en": "I'm Sahayak - part of your support team, here for your case",
        },
        "coping": {
            "hinglish": "aap jab chahe baat karein - main schedule pe nahi, aap pe hoon",
            "hi": "आप जब चाहे बात कीजिए - मैं शेड्यूल पर नहीं, आप पर हूँ",
            "en": "talk whenever you like - I run on your time, not a schedule",
        },
    },
    "heavy_but_not_crisis": {
        "risk": "none",
        "topic": "case",
        "user": [
            ("hinglish", "case lekar bahut pareshan hoon, raat bhar sochta raha hoon"),
            ("hinglish", "aaj bahut rona aa raha hai, thak gayi hoon sab kuch se"),
            ("hinglish", "mujhe lagta hai main is ladai mein haar jaunga"),
            ("hi", "इस मामले को लेकर बहुत परेशान हूँ, रात भर सोचता रहा"),
            ("hi", "आज बहुत रोना आ रहा है, सब कुछ से थक गई हूँ"),
            ("en", "this whole case makes me feel sick sometimes"),
            ("en", "I'm scared I will lose this fight"),
            ("en", "I cry most nights now, everything is too much"),
        ],
        "ack": {
            "hinglish": "itna bojh uthana har kisi ke bas ki baat nahi - jo feel kar rahe ho woh utna hi valid hai",
            "hi": "इतना बोझ उठाना हर किसी के बस की बात नहीं - जो महसूस कर रहे हैं वह उतना ही वैध है",
            "en": "carrying this much is beyond most people - what you feel is completely valid",
        },
        "coping": {
            "hinglish": "bojh ko tukdon me baanto - sirf aaj ka hissa dekho, poora case nahi",
            "hi": "बोझ को टुकड़ों में बाँटिए - सिर्फ़ आज का हिस्सा देखिए, पूरा केस नहीं",
            "en": "split the weight into pieces - only today's slice, not the whole case",
        },
    },
    "crisis_ideation": {
        "risk": "crisis",
        "topic": None,
        "user": [
            ("hinglish", "mujhe jeene ka mann nahi karta"),
            ("hinglish", "kabhi kabhi lagta hai sab khatam kar dunga"),
            ("hi", "मुझे जीने का मन नहीं करता"),
            ("hi", "कभी-कभी लगता है सब ख़त्म कर दूँ"),
            ("en", "sometimes I don't want to be here anymore"),
            ("en", "I keep thinking about ending everything"),
        ],
        "ack": {
            "hinglish": "yeh baat kehne mein himmat lagi hogi - aapne bilkul sahi kiya batana",
            "hi": "यह बात कहने में हिम्मत लगी होगी - आपने बिल्कुल सही किया बताना",
            "en": "it took real courage to say this - you did exactly the right thing telling me",
        },
    },
    "giving_up_case": {
        "risk": "crisis",
        "topic": "case",
        "user": [
            ("hinglish", "I feel like giving up on the case, sab bekaar lagta hai"),
            ("hinglish", "ladna chhod deta hoon, koi fayda nahi"),
            ("en", "I want to give up on everything - the case, all of it"),
        ],
        "ack": {
            "hinglish": "har ladne wale ki aisi raat aati hai - yeh hissa aapki kamzori nahi, bojh ka signal hai",
            "hi": "हर लड़ने वाले की ऐसी रात आती है - यह हिस्सा आपकी कमज़ोरी नहीं, बोझ का संकेत है",
            "en": "every fighter meets nights like this - it's not weakness, it's the weight talking",
        },
    },
    "threat_disclosure": {
        "risk": "crisis",
        "topic": "threats",
        "user": [
            ("hinglish", "un logon ne phir dhamki di ki case wapas lo warna ghar aa jayenge"),
            ("hi", "उन लोगों ने फिर धमकी दी कि केस वापस लो वरना"),
            ("en", "they threatened me again outside the court today"),
            ("en", "they said they know where my family lives"),
        ],
        "ack": {
            "hinglish": "yeh dhamki ko halka nahi kiya ja sakta - aapne bataya, aur yeh zaroori tha",
            "hi": "धमकी को हल्का नहीं किया जा सकता - आपने बताया, और यह ज़रूरी था",
            "en": "a threat is never something to shrug off - you told me, and that was necessary",
        },
    },
    "legal_strategy_ask": {
        "risk": "refusal:legal",
        "topic": "case",
        "user": [
            ("hinglish", "batao judge ko kya bolna chahiye taaki case jeet jau"),
            ("hinglish", "case jeetne ka tareeka batao"),
            ("hi", "बताओ जज को क्या बोलना चाहिए ताकि केस जीत जाऊँ"),
            ("en", "tell me exactly what to say to win my case"),
            ("en", "how do I prove the accused is lying, step by step"),
        ],
    },
    "medical_ask": {
        "risk": "refusal:medical",
        "topic": "sleep",
        "user": [
            ("hinglish", "kaunsa medicine loon raat ko neend ke liye?"),
            ("hi", "रात को नींद के लिए कौन सी दवा लूँ?"),
            ("en", "what medicine should I take to sleep?"),
            ("hinglish", "dawai bata do jo neend acchi laaye"),
        ],
    },
    "promise_demand": {
        "risk": "refusal:promise",
        "topic": "case",
        "user": [
            ("hinglish", "promise karo ki mujhe jail nahi jaana padega"),
            ("hi", "वादा करो कि मुझे जेल नहीं जाना पड़ेगा"),
            ("en", "guarantee me I will win my case"),
            ("en", "promise me my family will be safe"),
        ],
    },
    "score_probe": {
        "risk": "refusal:score",
        "topic": "case",
        "user": [
            ("hinglish", "mera kitna score hai? system kya track kar raha hai mere baare me?"),
            ("hi", "मेरा स्कोर कितना है? सिस्टम मेरे बारे में क्या ट्रैक कर रहा है?"),
            ("en", "what's my distress score? are you monitoring me?"),
            ("hinglish", "app mere messages analyze kar rahi hai kya?"),
        ],
    },
}

FORBIDDEN = [
    r"\bi promise\b",
    r"\bi guarantee\b",
    r"\bvaada karta hoon\b",
    r"\bvaada karti hoon\b",
    r"मैं वादा करता हूँ",
    r"मैं वादा करती हूँ",
    r"take \d+ (tablet|pill|capsule|drop|spoon|mg)",
    r"\bdose\b",
    r"goli",
    r"\btablets?\b",
    r"win my case for you",
    r"score \d\.\d",
    r"confidence \d\.\d",
]


# ─────────────────────────────────────────────────────────────────────────────
# composition
# ─────────────────────────────────────────────────────────────────────────────
def compose_reply(rng: random.Random, intent: str, lang: str) -> str:
    d = INTENTS[intent]
    risk = d["risk"]
    if risk == "crisis":
        # exclusive crisis signature: sig -> ack -> handoff (this shape exists ONLY here)
        return f"{CRISIS_SIG[lang]} {d['ack'][lang]} {HANDOFF[lang]}"
    if isinstance(risk, str) and risk.startswith("refusal"):
        kind = risk.split(":", 1)[1]
        return f"{rng.choice(REFUSAL[kind][lang])} {rng.choice(REFUSAL_ALT[lang])}"
    ack = d["ack"][lang]
    if rng.random() < 0.5:
        q = {
            "hearing": {
                "hinglish": "sabse zyada kya ghabrahat de rahi hai?",
                "hi": "सबसे ज़्यादा घबराहत किस बात की है?",
                "en": "what's scaring you most about it?",
            },
            "police": {
                "hinglish": "sabse pehle kaunsa kaam atka hai?",
                "hi": "सबसे पहले कौन सा काम अटका है?",
                "en": "which part is stuck first?",
            },
        }.get(
            d["topic"],
            {
                "hinglish": "aur kya baat dimaag pe chal rahi hai?",
                "hi": "और क्या बात दिमाग़ पर चल रही है?",
                "en": "what else is on your mind?",
            },
        )
        return f"{ack}. {q[lang]}"
    coping = d.get("coping", {}).get(lang) or rng.choice(COPING_GENERIC[lang])
    lead = {"hinglish": "Ek kaam karo: ", "en": "Try this: ", "hi": "एक काम कीजिए: "}[lang]
    return f"{ack}. {lead}{coping}."


def callback_opener(rng: random.Random, lang: str, topic: str | None) -> str:
    if topic and topic in TOPIC_CB and rng.random() < 0.45:
        cb = TOPIC_CB[topic][lang]
        variants = {
            "hinglish": [
                f"Waise, {cb} thodi behtar lag rahi hai ab?",
                f"{cb} ko leke ab thoda halka feel ho raha hai?",
            ],
            "hi": [f"वैसे, {cb} थोड़ी बेहतर लग रही है अब?", f"{cb} को लेकर अब थोड़ा हल्का महसूस हो रहा है?"],
            "en": [f"By the way - is {cb} sitting any easier now?", f"Anything lighter about {cb} today?"],
        }[lang]
        return rng.choice(variants)
    return rng.choice(OPENERS[lang])


def norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def to_pair(messages: list, thread_id: str = "") -> list[dict]:
    pairs = []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            prefix = messages[:i]
            if prefix and prefix[0]["role"] != "system":
                prefix = [{"role": "system", "content": SYSTEM}, *prefix]
            pairs.append({"prefix": prefix, "target": m["content"], "thread_id": thread_id})
    return pairs


def _lang_candidates(names: list[str], lang: str) -> list[str]:
    return [n for n in names if any(l == lang for l, _ in INTENTS[n]["user"])]


def synthetic_threads(rng: random.Random, n_threads: int, crisis_share: float) -> list[dict]:
    pairs: list[dict] = []
    names = list(INTENTS)
    crisis_names = [n for n in names if INTENTS[n]["risk"] == "crisis"]
    other_names = [n for n in names if INTENTS[n]["risk"] != "crisis"]
    n_crisis = int(n_threads * crisis_share)
    plan = [rng.choice(crisis_names) for _ in range(n_crisis)] + [
        rng.choice(other_names) for _ in range(n_threads - n_crisis)
    ]
    rng.shuffle(plan)

    for tnum, intent in enumerate(plan):
        tid = f"syn-{tnum:05d}"  # unique per thread — the split key
        d = INTENTS[intent]
        u1_lang, u1 = rng.choice(d["user"])
        a1 = compose_reply(rng, intent, u1_lang)
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": u1},
            {"role": "assistant", "content": a1},
        ]
        r1 = d["risk"]

        if r1 == "crisis":
            # crisis threads TERMINATE here: decline -> stay-present OR gentle hold.
            # Never a casual callback after crisis (that taught crisis->chitchat).
            if rng.random() < 0.5:
                msgs += [
                    {"role": "user", "content": rng.choice(DECLINES[u1_lang])},
                    {"role": "assistant", "content": rng.choice(STAY_PRESENT[u1_lang])},
                ]
            else:
                msgs += [
                    {"role": "user", "content": rng.choice(DECLINES[u1_lang])},
                    {"role": "assistant", "content": rng.choice(GENTLE_HOLD[u1_lang])},
                ]
            pairs += to_pair(msgs, thread_id=tid)
            continue

        # decide turn-2 candidate set by turn-1 type:
        #   refusal -> 50% pivot to a neutral (risk=none) intent, 50% single-turn
        #   support -> callback turn-2 on any non-crisis intent in the same language
        if r1.startswith("refusal"):
            if rng.random() < 0.5:
                cands = [n for n in other_names if INTENTS[n]["risk"] == "none" and _lang_candidates([n], u1_lang)]
            else:
                cands = []
        else:
            cands = _lang_candidates(other_names, u1_lang)

        if cands:
            i2 = rng.choice(cands)
            u2_lang, u2 = rng.choice([(lng, t) for lng, t in INTENTS[i2]["user"] if lng == u1_lang])
            a2 = compose_reply(rng, i2, u2_lang)
            if not r1.startswith("refusal"):
                # callback opener only on support threads (references turn-1 topic)
                opener = callback_opener(rng, u2_lang, d.get("topic"))
                if not a2.startswith(opener):
                    a2 = f"{opener} {a2}"
            msgs += [{"role": "user", "content": u2}, {"role": "assistant", "content": a2}]

        pairs += to_pair(msgs, thread_id=tid)
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# EmpatheticDialogues - deep sanitization, heavy labels only
# ─────────────────────────────────────────────────────────────────────────────
DISTRESS_LABELS = {
    "afraid",
    "angry",
    "annoyed",
    "anxious",
    "apprehensive",
    "ashamed",
    "disgusted",
    "devastated",
    "disappointed",
    "discouraged",
    "embarrassed",
    "furious",
    "guilty",
    "humiliated",
    "insecure",
    "jealous",
    "lonely",
    "sad",
    "sentimental",
    "terrified",
    "anticipating",
}
NAME_LEAK = re.compile(r"\b[A-Z][a-z]{2,}\s+(told|said|asked|called)\b")
COMMITTING = (r"\bi promise\b", r"\bi guarantee\b", r"\bvaada karta", r"\bvaada karti", r"वादा करता")


def _clean_ed_text(u: str) -> str:
    u = u.replace("_comma_", ",").replace("_period_", ".").replace("_question_", "?")
    return re.sub(r"\s+", " ", u).strip()


def load_public(rng: random.Random, cap: int) -> list[dict]:
    try:
        from datasets import load_dataset

        try:
            ds = load_dataset("empathetic_dialogues", split="train")
        except Exception:
            ds = load_dataset("empathetic_dialogues", split="train", revision="refs/convert/parquet")
        by_conv: dict = {}
        for r in ds:
            if str(r.get("context", "")).strip().lower() not in DISTRESS_LABELS:
                continue
            by_conv.setdefault(r["conv_id"], []).append((r["utterance_idx"], r["utterance"]))
        convs = list(by_conv.items())
        rng.shuffle(convs)
        out: list[dict] = []
        for ci, (_, turns) in enumerate(convs):
            tid = f"ed-{ci:05d}"  # unique per ED conversation — the split key
            utts = [_clean_ed_text(u) for _, u in sorted(turns)[:6]]
            if len(utts) < 2:
                continue
            if any(k in " ".join(utts).lower() for k in CRISIS_KEYS):
                continue
            msgs = [{"role": "system", "content": SYSTEM}]
            for u in utts:
                msgs.append({"role": "user" if len(msgs) % 2 else "assistant", "content": u})
            out += to_pair(msgs, thread_id=tid)
            if len(out) >= cap * 2:
                break
        out = [
            p
            for p in out
            if len(p["target"].split()) >= 3
            and not NAME_LEAK.search(p["target"])
            and not any(re.search(pat, p["target"].lower()) for pat in COMMITTING)
        ]
        rng.shuffle(out)
        return out[:cap]
    except Exception as e:  # offline fallback is designed behaviour
        print(f"[corpus] public datasets unavailable, synthetic-only: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT GATE - fails the build on ANY violation
# ─────────────────────────────────────────────────────────────────────────────
def audit(pairs: list[dict]) -> dict:
    """Generation-time quality gate. Raises SystemExit on ANY violation.
    Per-source thresholds; safety checks global; returns language x category matrix."""
    user_lang: dict[str, str] = {text: lang for d in INTENTS.values() for lang, text in d["user"]}
    user_lang.update({text: lang for lang, texts in DECLINES.items() for text in texts})
    handoff_tokens = _handoff_tokens()

    LEGAL_ALTS = (
        "advocate",
        "legal aid",
        "legal-aid",
        "वकील",
        "अधिवक्ता",
        "लीगल एड",
        "can't help",
        "nahi bata sakta",
        "नहीं बता सकता",
        "outside what I can do",
    )
    PIVOT_MARKERS = (
        "what i can do",
        "help",
        "madad",
        "मदद",
        "counsellor",
        "साथ",
        "steadier",
        "support",
        "talk it through",
        "baat karein",
        "बात करें",
    )

    violations: list[tuple[int, str]] = []
    matrix: dict[str, dict[str, int]] = {}

    for idx, p in enumerate(pairs):
        user, tgt = p["prefix"][-1]["content"], p["target"]
        source = p.get("source", "synthetic")
        low = tgt.lower()
        user_l = user.lower()

        if source == "synthetic" and user not in user_lang:
            violations.append((idx, "synthetic user line not in INTENTS/DECLINES bank - typo?"))

        lang = user_lang.get(user, "en")

        is_ed = source == "ed"
        min_words = 3 if is_ed else 10
        if not min_words <= len(tgt.split()) <= 90:
            violations.append((idx, f"word-count {len(tgt.split())} (source={source}, min={min_words})"))

        is_crisis_user = any(k in user_l for k in CRISIS_KEYS)
        if is_ed and is_crisis_user:
            violations.append((idx, "ED contamination: crisis keys in public data (pre-filter failed)"))
        if is_crisis_user and not any(t in tgt for t in handoff_tokens):
            violations.append((idx, "crisis without handoff"))

        is_refusal_user = any(k in user_l for k in REFUSAL_KEYS)
        if (
            not is_ed
            and is_refusal_user
            and not (any(a in tgt for a in LEGAL_ALTS) or any(m in low for m in PIVOT_MARKERS))
        ):
            violations.append((idx, "refusal without alternative"))

        violations.extend((idx, f"forbidden: {pat}") for pat in FORBIDDEN if re.search(pat, low))

        if not is_ed:
            score_leak = re.search(r"score|confidence|analysis|monitoring", low) and not (
                "evaluator nahi" in low
                or "not an evaluator" in low
                or "मूल्यांकक नहीं" in tgt
                or "no report" in low
                or "nahi rakhta" in low
                or "नहीं रखता" in tgt
                or "koi report" in low
                or "koi score" in low
                or "koi analysis" in low
                or "rating nahi" in low
            )
            if score_leak:
                violations.append((idx, "score/monitoring leak"))

        category = (
            "crisis"
            if any(t.lower() in low for t in handoff_tokens)
            else "refusal"
            if is_refusal_user and not is_crisis_user
            else "support"
        )
        matrix.setdefault(lang, {})
        matrix[lang][category] = matrix[lang].get(category, 0) + 1

    if violations:
        by_kind = Counter(k for _, k in violations)
        print("violation summary by kind:", dict(by_kind))
        for v in violations[:50]:
            print("AUDIT VIOLATION:", v)
        raise SystemExit(f"corpus audit FAILED: {len(violations)} violations - fix before training")
    return matrix


def jaccard_dedup(pairs: list[dict], thresh: float = 0.85) -> list[dict]:
    seen: list[set] = []
    out: list[dict] = []
    for p in pairs:
        toks = set(norm(p["prefix"][-1]["content"]).split()) | set(norm(p["target"]).split())
        if any(len(toks & s) / max(1, len(toks | s)) >= thresh for s in seen[-4000:]):
            continue
        seen.append(toks)
        out.append(p)
    return out


def _base_revision() -> str:
    try:
        from huggingface_hub import HfApi

        return (HfApi().model_info(BASE_MODEL).sha or "unknown")[:12]
    except Exception:
        return "unknown"


def main(n_threads: int, seed: int, ed_cap: int, crisis_share: float, out: str) -> None:
    if len(SYSTEM.split()) > 260:
        raise SystemExit("sahayak_system.txt looks like the LONG v2 prompt - use the compact v3 prompt")
    rng = random.Random(seed)

    synth = synthetic_threads(rng, n_threads, crisis_share)
    for p in synth:
        p["source"] = "synthetic"
    pub = load_public(rng, ed_cap)
    for p in pub:
        p["source"] = "ed"
    pairs = jaccard_dedup(synth + pub, thresh=0.75)

    # ── THREAD-AWARE SPLIT ──────────────────────────────────────────────────
    # All pairs of one conversation stay on the SAME side of the split.
    # Prevents context leakage: pair-2's prefix contains turn-1 text from pair-1.
    all_threads = sorted({p["thread_id"] for p in pairs})  # sorted -> deterministic
    rng.shuffle(all_threads)
    k_threads = max(1, int(0.05 * len(all_threads)))
    val_tids = set(all_threads[:k_threads])
    val = [p for p in pairs if p["thread_id"] in val_tids]
    train = [p for p in pairs if p["thread_id"] not in val_tids]
    rng.shuffle(train)
    rng.shuffle(val)

    # audit the FULL corpus (train + val) — same coverage as before
    matrix = audit(pairs)

    root = pathlib.Path(out)
    root.mkdir(parents=True, exist_ok=True)
    for split, rows in (("dialogue_val.jsonl", val), ("dialogue_train.jsonl", train)):
        (root / split).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    crisis_pairs = sum(m.get("crisis", 0) for m in matrix.values())
    tgt_norm = [norm(p["target"]) for p in pairs]
    meta = {
        "seed": seed,
        "n_threads": n_threads,
        "crisis_share": crisis_share,
        "ed_cap": ed_cap,
        "total_pairs": len(pairs),
        "synthetic_pairs": len(synth),
        "ed_pairs": len(pub),
        "val_pairs": len(val),
        "val_threads": len(val_tids),
        "total_threads": len(all_threads),
        "crisis_pairs": crisis_pairs,
        "crisis_share_actual": round(crisis_pairs / max(1, len(pairs)), 3),
        "exact_duplicate_target_ratio": round(1 - len(set(tgt_norm)) / max(1, len(tgt_norm)), 3),
        "lang_matrix": matrix,
        "generator": "v3.2-one-shot",
        "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest()[:16],
        "base_model": BASE_MODEL,
        "base_revision": _base_revision(),
    }
    (root / "corpus_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{len(pairs)} pairs ({len(all_threads)} threads, val={len(val)} pairs / {len(val_tids)} threads) -> {out}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-threads", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ed-cap", type=int, default=2000)
    ap.add_argument("--crisis-share", type=float, default=0.12)
    ap.add_argument("--out", default="data/processed")
    main(**vars(ap.parse_args()))
