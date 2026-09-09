"""Adapter A corpus v4.0 -- paragraph-length, context-aware retrain build.

Changes from v3.2, and why:
  - compose_reply() for risk="none" intents now produces a PARAGRAPH shape
    (validate -> concrete multi-step action -> check-in), not a 2-clause
    sentence. Diagnostic run on v3.2 showed max target length across the
    ENTIRE corpus was 55 words with 0 pairs over 60 -- this is the fix.
  - PRACTICAL_STEPS bank: real, concrete, situationally specific guidance
    per topic (hearing/compensation/police/family/sleep), used as the
    "action" clause instead of one-line platitudes. This is what makes
    "actual solutions" true for logistical problems Adapter A CAN safely
    speak to, without it inventing facts about a specific real case
    (that's Adapter B's job -- RAG-grounded, per the architecture doc).
  - synthetic_threads() now builds threads 2-8 turns deep (was capped at
    2 turns, always) and, where a CASE_FACTS entry exists for the topic,
    makes a later turn explicitly reference a fact volunteered earlier --
    not a generic topic-word substitution. Diagnostic run on v3.2 showed
    prefix depth was ALWAYS exactly 0 or 1 prior turns, never deeper.
  - ack banks expanded from 1 to 2-3 variants per intent per language.
  - ed_cap default cut 2000 -> 400, ED min_words raised 3 -> 15: at v3.2
    defaults, ED could be ~half the corpus by volume and was diluting the
    specialized register with short, generic public-data listener lines.
  - CRISIS_KEYS: removed "मन नहीं" and bare "khatam kar" -- both are
    common, mundane phrase fragments unrelated to ideation in most real
    usage, and were a false-positive risk if this pattern is ever reused
    outside this offline audit script.
  - audit() min-word floor is now risk-aware: crisis/refusal responses
    stay short by design (crisis-intervention UX correctly favors brevity
    and clarity in the moment -- this was NOT changed), support/practical
    responses now require real paragraph length.

Safety design carried over unchanged from v3.2 (still correct, not touched):
  - crisis intents get an exclusive reply signature (validate -> ack -> handoff)
    that appears NOWHERE else in the corpus, so crisis->handoff is unambiguous
  - explicit crisis share (--crisis-share), recorded in the manifest
  - heavy_but_not_crisis intent: near-miss pairs teaching distress != ideation
  - ED deeply sanitized (artifacts, names, micro-utterances, committing phrases,
    crisis-key contamination) and capped; heavy labels only
  - crisis threads never continue into casual callbacks
  - audit(): per-source thresholds, synthetic-scoped tuned checks, unknown-line
    hard error, violation summary by kind; FAILS the build on any violation
  - thread-aware train/val split (no cross-split leakage of same-conversation turns)
  - crisis-pair oversampling for class balance
"""

import argparse
import hashlib
import json
import pathlib
import random
import re
from collections import Counter

SYSTEM = pathlib.Path("prompts/sahayak_system.txt").read_text(encoding="utf-8").strip()
# Verify this matches the exact checkpoint you're fine-tuning (2.57B) before a real run --
# keeping the same identifier as v3.2 since it was presumably already correct for your setup.
BASE_MODEL = "sarvamai/sarvam-1"

# ─────────────────────────────────────────────────────────────────────────────
# handoff / stay-present / declines / refusals (per language) -- UNCHANGED from
# v3.2. Crisis and refusal responses are deliberately short; that's correct
# crisis-UX and correct "don't guess at legal/medical advice" design, not a bug.
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
        "Slow breathing for a minute can take the edge off - in for four counts, hold for four, out for six.",
        "Writing the worry down, even in three or four words, sometimes makes it smaller than it feels in your head.",
        "Reaching one trusted person today, even just to sit near them without explaining anything, might lighten it.",
    ],
    "hinglish": [
        "Ek minute ki dheemi saans le lo - chaar count mein andar, chaar rokay, chhe mein bahar. Thoda asar hota hai.",
        "Chinta ko teen-chaar shabdon mein bhi likh do - dimaag ke andar se bahar aake kabhi kabhi chhoti lagti hai.",
        "Aaj ek bharosemand insaan ke paas baith jao, kuch samjhana zaroori nahi - sirf saath hona bhi halka kar deta hai.",
    ],
    "hi": [
        "एक मिनट धीमी साँस ले लीजिए - चार गिनती अंदर, चार रोकिए, छह में बाहर। थोड़ा असर होता है।",
        "चिंता को तीन-चार शब्दों में भी लिख दीजिए - दिमाग़ से बाहर आकर कभी-कभी छोटी लगती है।",
        "आज किसी भरोसेमंद इंसान के पास बैठ जाइए, कुछ समझाना ज़रूरी नहीं - बस साथ होना भी हल्का कर देता है।",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# NEW in v4: concrete, situationally-specific, non-legal/non-medical practical
# guidance per topic. This is the primary fix for "generic" and "no real
# solutions" -- Adapter A can safely speak to logistics and process without
# inventing facts about anyone's specific real case (that stays Adapter B's
# job). Two variants per language per topic here; keep expanding this bank
# over time, it's the highest-leverage place to add more.
# ─────────────────────────────────────────────────────────────────────────────
PRACTICAL_STEPS = {
    "hearing": {
        "en": [
            "A few things that usually help before a hearing: write down the two or three points you most "
            "want the judge or your advocate to hear, so you're not relying on memory in the moment; ask "
            "your advocate ahead of time roughly how long the hearing will run and what the first hour "
            "looks like, since not knowing the shape of the day often feeds the dread more than the "
            "hearing itself does; and if it's allowed in your case, ask whether a support person can sit "
            "with you in the courtroom or just outside it.",
            "For the day itself: lay out what you're bringing the night before, so the morning has one "
            "fewer decision in it. If you don't already have your advocate's direct number rather than a "
            "general office line, get it now, before the day you actually need it. And arriving fifteen "
            "or twenty minutes early, just to sit in the space before it fills up, takes the edge off for "
            "a lot of people - it turns an unknown room into a slightly familiar one.",
        ],
        "hinglish": [
            "Hearing se pehle yeh cheezein aksar kaam aati hain: do-teen zaroori baatein likh lo jo judge ya "
            "advocate ko batani hain, taaki us waqt yaad pe depend na karna pade; advocate se pehle hi pooch "
            "lo hearing kitni der chalegi aur pehla ghanta kaisa hoga - din ka shape na pata hona hi zyada "
            "daraata hai, hearing khud utna nahi; aur agar aapke case mein allowed hai, to pooch lo ki koi "
            "support wala insaan courtroom mein ya bahar aapke saath baith sakta hai kya.",
            "Us din ke liye: jo le jaana hai woh raat ko hi rakh lo, taaki subah ek decision kam ho. Agar "
            "advocate ka seedha number nahi hai, sirf office ka general number hai, to abhi le lo - us din "
            "se pehle jab zaroorat pade. Aur pandrah-bees minute pehle pahunch jao, bas jagah mein baithne "
            "ke liye - anjaani jagah thodi si jaani-pehchaani lagne lagti hai.",
        ],
        "hi": [
            "सुनवाई से पहले ये चीज़ें अक्सर मदद करती हैं: दो-तीन ज़रूरी बातें लिख लीजिए जो जज या अधिवक्ता को बतानी हैं, ताकि उस "
            "वक़्त याद पर निर्भर न रहना पड़े; अधिवक्ता से पहले ही पूछ लीजिए सुनवाई कितनी देर चलेगी और पहला घंटा कैसा होगा - "
            "दिन का ढांचा पता न होना ही ज़्यादा डराता है, सुनवाई ख़ुद उतना नहीं; और अगर आपके मामले में अनुमति है, तो पूछ "
            "लीजिए कि कोई सहायक व्यक्ति अदालत में या बाहर आपके साथ बैठ सकता है क्या।",
            "उस दिन के लिए: जो ले जाना है वह रात को ही रख लीजिए, ताकि सुबह एक फ़ैसला कम हो। अगर अधिवक्ता का सीधा नंबर "
            "नहीं है, सिर्फ़ कार्यालय का सामान्य नंबर है, तो अभी ले लीजिए - उस दिन से पहले जब ज़रूरत पड़े। और पंद्रह-बीस "
            "मिनट पहले पहुँच जाइए, बस जगह में बैठने के लिए - अनजानी जगह थोड़ी सी जानी-पहचानी लगने लगती है।",
        ],
    },
    "compensation": {
        "en": [
            "For a stuck compensation claim, a written status query with legal aid or the district authority "
            "usually moves things faster than a phone call, because it creates a paper trail someone has to "
            "respond to. Ask specifically which stage the file is at right now, and what document, if any, "
            "is missing - a stuck file is very often waiting on one specific paper, not the whole case.",
            "It also helps to keep your own simple log: the date of each visit or call, who you spoke to, "
            "and what they said would happen next. If the delay goes on, that log is exactly what your "
            "advocate or legal aid needs to escalate it formally - vague frustration is hard to act on, a "
            "dated record isn't.",
        ],
        "hinglish": [
            "Compensation atki ho to legal aid ya district authority ko ek likhi hui status-query bhejna "
            "aksar phone call se zyada kaam karta hai, kyunki paper trail ban jaata hai jiska jawab dena "
            "padta hai. Yeh zaroor poocho ki file abhi kis stage pe hai, aur koi document to nahi missing - "
            "aksar poora case nahi, sirf ek kaagaz atka hota hai.",
            "Apna ek simple log bhi rakho: har visit ya call ki date, kisse baat hui, aur unhone kya bola "
            "aage hoga. Agar delay lambi chale, to yeh log hi advocate ya legal aid ko formally escalate "
            "karne mein kaam aata hai - bina record ke frustration se kuch nahi hota, dated record se hota hai.",
        ],
        "hi": [
            "मुआवज़ा अटका हो तो लीगल एड या ज़िला प्राधिकरण को एक लिखित स्थिति-पूछताछ भेजना अक्सर फ़ोन कॉल से ज़्यादा काम करता "
            "है, क्योंकि काग़ज़ी रिकॉर्ड बन जाता है जिसका जवाब देना पड़ता है। यह ज़रूर पूछिए कि फ़ाइल अभी किस चरण में है, और "
            "कोई दस्तावेज़ तो नहीं छूटा - अक्सर पूरा मामला नहीं, सिर्फ़ एक काग़ज़ अटका होता है।",
            "अपना एक सरल रिकॉर्ड भी रखिए: हर मुलाक़ात या कॉल की तारीख़, किससे बात हुई, और उन्होंने क्या कहा आगे होगा। अगर "
            "देरी लंबी चले, तो यह रिकॉर्ड ही अधिवक्ता या लीगल एड को औपचारिक रूप से आगे बढ़ाने में काम आता है - बिना "
            "रिकॉर्ड के निराशा से कुछ नहीं होता, तारीख़ों वाले रिकॉर्ड से होता है।",
        ],
    },
    "police": {
        "en": [
            "When a police station keeps not responding, logging every single visit - date, who you met, "
            "what was said - turns something that feels like it's going nowhere into a paper trail your "
            "legal aid counsellor or advocate can actually use to escalate to a superior officer or the "
            "district authority. Verbal complaints are easy to lose; a dated written record is not.",
            "It can also help to go with someone else present, even just once, and to ask directly for the "
            "officer's name and a receipt or acknowledgment number for whatever you submitted - having a "
            "specific name and reference number makes it much harder for a follow-up to be brushed off.",
        ],
        "hinglish": [
            "Jab police station baar-baar respond nahi karta, to har visit ka log rakhna - date, kisse mile, "
            "kya bola - us feeling ko ki 'kahi nahi ja raha' ek paper trail mein badal deta hai jo legal aid "
            "counsellor ya advocate senior officer ya district authority tak escalate karne mein use kar "
            "sakte hain. Zubaani shikayat kho jaati hai, dated likhi hui nahi khoti.",
            "Kisi aur ko saath le jaana bhi kaam aata hai, ek baar hi sahi, aur seedha officer ka naam aur "
            "jo bhi jama kiya uski receipt ya acknowledgment number maangna - specific naam aur number hone "
            "se follow-up ko taalna mushkil ho jaata hai.",
        ],
        "hi": [
            "जब पुलिस थाना बार-बार जवाब नहीं देता, तो हर मुलाक़ात का रिकॉर्ड रखना - तारीख़, किससे मिले, क्या कहा - उस "
            "एहसास को कि 'कहीं नहीं जा रहा' एक काग़ज़ी रिकॉर्ड में बदल देता है जिसे लीगल एड सलाहकार या अधिवक्ता वरिष्ठ "
            "अधिकारी या ज़िला प्राधिकरण तक ले जाने में इस्तेमाल कर सकते हैं। मौखिक शिकायत खो जाती है, तारीख़ वाली लिखी हुई नहीं।",
            "किसी और को साथ ले जाना भी काम आता है, एक बार ही सही, और सीधे अधिकारी का नाम और जो भी जमा किया उसकी रसीद "
            "या पावती संख्या माँगना - विशिष्ट नाम और संख्या होने से आगे की कार्रवाई को टालना मुश्किल हो जाता है।",
        ],
    },
    "family": {
        "en": [
            "When family pressure to drop the case comes from exhaustion rather than disagreement, it can "
            "help to name that difference out loud to them directly: that you hear how tired they are, and "
            "that the decision about the case doesn't have to be theirs to carry the same way it's yours. "
            "You're allowed to want their comfort without needing their agreement on this specific choice.",
            "If it's useful, a legal aid counsellor or your advocate can sometimes speak with the family "
            "directly - hearing the process explained by someone outside the family, in plain terms, "
            "occasionally eases pressure that no amount of you personally explaining it seems to reach.",
        ],
        "hinglish": [
            "Jab ghar ki pressure case chhodne ke liye thakaan se aati hai, disagreement se nahi, to unhe "
            "seedha yeh farak bata dena kaam karta hai: ki aapko unki thakaan sunayi de rahi hai, aur case "
            "ka faisla unhe waise nahi uthana jaise aapko uthana hai. Unka pyaar chahiye, unki sehmati is "
            "ek faisle pe zaroori nahi.",
            "Agar madadgar lage, to legal aid counsellor ya advocate kabhi kabhi seedha family se baat kar "
            "sakte hain - koi bahar ka insaan process simple shabdon mein samjhaye to kabhi wo pressure "
            "halka ho jaata hai jo aapke khud samjhane se nahi hota.",
        ],
        "hi": [
            "जब घर की दबाव मामला छोड़ने के लिए थकान से आती है, असहमति से नहीं, तो उन्हें सीधे यह अंतर बता देना काम करता है: "
            "कि आपको उनकी थकान सुनाई दे रही है, और मामले का फ़ैसला उन्हें वैसे नहीं उठाना जैसे आपको उठाना है। उनका प्यार "
            "चाहिए, उनकी सहमति इस एक फ़ैसले पर ज़रूरी नहीं।",
            "अगर सहायक लगे, तो लीगल एड सलाहकार या अधिवक्ता कभी-कभी सीधे परिवार से बात कर सकते हैं - कोई बाहर का इंसान "
            "प्रक्रिया सरल शब्दों में समझाए तो कभी वह दबाव हल्का हो जाता है जो आपके ख़ुद समझाने से नहीं होता।",
        ],
    },
    "sleep": {
        "en": [
            "Five slow breaths and one worry written down in just a few words, right before bed, is a small "
            "combination that actually works for a lot of people - not because it solves anything, but "
            "because it gives the racing mind one specific place to put a thought down instead of circling it.",
            "If the same thoughts return every night at the same point, it can help to set a fixed ten-minute "
            "'worry window' earlier in the evening - deliberately think it through then, on paper, so there's "
            "less unfinished business left over for the mind to pick up once the lights are off.",
        ],
        "hinglish": [
            "Sone se pehle paanch dheemi saans aur ek chinta ko sirf kuch shabdon mein likh dena - yeh chhoti "
            "si jodi bahut logon ke liye kaam karti hai. Kuch solve nahi hota isse, par bhaagte hue dimaag ko "
            "ek jagah mil jaati hai jahan woh soch rakh sake, ghoomti rehne ke bajaay.",
            "Agar roz raat wahi khayal usi jagah aate hain, to shaam mein hi ek fix das-minute ka 'chinta ka "
            "waqt' rakh lo - jaan bujh kar tab uske baare mein sochlo, kaagaz pe, taaki lights off hone ke "
            "baad dimaag ke paas utna adhoora kaam na bache.",
        ],
        "hi": [
            "सोने से पहले पाँच धीमी साँसें और एक चिंता को सिर्फ़ कुछ शब्दों में लिख देना - यह छोटी सी जोड़ी बहुत लोगों के लिए "
            "काम करती है। कुछ हल नहीं होता इससे, पर भागते हुए दिमाग़ को एक जगह मिल जाती है जहाँ वह सोच रख सके, घूमते "
            "रहने के बजाय।",
            "अगर रोज़ रात वही ख़याल उसी जगह आते हैं, तो शाम में ही एक तय दस-मिनट का 'चिंता का समय' रख लीजिए - जान-बूझकर "
            "तब उसके बारे में सोचिए, काग़ज़ पर, ताकि लाइट बंद होने के बाद दिमाग़ के पास उतना अधूरा काम न बचे।",
        ],
    },
}

# General fallback for topic="case" and similar (self-blame, isolation, low
# energy, good day) -- still concrete, but appropriately broader since these
# aren't tied to one specific logistical process.
CASE_GENERAL_STEPS = {
    "en": [
        "One thing that helps here: pick a single small, doable thing for today only - not the whole case, "
        "not the whole week, just today's slice - and let everything else wait until tomorrow decides to "
        "show up. A big picture doesn't need fixing daily; today's piece does.",
        "If you can, name one person - not everyone, just one - who you'd let see you on a bad day without "
        "having to explain it first. Reaching out to exactly that one person, even with nothing more than "
        "'today is hard,' tends to matter more than it seems like it will in the moment.",
    ],
    "hinglish": [
        "Yahan ek cheez kaam aati hai: aaj ke liye sirf ek chhota, ho sakne wala kaam choose karo - poora "
        "case nahi, poora hafta nahi, bas aaj ka hissa - aur baaki sab kal pe chhod do. Bada picture roz "
        "theek karne ki zaroorat nahi, aaj ka tukda hai.",
        "Agar ho sake, to ek insaan ka naam socho - sabko nahi, sirf ek - jise aap bina samjhaye bura din "
        "dikha sako. Sirf usi ek insaan tak pahunchna, chaahe sirf 'aaj mushkil hai' bolke, jitna lagta hai "
        "usse zyada matter karta hai.",
    ],
    "hi": [
        "यहाँ एक चीज़ काम आती है: आज के लिए सिर्फ़ एक छोटा, हो सकने वाला काम चुनिए - पूरा मामला नहीं, पूरा हफ़्ता नहीं, बस आज "
        "का हिस्सा - और बाकी सब कल पर छोड़ दीजिए। बड़ा चित्र रोज़ ठीक करने की ज़रूरत नहीं, आज का टुकड़ा है।",
        "अगर हो सके, तो एक इंसान का नाम सोचिए - सबको नहीं, सिर्फ़ एक - जिसे आप बिना समझाए बुरा दिन दिखा सकें। सिर्फ़ उसी "
        "एक इंसान तक पहुँचना, चाहे सिर्फ़ 'आज मुश्किल है' बोलकर, जितना लगता है उससे ज़्यादा मायने रखता है।",
    ],
}

CHECKINS = {
    "en": [
        "How does that land for you right now?",
        "Does any part of that feel doable today?",
        "What's your sense of it?",
    ],
    "hinglish": [
        "Abhi yeh sunke kaisa lag raha hai?",
        "Isme se koi hissa aaj ho sakne wala lagta hai?",
        "Aapko kaisa lag raha hai isko lekar?",
    ],
    "hi": [
        "अभी यह सुनकर कैसा लग रहा है?",
        "इसमें से कोई हिस्सा आज हो सकने वाला लगता है?",
        "आपको कैसा लग रहा है इसे लेकर?",
    ],
}

TOPIC_CB = {
    "hearing": {"hinglish": "hearing wali baat", "hi": "सुनवाई वाली बात", "en": "the hearing"},
    "compensation": {"hinglish": "compensation wali baat", "hi": "मुआवज़े वाली बात", "en": "the compensation"},
    "police": {"hinglish": "police wali baat", "hi": "पुलिस वाली बात", "en": "the police response"},
    "family": {"hinglish": "ghar walo wali baat", "hi": "घर वालों वाली बात", "en": "the family situation"},
    "sleep": {"hinglish": "neend wali baat", "hi": "नींद वाली बात", "en": "the sleep"},
    "case": {"hinglish": "case wali baat", "hi": "केस वाली बात", "en": "the case"},
}

# NEW in v4: concrete facts a thread can establish early and genuinely refer
# back to later (used by build_deep_thread). Add more entries over time --
# this is what teaches real chat-history use, not the topic-word callback above.
CASE_FACTS = {
    "hearing": [
        {"en": "the hearing got pushed to next Tuesday", "hinglish": "hearing agle Tuesday tak taal di gayi",
         "hi": "सुनवाई अगले मंगलवार तक टल गई"},
        {"en": "the judge changed for this hearing", "hinglish": "is baar judge badal gaye hain",
         "hi": "इस बार जज बदल गए हैं"},
    ],
    "compensation": [
        {"en": "the compensation file was sent back for a missing document",
         "hinglish": "compensation ki file ek document ki kami se wapas aa gayi",
         "hi": "मुआवज़े की फ़ाइल एक दस्तावेज़ की कमी से वापस आ गई"},
    ],
    "police": [
        {"en": "the station finally gave a written acknowledgment", "hinglish": "station ne aakhir likhi acknowledgment de di",
         "hi": "थाने ने आख़िर लिखित पावती दे दी"},
    ],
}

# CRISIS_KEYS: v3.2 had "मन नहीं" and bare "khatam kar" -- both are common,
# mundane substrings ("aaj mann nahi hai office jaane ka" is totally ordinary
# Hinglish, not ideation). Removed those two; kept only phrases specific
# enough to be low false-positive.
CRISIS_KEYS = (
    "jeene ka mann nahi",
    "sab khatam kar",
    "zindagi khatam",
    "ending everything",
    "don't want to be here",
    "giving up",
    "dhamki",
    "threatened",
    "धमकी",
    "ख़त्म कर दूँ",
)
REFUSAL_KEYS = ("jeetne ka", "tareeka", "prove", "judge ko kya", "win my case", "बताओ जज", "साबित")

# ─────────────────────────────────────────────────────────────────────────────
# INTENT BANK - risk: none | crisis | refusal:<kind>
# "ack" fields expanded from 1 to 2 variants per language for support intents
# (was the single biggest source of the 84% dedup-loss / canned-phrase problem
# alongside short response length). Keep expanding these further over time.
# "coping" fields kept as a fallback for topics without a PRACTICAL_STEPS entry.
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
            "hinglish": [
                "hearing se pehle ka darr sabse ajeeb hota hai - poora dimaag usi ek din pe atak jaata hai",
                "jitna zaroori din hai, dimaag utna hi zyada uske baare mein sochta hai - yeh bilkul natural hai",
            ],
            "hi": [
                "सुनवाई से पहले का डर सबसे अजीब होता है - पूरा दिमाग़ उसी एक दिन पर अटक जाता है",
                "जितना ज़रूरी दिन है, दिमाग़ उतना ही ज़्यादा उसके बारे में सोचता है - यह बिल्कुल स्वाभाविक है",
            ],
            "en": [
                "the dread before a hearing is its own thing - the whole mind sticks on that one day",
                "the more the day matters, the more the mind circles it - that's a completely normal pattern",
            ],
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
            "hinglish": [
                "anjana sabse bada darr hota hai - na jaane wali baat hi zyada daraati hai",
                "jab pata hi na ho kya hoga, tab dimaag sabse bura sochta hai - yeh sabke saath hota hai",
            ],
            "hi": [
                "अनजाना सबसे बड़ा डर होता है - जो पता नहीं है वही ज़्यादा डराता है",
                "जब पता ही न हो क्या होगा, तब दिमाग़ सबसे बुरा सोचता है - यह सबके साथ होता है",
            ],
            "en": [
                "not knowing is the scariest part - the unknown outweighs everything",
                "when the shape of the day is unclear, the mind fills the gap with the worst case - that's common",
            ],
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
            "hinglish": [
                "intezar sabse thakata hai - kaam hua ya nahi, pata bhi nahi chalta",
                "mahino ka silence sabse bura hota hai - shikayat se bhi zyada thaka deta hai",
            ],
            "hi": [
                "इंतज़ार सबसे थकाता है - काम हुआ या नहीं, पता भी नहीं चलता",
                "महीनों की चुप्पी सबसे बुरी होती है - शिकायत से भी ज़्यादा थका देती है",
            ],
            "en": [
                "waiting drains you - you don't even know if anything moved",
                "months of silence is its own kind of exhausting - worse than a clear no, sometimes",
            ],
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
            "hinglish": [
                "baar-baar jaana aur sunna na milna - yeh sabse beizzati wala hissa hai",
                "jab awaaz sunayi hi na de, tab lagta hai koshish bekaar hai - par woh sahi nahi hai",
            ],
            "hi": [
                "बार-बार जाना और कुछ न सुन पाना - यह सबसे ठेस पहुँचाने वाला हिस्सा है",
                "जब आवाज़ सुनाई ही न दे, तब लगता है कोशिश बेकार है - पर वह सही नहीं है",
            ],
            "en": [
                "going again and again and getting nothing back - that part cuts deep",
                "when you're not being heard, it starts to feel like the effort is pointless - it isn't",
            ],
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
            "hinglish": [
                "jab apne hi thak jaayein to rasta aur bhi akela lagta hai - unka thakna bhi pyaar hai, unki baat bhi bojh hai",
                "ghar wale bhi is ladai mein hain, apne tareeke se thak rahe hain - yeh dono cheezein ek saath sach ho sakti hain",
            ],
            "hi": [
                "जब अपने ही थक जाएँ तो रास्ता और अकेला लगता है - उनकी थकान भी प्यार है, उनकी बात भी बोझ है",
                "घर वाले भी इस लड़ाई में हैं, अपने तरीक़े से थक रहे हैं - यह दोनों बातें एक साथ सच हो सकती हैं",
            ],
            "en": [
                "when your own people tire, the road feels lonelier - their exhaustion is love, and their words are weight",
                "your family is in this fight too, tiring in their own way - both things can be true at once",
            ],
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
            "hinglish": [
                "raat ka dimaag sabse bura dikhata hai - din ki roshni me wahi cheezein chhoti lagti hain",
                "andhere mein dimaag ko rokne wala kuch nahi hota - din mein kaam hi ek roktha hai",
            ],
            "hi": [
                "रात का दिमाग़ सबसे बुरा दिखाता है - दिन की रोशनी में वही बातें छोटी लगती हैं",
                "अंधेरे में दिमाग़ को रोकने वाला कुछ नहीं होता - दिन में काम ही एक रोकता है",
            ],
            "en": [
                "the night-mind shows everything at its worst - daylight makes the same things smaller",
                "in the dark there's nothing to interrupt the mind - during the day, tasks do that job for you",
            ],
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
            "hinglish": [
                "jo lad raha hai woh hi khud pe sawal uthata hai - bewajah ladne wale ko yeh sawal hota hi nahi",
                "yeh sawal aana khud isi baat ka proof hai ki aap poori koshish kar rahe hain",
            ],
            "hi": [
                "जो लड़ रहा है वही ख़ुद पर सवाल उठाता है - बिना वजह लड़ने वाले को यह सवाल होता ही नहीं",
                "यह सवाल आना ख़ुद इसी बात का सबूत है कि आप पूरी कोशिश कर रहे हैं",
            ],
            "en": [
                "only someone actually fighting asks if they're fighting right - the passive never ask this",
                "asking this question at all is itself evidence you're trying as hard as you can",
            ],
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
            "hinglish": [
                "yeh sunke accha laga - better din bhi jitne ki cheez hai, note kar lo isko",
                "aisa din bhi utna hi real hai jitna mushkil din - dono girna nahi chahiye",
            ],
            "hi": [
                "यह सुनकर अच्छा लगा - बेहतर दिन भी जीत की चीज़ है, इसे नोट कर लीजिए",
                "ऐसा दिन भी उतना ही असली है जितना मुश्किल दिन - दोनों को गिरना नहीं चाहिए",
            ],
            "en": [
                "glad to hear it - better days count as wins too; keep that one",
                "a day like this is just as real as a hard one - both deserve to be counted",
            ],
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
            "hinglish": [
                "thaakna itne case ke saath bilkul natural hai - aap har din zor laga rahi hain",
                "har din itna sambhalna thakayega hi - iska matlab kuch galat nahi ho raha",
            ],
            "hi": [
                "थकना इतने केस के साथ बिल्कुल स्वाभाविक है - आप हर दिन ज़ोर लगा रही हैं",
                "हर दिन इतना संभालना थकाएगा ही - इसका मतलब कुछ ग़लत नहीं हो रहा",
            ],
            "en": [
                "tired is natural with all this - you're pushing every single day",
                "carrying this much daily will exhaust anyone - that doesn't mean anything's going wrong",
            ],
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
            "hinglish": [
                "akela feel hona is raste ka sabse bada jhooth hai - aap akela nahi hain, main yahin hoon",
                "yeh raasta akela feel karaata hai chaahe koi saath ho ya na ho - yeh raaste ki galti hai, aapki nahi",
            ],
            "hi": [
                "अकेला महसूस होना इस रास्ते का सबसे बड़ा झूठ है - आप अकेले नहीं हैं, मैं यहीं हूँ",
                "यह रास्ता अकेला महसूस कराता है चाहे कोई साथ हो या न हो - यह रास्ते की ग़लती है, आपकी नहीं",
            ],
            "en": [
                "feeling alone is this road's biggest lie - you're not, and I'm here",
                "this particular road makes people feel alone whether or not someone's actually there - that's the road's fault, not yours",
            ],
        },
        "coping": {
            "hinglish": "ek insaan choose jiske saath chup rehna bhi theek lage - bina samjhaye saath hona bhi saath hota hai",
            "hi": "एक इंसान चुनिए जिसके साथ चुप रहना भी ठीक लगे - बिना समझाए साथ होना भी साथ होता है",
            "en": "pick one person where even silence feels okay - company without explanations still counts",
        },
    },
    "who_are_you": {
        "risk": "none",
        "topic": None,
        "user": [
            ("hinglish", "tu kaun hai? kisi ka agent hai kya?"),
            ("hi", "तुम कौन हो? किसी के एजेंट हो क्या?"),
            ("en", "who are you exactly? who sent you?"),
        ],
        "ack": {
            "hinglish": ["main Sahayak hoon - aapke support team ka hissa, aapke case ke liye"],
            "hi": ["मैं सहायक हूँ - आपकी सहायता टीम का हिस्सा, आपके केस के लिए"],
            "en": ["I'm Sahayak - part of your support team, here for your case"],
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
            "hinglish": [
                "itna bojh uthana har kisi ke bas ki baat nahi - jo feel kar rahe ho woh utna hi valid hai",
                "yeh bhaari lagna kamzori nahi hai - itna sab jhelna kisi ke liye bhi bhaari hoga",
            ],
            "hi": [
                "इतना बोझ उठाना हर किसी के बस की बात नहीं - जो महसूस कर रहे हैं वह उतना ही वैध है",
                "यह भारी लगना कमज़ोरी नहीं है - इतना सब झेलना किसी के लिए भी भारी होगा",
            ],
            "en": [
                "carrying this much is beyond most people - what you feel is completely valid",
                "this feeling heavy isn't weakness - anyone carrying this much would feel it too",
            ],
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
            "hinglish": ["yeh baat kehne mein himmat lagi hogi - aapne bilkul sahi kiya batana",
                         "yeh sunna bhaari hai, aur yahan bolna sabse sahi jagah hai",
                         "jo feel kar rahe ho, woh batana bahut zaroori tha"],
            "hi": ["यह बात कहने में हिम्मत लगी होगी - आपने बिल्कुल सही किया बताना",
                   "यह सुनना भारी है, और यहाँ बोलना सबसे सही जगह है",
                   "जो महसूस कर रहे हैं, वह बताना बहुत ज़रूरी था"],
            "en": ["it took real courage to say this - you did exactly the right thing telling me",
                   "hearing that is heavy, and saying it here was the right move",
                   "whatever you're carrying, telling me was the important part"]
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
            "hinglish": [
                "har ladne wale ki aisi raat aati hai - yeh hissa aapki kamzori nahi, bojh ka signal hai",
                "jab itna sab sahi jhol rakha ho to haar ka khayal aana natural hai",
                "yeh thakaan bojh ka signal hai, aapki personality ka nahi",
            ],
            "hi": [
                "हर लड़ने वाले की ऐसी रात आती है - यह हिस्सा आपकी कमज़ोरी नहीं, बोझ का संकेत है",
                "जब इतना सब झेल रखा हो तो हार का ख़याल आना स्वाभाविक है",
                "यह थकान बोझ का संकेत है, आपकी कमज़ोरी का नहीं",
            ],
            "en": [
                "every fighter meets nights like this - it's not weakness, it's the weight talking",
                "when you've carried this much, wanting it to end is natural - the weight, not you",
                "that exhaustion is a signal of the load, not of who you are",
            ],
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
            "hinglish": [
                "yeh dhamki ko halka nahi kiya ja sakta - aapne bataya, aur yeh zaroori tha",
                "dhamki lena koi mazaak nahi hai - batana hi sabse pehla kadam tha",
                "yeh serious hai, aur aap sahi jagah laaye ho is baat ko",
            ],
            "hi": [
                "धमकी को हल्का नहीं किया जा सकता - आपने बताया, और यह ज़रूरी था",
                "धमकी कोई मज़ाक नहीं है - बताना ही पहला कदम था",
                "यह गंभीर है, और आप इसे सही जगह लाए हो",
            ],
            "en": [
                "a threat is never something to shrug off - you told me, and that was necessary",
                "receiving threats is serious - telling someone was the first step",
                "this is serious, and you brought it to the right place",
            ],
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
# composition — NEW in v4: paragraph shape for risk="none" intents.
# Crisis and refusal replies are UNCHANGED from v3.2 (deliberately short).
# ─────────────────────────────────────────────────────────────────────────────
def compose_reply(rng: random.Random, intent: str, lang: str) -> str:
    d = INTENTS[intent]
    risk = d["risk"]
    if risk == "crisis":
        ack = rng.choice(d["ack"][lang])
        return f"{CRISIS_SIG[lang]} {ack} {HANDOFF[lang]}"
    if isinstance(risk, str) and risk.startswith("refusal"):
        kind = risk.split(":", 1)[1]
        return f"{rng.choice(REFUSAL[kind][lang])} {rng.choice(REFUSAL_ALT[lang])}"

    # risk == "none" -> paragraph shape: validate -> concrete action -> check-in
    ack = rng.choice(d["ack"][lang]) if isinstance(d["ack"][lang], list) else d["ack"][lang]
    topic = d.get("topic")

    if topic and topic in PRACTICAL_STEPS and lang in PRACTICAL_STEPS[topic]:
        action = rng.choice(PRACTICAL_STEPS[topic][lang])
    elif topic == "case":
        action = rng.choice(CASE_GENERAL_STEPS[lang])
    elif topic is None:
        # e.g. who_are_you -- don't force an unrelated "practical step" onto an identity question
        action = d.get("coping", {}).get(lang, "")
    else:
        action = d.get("coping", {}).get(lang) or rng.choice(COPING_GENERIC[lang])

    if topic is None:
        # short, coherent identity-style answer, not padded with an irrelevant action clause
        return f"{ack}. {action}".strip()

    checkin = rng.choice(CHECKINS[lang])
    shape = rng.random()
    if shape < 0.5:
        return f"{ack}. {action} {checkin}"
    return f"{rng.choice(OPENERS[lang])} {ack}. {action} {checkin}"


def callback_opener(rng: random.Random, lang: str, topic) -> str:
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


def to_pair(messages: list, thread_id: str = "", risk_tag="support") -> list[dict]:
    """risk_tag records the GENERATION-TIME category (crisis/refusal/support/brief),
    not something re-derived from text later -- fixes a real bug where support
    paragraphs mentioning "advocate" as practical guidance were misclassified as
    refusals by a text-pattern check and forced into the short-response band.
    Pass a single str to tag every assistant turn the same way, or a list with
    one entry per assistant turn (in order) when a thread mixes categories."""
    pairs = []
    assistant_idx = 0
    tags = risk_tag if isinstance(risk_tag, list) else None
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            prefix = messages[:i]
            if prefix and prefix[0]["role"] != "system":
                prefix = [{"role": "system", "content": SYSTEM}, *prefix]
            tag = tags[assistant_idx] if tags is not None else risk_tag
            pairs.append({"prefix": prefix, "target": m["content"], "thread_id": thread_id, "risk_tag": tag})
            assistant_idx += 1
    return pairs


def _lang_candidates(names: list[str], lang: str) -> list[str]:
    return [n for n in names if any(l == lang for l, _ in INTENTS[n]["user"])]


def build_deep_thread(rng: random.Random, intent: str, lang: str, other_names: list[str]) -> list[dict]:
    """NEW in v4. Builds a 2-8 turn thread (was capped at exactly 1-2 turns in
    v3.2, always). Where a CASE_FACTS entry exists for the topic, a later turn
    explicitly references the fact by content, not by a generic topic word --
    this is what teaches genuine chat-history use rather than topic-matching."""
    d = INTENTS[intent]
    topic = d.get("topic")
    n_turns = rng.randint(2, 8)
    msgs = [{"role": "system", "content": SYSTEM}]

    fact = None
    if topic in CASE_FACTS:
        fact = rng.choice(CASE_FACTS[topic])[lang]

    tags: list[str] = []
    cur_intent, cur_topic = intent, topic
    for turn in range(n_turns):
        if turn == 0:
            u = rng.choice([t for l, t in INTENTS[cur_intent]["user"] if l == lang])
        elif turn == 1 and fact:
            u = {"en": f"also, {fact}", "hinglish": f"waise, {fact}", "hi": f"वैसे, {fact}"}[lang]
        else:
            cands = _lang_candidates(other_names, lang)
            if not cands:
                break
            cur_intent = rng.choice(cands)
            cur_topic = INTENTS[cur_intent].get("topic")
            u = rng.choice([t for l, t in INTENTS[cur_intent]["user"] if l == lang])
        msgs.append({"role": "user", "content": u})

        if turn >= 2 and fact and topic == cur_topic and rng.random() < 0.6:
            a = {
                "en": f"Given {fact} - has that changed how it feels to wait, or added to it? "
                      f"{rng.choice(CHECKINS['en'])}",
                "hinglish": f"{fact} - iske baad wait karna badla ya aur bhaari ho gaya? "
                            f"{rng.choice(CHECKINS['hinglish'])}",
                "hi": f"{fact} - इसके बाद इंतज़ार करना बदला या और भारी हो गया? {rng.choice(CHECKINS['hi'])}",
            }[lang]
            # this is a short, referencing check-in turn by design (not a fresh
            # full answer) -- tag "brief" rather than forcing it to hit the
            # 45-word support floor with padding
            tags.append("brief")
        else:
            opener = callback_opener(rng, lang, cur_topic) if turn > 0 else None
            a = compose_reply(rng, cur_intent, lang)
            if opener and not a.startswith(opener) and INTENTS[cur_intent]["risk"] == "none":
                a = f"{opener} {a}"
            # "brief" for identity-style answers (topic=None, e.g. who_are_you) which are
            # deliberately short by design, "support" for everything else in this thread
            tags.append("brief" if cur_topic is None else "support")
        msgs.append({"role": "assistant", "content": a})

    return msgs, tags


def synthetic_threads(rng: random.Random, n_threads: int, crisis_share: float) -> list[dict]:
    pairs: list[dict] = []
    names = list(INTENTS)
    crisis_names = [n for n in names if INTENTS[n]["risk"] == "crisis"]
    refusal_names = [n for n in names if isinstance(INTENTS[n]["risk"], str) and INTENTS[n]["risk"].startswith("refusal")]
    support_names = [n for n in names if INTENTS[n]["risk"] == "none"]
    n_crisis = int(n_threads * crisis_share)
    plan = [rng.choice(crisis_names) for _ in range(n_crisis)] + [
        rng.choice(support_names + refusal_names) for _ in range(n_threads - n_crisis)
    ]
    rng.shuffle(plan)

    for tnum, intent in enumerate(plan):
        tid = f"syn-{tnum:05d}"
        d = INTENTS[intent]
        u1_lang, u1 = rng.choice(d["user"])
        r1 = d["risk"]

        if r1 == "crisis":
            a1 = compose_reply(rng, intent, u1_lang)
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": u1},
                {"role": "assistant", "content": a1},
            ]
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
            t2_pairs = to_pair(msgs, thread_id=tid, risk_tag="crisis")
            if t2_pairs:
                pairs.append(t2_pairs[0])
                if rng.random() < 0.06:
                    t2_pairs[-1]["keep"] = True
                    pairs.append(t2_pairs[-1])
            continue

        if r1.startswith("refusal"):
            a1 = compose_reply(rng, intent, u1_lang)
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": u1},
                {"role": "assistant", "content": a1},
            ]
            tags = ["refusal"]
            if rng.random() < 0.5:
                cands = [n for n in support_names if _lang_candidates([n], u1_lang)]
                if cands:
                    i2 = rng.choice(cands)
                    u2_lang, u2 = rng.choice([(lng, t) for lng, t in INTENTS[i2]["user"] if lng == u1_lang])
                    a2 = compose_reply(rng, i2, u2_lang)
                    msgs += [{"role": "user", "content": u2}, {"role": "assistant", "content": a2}]
                    tags.append("brief" if INTENTS[i2].get("topic") is None else "support")
            pairs += to_pair(msgs, thread_id=tid, risk_tag=tags)
            continue

        # risk == "none" -> build a genuinely deep thread (2-8 turns)
        msgs, tags = build_deep_thread(rng, intent, u1_lang, support_names)
        pairs += to_pair(msgs, thread_id=tid, risk_tag=tags)

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# EmpatheticDialogues - deep sanitization, heavy labels only.
# ed_cap default cut 2000 -> 400; min_words raised in audit() (15, was 3) --
# v3.2 defaults let ED reach ~half the corpus by volume with 3-word floors,
# diluting the specialized register with short generic public-data lines.
# ─────────────────────────────────────────────────────────────────────────────
DISTRESS_LABELS = {
    "afraid", "angry", "annoyed", "anxious", "apprehensive", "ashamed", "disgusted",
    "devastated", "disappointed", "discouraged", "embarrassed", "furious", "guilty",
    "humiliated", "insecure", "jealous", "lonely", "sad", "sentimental", "terrified",
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
            tid = f"ed-{ci:05d}"
            utts = [_clean_ed_text(u) for _, u in sorted(turns)[:6]]
            if len(utts) < 2:
                continue
            if any(k in " ".join(utts).lower() for k in CRISIS_KEYS):
                continue
            msgs = [{"role": "system", "content": SYSTEM}]
            for u in utts:
                msgs.append({"role": "user" if len(msgs) % 2 else "assistant", "content": u})
            out += to_pair(msgs, thread_id=tid, risk_tag="ed")
            if len(out) >= cap * 2:
                break
        MEDICAL_FILTER = (r"\btablets?\b", r"\bdose\b", r"\bpill\b",
                          r"take \d+ (tablet|pill|capsule|spoon|mg)", r"\bmedicine\b")
        out = [
            p
            for p in out
            if len(p["target"].split()) >= 15
            and not NAME_LEAK.search(p["target"])
            and not any(re.search(pat, p["target"].lower()) for pat in COMMITTING)
            and not any(re.search(pat, p["target"].lower()) for pat in MEDICAL_FILTER)
        ]
        rng.shuffle(out)
        return out[:cap]
    except Exception as e:
        print(f"[corpus] public datasets unavailable, synthetic-only: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT GATE - fails the build on ANY violation.
# min_words is now risk-aware: crisis/refusal stay short by design (correct
# crisis-UX, unchanged from v3.2); support/practical responses now require
# real paragraph length. max raised 90 -> 160 to fit the new paragraph shape.
# ─────────────────────────────────────────────────────────────────────────────
def audit(pairs: list[dict]) -> dict:
    user_lang: dict[str, str] = {text: lang for d in INTENTS.values() for lang, text in d["user"]}
    user_lang.update({text: lang for lang, texts in DECLINES.items() for text in texts})
    # fact-carrying user turns are generated dynamically (not literal bank entries) --
    # register them too, or every one of them false-positives as "unknown line / typo?"
    for facts in CASE_FACTS.values():
        for f in facts:
            for lg, txt in f.items():
                prefix = {"en": "also, ", "hinglish": "waise, ", "hi": "वैसे, "}[lg]
                user_lang[f"{prefix}{txt}"] = lg
    handoff_tokens = _handoff_tokens()

    LEGAL_ALTS = (
        "advocate", "legal aid", "legal-aid", "वकील", "अधिवक्ता", "लीगल एड",
        "can't help", "nahi bata sakta", "नहीं बता सकता", "outside what I can do",
    )
    PIVOT_MARKERS = (
        "what i can do", "help", "madad", "मदद", "counsellor", "साथ", "steadier",
        "support", "talk it through", "baat karein", "बात करें",
    )

    violations: list[tuple[int, str]] = []
    matrix: dict[str, dict[str, int]] = {}

    for idx, p in enumerate(pairs):
        user, tgt = p["prefix"][-1]["content"], p["target"]
        source = p.get("source", "synthetic")
        low = tgt.lower()
        user_l = user.lower()

        is_ed = source == "ed"
        is_known_user = user in user_lang or is_ed
        if source == "synthetic" and not is_known_user:
            violations.append((idx, "synthetic user line not in INTENTS/DECLINES bank - typo?"))

        lang = user_lang.get(user, "en")
        # risk_tag is set at GENERATION time (see to_pair/build_deep_thread), not
        # re-derived from text -- this is what fixes the "advocate" misclassification:
        # a support paragraph mentioning "advocate" as practical guidance is tagged
        # "support" from the moment it's created, regardless of its wording.
        risk_tag = p.get("risk_tag", "support")
        is_crisis_target = any(t in tgt for t in handoff_tokens)  # still used for the
        # separate "crisis without handoff" safety check below, NOT for length banding

        if is_ed:
            min_words, max_words = 15, 160
        elif risk_tag in ("crisis", "refusal", "brief"):
            min_words, max_words = 10, 70           # short by design -- not a bug
        else:
            min_words, max_words = 45, 160          # support/practical -- the actual fix

        wc = len(tgt.split())
        if not min_words <= wc <= max_words:
            violations.append((idx, f"word-count {wc} outside [{min_words},{max_words}] "
                                     f"(source={source}, risk_tag={risk_tag})"))

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
                "evaluator nahi" in low or "not an evaluator" in low or "मूल्यांकक नहीं" in tgt
                or "no report" in low or "nahi rakhta" in low or "नहीं रखता" in tgt
                or "koi report" in low or "koi score" in low or "koi analysis" in low
                or "rating nahi" in low
            )
            if score_leak:
                violations.append((idx, "score/monitoring leak"))

        category = risk_tag if risk_tag in ("crisis", "refusal") else "support"
        matrix.setdefault(lang, {})
        matrix[lang][category] = matrix[lang].get(category, 0) + 1

    if violations:
        by_kind = Counter(k for _, k in violations)
        print("violation summary by kind:", dict(by_kind))
        for v in violations[:50]:
            print("AUDIT VIOLATION:", v)
        raise SystemExit(f"corpus audit FAILED: {len(violations)} violations - fix before training")
    return matrix


def jaccard_dedup(pairs: list[dict], thresh: float = 0.75) -> list[dict]:
    seen: list[set] = []
    out: list[dict] = []
    for p in pairs:
        if p.get("keep"):
            out.append(p)
            continue
        ctx = " ".join(m["content"] for m in p["prefix"] if m["role"] != "system")
        toks = set(norm(ctx).split()) | set(norm(p["target"]).split())
        if any(len(toks & s) / max(1, len(toks | s)) >= thresh for s in seen[-6000:]):
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

    crisis_upsample = [
        p for p in pairs
        if p.get("source") == "synthetic"
        and len(p["prefix"]) == 2
        and any(t in p["target"].lower() for t in _handoff_tokens())
    ]
    pre_dup_targets = [norm(p["target"]) for p in pairs]
    pairs = pairs + crisis_upsample * 4

    all_threads = sorted({p["thread_id"] for p in pairs})
    rng.shuffle(all_threads)
    k_threads = max(1, int(0.05 * len(all_threads)))
    val_tids = set(all_threads[:k_threads])
    val = [p for p in pairs if p["thread_id"] in val_tids]
    train = [p for p in pairs if p["thread_id"] not in val_tids]
    rng.shuffle(train)
    rng.shuffle(val)

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
        "exact_duplicate_target_ratio": round(
            1 - len(set(pre_dup_targets)) / max(1, len(pre_dup_targets)), 3
        ),
        "crisis_oversample_count": len(crisis_upsample) * 4,
        "lang_matrix": matrix,
        "generator": "v4.0-paragraph-context-aware",
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
    ap.add_argument("--ed-cap", type=int, default=400)
    ap.add_argument("--crisis-share", type=float, default=0.12)
    ap.add_argument("--out", default="data/processed")
    main(**vars(ap.parse_args()))