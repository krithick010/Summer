import json, logging, re, time, httpx
from config import OLLAMA_BASE_URL, EXTRACTOR_MODEL, OLLAMA_NUM_CTX
from core.llm_engine import log_ollama_timing

log = logging.getLogger(__name__)

SCHEMA = {
  "type": "object",
  "properties": {"facts": {"type": "array", "items": {
    "type": "object",
    "properties": {
      "subject": {"type": "string"},
      "attribute": {"type": "string"},
      "value": {"type": "string"}},
    "required": ["subject", "attribute", "value"]}}},
  "required": ["facts"]}

PROMPT = """You extract NEW personal facts from the user's LATEST message only.
Earlier lines are just context to work out who "her/his/it" refers to. Never take values from them.

subject = the PERSON the fact is about: mom, dad, grandma, a name, or "user" for the user himself.
attribute = one short word: email, phone, birthday, address, allergy, favorite_food, opinion, etc.
Never put the person inside the attribute.
Never invent or guess. If the latest message states no new fact (questions, greetings, chat), return {"facts": []}.

Examples:
latest: "my grandma's number is 555-0199 and email is g@x.com"
-> {"facts": [{"subject": "grandma", "attribute": "phone", "value": "555-0199"}, {"subject": "grandma", "attribute": "email", "value": "g@x.com"}]}
latest: "what's mom's email?"
-> {"facts": []}

Context:
{context}

Latest user message:
{latest}"""

RELATIONS = {"mom", "dad", "grandma", "grandpa", "brother", "sister", "wife", "husband", "friend"}
SUBJECT_ALIASES = {"mother": "mom", "mama": "mom", "mummy": "mom", "father": "dad", "papa": "dad",
                   "grandmother": "grandma", "granny": "grandma", "grandfather": "grandpa",
                   "me": "user", "i": "user", "myself": "user"}
ATTR_ALIASES = {"phone_number": "phone", "number": "phone", "mobile": "phone",
                "mail": "email", "gmail": "email", "email_address": "email"}


def _squash(s: str) -> str:
    return re.sub(r"[\s\-()]", "", s.lower())


def _clean(f: dict, latest: str):
    subject = str(f.get("subject", "")).strip().lower().replace(" ", "_")
    attr = str(f.get("attribute", "")).strip().lower().replace(" ", "_")
    value = str(f.get("value", "")).strip()
    subject = SUBJECT_ALIASES.get(subject, subject)
    for rel in RELATIONS:                      # e.g. subject "user", attribute "grandma_phone_number"
        if attr.startswith(rel + "_"):
            subject, attr = rel, attr[len(rel) + 1:]
            break
    attr = ATTR_ALIASES.get(attr, attr)
    if not (subject and attr and value):
        return None
    v = value.lower().replace("_", " ")
    if v in {"unknown", "n/a", "none", "null", "unspecified"} or "not provided" in v or "not available" in v:
        return None
    if _squash(value) not in _squash(latest):  # value must be in the LATEST message
        return None
    return {"subject": subject, "attribute": attr, "value": value}


async def extract_facts(history, message):
    recent = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")][-4:]
    context = "\n".join(f'{m["role"]}: {m["content"]}' for m in recent) or "(none)"
    prompt = PROMPT.replace("{context}", context).replace("{latest}", message)
    payload = {"model": EXTRACTOR_MODEL, "stream": False, "format": SCHEMA, "keep_alive": "30m",
               "messages": [{"role": "user", "content": prompt}],
               "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX, "num_predict": 200}}
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
    data = r.json()
    log_ollama_timing(
        log,
        "extractor",
        data,
        EXTRACTOR_MODEL,
        (time.perf_counter() - t0) * 1000.0,
    )
    raw = json.loads(data["message"]["content"]).get("facts", [])
    facts = []
    for f in raw:
        cleaned = _clean(f, message)
        if cleaned:
            facts.append(cleaned)
        else:
            log.warning("Dropped fact: %s", f)
    return facts