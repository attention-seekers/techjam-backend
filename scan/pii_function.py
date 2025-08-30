import re
import spacy
nlp = spacy.load("en_core_web_sm")

PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s\-]{7,})\b")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ADDRESS_HINTS = re.compile(r"\b(st|street|ave|avenue|road|rd|blk|block|floor|unit|#\d+)\b", re.I)

def pii_from_text(span_text: str):
    hits = []
    if EMAIL_RE.search(span_text): hits.append(("email", 0.98))
    if PHONE_RE.search(span_text): hits.append(("phone_number", 0.95))
    if ADDRESS_HINTS.search(span_text): hits.append(("address_hint", 0.70))

    doc = nlp(span_text)
    for ent in doc.ents:
        if ent.label_ in ("PERSON","GPE","ORG","LOC"):
            hits.append((ent.label_.lower(), 0.60))
    return hits
