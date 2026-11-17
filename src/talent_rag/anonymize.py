"""Redact personal data so that neither the embeddings, the LLM prompt nor the ranking can depend on it.

Removed / neutralised: e-mail addresses, phone numbers, URLs, candidate names, titles (Mr/Mrs/...), gendered pronouns
(he/she -> they) and lines that state protected characteristics (date of birth, marital status, gender, nationality,
religion, father's name ...). Job-relevant text (skills, employers, education, experience) is kept.

This reduces - it does not eliminate - bias risk: employers, colleges, locations and writing style can still act as
proxies. See ``bias.py`` for the audit.
"""

from __future__ import annotations

import re

from .lexicons import NEUTRAL_PRONOUNS, PERSONAL_DETAIL_KEYS, TITLES

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_RE = re.compile(r"(?:https?://|www\.)\S+|(?:linkedin|github)\.com/\S+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<![\w/])(?:\+?\d[\d\s().-]{7,}\d)(?![\w/])")
_KEYS = "|".join(sorted((re.escape(k) for k in PERSONAL_DETAIL_KEYS), key=len, reverse=True))
PERSONAL_LINE_RE = re.compile(rf"^\s*(?:{_KEYS})\b[^\n]*$", re.IGNORECASE | re.MULTILINE)
PRONOUN_RE = re.compile(r"\b(" + "|".join(NEUTRAL_PRONOUNS) + r")\b", re.IGNORECASE)
TITLE_RE = re.compile(r"\b(?i:" + "|".join(sorted(TITLES - {"dr"})) + r")\.?\s+(?=[A-Z\[])")
# "DOB: 3/4/1995 | Gender: Male" written inside a line (the whole-line rule only sees lines that START with the key)
_INLINE_KEYS = "|".join(sorted((re.escape(k) for k in PERSONAL_DETAIL_KEYS if k not in {"age", "photo", "hobbies", "passport", "visa status"}), key=len, reverse=True))
INLINE_DETAIL_RE = re.compile(rf"\b(?:{_INLINE_KEYS})\b\s*[:\-]\s*[^\n,;|]{{1,30}}", re.IGNORECASE)


def _match_case(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    return repl.capitalize() if src[:1].isupper() else repl


# words that can appear in a first line but are never a person's name (job titles, headings, skills, pronouns)
_NOT_NAMES = {
    "resume", "curriculum", "vitae", "cv", "profile", "summary", "objective", "skills", "education", "experience", "contact",
    "data", "software", "web", "java", "python", "machine", "learning", "devops", "hr", "qa", "mechanical", "senior", "junior",
    "lead", "intern", "fresher", "graduate", "full", "stack", "front", "back", "end", "cloud", "network", "systems", "business",
    "she", "he", "him", "her", "his", "they", "the", "and", "of", "for", "with",
}
_EMAIL_NOT_NAMES = {"info", "contact", "hr", "jobs", "admin", "mail", "email", "resume", "cv", "career", "careers", "hello", "office"}
_ROLE_WORD_RE = re.compile(r"(analyst|engineer|developer|manager|accountant|designer|scientist|administrator|consultant|architect|officer|executive|specialist|technician|programmer|tester|recruiter|auditor|lead|director)s?$", re.IGNORECASE)


def guess_names(text: str) -> list[str]:
    """Name tokens of the candidate when no annotation is available.

    Deliberately conservative: the first non-empty line counts only if it is 2-3 capitalised words that are neither job-title
    words, headings nor pronouns; plus the tokens of the e-mail local part. A wrong guess would redact job-relevant words,
    so when in doubt nothing is guessed (and the whole e-mail address is redacted anyway).
    """
    names: list[str] = []
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    toks = first.split()
    if 2 <= len(toks) <= 3 and all(re.fullmatch(r"[A-Z][a-zA-Z.'-]+|[A-Z]{2,}", t) for t in toks):
        if not any(t.lower().strip(".") in _NOT_NAMES or _ROLE_WORD_RE.search(t) for t in toks):
            names.extend(t.strip(".") for t in toks)
    m = EMAIL_RE.search(text)
    if m:
        names.extend(t for t in re.split(r"[._\-\d]+", m.group(0).split("@")[0]) if len(t) > 2 and t.lower() not in _EMAIL_NOT_NAMES)
    return names


def anonymize(text: str, names: list[str] | tuple[str, ...] = (), neutralize_pronouns: bool = True) -> str:
    """Return ``text`` with personal data redacted (see module docstring).

    ``names`` are the known name(s) of the candidate (e.g. the annotated ``Name``); only if none are given the names are guessed.
    """
    out = PERSONAL_LINE_RE.sub("[REDACTED PERSONAL DETAIL]", text)
    out = INLINE_DETAIL_RE.sub("[REDACTED PERSONAL DETAIL]", out)
    out = EMAIL_RE.sub("[EMAIL]", out)
    out = URL_RE.sub("[URL]", out)
    out = PHONE_RE.sub("[PHONE]", out)
    all_names = {n for n in (list(names) or guess_names(text)) if len(n) > 1}
    # also treat each token of a multi-word name as a name
    all_names |= {t for n in list(all_names) for t in n.split() if len(t) > 1}
    for name in sorted(all_names, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(name)}\b", "[NAME]", out, flags=re.IGNORECASE)
    out = TITLE_RE.sub("", out)
    if neutralize_pronouns:
        out = PRONOUN_RE.sub(lambda m: _match_case(m.group(0), NEUTRAL_PRONOUNS[m.group(0).lower()]), out)
    return out
