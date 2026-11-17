"""Explain *why* the top candidates fit: LLM-generated, grounded in the resumes, privacy-preserving.

* Candidates are shown to the LLM as ``Candidate A/B/C`` with redacted evidence (no names, e-mails, phones, pronouns).
* The prompt forbids inferring or mentioning protected characteristics and demands JSON with *exact quotes*.
* Every quote is **verified** against the evidence text (``grounded_ratio``) - hallucinated quotes are flagged.
* If no LLM is configured or the call/parse fails, a deterministic **template explainer** (no LLM) is used instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .anonymize import anonymize
from .index import Hit
from .llm import LLMClient, LLMError
from .query import Coverage, QuerySpec

SYSTEM_PROMPT = (
    "You are an assistant that helps a recruiter understand why candidates match a job request. "
    "Use ONLY the facts in the candidate evidence provided. Never guess. If something is not stated, say it is not stated. "
    "Candidate names and contact details are removed on purpose: never try to infer or mention gender, ethnicity, age, nationality, "
    "religion, marital status or any other protected characteristic, and do not let such guesses influence your assessment. "
    "Judge only skills, experience, roles and education. Reply with a single JSON object and nothing else."
)

SCHEMA_HINT = (
    '{"candidates":[{"candidate_id":"A","fit_score":1-5,"summary":"2-3 sentences on why this candidate fits the request",'
    '"matched_requirements":["..."],"gaps":["..."],"evidence_quotes":["exact quote (max 25 words) copied from the evidence"]}],'
    '"overall":"1-2 sentences comparing the candidates"}'
)

LABELS = "ABCDEFGHIJ"


@dataclass
class CandidateExplanation:
    candidate_id: str  # A / B / C
    resume_id: str
    fit_score: int | None
    summary: str
    matched_requirements: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    evidence_quotes: list[str] = field(default_factory=list)
    ungrounded_quotes: list[str] = field(default_factory=list)


@dataclass
class Explanation:
    backend: str
    candidates: list[CandidateExplanation]
    overall: str
    grounded_ratio: float | None = None
    fallback_reason: str | None = None
    raw: str | None = None


def candidate_payload(hit: Hit, label: str, coverage: Coverage | None) -> dict:
    """What the LLM is allowed to see for one candidate (redacted)."""
    r = hit.resume
    evidence = "\n---\n".join(anonymize(e.text, names=r.get("Name")) for e in hit.evidence)
    facts = {"role": "; ".join(dict.fromkeys(r.designations)) or r.category or "not stated"}
    if coverage and coverage.years is not None:
        facts["years_of_experience"] = f"{coverage.years:g}"
    if coverage:
        facts["requested_skills_found"] = ", ".join(coverage.matched_skills) or "none"
        facts["requested_skills_not_found"] = ", ".join(coverage.missing_skills) or "none"
    return {"candidate_id": label, "resume_id": r.id, "facts": facts, "evidence": evidence}


def build_prompt(query: str, payloads: list[dict]) -> str:
    blocks = []
    for p in payloads:
        facts = "\n".join(f"- {k}: {v}" for k, v in p["facts"].items())
        blocks.append(f"### Candidate {p['candidate_id']}\nFacts:\n{facts}\nEvidence (verbatim resume excerpts):\n{p['evidence']}")
    return (
        f"Job request: {query}\n\n" + "\n\n".join(blocks) +
        f"\n\nFor each candidate explain why they fit the request, list matched requirements and gaps, give fit_score 1-5, "
        f"and copy 1-3 short exact quotes from that candidate's evidence. Answer as JSON with this shape:\n{SCHEMA_HINT}"
    )


def parse_llm_json(text: str) -> dict | None:
    """Extract the first JSON object from an LLM answer (tolerates code fences / chatter)."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1]):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("candidates"), list):
            return obj
    return None


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def verify_quotes(quotes: list[str], evidence: str) -> tuple[list[str], list[str]]:
    """Split quotes into (grounded, ungrounded): a quote is grounded if it appears verbatim (normalised) in the evidence."""
    ev = _norm(evidence)
    good, bad = [], []
    for q in quotes:
        (good if q and _norm(q) and _norm(q) in ev else bad).append(q)
    return good, bad


def _best_sentence(text: str, terms: list[str], max_words: int = 25) -> str:
    """The evidence sentence that mentions the most of the requested ``terms`` (ties: the more informative, non-header line)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    best, best_key = "", (-1, 0)
    for s in sentences:
        if len(s.split()) < 3 or s.startswith("[") and s.endswith("]"):
            continue
        hits = sum(bool(re.search(rf"(?<![\w+#.]){re.escape(t)}(?![\w+#])", s, re.IGNORECASE)) for t in terms)
        key = (hits, 0 if s.lower().startswith("role:") else 1)
        if hits and key > best_key:
            best, best_key = s, key
    if not best:
        best = next((s for s in sentences if len(s.split()) >= 3), sentences[0] if sentences else "")
    return " ".join(best.split()[:max_words]).strip("\u2022- ")


def _years_phrase(value: str) -> str:
    return f"{value} year" + ("" if float(value) == 1 else "s")


def template_explanation(spec: QuerySpec, payloads: list[dict], coverages: list[Coverage | None]) -> Explanation:
    """Deterministic explanation without an LLM (clearly labelled as such)."""
    out = []
    for p, cov in zip(payloads, coverages):
        matched, gaps = [], []
        score = None
        if cov:
            matched += [f"skill: {s}" for s in cov.matched_skills]
            gaps += [f"skill not found: {s}" for s in cov.missing_skills]
            if cov.years_ok is True:
                matched.append(f"experience: {_years_phrase(f'{cov.years:g}')} fits the request")
            elif cov.years_ok is False:
                gaps.append(f"experience: {_years_phrase(f'{cov.years:g}')} is outside the requested range")
            if cov.role_ok is True:
                matched.append(f"role matches '{spec.role}'")
            elif cov.role_ok is False:
                gaps.append(f"role differs from '{spec.role}'")
            score = max(1, min(5, round(1 + 4 * cov.score)))
        role = p["facts"].get("role", "not stated")
        yrs = p["facts"].get("years_of_experience")
        parts = [f"Role: {role}" + (f", {_years_phrase(yrs)} of experience." if yrs else ".")]
        if matched:
            parts.append("Meets " + "; ".join(m.split(": ", 1)[-1] for m in matched[:4]) + ".")
        if gaps:
            parts.append("Gaps: " + "; ".join(g.split(": ", 1)[-1] for g in gaps[:3]) + ".")
        quote = _best_sentence(p["evidence"], (cov.matched_skills if cov else []) or spec.skills)
        out.append(CandidateExplanation(p["candidate_id"], p["resume_id"], score, " ".join(parts), matched, gaps, [quote] if quote else []))
    best = max(out, key=lambda c: (c.fit_score or 0), default=None)
    overall = f"Candidate {best.candidate_id} meets the most requirements." if best and best.fit_score else "No requirement could be checked automatically."
    return Explanation("template (no LLM)", out, overall, grounded_ratio=1.0)


def explain_candidates(
    query: str,
    hits: list[Hit],
    spec: QuerySpec,
    coverages: list[Coverage | None],
    llm: LLMClient | None = None,
) -> Explanation:
    """Explain the given hits. Falls back to the template explainer if ``llm`` is ``None`` or fails."""
    if not hits:
        return Explanation("none", [], "No candidates were retrieved.")
    labels = [LABELS[i] for i in range(len(hits))]
    payloads = [candidate_payload(h, lab, c) for h, lab, c in zip(hits, labels, coverages)]
    if llm is None:
        return template_explanation(spec, payloads, coverages)
    try:
        raw = llm.generate(SYSTEM_PROMPT, build_prompt(query, payloads))
    except LLMError as exc:
        fb = template_explanation(spec, payloads, coverages)
        fb.fallback_reason = f"LLM call failed: {exc}"
        return fb
    obj = parse_llm_json(raw)
    if obj is None:
        fb = template_explanation(spec, payloads, coverages)
        fb.fallback_reason = "LLM answer was not valid JSON"
        fb.raw = raw
        return fb
    by_label = {p["candidate_id"]: p for p in payloads}
    cands: list[CandidateExplanation] = []
    total = grounded = 0
    for item in obj["candidates"]:
        lab = str(item.get("candidate_id", "")).strip().upper()[:1]
        if lab not in by_label:
            continue
        quotes = [str(q) for q in (item.get("evidence_quotes") or [])]
        good, bad = verify_quotes(quotes, by_label[lab]["evidence"])
        total += len(quotes)
        grounded += len(good)
        try:
            score = int(item.get("fit_score"))
            score = max(1, min(5, score))
        except (TypeError, ValueError):
            score = None
        cands.append(CandidateExplanation(
            lab, by_label[lab]["resume_id"], score, str(item.get("summary", "")).strip(),
            [str(x) for x in item.get("matched_requirements") or []], [str(x) for x in item.get("gaps") or []], good, bad,
        ))
    if not cands:
        fb = template_explanation(spec, payloads, coverages)
        fb.fallback_reason = "LLM answer contained no known candidates"
        fb.raw = raw
        return fb
    cands.sort(key=lambda c: c.candidate_id)
    return Explanation(llm.name, cands, str(obj.get("overall", "")).strip(), (grounded / total) if total else None, None, raw)
