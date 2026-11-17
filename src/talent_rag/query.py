"""Understand a recruiter's query and check candidates against its requirements.

A *skill vocabulary* (built-in terms + skills mined from the annotated resumes) finds the skills a query asks for;
seniority words and "N years" patterns become experience requirements; role phrases ("data analyst") become role terms.
The resulting :class:`QuerySpec` is used to show *requirement coverage* next to every candidate and to ground explanations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .data import Resume, split_skill_blob, years_of_experience

BUILTIN_SKILLS = """
sql;mysql;postgresql;oracle;mongodb;redis;python;java;javascript;typescript;c++;c#;go;rust;php;ruby;scala;kotlin;swift;r;matlab;
html;css;react;angular;vue;node.js;django;flask;spring boot;hibernate;redux;tailwind css;rest apis;microservices;kafka;
pandas;numpy;scikit-learn;tensorflow;pytorch;keras;nlp;deep learning;machine learning;computer vision;mlflow;statistics;data visualization;etl;
tableau;power bi;excel;google analytics;looker;spark;hadoop;airflow;dbt;
aws;azure;gcp;docker;kubernetes;jenkins;terraform;ansible;linux;ci/cd;git;github;gitlab;prometheus;grafana;bash;
selenium;testng;junit;jira;postman;cypress;manual testing;api testing;test automation;
tally;quickbooks;taxation;auditing;financial reporting;sap fico;sap;accounts payable;accounts receivable;budgeting;gaap;ifrs;
recruitment;talent acquisition;payroll;employee relations;hris;onboarding;performance management;training and development;labour law;
autocad;solidworks;catia;ansys;gd&t;lean manufacturing;six sigma;project management;cnc;thermodynamics;
agile;scrum;communication;teamwork;problem solving;leadership;ms office;powerpoint;word
"""
ALIASES = {
    "powerbi": "power bi", "power-bi": "power bi", "nodejs": "node.js", "node js": "node.js", "postgres": "postgresql", "js": "javascript",
    "k8s": "kubernetes", "ml": "machine learning", "dl": "deep learning", "cicd": "ci/cd", "ms excel": "excel", "microsoft excel": "excel",
    "scikit learn": "scikit-learn", "sklearn": "scikit-learn", "restful apis": "rest apis", "rest api": "rest apis", "reactjs": "react",
}
SENIORITY = {
    "intern": (0, 1), "fresher": (0, 1), "entry": (0, 2), "entry-level": (0, 2), "graduate": (0, 2), "junior": (0, 2), "jr": (0, 2),
    "mid": (2, 5), "intermediate": (2, 5), "mid-level": (2, 5), "senior": (5, None), "sr": (5, None), "experienced": (5, None),
    "lead": (7, None), "principal": (8, None), "staff": (8, None),
}
ROLE_NOUNS = (
    "analyst|engineer|developer|manager|accountant|designer|scientist|administrator|consultant|architect|officer|executive|"
    "specialist|technician|programmer|tester|recruiter|auditor"
)
_ROLE_RE = re.compile(rf"((?:[a-z][a-z+#./-]*\s+){{0,2}}(?:{ROLE_NOUNS}))\b", re.IGNORECASE)
_STOP_ROLE = {"a", "an", "the", "me", "find", "who", "with", "for", "as", "and", "or", "of", "to", "need", "looking", "candidate", "someone", "person", "hire"}


ACRONYMS = {"sql", "aws", "gcp", "nlp", "etl", "ci/cd", "css", "html", "hris", "cnc", "gaap", "ifrs", "sap", "sap fico", "api", "rest apis", "kpi", "php", "gd&t", "ms office"}
PRETTY = {
    "node.js": "Node.js", "power bi": "Power BI", "javascript": "JavaScript", "typescript": "TypeScript", "mysql": "MySQL",
    "postgresql": "PostgreSQL", "mongodb": "MongoDB", "pytorch": "PyTorch", "tensorflow": "TensorFlow", "scikit-learn": "scikit-learn",
    "c++": "C++", "c#": "C#", "github": "GitHub", "gitlab": "GitLab", "mlflow": "MLflow", "quickbooks": "QuickBooks", "solidworks": "SolidWorks",
    "autocad": "AutoCAD", "catia": "CATIA", "ansys": "ANSYS", "powerpoint": "PowerPoint", "rest apis": "REST APIs", "spring boot": "Spring Boot",
    "testng": "TestNG", "junit": "JUnit", "devops": "DevOps", "tailwind css": "Tailwind CSS", "google analytics": "Google Analytics",
}


def prettify(term: str) -> str:
    """Display form of a lower-case skill term (SQL, Power BI, Node.js, ...)."""
    low = term.lower()
    if low in PRETTY:
        return PRETTY[low]
    if low in ACRONYMS:
        return low.upper()
    return term if any(c.isupper() for c in term) else term.title()


def _term_pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<![\w+#.])" + re.escape(term) + r"(?![\w+#])", re.IGNORECASE)


class SkillVocabulary:
    """Canonical skill terms with fast case-insensitive, boundary-aware matching."""

    def __init__(self, terms: list[str]) -> None:
        canon: dict[str, str] = {}
        for t in terms:
            t = re.sub(r"\s+", " ", t.strip())
            if t:
                canon.setdefault(t.lower(), t)
        self.canon = canon
        self._patterns = {k: _term_pattern(k) for k in canon}
        self._aliases = {a: c for a, c in ALIASES.items() if c in canon}
        self._alias_patterns = {a: _term_pattern(a) for a in self._aliases}
        self._cache: dict[str, set[str]] = {}

    def __len__(self) -> int:
        return len(self.canon)

    def __contains__(self, term: str) -> bool:
        return self.normalize(term) in self.canon

    def normalize(self, term: str) -> str:
        t = re.sub(r"\s+", " ", term.strip().lower())
        return ALIASES.get(t, t)

    @classmethod
    def builtin(cls) -> SkillVocabulary:
        return cls([prettify(t) for t in re.split(r"[;\n]+", BUILTIN_SKILLS) if t.strip()])

    @classmethod
    def from_resumes(cls, resumes: list[Resume], min_resumes: int = 2) -> SkillVocabulary:
        """Built-in terms plus skill phrases that appear in the Skills section of at least ``min_resumes`` resumes."""
        counts: dict[str, int] = {}
        display: dict[str, str] = {}
        for r in resumes:
            seen = set()
            for blob in r.get("Skills"):
                for s in split_skill_blob(blob):
                    k = s.lower()
                    if k not in seen and len(k) > 1 and not k.isdigit():
                        seen.add(k)
                        display.setdefault(k, s)
            for k in seen:
                counts[k] = counts.get(k, 0) + 1
        mined = [display[k] for k, c in counts.items() if c >= min_resumes]
        builtin = [prettify(t) for t in BUILTIN_SKILLS.replace("\n", "").split(";") if t.strip()]
        return cls([*mined, *builtin])  # mined terms first: they keep the casing used in the resumes

    def skills_of(self, resume: Resume) -> set[str]:
        """Lower-case skills mentioned in a resume (cached per resume id)."""
        key = f"{resume.id}:{len(resume.text)}"
        if key not in self._cache:
            self._cache[key] = {s.lower() for s in self.find(resume.text)}
        return self._cache[key]

    def find(self, text: str) -> list[str]:
        """Canonical skills mentioned in ``text`` (unique, in order of first appearance)."""
        hits: list[tuple[int, str]] = []
        for key, pat in self._patterns.items():
            m = pat.search(text)
            if m:
                hits.append((m.start(), self.canon[key]))
        for alias, canon_key in self._aliases.items():
            m = self._alias_patterns[alias].search(text)
            if m and not any(c.lower() == canon_key for _, c in hits):
                hits.append((m.start(), self.canon[canon_key]))
        hits.sort()
        # drop skills that are only a part of a longer matched skill ("sap" inside "sap fico", "sql" inside "mysql" is already blocked)
        longer = [c.lower() for _, c in hits]
        return [c for _, c in hits if not any(c.lower() != o and re.search(rf"\b{re.escape(c.lower())}\b", o) and self._covered(text, c, o) for o in longer)]

    @staticmethod
    def _covered(text: str, short: str, long: str) -> bool:
        """True if every occurrence of ``short`` lies inside an occurrence of ``long``."""
        spans = [m.span() for m in re.finditer(re.escape(long), text, re.IGNORECASE)]
        for m in re.finditer(rf"(?<![\w+#.]){re.escape(short.lower())}(?![\w+#])", text.lower()):
            if not any(a <= m.start() and m.end() <= b for a, b in spans):
                return False
        return True


@dataclass
class QuerySpec:
    raw: str
    skills: list[str] = field(default_factory=list)
    min_years: float | None = None
    max_years: float | None = None
    seniority: str | None = None
    role: str | None = None

    @property
    def role_terms(self) -> list[str]:
        return self.role.split() if self.role else []

    def requirements(self) -> list[str]:
        reqs = [f"skill: {s}" for s in self.skills]
        if self.role:
            reqs.append(f"role: {self.role}")
        lo, hi = self.min_years, self.max_years
        if lo is not None and hi is not None:
            reqs.append(f"experience: {lo:g}-{hi:g} years")
        elif lo is not None:
            reqs.append(f"experience: {lo:g}+ years")
        elif hi is not None:
            reqs.append(f"experience: up to {hi:g} years")
        return reqs


def parse_query(text: str, vocab: SkillVocabulary) -> QuerySpec:
    spec = QuerySpec(raw=text, skills=vocab.find(text))
    low = text.lower()
    for word, (lo, hi) in SENIORITY.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            spec.seniority, spec.min_years, spec.max_years = word, float(lo) if lo else None, float(hi) if hi is not None else None
            break
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-\u2013to]+\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", low)
    if m:
        spec.min_years, spec.max_years = float(m.group(1)), float(m.group(2))
    else:
        m = re.search(r"(?:at least|minimum of|min\.?|over|more than)?\s*(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)|(?:at least|minimum of|over|more than)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", low)
        if m:
            spec.min_years, spec.max_years = float(m.group(1) or m.group(2)), None
        else:
            m = re.search(r"(?:less than|under|up to|max(?:imum)?(?: of)?)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", low)
            if m:
                spec.min_years, spec.max_years = None, float(m.group(1))
            else:
                m = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", low)
                if m:
                    spec.min_years = float(m.group(1))
    role = None
    for m in _ROLE_RE.finditer(text):
        words = [w for w in re.split(r"\s+", m.group(1).lower()) if w not in _STOP_ROLE and w not in SENIORITY]
        if words and words[-1].rstrip("s") in ROLE_NOUNS.split("|"):
            role = " ".join(words)
            break
    spec.role = role
    return spec


@dataclass
class Coverage:
    matched_skills: list[str]
    missing_skills: list[str]
    years: float | None
    years_ok: bool | None
    role_ok: bool | None
    score: float  # share of checkable requirements that are met (0..1)


def requirement_coverage(spec: QuerySpec, resume: Resume, vocab: SkillVocabulary) -> Coverage:
    have = vocab.skills_of(resume)
    matched = [s for s in spec.skills if s.lower() in have]
    missing = [s for s in spec.skills if s.lower() not in have]
    years = years_of_experience(resume)
    years_ok = None
    if years is not None and (spec.min_years is not None or spec.max_years is not None):
        years_ok = (spec.min_years is None or years >= spec.min_years) and (spec.max_years is None or years <= spec.max_years)
    role_ok = None
    if spec.role:
        hay = " ".join([*resume.designations, resume.category or "", resume.text[:300]]).lower()
        role_ok = all(re.search(rf"\b{re.escape(t.rstrip('s'))}", hay) for t in spec.role_terms)
    checks = [True] * len(matched) + [False] * len(missing) + [c for c in (years_ok, role_ok) if c is not None]
    score = sum(checks) / len(checks) if checks else 1.0
    return Coverage(matched, missing, years, years_ok, role_ok, score)
