"""Resume loading and structured-field helpers.

Supported inputs
----------------
* **Kaggle "Resume Entities for NER"** (Dataturks): a JSON-lines file where every line is
  ``{"content": "<resume text>", "annotation": [{"label": ["Skills"], "points": [{"start": 10, "end": 50, "text": "..."}]}], ...}``
  Labels include Name, Designation, Companies worked at, Skills, College Name, Degree, Graduation Year,
  Years of Experience, Location and Email Address.
* a JSON array with the same objects,
* a CSV with a text column (``Resume_str`` / ``Resume`` / ``text`` / ``content``) and an optional ``Category`` column,
* a folder of ``.txt`` files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

CANONICAL_LABELS = {
    "name": "Name",
    "college name": "College Name",
    "degree": "Degree",
    "graduation year": "Graduation Year",
    "years of experience": "Years of Experience",
    "companies worked at": "Companies worked at",
    "designation": "Designation",
    "skills": "Skills",
    "location": "Location",
    "email address": "Email Address",
}

TEXT_COLS = ("content", "resume_str", "resume", "text", "resume_text", "cv")
CATEGORY_COLS = ("category", "label", "role", "job_title")


@dataclass
class Resume:
    """One resume with its (optional) annotated entities."""

    id: str
    text: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    spans: list[tuple[str, int, int]] = field(default_factory=list)  # (label, start, end_exclusive)
    category: str | None = None

    def get(self, *labels: str) -> list[str]:
        out: list[str] = []
        for label in labels:
            out.extend(self.entities.get(label, []))
        return out

    @property
    def name(self) -> str:
        names = self.get("Name")
        if names:
            return names[0]
        first = next((ln.strip() for ln in self.text.splitlines() if ln.strip()), "")
        return first if 0 < len(first.split()) <= 4 else self.id

    @property
    def designations(self) -> list[str]:
        return self.get("Designation")

    @property
    def title(self) -> str:
        d = self.designations
        return d[0] if d else (self.category or "")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\u2022", " ")).strip(" \t\r\n:-,;")


def _canon(label: str) -> str:
    return CANONICAL_LABELS.get(label.strip().lower(), label.strip())


def parse_dataturks_record(obj: dict, index: int) -> Resume:
    text = obj.get("content") or ""
    entities: dict[str, list[str]] = {}
    spans: list[tuple[str, int, int]] = []
    for ann in obj.get("annotation") or []:
        labels = ann.get("label") or []
        labels = [labels] if isinstance(labels, str) else labels
        for point in ann.get("points") or []:
            snippet = point.get("text", "")
            start, end = int(point.get("start", 0)), int(point.get("end", 0))
            # Dataturks end offsets are inclusive; accept exclusive ones too.
            if text[start : end + 1].strip() == snippet.strip():
                end_excl = end + 1
            elif text[start:end].strip() == snippet.strip():
                end_excl = end
            else:
                end_excl = end + 1
            for label in labels:
                label = _canon(label)
                # keep the raw layout (bullets / line breaks) of skills blocks so they can be split into skills later
                value = snippet.strip() if label == "Skills" else _clean(snippet)
                if value and value not in entities.setdefault(label, []):
                    entities[label].append(value)
                spans.append((label, start, end_excl))
    meta = obj.get("metadata") or {}
    rid = str(meta.get("id") or obj.get("id") or f"R{index + 1:04d}")
    return Resume(id=rid, text=text, entities=entities, spans=spans)


def _read_json_records(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if raw.startswith("["):
        return json.loads(raw)
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_resumes(path: str | Path, limit: int = 0) -> list[Resume]:
    """Load resumes from a Dataturks JSON(L), CSV or a folder of .txt files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume data not found: {path}. See docs/DATASET.md for how to download the Kaggle dataset.")
    resumes: list[Resume] = []
    if path.is_dir():
        for i, f in enumerate(sorted(path.glob("*.txt"))):
            resumes.append(Resume(id=f.stem, text=f.read_text(encoding="utf-8", errors="replace")))
        if not resumes:
            raise ValueError(f"No .txt resumes found in {path}")
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
        cols = {c.strip().lower(): c for c in df.columns}
        text_col = next((cols[c] for c in TEXT_COLS if c in cols), None)
        if text_col is None:
            raise ValueError(f"No text column in {path}. Columns: {list(df.columns)}; expected one of {TEXT_COLS}.")
        cat_col = next((cols[c] for c in CATEGORY_COLS if c in cols), None)
        id_col = next((cols[c] for c in ("id", "resume_id") if c in cols), None)
        for i, row in df.iterrows():
            text = row[text_col]
            if not isinstance(text, str) or not text.strip():
                continue
            rid = str(row[id_col]) if id_col else f"R{len(resumes) + 1:04d}"
            resumes.append(Resume(id=rid, text=text, category=str(row[cat_col]) if cat_col else None))
    else:
        for i, obj in enumerate(_read_json_records(path)):
            if not obj.get("content"):
                continue
            resumes.append(parse_dataturks_record(obj, len(resumes)))
        if not resumes:
            raise ValueError(f"No resumes with a 'content' field found in {path}.")
    ids = [r.id for r in resumes]
    if len(set(ids)) != len(ids):  # make ids unique
        for i, r in enumerate(resumes):
            r.id = f"R{i + 1:04d}"
    return resumes[:limit] if limit else resumes


# --------------------------------------------------------------------------- #
# Structured helpers
# --------------------------------------------------------------------------- #
_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE)
_MONTHS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*months?", re.IGNORECASE)


def parse_years(text: str) -> float | None:
    """'5 years' -> 5.0, '18 months' -> 1.5, 'Fresher' -> 0.0; ``None`` if nothing found."""
    m = _YEARS_RE.search(text)
    if m:
        v = float(m.group(1))
        return v if v <= 50 else None
    m = _MONTHS_RE.search(text)
    if m:
        return float(m.group(1)) / 12
    if re.search(r"\bfresher\b|\bentry[- ]level\b", text, re.IGNORECASE):
        return 0.0
    return None


def years_of_experience(resume: Resume) -> float | None:
    """Annotated 'Years of Experience' if present, otherwise the largest 'N years' near the word 'experience'."""
    for value in resume.get("Years of Experience"):
        y = parse_years(value)
        if y is not None:
            return y
    best = None
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)[^.\n]{0,40}experience|experience[^.\n]{0,30}?(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", resume.text, re.IGNORECASE):
        v = float(m.group(1) or m.group(2))
        if v <= 50:
            best = max(best or 0.0, v)
    return best


def split_skill_blob(blob: str) -> list[str]:
    """Split an annotated skills section into individual short skill phrases."""
    parts = re.split(r"[\n\r,;|\u2022•·●▪]+|\s{2,}|\s-\s", blob)
    out: list[str] = []
    for p in parts:
        p = p.strip(" \t\r\n-_:;,|*\u2022\u00b7\u25cf\u25aa").rstrip(".")  # keep '+', '#' and inner dots: C++, C#, Node.js, .NET
        if ":" in p:
            p = p.split(":")[-1].strip()
        if 1 < len(p) <= 40 and len(p.split()) <= 4 and p.lower() not in {s.lower() for s in out}:
            out.append(p)
    return out


def resume_frame(resumes: list[Resume]) -> pd.DataFrame:
    """Overview table (one row per resume)."""
    rows = []
    for r in resumes:
        rows.append(
            {
                "id": r.id, "name": r.name, "title": r.title, "years": years_of_experience(r),
                "words": len(r.text.split()), "annotated_entities": sum(len(v) for v in r.entities.values()),
            }
        )
    return pd.DataFrame(rows)


def dataset_statistics(resumes: list[Resume]) -> dict:
    words = [len(r.text.split()) for r in resumes]
    labels: dict[str, int] = {}
    for r in resumes:
        for label, values in r.entities.items():
            labels[label] = labels.get(label, 0) + len(values)
    years = [y for y in (years_of_experience(r) for r in resumes) if y is not None]
    return {
        "resumes": len(resumes),
        "avg_words": sum(words) / max(len(words), 1),
        "min_words": min(words, default=0),
        "max_words": max(words, default=0),
        "annotated_resumes": sum(1 for r in resumes if r.entities),
        "entities_by_label": dict(sorted(labels.items(), key=lambda kv: -kv[1])),
        "with_known_experience": len(years),
        "median_years": float(pd.Series(years).median()) if years else None,
    }
