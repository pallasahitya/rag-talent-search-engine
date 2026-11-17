"""Split resumes into retrievable chunks (section-aware sliding windows) plus a structured *profile* chunk."""

from __future__ import annotations

import re
from dataclasses import dataclass

SECTION_NAMES = [
    "summary", "profile", "objective", "career objective", "professional summary", "skills", "technical skills", "key skills",
    "core competencies", "experience", "work experience", "professional experience", "employment history", "projects",
    "education", "academic qualifications", "qualifications", "certifications", "achievements", "awards", "languages",
    "personal details", "personal information", "interests", "references", "additional information",
]
_HEADER_RE = re.compile(r"^\s*(?:" + "|".join(sorted((re.escape(s) for s in SECTION_NAMES), key=len, reverse=True)) + r")\s*:?\s*$", re.IGNORECASE)


@dataclass
class Chunk:
    chunk_id: str
    resume_id: str
    section: str
    text: str

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "resume_id": self.resume_id, "section": self.section, "text": self.text}


_MARKERS_RE = re.compile(r"\[(?:REDACTED PERSONAL DETAIL|NAME|EMAIL|PHONE|URL)\]")


def is_noise(text: str) -> bool:
    """True if nothing is left once redaction markers and punctuation are removed."""
    return not re.sub(r"[\W_]+", "", _MARKERS_RE.sub("", text))


def split_sections(text: str) -> list[tuple[str, str]]:
    """``[(section_name, section_text), ...]``; text before the first header is the 'Header' section."""
    sections: list[tuple[str, list[str]]] = [("Header", [])]
    for line in text.splitlines():
        if _HEADER_RE.match(line):
            sections.append((line.strip().rstrip(":").title(), []))
        else:
            sections[-1][1].append(line)
    out = [(name, "\n".join(lines).strip()) for name, lines in sections]
    return [(n, t) for n, t in out if t and not is_noise(t)]


def window_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = max(1, size - overlap)
    windows = []
    for start in range(0, len(words), step):
        windows.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return windows


def chunk_resume(resume_id: str, text: str, size: int = 120, overlap: int = 30, profile: str = "") -> list[Chunk]:
    """Profile chunk first (if given), then section windows. Short sections stay whole."""
    chunks: list[Chunk] = []
    if profile.strip():
        chunks.append(Chunk(f"{resume_id}::0", resume_id, "Profile", profile.strip()))
    for section, body in split_sections(text):
        for piece in window_words(body, size, overlap):
            chunks.append(Chunk(f"{resume_id}::{len(chunks)}", resume_id, section, f"{section}: {piece}" if section != "Header" else piece))
    if not chunks:
        chunks.append(Chunk(f"{resume_id}::0", resume_id, "Resume", text.strip() or "(empty)"))
    return chunks
