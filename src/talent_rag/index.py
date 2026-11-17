"""The talent index: chunked resumes -> embeddings -> vector store (+ BM25), with dense / BM25 / hybrid search.

Search flow
-----------
1. **dense**  - embed the query, find the nearest chunks in the vector store, give each resume the score of its best chunk;
2. **bm25**   - lexical BM25 over whole resumes;
3. **hybrid** - reciprocal rank fusion (RRF) of the two resume rankings (robust, no score calibration needed).
For every hit the best-matching chunks are returned as *evidence* (used to ground the LLM explanation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__
from .anonymize import anonymize
from .bm25 import BM25
from .chunking import Chunk, chunk_resume
from .config import CHUNK_OVERLAP, CHUNK_WORDS, RRF_K
from .data import Resume, split_skill_blob, years_of_experience
from .embedders import Embedder, get_embedder, load_embedder
from .query import SkillVocabulary
from .stores import ChromaStore, FaissStore, VectorStore, make_store

MODES = ("dense", "bm25", "hybrid")


@dataclass
class Evidence:
    section: str
    text: str
    score: float


@dataclass
class Hit:
    resume: Resume
    rank: int
    score: float  # the score used for ranking (similarity, BM25 or RRF depending on the mode)
    dense_score: float | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    evidence: list[Evidence] = field(default_factory=list)


def profile_text(resume: Resume, vocab: SkillVocabulary) -> str:
    """Structured one-paragraph profile (role, experience, skills, education) - contains no personal identifiers."""
    parts = []
    roles = list(dict.fromkeys(resume.designations)) or ([resume.category] if resume.category else [])
    if roles:
        parts.append("Role: " + "; ".join(roles))
    years = years_of_experience(resume)
    if years is not None:
        parts.append(f"Experience: {years:g} years")
    skills = [s for blob in resume.get("Skills") for s in split_skill_blob(blob)] or vocab.find(resume.text)
    if skills:
        parts.append("Skills: " + ", ".join(dict.fromkeys(skills)))
    if resume.get("Degree"):
        parts.append("Education: " + "; ".join(dict.fromkeys(resume.get("Degree"))))
    if resume.get("Companies worked at"):
        parts.append("Employers: " + "; ".join(dict.fromkeys(resume.get("Companies worked at"))))
    return ". ".join(parts)


def embedding_text(resume: Resume, anonymize_text: bool) -> str:
    return anonymize(resume.text, names=resume.get("Name")) if anonymize_text else resume.text


class TalentIndex:
    def __init__(
        self,
        resumes: list[Resume],
        chunks: list[Chunk],
        embeddings: np.ndarray,
        embedder: Embedder,
        store: VectorStore,
        config: dict,
    ) -> None:
        self.resumes, self.chunks, self.embeddings = resumes, chunks, np.asarray(embeddings, dtype=np.float32)
        self.embedder, self.store, self.config = embedder, store, config
        self.resume_by_id = {r.id: r for r in resumes}
        pos = {r.id: i for i, r in enumerate(resumes)}
        self.chunk_resume = np.array([pos[c.resume_id] for c in chunks])
        self._chunks_of: dict[int, list[int]] = {}
        for ci, ri in enumerate(self.chunk_resume):
            self._chunks_of.setdefault(int(ri), []).append(ci)
        self.vocab = SkillVocabulary.from_resumes(resumes)
        docs = [" ".join(chunks[ci].text for ci in self._chunks_of.get(i, [])) for i in range(len(resumes))]
        self.bm25 = BM25(docs)

    # ---------------------------------------------------------------- build
    @classmethod
    def build(
        cls,
        resumes: list[Resume],
        embedder: str | Embedder = "lsa",
        store: str = "faiss",
        chunk_words: int = CHUNK_WORDS,
        chunk_overlap: int = CHUNK_OVERLAP,
        anonymize_text: bool = True,
        out_dir: str | Path | None = None,
    ) -> TalentIndex:
        if not resumes:
            raise ValueError("No resumes to index.")
        emb = get_embedder(embedder) if isinstance(embedder, str) else embedder
        vocab = SkillVocabulary.from_resumes(resumes)
        chunks: list[Chunk] = []
        for r in resumes:
            chunks.extend(chunk_resume(r.id, embedding_text(r, anonymize_text), chunk_words, chunk_overlap, profile_text(r, vocab)))
        texts = [c.text for c in chunks]
        if emb.needs_fit and not emb.fitted:
            emb.fit(texts)
        vectors = emb.embed_documents(texts)
        out = Path(out_dir) if out_dir else None
        vs = make_store(store, vectors.shape[1], out)
        vs.add(vectors)
        config = {
            "version": __version__, "embedder": emb.spec(), "store": store, "chunk_words": chunk_words,
            "chunk_overlap": chunk_overlap, "anonymize": anonymize_text, "n_resumes": len(resumes), "n_chunks": len(chunks),
        }
        index = cls(resumes, chunks, vectors, emb, vs, config)
        if out:
            index.save(out)
        return index

    def rebuild(self, resumes: list[Resume]) -> TalentIndex:
        """A new index over (modified) ``resumes`` that re-uses this index's *fitted* embedder and settings (used by the bias audit)."""
        return TalentIndex.build(
            resumes, self.embedder, "numpy", self.config["chunk_words"], self.config["chunk_overlap"], self.config["anonymize"]
        )

    # ------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "resumes.jsonl", "w", encoding="utf-8") as f:
            for r in self.resumes:
                f.write(json.dumps({"id": r.id, "text": r.text, "entities": r.entities, "spans": r.spans, "category": r.category}, ensure_ascii=False) + "\n")
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(d / "embeddings.npy", self.embeddings)
        self.embedder.save(d / "embedder")
        self.store.save(d)
        (d / "config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path, store: str | None = None) -> TalentIndex:
        d = Path(directory)
        if not (d / "config.json").exists():
            raise FileNotFoundError(f"No index found in {d}. Build one first: python -m talent_rag build-index")
        config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        resumes = []
        for line in (d / "resumes.jsonl").read_text(encoding="utf-8").splitlines():
            o = json.loads(line)
            resumes.append(Resume(o["id"], o["text"], o["entities"], [tuple(s) for s in o["spans"]], o.get("category")))
        chunks = [Chunk(**json.loads(line)) for line in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        embeddings = np.load(d / "embeddings.npy")
        embedder = load_embedder(d / "embedder")
        kind = store or config["store"]
        if kind == "faiss" and (d / "faiss.index").exists():
            vs: VectorStore = FaissStore.load(d, embeddings.shape[1])
        elif kind == "chroma" and (d / "chroma").exists():
            vs = ChromaStore(d / "chroma", reset=False)
            if len(vs) != len(embeddings):
                vs = ChromaStore(d / "chroma", reset=True)
                vs.add(embeddings)
        else:  # rebuild the requested store from the saved embeddings
            vs = make_store(kind, embeddings.shape[1], d)
            vs.add(embeddings)
        return cls(resumes, chunks, embeddings, embedder, vs, {**config, "store": kind})

    # ---------------------------------------------------------------- search
    def _dense_resume_scores(self, qv: np.ndarray, pool: int) -> dict[int, float]:
        idx, sims = self.store.search(qv, min(pool, len(self.chunks)))
        best: dict[int, float] = {}
        for ci, s in zip(idx, sims):
            ri = int(self.chunk_resume[int(ci)])
            if ri not in best or s > best[ri]:
                best[ri] = float(s)
        return best

    def search(self, query: str, k: int = 3, mode: str = "hybrid", n_evidence: int = 2, rrf_k: int = RRF_K) -> list[Hit]:
        """Top-``k`` resumes for a natural-language ``query``."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not query.strip():
            raise ValueError("Empty query.")
        qv = self.embedder.embed_query(query)
        n = len(self.resumes)
        dense_rank: dict[int, int] = {}
        dense_score: dict[int, float] = {}
        if mode in ("dense", "hybrid"):
            scores = self._dense_resume_scores(qv, max(100, k * 40, len(self.chunks) if len(self.chunks) <= 5000 else 0))
            for r, ri in enumerate(sorted(scores, key=lambda i: (-scores[i], i)), start=1):
                dense_rank[ri], dense_score[ri] = r, scores[ri]
        bm_rank: dict[int, int] = {}
        bm_score: dict[int, float] = {}
        if mode in ("bm25", "hybrid"):
            s = self.bm25.scores(query)
            order = [int(i) for i in np.argsort(-s, kind="stable") if s[i] > 0]
            for r, ri in enumerate(order, start=1):
                bm_rank[ri], bm_score[ri] = r, float(s[ri])
        if mode == "dense":
            final = {ri: dense_score[ri] for ri in dense_rank}
        elif mode == "bm25":
            final = {ri: bm_score[ri] for ri in bm_rank}
        else:
            final = {}
            for ri in set(dense_rank) | set(bm_rank):
                final[ri] = (1.0 / (rrf_k + dense_rank[ri]) if ri in dense_rank else 0.0) + (1.0 / (rrf_k + bm_rank[ri]) if ri in bm_rank else 0.0)
        ranked = sorted(final, key=lambda i: (-final[i], i))[: min(k, n)]
        hits = []
        for rank, ri in enumerate(ranked, start=1):
            cids = self._chunks_of[ri]
            sims = self.embeddings[cids] @ qv
            top = np.argsort(-sims, kind="stable")[:n_evidence]
            evidence = [Evidence(self.chunks[cids[j]].section, self.chunks[cids[j]].text, float(sims[j])) for j in top]
            hits.append(Hit(self.resumes[ri], rank, final[ri], dense_score.get(ri), bm_score.get(ri), dense_rank.get(ri), bm_rank.get(ri), evidence))
        return hits

    def rank_all(self, query: str, mode: str = "hybrid") -> list[str]:
        """Resume ids of the whole collection ordered by relevance (used by evaluation / bias audit)."""
        return [h.resume.id for h in self.search(query, k=len(self.resumes), mode=mode, n_evidence=1)]
