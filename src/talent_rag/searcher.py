"""High-level search: retrieval + requirement coverage + optional re-ranking + optional LLM explanation."""

from __future__ import annotations

from dataclasses import dataclass

from .explain import Explanation, explain_candidates
from .index import Hit, TalentIndex
from .llm import LLMClient
from .query import Coverage, QuerySpec, SkillVocabulary, parse_query, requirement_coverage


@dataclass
class SearchResult:
    query: str
    spec: QuerySpec
    mode: str
    hits: list[Hit]
    coverages: list[Coverage]
    reranked: bool
    explanation: Explanation | None = None


class TalentSearch:
    """Natural-language talent search over a :class:`TalentIndex`.

    ``rerank_weight`` blends the retrieval rank with *requirement coverage* (requested skills, years, role):
    ``final = (1 - w) * retrieval_rank_score + w * coverage``. ``w = 0`` keeps the pure retrieval order.
    """

    def __init__(self, index: TalentIndex) -> None:
        self.index = index
        self.vocab: SkillVocabulary = index.vocab

    def search(
        self,
        query: str,
        k: int = 3,
        mode: str = "hybrid",
        rerank_weight: float = 0.5,
        explain: bool = False,
        llm: LLMClient | None = None,
        n_evidence: int = 2,
        pool: int | None = None,
    ) -> SearchResult:
        spec = parse_query(query, self.vocab)
        use_rerank = rerank_weight > 0 and bool(spec.requirements())
        pool_size = max(pool or 0, k * 5, 15) if use_rerank else k
        hits = self.index.search(query, pool_size, mode, n_evidence)
        covs = [requirement_coverage(spec, h.resume, self.vocab) for h in hits]
        if use_rerank:
            n = max(len(hits), 1)
            blend = [(1 - rerank_weight) * (1 - i / n) + rerank_weight * c.score for i, c in enumerate(covs)]
            order = sorted(range(len(hits)), key=lambda i: (-blend[i], i))[:k]
            hits, covs = [hits[i] for i in order], [covs[i] for i in order]
            for rank, h in enumerate(hits, start=1):
                h.rank = rank
        else:
            hits, covs = hits[:k], covs[:k]
        result = SearchResult(query, spec, mode, hits, covs, use_rerank)
        if explain:
            result.explanation = explain_candidates(query, hits, spec, covs, llm)
        return result

    def ranked_ids(self, query: str, k: int = 10, mode: str = "hybrid", rerank_weight: float = 0.0) -> list[str]:
        return [h.resume.id for h in self.search(query, k, mode, rerank_weight).hits]
