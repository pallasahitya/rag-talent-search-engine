"""LangChain integration: use the talent index as a standard LangChain ``BaseRetriever``.

    from talent_rag.index import TalentIndex
    from talent_rag.langchain_adapter import TalentRetriever

    retriever = TalentRetriever(index=TalentIndex.load("outputs/demo/index"), k=3)
    docs = retriever.invoke("Junior Data Analyst who knows SQL and Tableau")   # -> list[Document]

Each ``Document`` holds the redacted evidence text of one candidate; the metadata carries the resume id, rank, score and role.
Requires ``pip install langchain-core``. The retriever can be dropped into any LangChain chain (RAG prompt -> chat model).
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as exc:  # pragma: no cover
    raise ImportError("The LangChain adapter needs langchain-core:  pip install langchain-core") from exc

from .anonymize import anonymize
from .searcher import TalentSearch


class TalentRetriever(BaseRetriever):
    """LangChain retriever returning one Document (redacted evidence) per candidate."""

    index: Any
    k: int = 3
    mode: str = "hybrid"
    rerank_weight: float = 0.5

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None) -> list[Document]:
        result = TalentSearch(self.index).search(query, self.k, self.mode, self.rerank_weight)
        docs = []
        for hit, cov in zip(result.hits, result.coverages):
            text = "\n---\n".join(anonymize(e.text, names=hit.resume.get("Name")) for e in hit.evidence)
            docs.append(Document(page_content=text, metadata={
                "resume_id": hit.resume.id, "rank": hit.rank, "score": hit.score, "role": hit.resume.title,
                "matched_skills": cov.matched_skills, "missing_skills": cov.missing_skills,
            }))
        return docs
