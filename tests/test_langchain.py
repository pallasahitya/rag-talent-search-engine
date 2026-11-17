import pytest

pytest.importorskip("langchain_core")

from talent_rag.langchain_adapter import TalentRetriever  # noqa: E402


def test_retriever_returns_redacted_documents_with_metadata(index):
    retriever = TalentRetriever(index=index, k=3)
    docs = retriever.invoke("Find me a Junior Data Analyst who knows SQL and Tableau")
    assert len(docs) == 3
    for d, rank in zip(docs, (1, 2, 3)):
        assert d.metadata["rank"] == rank and d.metadata["resume_id"].startswith("R") and "Data Analyst" in d.metadata["role"]
        assert d.page_content and "@" not in d.page_content
        assert index.resume_by_id[d.metadata["resume_id"]].name not in d.page_content
    assert docs[0].metadata["matched_skills"] == ["SQL", "Tableau"]


def test_retriever_options(index):
    docs = TalentRetriever(index=index, k=2, mode="dense", rerank_weight=0.0).invoke("devops engineer with AWS")
    assert len(docs) == 2
