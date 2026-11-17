import numpy as np
import pytest

from talent_rag.index import MODES, TalentIndex, profile_text
from talent_rag.searcher import TalentSearch

Q = "Find me a Junior Data Analyst who knows SQL and Tableau"


def test_index_structure(index, resumes):
    assert index.config["n_resumes"] == 45 and index.config["n_chunks"] == len(index.chunks) == len(index.embeddings)
    assert index.embeddings.shape[1] == index.embedder.spec()["dim"] or index.embeddings.shape[1] <= 64
    assert np.allclose(np.linalg.norm(index.embeddings, axis=1), 1.0, atol=1e-5)
    assert {c.section for c in index.chunks} >= {"Profile", "Skills", "Experience"}
    assert all(len(index._chunks_of[i]) >= 2 for i in range(len(resumes)))


def test_profile_chunk_contains_job_facts_but_no_personal_identifiers(resumes, index):
    r = resumes[0]
    prof = profile_text(r, index.vocab)
    assert "Role:" in prof and "Skills:" in prof and "Experience:" in prof
    assert r.name.split()[0] not in prof and "@" not in prof


def test_embedded_text_is_anonymised(index, resumes):
    blob = " ".join(c.text for c in index.chunks)
    for r in resumes[:15]:
        assert r.get("Email Address")[0] not in blob
        assert r.name not in blob
    assert "[NAME]" in blob or "[EMAIL]" in blob
    assert "Date of Birth" not in blob and "Marital Status" not in blob


def test_raw_index_keeps_original_text(raw_index, resumes):
    blob = " ".join(c.text for c in raw_index.chunks)
    assert resumes[0].get("Email Address")[0] in blob


@pytest.mark.parametrize("mode", MODES)
def test_search_returns_ranked_hits_with_evidence(index, mode):
    hits = index.search(Q, 5, mode)
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert all(h.evidence and h.evidence[0].text for h in hits)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_relevant_candidates_come_first(index):
    for mode in MODES:
        top = index.search(Q, 3, mode)
        assert all("Data Analyst" in h.resume.title for h in top), mode


def test_hybrid_fuses_both_rankings(index):
    hits = index.search("senior devops engineer AWS Kubernetes Terraform", 3, "hybrid")
    assert all(h.dense_rank is not None or h.bm25_rank is not None for h in hits)
    h = hits[0]
    expected = (1 / (60 + h.dense_rank) if h.dense_rank else 0) + (1 / (60 + h.bm25_rank) if h.bm25_rank else 0)
    assert h.score == pytest.approx(expected)


def test_k_is_clamped_and_empty_query_rejected(index):
    assert len(index.search(Q, 1000, "dense")) == 45
    assert len(index.rank_all(Q)) == 45 and len(set(index.rank_all(Q))) == 45
    with pytest.raises(ValueError):
        index.search("   ")
    with pytest.raises(ValueError):
        index.search(Q, mode="magic")


def test_unknown_terms_only_bm25_returns_nothing_but_dense_still_ranks(index):
    assert index.search("zzzzqqqq", 3, "bm25") == []
    assert len(index.search("zzzzqqqq", 3, "dense")) == 3


@pytest.mark.parametrize("store", ["numpy", "faiss", "chroma"])
def test_save_load_roundtrip_and_store_equivalence(resumes, tmp_path, store):
    built = TalentIndex.build(resumes, "lsa:48", store, 80, 20, True, tmp_path / store)
    ref = [h.resume.id for h in built.search(Q, 5)]
    loaded = TalentIndex.load(tmp_path / store)
    assert [h.resume.id for h in loaded.search(Q, 5)] == ref
    assert loaded.config["store"] == store and len(loaded.resumes) == 45
    other = TalentIndex.load(tmp_path / store, store="numpy")  # same embeddings, different store
    assert [h.resume.id for h in other.search(Q, 5)] == ref


def test_load_missing_index_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build-index"):
        TalentIndex.load(tmp_path / "missing")


def test_rebuild_reuses_the_fitted_embedder(index, resumes):
    clone = index.rebuild(resumes)
    assert clone.embedder is index.embedder
    assert [h.resume.id for h in clone.search(Q, 5)] == [h.resume.id for h in index.search(Q, 5)]


def test_searcher_reranking_prefers_requirement_coverage(index):
    s = TalentSearch(index)
    plain = s.search(Q, 3, "dense", rerank_weight=0.0)
    reranked = s.search(Q, 3, "dense", rerank_weight=0.8)
    assert not plain.reranked and reranked.reranked
    assert [h.rank for h in reranked.hits] == [1, 2, 3]
    assert sum(c.score for c in reranked.coverages) >= sum(c.score for c in plain.coverages)
    assert reranked.coverages[0].score == 1.0 and reranked.spec.skills == ["SQL", "Tableau"]


def test_rerank_is_skipped_when_no_requirements_are_detected(index):
    res = TalentSearch(index).search("hello world", 3, "dense", rerank_weight=0.9)
    assert not res.reranked and len(res.hits) == 3


def test_searcher_explain_integration(index):
    res = TalentSearch(index).search(Q, 3, explain=True)
    assert res.explanation.backend.startswith("template") and len(res.explanation.candidates) == 3
    assert [c.resume_id for c in res.explanation.candidates] == [h.resume.id for h in res.hits]
