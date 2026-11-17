import math

import pytest

from talent_rag.evaluate import (
    build_benchmark,
    compare_configs,
    evaluate_index,
    markdown_table,
    retrieval_metrics,
    role_phrase,
    write_evaluation,
)


def test_role_phrase():
    assert role_phrase("Senior Java Developer") == "java developer"
    assert role_phrase("Junior Data Analyst") == "data analyst"
    assert role_phrase("Machine Learning Engineer") == "machine learning engineer"
    assert role_phrase("Chief Cook") is None and role_phrase("") is None


def test_metrics_with_hand_computed_values():
    m = retrieval_metrics(["a", "x", "b", "y", "z"], {"a", "b", "c"})
    assert m["precision@3"] == pytest.approx(2 / 3) and m["hit@3"] == 1.0 and m["mrr"] == 1.0
    assert m["recall@10"] == pytest.approx(2 / 3)
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert m["ndcg@10"] == pytest.approx(dcg / idcg)
    miss = retrieval_metrics(["x", "y", "z", "a"], {"a"})
    assert miss["hit@3"] == 0.0 and miss["mrr"] == pytest.approx(0.25) and miss["precision@3"] == 0.0
    assert retrieval_metrics(["a"], set())["recall@10"] == 0.0


def test_benchmark_is_valid_deterministic_and_diverse(resumes, index, benchmark):
    assert len(benchmark) == 30 and {q.kind for q in benchmark} == {"skills", "role+skill"}
    again = build_benchmark(resumes, index.vocab, 30, seed=0)
    assert [q.text for q in again] == [q.text for q in benchmark]
    assert [q.text for q in build_benchmark(resumes, index.vocab, 30, seed=1)] != [q.text for q in benchmark]
    by_id = {r.id: r for r in resumes}
    for q in benchmark:
        assert 3 <= len(q.relevant) <= 22
        for rid in q.relevant:  # relevance judgements are true by construction
            have = {s.lower() for s in index.vocab.find(by_id[rid].text)}
            assert all(s.lower() in have for s in q.skills)
            if q.role:
                assert any(role_phrase(d) == q.role for d in by_id[rid].designations)


def test_evaluate_index_returns_means_and_per_query_table(index, benchmark):
    means, per_query = evaluate_index(index, benchmark, "hybrid", 0.0)
    assert len(per_query) == len(benchmark) and set(means) >= {"precision@3", "recall@10", "mrr", "ndcg@10", "ms_per_query"}
    assert means["mrr"] > 0.8 and means["ndcg@10"] > 0.8 and means["ms_per_query"] > 0


def test_lexical_beats_dense_on_a_mention_defined_benchmark_and_rerank_helps(index, benchmark):
    dense, _ = evaluate_index(index, benchmark, "dense", 0.0)
    bm25, _ = evaluate_index(index, benchmark, "bm25", 0.0)
    rerank, _ = evaluate_index(index, benchmark, "dense", 0.5)
    assert bm25["ndcg@10"] >= dense["ndcg@10"]  # relevance = skill mention, so lexical matching is favoured (documented caveat)
    assert rerank["ndcg@10"] >= dense["ndcg@10"]


def test_compare_configs_and_report(resumes, benchmark, tmp_path):
    df = compare_configs(resumes, ["lsa:32"], ["dense", "bm25"], benchmark[:10], [True, False], [0.0, 0.5], progress=lambda *_: None)
    assert len(df) == 2 * 2 * 2 and df["ndcg@10"].is_monotonic_decreasing
    write_evaluation(df, benchmark[:10], tmp_path, len(resumes))
    for name in ("retrieval_comparison.csv", "retrieval_comparison.png", "retrieval_report.md", "benchmark_queries.csv"):
        assert (tmp_path / name).exists(), name
    assert "Relevance is defined by skill/role mentions" in (tmp_path / "retrieval_report.md").read_text()
    assert markdown_table(df.head(1)).count("|") > 10
