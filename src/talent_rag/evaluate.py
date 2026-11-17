"""Retrieval benchmark generated from the resume annotations, and configuration comparison.

For real data this gives *annotation-derived* relevance judgements:

* **skill queries** - "Looking for someone with SQL and Tableau": relevant = resumes that mention both skills;
* **role + skill queries** - "Find me a data analyst who knows SQL": relevant = resumes whose Designation matches the role
  *and* that mention the skill.

IMPORTANT: because relevance is defined by skill/role *mentions*, lexical methods (BM25) have a natural advantage. Treat the
numbers as a way to compare configurations and catch regressions - not as absolute quality, and not as a substitute for
judging real recruiter queries.
"""

from __future__ import annotations

import itertools
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Resume
from .index import TalentIndex
from .query import SkillVocabulary
from .searcher import TalentSearch

SENIORITY_WORDS = {"junior", "senior", "sr", "jr", "lead", "principal", "associate", "trainee", "intern", "graduate", "chief", "head", "assistant", "staff"}
ROLE_NOUN_RE = re.compile(
    r"(analyst|engineer|developer|manager|accountant|designer|scientist|administrator|consultant|architect|officer|executive|specialist|technician|programmer|tester)\b",
    re.IGNORECASE,
)


@dataclass
class BenchQuery:
    text: str
    kind: str  # "skills" | "role+skill"
    skills: list[str]
    role: str | None
    relevant: set[str] = field(default_factory=set)


def role_phrase(designation: str) -> str | None:
    """'Senior Java Developer' -> 'java developer'; ``None`` if it has no recognised role noun."""
    words = [w for w in re.findall(r"[A-Za-z+#.]+", designation.lower()) if w not in SENIORITY_WORDS]
    for i, w in enumerate(words):
        if ROLE_NOUN_RE.fullmatch(w):
            return " ".join(words[max(0, i - 2) : i + 1])
    return None


def build_benchmark(
    resumes: list[Resume], vocab: SkillVocabulary, n_queries: int = 40, seed: int = 0, min_relevant: int = 3, max_fraction: float = 0.5,
) -> list[BenchQuery]:
    rng = random.Random(seed)
    skills = {r.id: {s.lower() for s in vocab.find(r.text)} for r in resumes}
    display = {s.lower(): s for r in resumes for s in vocab.find(r.text)}
    roles: dict[str, set[str]] = {}
    for r in resumes:
        for d in r.designations:
            rp = role_phrase(d)
            if rp:
                roles.setdefault(rp, set()).add(r.id)
    n = len(resumes)
    ok = lambda rel: min_relevant <= len(rel) <= max(min_relevant, int(max_fraction * n))  # noqa: E731

    candidates: list[BenchQuery] = []
    counts: dict[tuple[str, str], set[str]] = {}
    for rid, ss in skills.items():
        for a, b in itertools.combinations(sorted(ss), 2):
            counts.setdefault((a, b), set()).add(rid)
    skill_templates = ["Looking for someone with experience in {a} and {b}", "Find me a candidate who knows {a} and {b}", "Candidate with {a} and {b} skills"]
    for (a, b), rel in sorted(counts.items()):
        if ok(rel):
            tpl = skill_templates[(len(a) + len(b)) % len(skill_templates)]  # deterministic template choice
            candidates.append(BenchQuery(tpl.format(a=display[a], b=display[b]), "skills", [display[a], display[b]], None, set(rel)))
    role_templates = ["Find me a {role} who knows {s}", "{role} with {s} experience", "Looking for a {role} skilled in {s}"]
    for rp, rids in sorted(roles.items()):
        if len(rids) < min_relevant:
            continue
        skill_counts: dict[str, int] = {}
        for rid in rids:
            for s in skills[rid]:
                skill_counts[s] = skill_counts.get(s, 0) + 1
        for s, c in sorted(skill_counts.items()):
            rel = {rid for rid in rids if s in skills[rid]}
            if c >= 2 and len(rel) < len(rids) and ok(rel):
                tpl = role_templates[(len(rp) + len(s)) % len(role_templates)]
                candidates.append(BenchQuery(tpl.format(role=rp.title() if tpl[0] == "{" else rp, s=display[s]), "role+skill", [display[s]], rp, rel))
    rng.shuffle(candidates)
    seen, out = set(), []
    for q in candidates:
        key = (tuple(sorted(q.relevant)), q.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= n_queries:
            break
    return out


def retrieval_metrics(ranked: list[str], relevant: set[str], ks: tuple[int, int, int] = (3, 10, 10)) -> dict[str, float]:
    """precision@3, recall@10, hit@3, MRR, nDCG@10 (binary relevance)."""
    k_p, k_r, k_n = ks
    rel = [1 if r in relevant else 0 for r in ranked]
    precision = sum(rel[:k_p]) / k_p
    recall = sum(rel[:k_r]) / len(relevant) if relevant else 0.0
    hit = float(any(rel[:k_p]))
    mrr = next((1.0 / (i + 1) for i, v in enumerate(rel) if v), 0.0)
    dcg = sum(v / np.log2(i + 2) for i, v in enumerate(rel[:k_n]))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k_n)))
    return {"precision@3": precision, "recall@10": recall, "hit@3": hit, "mrr": mrr, "ndcg@10": dcg / idcg if idcg else 0.0}


def evaluate_index(index: TalentIndex, benchmark: list[BenchQuery], mode: str = "hybrid", rerank_weight: float = 0.0) -> tuple[dict, pd.DataFrame]:
    """Mean metrics (+ latency) and a per-query table for one index / mode / re-rank setting."""
    searcher = TalentSearch(index)
    rows = []
    t0 = time.perf_counter()
    for q in benchmark:
        ranked = searcher.ranked_ids(q.text, 10, mode, rerank_weight)
        rows.append({"query": q.text, "kind": q.kind, "relevant": len(q.relevant), **retrieval_metrics(ranked, q.relevant)})
    seconds = time.perf_counter() - t0
    per_query = pd.DataFrame(rows)
    means = per_query[["precision@3", "recall@10", "hit@3", "mrr", "ndcg@10"]].mean().to_dict()
    means["ms_per_query"] = 1000 * seconds / max(len(benchmark), 1)
    return means, per_query


def compare_configs(
    resumes: list[Resume],
    embedders: list[str],
    modes: list[str],
    benchmark: list[BenchQuery],
    anonymize_options: list[bool] | None = None,
    rerank_weights: list[float] | None = None,
    store: str = "numpy",
    progress=print,
) -> pd.DataFrame:
    """Evaluate every combination of embedder x anonymisation x mode x re-rank weight on the same benchmark."""
    rows = []
    for emb in embedders:
        for anon in anonymize_options or [True]:
            progress(f"building index: embedder={emb} anonymize={anon}")
            index = TalentIndex.build(resumes, emb, store, anonymize_text=anon)
            for mode in modes:
                for w in rerank_weights or [0.0]:
                    means, _ = evaluate_index(index, benchmark, mode, w)
                    rows.append({"embedder": emb, "anonymized": anon, "mode": mode, "rerank_weight": w, **means})
    return pd.DataFrame(rows).sort_values(["ndcg@10", "mrr"], ascending=False).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def write_evaluation(df: pd.DataFrame, benchmark: list[BenchQuery], out_dir: str | Path, n_resumes: int) -> None:
    from .visualize import plot_config_comparison

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "retrieval_comparison.csv", index=False)
    pd.DataFrame([{"query": q.text, "kind": q.kind, "relevant_resumes": len(q.relevant)} for q in benchmark]).to_csv(out / "benchmark_queries.csv", index=False)
    plot_config_comparison(df, out / "retrieval_comparison.png")
    kinds = pd.Series([q.kind for q in benchmark]).value_counts().to_dict()
    (out / "retrieval_report.md").write_text(
        "\n".join([
            "# Retrieval evaluation", "",
            f"{len(benchmark)} auto-generated queries ({kinds}) over {n_resumes} resumes; relevance = the resume mentions the requested skills / has the requested role.", "",
            "> Relevance is defined by skill/role mentions, so lexical methods have a natural advantage. "
            "Use this to compare configurations, not as absolute quality.", "",
            markdown_table(df.drop(columns=["rerank_weight"]) if df["rerank_weight"].nunique() == 1 else df), "",
        ]),
        encoding="utf-8",
    )
