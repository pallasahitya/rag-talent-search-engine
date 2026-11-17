"""Bias audit for the search engine (bonus feature).

Two complementary checks, both on the *whole retrieval pipeline*:

1. **Counterfactual test** - every resume with a gender-coded first name is rewritten with the counterpart name and swapped
   pronouns, a second index is built with the *same fitted embedder*, and the same queries are run on both. If the search
   does not depend on gender cues, rankings and scores must not change. (With anonymisation switched on the texts become
   identical, so the difference must be exactly zero - this verifies the mitigation.)
2. **Qualified-selection parity** - among the *qualified* candidates of each benchmark query (those the benchmark marks
   relevant) compare how often each gender-coded group is selected into the top-k; report the disparate-impact ratio
   (min/max selection rate; below 0.8 is flagged, the "four-fifths" rule of thumb) and a Fisher exact test.

LIMITS - read before trusting a green result: gender is *inferred from first names* with a small lexicon (cultures are
incompletely covered, many names are 'unknown'); samples are small; the audit cannot see proxies it was not given (employer,
college, location, writing style); and passing an audit is not legal or ethical clearance. Employment screening is a
high-risk use of AI: keep a human in the loop and follow the rules that apply to you.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .data import Resume
from .evaluate import BenchQuery
from .index import TalentIndex
from .lexicons import FEMALE_FIRST_NAMES, FEMALE_TO_MALE, MALE_FIRST_NAMES, MALE_TO_FEMALE
from .searcher import TalentSearch

FEMALE, MALE, UNKNOWN = "female-coded", "male-coded", "unknown"
_M, _F = {n.lower() for n in MALE_FIRST_NAMES}, {n.lower() for n in FEMALE_FIRST_NAMES}
_PAIR_M2F = dict(zip(MALE_FIRST_NAMES, FEMALE_FIRST_NAMES))
_PAIR_F2M = dict(zip(FEMALE_FIRST_NAMES, MALE_FIRST_NAMES))
DISPARATE_IMPACT_THRESHOLD = 0.8
MIN_QUALIFIED_PER_GROUP = 5


def first_name(resume: Resume) -> str:
    name = resume.name
    return name.split()[0] if name and name != resume.id else ""


def infer_group(resume: Resume) -> str:
    """Gender-coded group from the first name (AUDIT ONLY - never used for ranking)."""
    f = first_name(resume).lower()
    return MALE if f in _M else FEMALE if f in _F else UNKNOWN


def _swap_case(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    return repl.capitalize() if src[:1].isupper() else repl


def swap_gender(resume: Resume) -> Resume | None:
    """Counterfactual copy with the counterpart first name, e-mail local part and swapped pronouns (``None`` if name unknown)."""
    group, first = infer_group(resume), first_name(resume)
    if group == UNKNOWN:
        return None
    new_first = (_PAIR_M2F if group == MALE else _PAIR_F2M)[next(n for n in (MALE_FIRST_NAMES + FEMALE_FIRST_NAMES) if n.lower() == first.lower())]
    mapping = MALE_TO_FEMALE if group == MALE else FEMALE_TO_MALE
    # e-mail first (keeps the lower-case local part), then the exact-case / UPPER-case name occurrences
    text = re.sub(
        r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
        lambda m: re.sub(rf"^{re.escape(first.lower())}(?=[._\d])", new_first.lower(), m.group(0), flags=re.IGNORECASE),
        resume.text,
    )
    text = re.sub(rf"\b{re.escape(first)}\b", new_first, text)
    text = re.sub(rf"\b{re.escape(first.upper())}\b", new_first.upper(), text)
    text = re.sub(r"\b(" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r")(?=\b|\.)", lambda m: _swap_case(m.group(0), mapping[m.group(0).lower()]), text, flags=re.IGNORECASE)
    entities = {k: [re.sub(rf"\b{re.escape(first)}\b", new_first, v, flags=re.IGNORECASE) if k == "Name" else v for v in vals] for k, vals in resume.entities.items()}
    return Resume(resume.id, text, entities, [], resume.category)


@dataclass
class CounterfactualResult:
    anonymized: bool
    swapped_resumes: int
    queries: int
    top_k_overlap: float  # mean Jaccard of the top-k sets (1.0 = identical)
    topk_changed_queries: float  # share of queries whose top-k set changed
    spearman: float  # mean rank correlation over the full ranking
    mean_abs_score_delta: float
    max_abs_score_delta: float
    delta_m2f: float  # mean score change of resumes that were male-coded -> female-coded
    delta_f2m: float
    per_query: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def severity(self) -> str:
        """``ranking`` if the top-k changed, ``scores`` if only similarity scores moved, else ``none``."""
        if self.topk_changed_queries > 0:
            return "ranking"
        return "scores" if self.max_abs_score_delta > 1e-6 else "none"

    def flagged(self) -> bool:
        return self.severity != "none"


def counterfactual_audit(index: TalentIndex, queries: list[str], k: int = 3, mode: str = "dense") -> CounterfactualResult:
    swapped = {}
    cf_resumes = []
    for r in index.resumes:
        cf = swap_gender(r)
        if cf is not None:
            swapped[r.id] = infer_group(r)
        cf_resumes.append(cf or r)
    cf_index = index.rebuild(cf_resumes)
    rows, deltas_m2f, deltas_f2m = [], [], []
    for q in queries:
        a = index.search(q, len(index.resumes), mode, n_evidence=1)
        b = cf_index.search(q, len(index.resumes), mode, n_evidence=1)
        sa, sb = {h.resume.id: h.score for h in a}, {h.resume.id: h.score for h in b}
        ra, rb = {h.resume.id: h.rank for h in a}, {h.resume.id: h.rank for h in b}
        ids = [i for i in ra if i in rb]
        top_a, top_b = {h.resume.id for h in a[:k]}, {h.resume.id for h in b[:k]}
        rho = stats.spearmanr([ra[i] for i in ids], [rb[i] for i in ids]).statistic if len(ids) > 2 else 1.0
        d = np.array([sb[i] - sa[i] for i in ids if i in sa and i in sb])
        for i in ids:
            if i in swapped:
                (deltas_m2f if swapped[i] == MALE else deltas_f2m).append(sb[i] - sa[i])
        rows.append({
            "query": q, "top_k_jaccard": len(top_a & top_b) / len(top_a | top_b) if top_a | top_b else 1.0,
            "top_k_changed": top_a != top_b, "spearman": 1.0 if np.isnan(rho) else float(rho),
            "mean_abs_delta": float(np.abs(d).mean()) if len(d) else 0.0, "max_abs_delta": float(np.abs(d).max()) if len(d) else 0.0,
        })
    pq = pd.DataFrame(rows)
    return CounterfactualResult(
        index.config["anonymize"], len(swapped), len(queries), float(pq["top_k_jaccard"].mean()), float(pq["top_k_changed"].mean()),
        float(pq["spearman"].mean()), float(pq["mean_abs_delta"].mean()), float(pq["max_abs_delta"].max()),
        float(np.mean(deltas_m2f)) if deltas_m2f else 0.0, float(np.mean(deltas_f2m)) if deltas_f2m else 0.0, pq,
    )


@dataclass
class ParityResult:
    table: pd.DataFrame
    disparate_impact: float | None
    p_value: float | None
    flagged: bool
    notes: list[str]


def qualified_selection_parity(
    index: TalentIndex, benchmark: list[BenchQuery], k: int = 3, mode: str = "hybrid", rerank_weight: float = 0.0,
) -> ParityResult:
    """Selection rate of each group among qualified candidates (top-k membership), pooled over the benchmark queries."""
    groups = {r.id: infer_group(r) for r in index.resumes}
    searcher = TalentSearch(index)
    qualified = {FEMALE: 0, MALE: 0, UNKNOWN: 0}
    selected = {FEMALE: 0, MALE: 0, UNKNOWN: 0}
    for q in benchmark:
        top = set(searcher.ranked_ids(q.text, k, mode, rerank_weight))
        for rid in q.relevant:
            g = groups.get(rid, UNKNOWN)
            qualified[g] += 1
            selected[g] += rid in top
    table = pd.DataFrame([
        {"group": g, "qualified_slots": qualified[g], "selected": selected[g], "selection_rate": selected[g] / qualified[g] if qualified[g] else float("nan")}
        for g in (FEMALE, MALE, UNKNOWN)
    ])
    notes = []
    di = p = None
    f, m = table.set_index("group").loc[FEMALE], table.set_index("group").loc[MALE]
    if min(f.qualified_slots, m.qualified_slots) < MIN_QUALIFIED_PER_GROUP:
        notes.append(f"Too few qualified candidates per group (< {MIN_QUALIFIED_PER_GROUP}) for a meaningful parity test.")
    elif max(f.selection_rate, m.selection_rate) > 0:
        di = min(f.selection_rate, m.selection_rate) / max(f.selection_rate, m.selection_rate)
        p = float(stats.fisher_exact([[f.selected, f.qualified_slots - f.selected], [m.selected, m.qualified_slots - m.selected]])[1])
    flagged = di is not None and di < DISPARATE_IMPACT_THRESHOLD and (p is not None and p < 0.05)
    if di is not None and di < DISPARATE_IMPACT_THRESHOLD and not flagged:
        notes.append("Selection rates differ by more than the four-fifths rule of thumb but the difference is not statistically significant (small sample).")
    if table.set_index("group").loc[UNKNOWN, "qualified_slots"] > 0.5 * table["qualified_slots"].sum():
        notes.append("More than half of the qualified candidates have names that are not in the gender lexicon ('unknown'): the audit covers a minority of the pool.")
    return ParityResult(table, di, p, flagged, notes)


@dataclass
class BiasReport:
    counterfactual: list[CounterfactualResult]
    parity: ParityResult
    flags: list[str]
    warnings: list[str]


def run_bias_audit(
    resumes: list[Resume],
    benchmark: list[BenchQuery],
    embedder: str = "lsa",
    k: int = 3,
    mode: str = "hybrid",
    rerank_weight: float = 0.0,
    progress=print,
) -> BiasReport:
    """Audit with anonymisation ON and OFF so the effect of the mitigation is visible."""
    queries = [q.text for q in benchmark]
    cf: list[CounterfactualResult] = []
    parity = None
    for anon in (True, False):
        progress(f"auditing index with anonymisation={'on' if anon else 'off'} ...")
        index = TalentIndex.build(resumes, embedder, "numpy", anonymize_text=anon)
        cf.append(counterfactual_audit(index, queries, k, "dense"))
        if anon:
            parity = qualified_selection_parity(index, benchmark, k, mode, rerank_weight)
    flags, warns = [], []
    for res in cf:
        label = "ON" if res.anonymized else "OFF"
        if res.severity == "ranking":
            flags.append(f"Anonymisation {label}: swapping gendered names/pronouns CHANGED THE TOP-{k} for {100 * res.topk_changed_queries:.0f}% of queries "
                         f"(max score change {res.max_abs_score_delta:.4f}).")
        elif res.severity == "scores":
            flags.append(f"Anonymisation {label}: weak signal - gendered cues move similarity scores (max change {res.max_abs_score_delta:.4f}) "
                         f"but the top-{k} did not change for any query.")
    if parity.flagged:
        flags.append(f"Qualified-selection parity: disparate-impact ratio {parity.disparate_impact:.2f} (< {DISPARATE_IMPACT_THRESHOLD}, p = {parity.p_value:.3f}).")
    warns.extend(parity.notes)
    warns.append("Group membership is inferred from first names with a small lexicon; this is an audit aid, not a verdict.")
    return BiasReport(cf, parity, flags, warns)


def write_bias_report(report: BiasReport, out_dir: str | Path, k: int = 3) -> None:
    from .evaluate import markdown_table
    from .visualize import plot_bias

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [{
        "anonymisation": "on" if r.anonymized else "off", "swapped_resumes": r.swapped_resumes, "queries": r.queries,
        f"top{k}_overlap": r.top_k_overlap, "queries_with_changed_topk": r.topk_changed_queries, "spearman": r.spearman,
        "mean_abs_score_delta": r.mean_abs_score_delta, "max_abs_score_delta": r.max_abs_score_delta,
        "delta_male_to_female": r.delta_m2f, "delta_female_to_male": r.delta_f2m,
    } for r in report.counterfactual]
    cf_df = pd.DataFrame(rows)
    cf_df.to_csv(out / "bias_counterfactual.csv", index=False)
    report.parity.table.to_csv(out / "bias_parity.csv", index=False)
    payload = {
        "flags": report.flags, "warnings": report.warnings, "counterfactual": rows,
        "parity": {"table": report.parity.table.to_dict("records"), "disparate_impact": report.parity.disparate_impact,
                   "p_value": report.parity.p_value, "flagged": report.parity.flagged},
    }
    (out / "bias_report.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    plot_bias(report, out / "bias_report.png")
    status = "⚠️ FLAGGED" if report.flags else "✅ no flag raised"
    di = "n/a" if report.parity.disparate_impact is None else f"{report.parity.disparate_impact:.2f}"
    lines = [
        "# Bias audit", "", f"**Status: {status}**", "",
        *( [f"* {f}" for f in report.flags] or ["* No check exceeded its threshold on this data."] ), "",
        "## 1. Counterfactual test (names + pronouns swapped, same fitted embedder)", "",
        markdown_table(cf_df, "{:.4f}"), "",
        "A value of 1.0 overlap / 0.0 delta means the search ignored the swap. Anonymisation ON should give exactly that.", "",
        "## 2. Qualified-selection parity (top-%d membership among qualified candidates)" % k, "",
        markdown_table(report.parity.table), "",
        f"Disparate-impact ratio (female-coded vs male-coded): **{di}**" + ("" if report.parity.p_value is None else f" | Fisher exact p = {report.parity.p_value:.3f}"), "",
        "## Warnings and limits", "", *[f"* {w}" for w in report.warnings], "",
        "This audit is an aid for human review. It is not legal or ethical clearance; automated hiring tools are high-risk and need human oversight.", "",
    ]
    (out / "bias_report.md").write_text("\n".join(lines), encoding="utf-8")
