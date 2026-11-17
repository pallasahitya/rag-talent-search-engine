"""Command line interface:  ``python -m talent_rag <command>``  (or ``talent-rag``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .config import DEFAULT_INDEX_DIR, DEFAULT_OUTPUT_DIR, EMBEDDING_PRESETS, RAW_DIR, SAMPLE_DATA, TOP_K
from .data import dataset_statistics, load_resumes
from .evaluate import build_benchmark, compare_configs, write_evaluation
from .index import MODES, TalentIndex
from .llm import make_llm
from .searcher import SearchResult, TalentSearch

DEFAULT_DATA = RAW_DIR / "Entity Recognition in Resumes.json"
EXAMPLE_QUERIES = [
    "Find me a Junior Data Analyst who knows SQL and Tableau",
    "Senior DevOps Engineer with 6+ years of experience in AWS and Kubernetes",
    "Machine learning engineer with PyTorch and NLP experience",
]


def result_to_dict(res: SearchResult, show_names: bool = True) -> dict:
    out = {
        "query": res.query, "mode": res.mode, "reranked": res.reranked,
        "requirements": res.spec.requirements(),
        "candidates": [],
    }
    for hit, cov in zip(res.hits, res.coverages):
        out["candidates"].append({
            "rank": hit.rank, "resume_id": hit.resume.id, **({"name": hit.resume.name} if show_names else {}),
            "role": hit.resume.title, "years": cov.years, "score": hit.score, "dense_rank": hit.dense_rank, "bm25_rank": hit.bm25_rank,
            "coverage": cov.score, "matched_skills": cov.matched_skills, "missing_skills": cov.missing_skills,
            "evidence": [{"section": e.section, "text": e.text, "similarity": e.score} for e in hit.evidence],
        })
    if res.explanation:
        ex = res.explanation
        out["explanation"] = {
            "backend": ex.backend, "overall": ex.overall, "grounded_ratio": ex.grounded_ratio, "fallback_reason": ex.fallback_reason,
            "candidates": [c.__dict__ for c in ex.candidates],
        }
    return out


def format_result(res: SearchResult, show_names: bool = True) -> str:
    lines = [f"\nQuery: {res.query}", f"Understood requirements: {', '.join(res.spec.requirements()) or '(none detected)'}  "
             f"[mode={res.mode}{', requirement re-ranking' if res.reranked else ''}]"]
    labels = {c.resume_id: c for c in (res.explanation.candidates if res.explanation else [])}
    for hit, cov in zip(res.hits, res.coverages):
        who = f"{hit.resume.name} ({hit.resume.id})" if show_names else hit.resume.id
        yrs = "?" if cov.years is None else f"{cov.years:g}y"
        lines.append(f"\n#{hit.rank}  {who}  - {hit.resume.title or 'role n/a'}, {yrs}   coverage {100 * cov.score:.0f}%")
        lines.append(f"    skills matched: {', '.join(cov.matched_skills) or '-'}   missing: {', '.join(cov.missing_skills) or '-'}")
        ex = labels.get(hit.resume.id)
        if ex:
            lines.append(f"    why [{ex.candidate_id}, fit {ex.fit_score}/5]: {ex.summary}")
            for g in ex.gaps[:3]:
                lines.append(f"      gap: {g}")
            for q in ex.evidence_quotes[:1]:
                lines.append(f"      evidence: \"{q}\"")
            for q in ex.ungrounded_quotes:
                lines.append(f"      WARNING ungrounded quote: \"{q}\"")
    if res.explanation:
        ex = res.explanation
        note = f"  (fallback: {ex.fallback_reason})" if ex.fallback_reason else ""
        gr = "" if ex.grounded_ratio is None else f", quotes grounded {100 * ex.grounded_ratio:.0f}%"
        lines.append(f"\nOverall [{ex.backend}{gr}]{note}: {ex.overall}")
    return "\n".join(lines)


def _load_index(args: argparse.Namespace) -> TalentIndex:
    return TalentIndex.load(args.index, getattr(args, "store", None))


def cmd_info(args: argparse.Namespace) -> int:
    import importlib

    print(f"talent-rag {__version__}")
    for mod in ("numpy", "pandas", "sklearn", "faiss", "chromadb", "torch", "transformers", "langchain_core", "streamlit"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:<15} {getattr(m, '__version__', 'installed')}")
        except ImportError:
            print(f"  {mod:<15} not installed")
    print("embedding presets:", ", ".join(f"{k} -> {v['model']}" for k, v in EMBEDDING_PRESETS.items()), "| plus 'lsa' (offline)")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    resumes = load_resumes(args.data, args.limit)
    st = dataset_statistics(resumes)
    for k, v in st.items():
        print(f"{k}: {round(v, 1) if isinstance(v, float) else v}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    resumes = load_resumes(args.data, args.limit)
    index = TalentIndex.build(resumes, args.embedder, args.store, args.chunk_words, args.chunk_overlap, not args.no_anonymize, args.out)
    print(f"indexed {index.config['n_resumes']} resumes -> {index.config['n_chunks']} chunks | embedder {index.embedder.name} | store {args.store} | "
          f"anonymised embeddings: {index.config['anonymize']}\nsaved to {Path(args.out).resolve()}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    index = _load_index(args)
    llm = make_llm(args.explain)
    explain = args.explain.lower() not in {"none", "off"}
    res = TalentSearch(index).search(args.query, args.k, args.mode, args.rerank, explain, llm)
    if args.json:
        print(json.dumps(result_to_dict(res, not args.hide_names), indent=2, ensure_ascii=False, default=float))
    else:
        print(format_result(res, not args.hide_names))
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    index = _load_index(args)
    search = TalentSearch(index)
    llm = make_llm(args.explain)
    explain = args.explain.lower() not in {"none", "off"}
    print(f"Talent search over {len(index.resumes)} resumes. Ask in plain language (empty line to quit).")
    while True:
        try:
            q = input("\nrecruiter> ").strip()
        except EOFError:
            break
        if not q:
            break
        print(format_result(search.search(q, args.k, args.mode, args.rerank, explain, llm), not args.hide_names))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    resumes = load_resumes(args.data, args.limit)
    from .query import SkillVocabulary

    bench = build_benchmark(resumes, SkillVocabulary.from_resumes(resumes), args.queries, args.seed)
    if not bench:
        print("error: could not generate benchmark queries (too few annotated resumes).", file=sys.stderr)
        return 2
    df = compare_configs(resumes, args.embedders, args.modes, bench, [True, False] if args.compare_anonymization else [True], args.rerank_weights, args.store)
    write_evaluation(df, bench, args.out, len(resumes))
    print(df.round(3).to_string(index=False))
    print(f"\nwritten to {Path(args.out).resolve()}")
    return 0


def cmd_bias(args: argparse.Namespace) -> int:
    from .bias import run_bias_audit, write_bias_report
    from .query import SkillVocabulary

    resumes = load_resumes(args.data, args.limit)
    bench = build_benchmark(resumes, SkillVocabulary.from_resumes(resumes), args.queries, args.seed)
    report = run_bias_audit(resumes, bench, args.embedder, args.k, args.mode, args.rerank)
    write_bias_report(report, args.out, args.k)
    print("FLAGS:" if report.flags else "No flag raised.")
    for f in report.flags:
        print(" -", f)
    for w in report.warnings:
        print(" note:", w)
    print(f"report written to {Path(args.out).resolve() / 'bias_report.md'}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Index the bundled sample, run example queries (+ explanations), evaluate retrieval and audit bias."""
    out = Path(args.out)
    resumes = load_resumes(SAMPLE_DATA)
    index = TalentIndex.build(resumes, args.embedder, args.store, anonymize_text=True, out_dir=out / "index")
    search = TalentSearch(index)
    results = [search.search(q, TOP_K, "hybrid", 0.5, explain=True) for q in EXAMPLE_QUERIES]
    (out / "search_results.json").write_text(json.dumps([result_to_dict(r) for r in results], indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (out / "search_results.md").write_text(
        "# Example searches (bundled SYNTHETIC sample, offline template explainer)\n\n" + "\n\n".join("```\n" + format_result(r) + "\n```" for r in results) + "\n",
        encoding="utf-8",
    )
    for r in results:
        print(format_result(r))
    ns = argparse.Namespace(data=str(SAMPLE_DATA), limit=0, queries=30, seed=0, embedders=[args.embedder], modes=list(MODES),
                            compare_anonymization=True, rerank_weights=[0.0, 0.5], store="numpy", out=str(out / "evaluation"))
    print("\n== retrieval evaluation ==")
    cmd_evaluate(ns)
    print("\n== bias audit ==")
    cmd_bias(argparse.Namespace(data=str(SAMPLE_DATA), limit=0, queries=30, seed=0, embedder=args.embedder, k=TOP_K, mode="hybrid", rerank=0.0, out=str(out / "bias")))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="talent-rag", description="RAG-powered talent search engine over resumes.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def data_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", default=str(DEFAULT_DATA), help="Kaggle JSON(L) / CSV / folder of .txt resumes")
        p.add_argument("--limit", type=int, default=0, help="only the first N resumes (0 = all)")

    def search_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--index", default=str(DEFAULT_INDEX_DIR))
        p.add_argument("--store", choices=["numpy", "faiss", "chroma"], default=None, help="vector store (default: the one the index was built with)")
        p.add_argument("-k", type=int, default=TOP_K, help="number of candidates (default 3 = what is passed to the LLM)")
        p.add_argument("--mode", choices=list(MODES), default="hybrid")
        p.add_argument("--rerank", type=float, default=0.5, help="weight of requirement coverage in re-ranking (0 = off)")
        p.add_argument("--explain", default="template", help="none | template | openai:<model> | gemini:<model> | ollama:<model>")
        p.add_argument("--hide-names", action="store_true", help="show resume ids only")

    sub.add_parser("info", help="installed libraries and embedding presets").set_defaults(func=cmd_info)

    p = sub.add_parser("stats", help="dataset statistics")
    data_args(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("build-index", help="chunk, embed and index the resumes")
    data_args(p)
    p.add_argument("--out", default=str(DEFAULT_INDEX_DIR))
    p.add_argument("--embedder", default="lsa", help="lsa (offline) | minilm | bge-small | e5-small | hf:<model>")
    p.add_argument("--store", choices=["numpy", "faiss", "chroma"], default="faiss")
    p.add_argument("--chunk-words", type=int, default=120)
    p.add_argument("--chunk-overlap", type=int, default=30)
    p.add_argument("--no-anonymize", action="store_true", help="embed the raw resume text (not recommended)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("search", help="find candidates for a natural-language query")
    p.add_argument("query")
    search_args(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("chat", help="interactive search in the terminal")
    search_args(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("evaluate", help="compare embedders / modes on an annotation-derived benchmark")
    data_args(p)
    p.add_argument("--embedders", nargs="+", default=["lsa"])
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    p.add_argument("--rerank-weights", nargs="+", type=float, default=[0.0, 0.5])
    p.add_argument("--compare-anonymization", action="store_true")
    p.add_argument("--queries", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--store", choices=["numpy", "faiss", "chroma"], default="numpy")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR / "evaluation"))
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("bias-audit", help="counterfactual + qualified-selection parity audit")
    data_args(p)
    p.add_argument("--embedder", default="lsa")
    p.add_argument("-k", type=int, default=TOP_K)
    p.add_argument("--mode", choices=list(MODES), default="hybrid")
    p.add_argument("--rerank", type=float, default=0.0)
    p.add_argument("--queries", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR / "bias"))
    p.set_defaults(func=cmd_bias)

    p = sub.add_parser("demo", help="run everything on the bundled synthetic sample (offline)")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR / "demo"))
    p.add_argument("--embedder", default="lsa")
    p.add_argument("--store", choices=["numpy", "faiss", "chroma"], default="faiss")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    pd.set_option("display.width", 180)
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
