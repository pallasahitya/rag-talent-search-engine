"""Streamlit chat UI for the talent search engine.

Run:  streamlit run app/streamlit_app.py
Set TALENT_INDEX to use another index folder (default: outputs/demo/index, built by `python -m talent_rag demo`).
LLM keys are read from the environment: OPENAI_API_KEY / GEMINI_API_KEY; Ollama needs a running local server.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from talent_rag.config import DEFAULT_OUTPUT_DIR, SAMPLE_DATA
from talent_rag.data import load_resumes
from talent_rag.index import MODES, TalentIndex
from talent_rag.llm import make_llm
from talent_rag.searcher import TalentSearch

st.set_page_config(page_title="RAG Talent Search", page_icon="🧭", layout="wide")

DEFAULT_INDEX = os.environ.get("TALENT_INDEX", str(DEFAULT_OUTPUT_DIR / "demo" / "index"))


@st.cache_resource(show_spinner="Loading index ...")
def load_index(path: str) -> TalentIndex:
    return TalentIndex.load(path)


def build_demo_index(path: str) -> None:
    TalentIndex.build(load_resumes(SAMPLE_DATA), "lsa", "numpy", anonymize_text=True, out_dir=path)


st.title("🧭 RAG-powered talent search")
st.caption("Ask in plain language, e.g. *Find me a Junior Data Analyst who knows SQL and Tableau*. "
           "Candidates come from a vector database; explanations are grounded in redacted resume text.")

with st.sidebar:
    st.header("Settings")
    index_path = st.text_input("Index folder", value=DEFAULT_INDEX)
    mode = st.selectbox("Retrieval mode", list(MODES), index=list(MODES).index("hybrid"))
    k = st.slider("Candidates (top-k)", 1, 10, 3)
    rerank = st.slider("Requirement re-ranking weight", 0.0, 1.0, 0.5, 0.1, help="0 = pure retrieval order; higher = favour candidates that meet the requested skills / years / role")
    backend = st.selectbox("Explanation", ["template (offline, no LLM)", "openai", "gemini", "ollama (local)"])
    model_name = st.text_input("Model name", value="", placeholder="e.g. gpt-4o-mini / gemini-2.5-flash / llama3.1:8b") if backend != "template (offline, no LLM)" else ""
    show_names = st.checkbox("Show candidate names", value=True, help="Names are never sent to the LLM.")
    if st.button("Clear chat"):
        st.session_state["messages"] = []

if not (Path(index_path) / "config.json").exists():
    st.warning(f"No index found in `{index_path}`.")
    if st.button("Build a demo index from the bundled synthetic sample"):
        build_demo_index(index_path)
        st.rerun()
    st.stop()

index = load_index(index_path)
searcher = TalentSearch(index)
st.sidebar.caption(f"{len(index.resumes)} resumes · {len(index.chunks)} chunks · embedder {index.embedder.name} · store {index.config['store']} · anonymised embeddings: {index.config['anonymize']}")


def llm_from_settings():
    if backend.startswith("template"):
        return None
    provider = backend.split()[0]
    return make_llm(f"{provider}:{model_name}" if model_name else provider)


def render_result(res, show_names: bool) -> None:
    st.markdown(f"**Understood requirements:** {', '.join(res.spec.requirements()) or '(none detected)'}")
    labels = {c.resume_id: c for c in (res.explanation.candidates if res.explanation else [])}
    for hit, cov in zip(res.hits, res.coverages):
        who = f"{hit.resume.name} ({hit.resume.id})" if show_names else hit.resume.id
        yrs = "?" if cov.years is None else f"{cov.years:g} y"
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**#{hit.rank} · {who}** — {hit.resume.title or 'role n/a'} · {yrs}")
            c2.metric("Coverage", f"{100 * cov.score:.0f}%")
            st.markdown(f"✅ matched: {', '.join(cov.matched_skills) or '—'} &nbsp;&nbsp; ❌ missing: {', '.join(cov.missing_skills) or '—'}")
            ex = labels.get(hit.resume.id)
            if ex:
                st.markdown(f"**Why (fit {ex.fit_score}/5):** {ex.summary}")
                for g in ex.gaps[:3]:
                    st.markdown(f"- gap: {g}")
                for q in ex.evidence_quotes[:1]:
                    st.markdown(f"> {q}")
                for q in ex.ungrounded_quotes:
                    st.warning(f"Ungrounded quote (not found in the resume evidence): {q}")
            with st.expander("Evidence snippets"):
                for e in hit.evidence:
                    st.caption(f"{e.section} · similarity {e.score:.2f}")
                    st.text(e.text)
    if res.explanation:
        ex = res.explanation
        extra = f" · quotes grounded {100 * ex.grounded_ratio:.0f}%" if ex.grounded_ratio is not None else ""
        st.info(f"**Overall** ({ex.backend}{extra}): {ex.overall}")
        if ex.fallback_reason:
            st.warning(f"Fell back to the offline explainer — {ex.fallback_reason}")


st.session_state.setdefault("messages", [])
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_result(msg["result"], msg["show_names"])

if query := st.chat_input("Describe the candidate you are looking for ..."):
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        try:
            result = searcher.search(query, k, mode, rerank, explain=True, llm=llm_from_settings())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Search failed: {exc}")
        else:
            render_result(result, show_names)
            st.session_state["messages"].append({"role": "assistant", "result": result, "show_names": show_names})

with st.expander("⚖️ About fairness and privacy"):
    st.markdown(
        "* Names, e-mails, phones, pronouns and personal-detail lines are **redacted** before embedding and before anything is sent to an LLM.\n"
        "* Run `python -m talent_rag bias-audit` to test whether gendered names/pronouns change the results.\n"
        "* This tool supports a human decision; it must not be the only basis for rejecting anyone."
    )
