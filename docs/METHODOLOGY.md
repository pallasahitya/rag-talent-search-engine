# Methodology

## Pipeline
```
resume text ──► (anonymise) ──► chunk (section-aware windows + profile chunk) ──► embed ──► vector store (FAISS | ChromaDB | NumPy)
                                                       └────────────────────────────► BM25 (resume level)
query ──► parse (skills, years, role) ──► dense search + BM25 ──► reciprocal-rank fusion ──► requirement re-ranking
      ──► top-3 candidates + evidence ──► LLM (or template) explanation ──► grounding check ──► answer
```

## 1. Chunking (`chunking.py`)
A resume is split by section headers (Summary, Skills, Experience, Education ...) and long sections into overlapping windows
(120 words, 30 overlap). A **profile chunk** built from the annotations (role, years, skills, degree, employers) is added first:
it lets short queries such as "data analyst SQL" match a compact, high-signal text. Sections that contain only redaction markers
are dropped. Each chunk keeps its resume id and section.

## 2. Embeddings (`embedders.py`)
* `lsa` - TF-IDF (1-2 grams) + truncated SVD, fitted on the corpus: offline, fast, deterministic; the default for tests/no internet.
* `minilm`, `bge-small`, `e5-small` or `hf:<model>` - real Hugging Face sentence encoders (mean/CLS pooling, query/document prefixes where the
  model requires them). All vectors are L2-normalised, so cosine similarity = dot product.
Real encoders need a one-time model download.

## 3. Vector databases (`stores.py`)
The same interface over **NumPy** (exact), **FAISS** (`IndexFlatIP`, exact inner product) and **ChromaDB** (persistent, cosine).
Tests verify that all three return identical rankings and survive save/load. Exact search is used on purpose: with a few hundred
resumes approximate indexes add error and no speed benefit.

## 4. Retrieval modes (`index.py`)
* **dense** - nearest chunks; a resume's score = its best chunk.
* **bm25** - own Okapi BM25 implementation over whole resumes (good for exact skill names such as "C++", "Node.js").
* **hybrid** (default) - **reciprocal rank fusion**: `score = sum(1 / (60 + rank))` over the two rankings. No score calibration needed and
  robust when one method misses.
Evidence snippets (the best-matching chunks) are returned with every hit.

## 5. Query understanding and re-ranking (`query.py`, `searcher.py`)
A skill vocabulary (built-in terms + skills mined from the annotated resumes, with aliases such as PowerBI -> Power BI) finds the requested
skills; seniority words (junior/senior/...) and "N+ years" / "3-5 years" become experience ranges; role phrases ("data analyst") become role terms.
Each candidate gets a **requirement coverage** (share of skill/experience/role checks that pass; unknown experience is not counted against it).
Optionally the final order blends retrieval rank and coverage: `(1 - w) * rank_score + w * coverage` (default `w = 0.5`; `--rerank 0` disables it).
Plain semantic search does not understand "junior", so this step keeps senior profiles out of a junior search.

## 6. LLM-based evaluation / explanation (`explain.py`, `llm.py`)
The top-3 candidates are passed to an LLM that explains *why* they fit. Design choices:
* **Privacy** - the LLM sees `Candidate A/B/C` with redacted evidence, never names, e-mails, phones or resume ids.
* **Safety instructions** - use only the evidence; say "not stated" when unknown; never infer or mention protected characteristics.
* **Structured output** - JSON with fit score (1-5), summary, matched requirements, gaps and **exact quotes**.
* **Grounding check** - each quote is verified to occur verbatim in the candidate's evidence; the share of grounded quotes is reported and
  hallucinated quotes are flagged in the UI/CLI.
* **Graceful fallback** - if the LLM is not configured, fails, or returns invalid JSON, a deterministic **template explainer** (no LLM) is used and the
  reason is shown. The template is clearly labelled and never pretends to be an LLM.
Backends (standard library HTTP, no SDK): OpenAI-compatible APIs, Google Gemini, and a local Ollama server (Llama 3). See `docs/LLM_SETUP.md`.

## 7. Evaluation (`evaluate.py`)
Retrieval is evaluated on an **annotation-derived benchmark**: queries are generated from skills and roles found in the resumes, and a resume
is *relevant* if it mentions all requested skills (and has the requested role). Metrics: precision@3, recall@10, hit@3, MRR, nDCG@10.
Caveats: relevance is defined by *mentions*, so BM25 is naturally favoured and re-ranking by coverage looks perfect by construction;
use the benchmark to compare configurations and catch regressions, not as absolute quality. Judge real quality with real recruiter queries.

## 8. Bias audit (`bias.py`)
Counterfactual test (names + pronouns swapped, same fitted embedder, rankings compared) and qualified-selection parity (selection rate of
gender-coded groups among qualified candidates, disparate-impact ratio and Fisher exact test). With anonymisation on, the counterfactual
texts are identical, so any difference must be exactly zero - the audit verifies this. Limits are listed in `docs/BIAS_AND_PRIVACY.md`.

## 9. Known limitations
* LSA is a lexical-semantic baseline; real sentence encoders should be used for production-quality semantic matching.
* Skill vocabulary and role parsing are English and tech/finance/HR/engineering oriented; add terms for other fields.
* Chunk windows are word-based; very unusual resume layouts may segment poorly.
* Retrieval cannot judge quality of experience, only what is written; it should assist, not replace, a human screener.
