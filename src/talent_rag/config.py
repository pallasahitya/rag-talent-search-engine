"""Defaults and paths."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA = PROJECT_ROOT / "data" / "sample" / "resumes_sample.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_INDEX_DIR = DEFAULT_OUTPUT_DIR / "index"

CHUNK_WORDS = 120
CHUNK_OVERLAP = 30
TOP_K = 3  # the brief: pass the top 3 resumes to the LLM
RRF_K = 60  # reciprocal-rank-fusion constant

# Embedding presets: name -> (hub id, pooling, query prefix, document prefix)
EMBEDDING_PRESETS: dict[str, dict] = {
    "minilm": {"model": "sentence-transformers/all-MiniLM-L6-v2", "pooling": "mean", "query_prefix": "", "doc_prefix": ""},
    "bge-small": {
        "model": "BAAI/bge-small-en-v1.5", "pooling": "cls",
        "query_prefix": "Represent this sentence for searching relevant passages: ", "doc_prefix": "",
    },
    "e5-small": {"model": "intfloat/e5-small-v2", "pooling": "mean", "query_prefix": "query: ", "doc_prefix": "passage: "},
}

# LLM defaults (model names change often - override with --explain provider:model)
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
