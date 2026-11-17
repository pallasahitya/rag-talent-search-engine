"""Text embedders. All return L2-normalised float32 vectors (cosine similarity = dot product).

* :class:`LsaEmbedder`  - TF-IDF + truncated SVD (latent semantic analysis). Fully offline, fits on your corpus in
  seconds; a strong lexical-semantic baseline and the default for tests / no-internet use.
* :class:`HFEmbedder`   - any Hugging Face encoder with mean or CLS pooling (e.g. ``all-MiniLM-L6-v2``, ``bge-small``,
  ``e5-small``): real *semantic* embeddings. Needs ``torch`` + ``transformers`` and a one-time model download.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .config import EMBEDDING_PRESETS


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


class Embedder(ABC):
    name: str = "embedder"
    needs_fit: bool = False

    @property
    def fitted(self) -> bool:
        return True

    def fit(self, texts: list[str]) -> Embedder:
        return self

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...

    @abstractmethod
    def spec(self) -> dict: ...

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "embedder.json").write_text(json.dumps(self.spec(), indent=2), encoding="utf-8")


class LsaEmbedder(Embedder):
    """TF-IDF (1-2 grams, sub-linear tf) followed by TruncatedSVD, then L2 normalisation."""

    needs_fit = True

    def __init__(self, dim: int = 128, ngram_range: tuple[int, int] = (1, 2), min_df: int = 1, random_state: int = 0) -> None:
        self.dim, self.ngram_range, self.min_df, self.random_state = dim, tuple(ngram_range), min_df, random_state
        self.name = f"lsa-{dim}"
        self.vectorizer = None
        self.svd = None

    @property
    def fitted(self) -> bool:
        return self.vectorizer is not None

    def fit(self, texts: list[str]) -> LsaEmbedder:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            lowercase=True, token_pattern=r"(?u)\b[\w+#.]*\w[+#]*\b", ngram_range=self.ngram_range,
            sublinear_tf=True, min_df=self.min_df, stop_words="english",
        )
        x = self.vectorizer.fit_transform(texts)
        k = max(2, min(self.dim, x.shape[1] - 1, x.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        self.svd.fit(x)
        return self

    def _transform(self, texts: list[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("LsaEmbedder must be fitted on the corpus before use.")
        return _normalize(self.svd.transform(self.vectorizer.transform(texts)))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._transform(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._transform([text])[0]

    def spec(self) -> dict:
        return {"type": "lsa", "dim": self.dim, "ngram_range": list(self.ngram_range), "min_df": self.min_df}

    def save(self, directory: Path) -> None:
        import joblib

        super().save(directory)
        joblib.dump({"vectorizer": self.vectorizer, "svd": self.svd}, directory / "lsa.joblib")

    @classmethod
    def load(cls, directory: Path) -> LsaEmbedder:
        import joblib

        spec = json.loads((directory / "embedder.json").read_text(encoding="utf-8"))
        obj = cls(spec["dim"], tuple(spec["ngram_range"]), spec["min_df"])
        parts = joblib.load(directory / "lsa.joblib")  # only load indexes you built yourself (pickle)
        obj.vectorizer, obj.svd = parts["vectorizer"], parts["svd"]
        return obj


class HFEmbedder(Embedder):
    """Sentence embeddings from a Hugging Face encoder (mean or CLS pooling)."""

    def __init__(
        self,
        model: str,
        pooling: str = "mean",
        query_prefix: str = "",
        doc_prefix: str = "",
        max_length: int = 256,
        batch_size: int = 32,
        device: str = "",
        name: str | None = None,
    ) -> None:
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name, self.pooling = model, pooling
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.max_length, self.batch_size = max_length, batch_size
        self.name = name or model.split("/")[-1]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.device).eval()

    def _encode(self, texts: list[str]) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(texts[i : i + self.batch_size], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
                hidden = self.model(**batch).last_hidden_state
                if self.pooling == "cls":
                    vec = hidden[:, 0]
                else:
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(vec.float().cpu().numpy())
        return _normalize(np.concatenate(out))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]

    def spec(self) -> dict:
        return {
            "type": "hf", "model": self.model_name, "pooling": self.pooling, "query_prefix": self.query_prefix,
            "doc_prefix": self.doc_prefix, "max_length": self.max_length, "name": self.name,
        }

    @classmethod
    def load(cls, directory: Path) -> HFEmbedder:
        spec = json.loads((directory / "embedder.json").read_text(encoding="utf-8"))
        spec.pop("type")
        return cls(**spec)


def get_embedder(spec: str = "lsa") -> Embedder:
    """``lsa`` / ``lsa:256`` / a preset (``minilm``, ``bge-small``, ``e5-small``) / ``hf:<hub id or local folder>``."""
    spec = spec.strip()
    if spec == "lsa" or spec.startswith("lsa:"):
        return LsaEmbedder(int(spec.split(":")[1]) if ":" in spec else 128)
    if spec in EMBEDDING_PRESETS:
        p = EMBEDDING_PRESETS[spec]
        return HFEmbedder(p["model"], p["pooling"], p["query_prefix"], p["doc_prefix"], name=spec)
    if spec.startswith("hf:"):
        return HFEmbedder(spec[3:])
    raise ValueError(f"Unknown embedder {spec!r}. Use 'lsa', 'lsa:<dim>', one of {list(EMBEDDING_PRESETS)} or 'hf:<model>'.")


def load_embedder(directory: str | Path) -> Embedder:
    directory = Path(directory)
    kind = json.loads((directory / "embedder.json").read_text(encoding="utf-8"))["type"]
    return LsaEmbedder.load(directory) if kind == "lsa" else HFEmbedder.load(directory)
