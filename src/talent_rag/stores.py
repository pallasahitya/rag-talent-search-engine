"""Vector stores: exact NumPy search, FAISS and ChromaDB behind one tiny interface.

Vectors are L2-normalised, so the inner product equals cosine similarity. Row ``i`` of the matrix is chunk ``i``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class VectorStore(ABC):
    kind: str

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(row_indices, similarities)`` sorted by decreasing similarity."""

    def save(self, directory: Path) -> None:  # pragma: no cover - default: nothing to persist
        return None

    def __len__(self) -> int:
        return self.size

    size: int = 0


class NumpyStore(VectorStore):
    kind = "numpy"

    def __init__(self) -> None:
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.size = 0

    def add(self, vectors: np.ndarray) -> None:
        v = np.asarray(vectors, dtype=np.float32)
        self.matrix = v if self.size == 0 else np.vstack([self.matrix, v])
        self.size = len(self.matrix)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.size)
        sims = self.matrix @ np.asarray(query, dtype=np.float32)
        order = np.argsort(-sims, kind="stable")[:k]
        return order, sims[order]


class FaissStore(VectorStore):
    kind = "faiss"

    def __init__(self, dim: int, index=None) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install faiss-cpu   (or use --store numpy / chroma)") from exc
        self._faiss = faiss
        self.index = index if index is not None else faiss.IndexFlatIP(dim)
        self.size = self.index.ntotal

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.size = self.index.ntotal

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.size)
        sims, idx = self.index.search(np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1), k)
        return idx[0], sims[0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(directory / "faiss.index"))

    @classmethod
    def load(cls, directory: Path, dim: int) -> FaissStore:
        import faiss

        return cls(dim, faiss.read_index(str(Path(directory) / "faiss.index")))


class ChromaStore(VectorStore):
    """ChromaDB collection with cosine distance (similarity = 1 - distance). Persisted under ``<dir>/chroma``."""

    kind = "chroma"
    COLLECTION = "resume_chunks"

    def __init__(self, directory: Path | None = None, reset: bool = True) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install chromadb   (or use --store numpy / faiss)") from exc
        self.client = chromadb.PersistentClient(path=str(directory)) if directory else chromadb.EphemeralClient()
        if reset:
            try:
                self.client.delete_collection(self.COLLECTION)
            except Exception:  # noqa: BLE001 - collection did not exist yet
                pass
        try:
            self.collection = self.client.get_or_create_collection(self.COLLECTION, configuration={"hnsw": {"space": "cosine"}})
        except TypeError:  # older chromadb versions
            self.collection = self.client.get_or_create_collection(self.COLLECTION, metadata={"hnsw:space": "cosine"})
        self.size = self.collection.count()

    def add(self, vectors: np.ndarray, batch: int = 2000) -> None:
        start = self.size
        for i in range(0, len(vectors), batch):
            part = vectors[i : i + batch]
            ids = [str(start + i + j) for j in range(len(part))]
            self.collection.add(ids=ids, embeddings=np.asarray(part, dtype=np.float32).tolist())
        self.size = self.collection.count()

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.size)
        res = self.collection.query(query_embeddings=[np.asarray(query, dtype=np.float32).tolist()], n_results=k)
        idx = np.array([int(i) for i in res["ids"][0]])
        sims = 1.0 - np.array(res["distances"][0], dtype=np.float64)
        return idx, sims


def make_store(kind: str, dim: int, directory: Path | None = None) -> VectorStore:
    if kind == "numpy":
        return NumpyStore()
    if kind == "faiss":
        return FaissStore(dim)
    if kind == "chroma":
        return ChromaStore(directory / "chroma" if directory else None)
    raise ValueError(f"Unknown store {kind!r}; choose numpy, faiss or chroma.")
