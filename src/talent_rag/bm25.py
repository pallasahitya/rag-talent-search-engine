"""A small, dependency-free BM25 (Okapi) index used for the lexical half of hybrid search."""

from __future__ import annotations

import re

import numpy as np
from scipy import sparse

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.]*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it its of on or that the their to was were will with who whom which "
    "find me looking candidate candidates someone person who knows know experience experienced skilled skills years year".split()
)


def tokenize(text: str) -> list[str]:
    toks = [t.rstrip(".") for t in TOKEN_RE.findall(text.lower())]
    return [t for t in toks if t and t not in STOPWORDS]


class BM25:
    """Okapi BM25 over a list of documents (``k1`` / ``b`` as in the original paper)."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        docs = [tokenize(d) for d in documents]
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        for i, toks in enumerate(docs):
            counts: dict[int, int] = {}
            for t in toks:
                j = self.vocab.setdefault(t, len(self.vocab))
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(i)
                cols.append(j)
                vals.append(c)
        self.n = len(docs)
        self.tf = sparse.csc_matrix((vals, (rows, cols)), shape=(self.n, max(len(self.vocab), 1)), dtype=np.float64)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avg_len = float(self.doc_len.mean()) if self.n else 0.0
        df = np.asarray((self.tf > 0).sum(axis=0)).ravel()
        self.idf = np.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for ``query``."""
        out = np.zeros(self.n)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avg_len or 1.0))
        for t in set(tokenize(query)):
            j = self.vocab.get(t)
            if j is None:
                continue
            col = self.tf.getcol(j)
            idx, tf = col.indices, col.data
            out[idx] += self.idf[j] * tf * (self.k1 + 1) / (tf + norm[idx])
        return out

    def top(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        return order, s[order]
