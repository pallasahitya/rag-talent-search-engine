"""Charts for the evaluation and bias reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_config_comparison(df: pd.DataFrame, path: str | Path) -> Path:
    d = df.copy()
    d["label"] = d["embedder"] + "\n" + d["mode"] + np.where(d["rerank_weight"] > 0, "+rerank", "") + np.where(d["anonymized"], "", "\n(raw text)")
    d = d.sort_values("ndcg@10", ascending=False)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(d) + 3), 4.8))
    x = np.arange(len(d))
    for i, (col, name) in enumerate([("precision@3", "precision@3"), ("mrr", "MRR"), ("ndcg@10", "nDCG@10")]):
        ax.bar(x + (i - 1) * 0.27, d[col], 0.27, label=name, color=COLORS[i], edgecolor="#444")
    ax.set_xticks(x)
    ax.set_xticklabels(d["label"], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval quality on the annotation-derived benchmark")
    ax.legend()
    return _save(fig, path)


def plot_bias(report, path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cf = report.counterfactual
    labels = ["anonymised" if r.anonymized else "raw text" for r in cf]
    axes[0].bar(labels, [100 * r.topk_changed_queries for r in cf], color=[COLORS[2] if r.anonymized else COLORS[3] for r in cf], edgecolor="#444")
    axes[0].set_ylabel("% of queries whose top-k changed")
    axes[0].set_title("Counterfactual name/pronoun swap")
    t = report.parity.table
    axes[1].bar(t["group"], t["selection_rate"].fillna(0), color=COLORS[:3], edgecolor="#444")
    axes[1].set_ylabel("selection rate among qualified candidates")
    axes[1].set_title("Qualified-selection parity")
    axes[1].tick_params(axis="x", labelsize=8)
    return _save(fig, path)
