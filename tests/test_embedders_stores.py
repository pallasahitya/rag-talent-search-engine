import numpy as np
import pytest

from talent_rag.embedders import HFEmbedder, LsaEmbedder, get_embedder, load_embedder
from talent_rag.stores import ChromaStore, FaissStore, NumpyStore, make_store


@pytest.fixture(scope="module")
def corpus(resumes):
    return [r.text for r in resumes]


def test_lsa_vectors_are_normalised_and_deterministic(corpus):
    emb = get_embedder("lsa:32").fit(corpus)
    x = emb.embed_documents(corpus[:10])
    assert x.shape == (10, 32) and x.dtype == np.float32
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-5)
    assert np.allclose(x, get_embedder("lsa:32").fit(corpus).embed_documents(corpus[:10]), atol=1e-5)


def test_lsa_similar_documents_are_closer(resumes):
    emb = get_embedder("lsa:48").fit([r.text for r in resumes])
    da = [r for r in resumes if "Data Analyst" in r.title]
    other = next(r for r in resumes if "Mechanical" in r.title)
    q = emb.embed_query("data analyst with SQL and Tableau dashboards")
    assert emb.embed_documents([da[0].text])[0] @ q > emb.embed_documents([other.text])[0] @ q


def test_lsa_must_be_fitted():
    with pytest.raises(RuntimeError, match="fitted"):
        LsaEmbedder().embed_query("x")


def test_lsa_save_and_load_roundtrip(corpus, tmp_path):
    emb = get_embedder("lsa:16").fit(corpus)
    emb.save(tmp_path)
    loaded = load_embedder(tmp_path)
    assert np.allclose(loaded.embed_query("sql tableau"), emb.embed_query("sql tableau"))
    assert loaded.spec() == emb.spec()


def test_unknown_embedder_name():
    with pytest.raises(ValueError, match="Unknown embedder"):
        get_embedder("nope")


def test_hf_embedder_with_tiny_local_encoder(tiny_encoder, corpus, tmp_path):
    emb = HFEmbedder(str(tiny_encoder), pooling="mean", max_length=64, batch_size=4, device="cpu")
    x = emb.embed_documents(corpus[:9])  # 9 docs with batch size 4 -> 3 batches
    assert x.shape == (9, 32) and np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-4)
    assert np.allclose(x, emb.embed_documents(corpus[:9]), atol=1e-5)  # deterministic
    assert np.allclose(emb.embed_documents(corpus[:1])[0], x[0], atol=1e-4)  # independent of batch composition / padding
    emb.save(tmp_path)
    again = load_embedder(tmp_path)
    assert isinstance(again, HFEmbedder) and np.allclose(again.embed_query("sql"), emb.embed_query("sql"), atol=1e-5)


def test_hf_pooling_and_prefixes_change_the_vector(tiny_encoder):
    mean = HFEmbedder(str(tiny_encoder), "mean", device="cpu").embed_query("data analyst sql")
    cls = HFEmbedder(str(tiny_encoder), "cls", device="cpu").embed_query("data analyst sql")
    pre = HFEmbedder(str(tiny_encoder), "mean", query_prefix="query: ", device="cpu").embed_query("data analyst sql")
    assert not np.allclose(mean, cls) and not np.allclose(mean, pre)
    with pytest.raises(ValueError):
        HFEmbedder(str(tiny_encoder), "max", device="cpu")


@pytest.fixture(scope="module")
def vectors():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 16)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.mark.parametrize("kind", ["numpy", "faiss", "chroma"])
def test_every_store_finds_the_exact_nearest_neighbours(kind, vectors, tmp_path):
    store = make_store(kind, 16, tmp_path)
    store.add(vectors)
    q = vectors[7] + 0.01
    idx, sims = store.search(q / np.linalg.norm(q), 5)
    expected = np.argsort(-(vectors @ (q / np.linalg.norm(q))))[:5]
    assert idx.tolist() == expected.tolist() and idx[0] == 7
    assert np.all(np.diff(sims) <= 1e-6) and len(store) == 50
    assert len(store.search(vectors[0], 500)[0]) == 50  # k is clamped to the collection size


def test_stores_can_be_extended(vectors):
    for store in (NumpyStore(), FaissStore(16), ChromaStore(None)):
        store.add(vectors[:20])
        store.add(vectors[20:])
        assert len(store) == 50 and store.search(vectors[33], 1)[0][0] == 33


def test_faiss_and_chroma_persistence(vectors, tmp_path):
    f = FaissStore(16)
    f.add(vectors)
    f.save(tmp_path / "f")
    assert FaissStore.load(tmp_path / "f", 16).search(vectors[3], 1)[0][0] == 3
    c = ChromaStore(tmp_path / "c")
    c.add(vectors)
    del c
    reopened = ChromaStore(tmp_path / "c", reset=False)
    assert len(reopened) == 50 and reopened.search(vectors[9], 1)[0][0] == 9


def test_unknown_store():
    with pytest.raises(ValueError):
        make_store("pinecone", 4)
