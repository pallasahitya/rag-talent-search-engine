import warnings

import pytest

warnings.filterwarnings("ignore")

from talent_rag.config import SAMPLE_DATA  # noqa: E402
from talent_rag.data import load_resumes  # noqa: E402
from talent_rag.evaluate import build_benchmark  # noqa: E402
from talent_rag.index import TalentIndex  # noqa: E402


@pytest.fixture(scope="session")
def resumes():
    return load_resumes(SAMPLE_DATA)


@pytest.fixture(scope="session")
def index(resumes):
    return TalentIndex.build(resumes, "lsa:64", "numpy", 80, 20, True)


@pytest.fixture(scope="session")
def raw_index(resumes):
    """Same settings but WITHOUT anonymised embeddings (for bias tests)."""
    return TalentIndex.build(resumes, "lsa:64", "numpy", 80, 20, False)


@pytest.fixture(scope="session")
def benchmark(resumes, index):
    return build_benchmark(resumes, index.vocab, 30, seed=0)


@pytest.fixture(scope="session")
def tiny_encoder(tmp_path_factory, resumes):
    from talent_rag.testing import build_tiny_encoder

    return build_tiny_encoder(tmp_path_factory.mktemp("enc"), [r.text for r in resumes])
