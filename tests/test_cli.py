import json

import pytest

from talent_rag.cli import main
from talent_rag.config import SAMPLE_DATA

Q = "Find me a Junior Data Analyst who knows SQL and Tableau"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("cli_index")
    assert main(["build-index", "--data", str(SAMPLE_DATA), "--out", str(out), "--embedder", "lsa:48", "--store", "faiss"]) == 0
    return out


def test_info_and_stats(capsys):
    assert main(["info"]) == 0
    assert main(["stats", "--data", str(SAMPLE_DATA)]) == 0
    out = capsys.readouterr().out
    assert "talent-rag" in out and "resumes: 45" in out and "Skills" in out


@pytest.mark.parametrize("store", ["numpy", "chroma"])
def test_build_index_with_other_stores(tmp_path, store, capsys):
    assert main(["build-index", "--data", str(SAMPLE_DATA), "--out", str(tmp_path), "--embedder", "lsa:32", "--store", store, "--limit", "20"]) == 0
    assert "indexed 20 resumes" in capsys.readouterr().out


def test_search_text_output(built, capsys):
    assert main(["search", Q, "--index", str(built)]) == 0
    out = capsys.readouterr().out
    assert "Understood requirements" in out and "#1" in out and "#3" in out and "Overall [template" in out and "Data Analyst" in out


def test_search_json_output_and_hidden_names(built, capsys):
    assert main(["search", Q, "--index", str(built), "--json", "--hide-names", "--explain", "none"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["candidates"]) == 3 and "name" not in data["candidates"][0] and "explanation" not in data
    assert data["requirements"] and data["candidates"][0]["evidence"]


def test_search_with_an_unreachable_llm_falls_back_to_the_template(built, capsys, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens here
    monkeypatch.setattr("talent_rag.llm.time.sleep", lambda s: None)
    assert main(["search", Q, "--index", str(built), "--explain", "ollama:fake-model"]) == 0
    out = capsys.readouterr().out
    assert "template (no LLM)" in out and "fallback: LLM call failed" in out and "#1" in out


def test_search_with_other_store_and_mode(built, capsys):
    assert main(["search", Q, "--index", str(built), "--store", "numpy", "--mode", "dense", "-k", "2", "--explain", "none", "--rerank", "0"]) == 0
    assert capsys.readouterr().out.count("\n#") == 2


def test_chat_command(built, monkeypatch, capsys):
    answers = iter([Q, ""])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert main(["chat", "--index", str(built)]) == 0
    assert "#1" in capsys.readouterr().out


def test_evaluate_and_bias_commands(tmp_path, capsys):
    ev = tmp_path / "ev"
    assert main(["evaluate", "--data", str(SAMPLE_DATA), "--embedders", "lsa:32", "--modes", "dense", "bm25", "--queries", "10", "--out", str(ev)]) == 0
    assert (ev / "retrieval_comparison.csv").exists()
    bi = tmp_path / "bias"
    assert main(["bias-audit", "--data", str(SAMPLE_DATA), "--embedder", "lsa:32", "--queries", "10", "--out", str(bi)]) == 0
    assert (bi / "bias_report.md").exists() and "report written" in capsys.readouterr().out


def test_errors_are_reported_with_exit_code_2(tmp_path, capsys):
    assert main(["stats", "--data", "nope.json"]) == 2
    assert "DATASET" in capsys.readouterr().err
    assert main(["search", "x", "--index", str(tmp_path / "none")]) == 2
    assert "build-index" in capsys.readouterr().err


def test_demo_command(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path), "--store", "numpy"]) == 0
    for name in ("search_results.md", "search_results.json", "evaluation/retrieval_comparison.csv", "bias/bias_report.md", "index/config.json"):
        assert (tmp_path / name).exists(), name
    assert "Junior Data Analyst" in (tmp_path / "search_results.md").read_text()


def test_requires_a_command():
    with pytest.raises(SystemExit):
        main([])
