import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.fixture
def app_index(resumes, tmp_path, monkeypatch):
    from talent_rag.index import TalentIndex

    TalentIndex.build(resumes, "lsa:48", "numpy", out_dir=tmp_path)
    monkeypatch.setenv("TALENT_INDEX", str(tmp_path))
    return tmp_path


def run_app():
    from pathlib import Path

    app = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    return streamlit_testing.AppTest.from_file(str(app), default_timeout=120).run()


def test_chat_flow_renders_candidates_and_explanation(app_index):
    at = run_app()
    assert not at.exception
    at.chat_input[0].set_value("Find me a Junior Data Analyst who knows SQL and Tableau").run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Understood requirements" in text and "#1" in text and "SQL" in text
    assert any("Overall" in i.value for i in at.info)
    assert len(at.session_state["messages"]) == 2


def test_missing_index_offers_to_build_a_demo_index(tmp_path, monkeypatch):
    monkeypatch.setenv("TALENT_INDEX", str(tmp_path / "nothing"))
    at = run_app()
    assert not at.exception and at.warning and "No index found" in at.warning[0].value
    assert any("Build a demo index" in b.label for b in at.button)
