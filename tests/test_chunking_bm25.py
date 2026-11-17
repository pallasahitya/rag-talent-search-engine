
from talent_rag.bm25 import BM25, tokenize
from talent_rag.chunking import chunk_resume, split_sections, window_words

RESUME = """Jane Doe
Data Analyst

Summary
Analyst with SQL experience.

Skills
SQL, Tableau, Excel

Experience
Built dashboards. """ + "Wrote queries. " * 80 + """

Personal Details
[REDACTED PERSONAL DETAIL]
[REDACTED PERSONAL DETAIL]
"""


def test_sections_are_detected_and_noise_sections_dropped():
    names = [n for n, _ in split_sections(RESUME)]
    assert names == ["Header", "Summary", "Skills", "Experience"]  # the all-redacted 'Personal Details' is dropped


def test_windows_overlap_and_cover_everything():
    text = " ".join(f"w{i}" for i in range(100))
    ws = window_words(text, 40, 10)
    assert all(len(w.split()) <= 40 for w in ws)
    assert ws[0].split()[-10:] == ws[1].split()[:10]  # overlap
    assert set(" ".join(ws).split()) == set(text.split())
    assert window_words("short text", 40, 10) == ["short text"] and window_words("", 40, 10) == []


def test_chunks_have_profile_first_and_stable_ids():
    chunks = chunk_resume("R1", RESUME, 60, 15, profile="Role: Data Analyst. Skills: SQL")
    assert chunks[0].section == "Profile" and chunks[0].chunk_id == "R1::0"
    assert [c.chunk_id for c in chunks] == [f"R1::{i}" for i in range(len(chunks))]
    assert all(c.resume_id == "R1" for c in chunks)
    assert any(c.section == "Experience" and c.text.startswith("Experience:") for c in chunks)
    assert len([c for c in chunks if c.section == "Experience"]) > 1  # long section is windowed


def test_unstructured_resume_and_empty_text_still_produce_a_chunk():
    assert len(chunk_resume("R2", "just one line of text without headers")) == 1
    assert chunk_resume("R3", "")[0].text == "(empty)"


def test_tokenizer_keeps_technical_tokens_and_drops_stopwords():
    assert tokenize("Find me a candidate with C++, C#, Node.js and .NET") == ["c++", "c#", "node.js", "net"]
    assert tokenize("the and of") == []


def test_bm25_ranks_matching_document_first_and_uses_idf():
    docs = ["sql tableau dashboards analyst", "java spring boot microservices", "excel payroll recruitment", "sql sql sql queries"]
    bm = BM25(docs)
    idx, sc = bm.top("tableau sql", 4)
    assert idx[0] == 0 and sc[0] > sc[1] > 0
    assert bm.idf[bm.vocab["tableau"]] > bm.idf[bm.vocab["sql"]]  # rarer term is more informative
    assert bm.scores("zzzz nothing").sum() == 0
    assert bm.scores("").sum() == 0


def test_bm25_length_normalisation():
    short, long = "sql analyst", "sql analyst " + "filler words here " * 30
    s = BM25([short, long]).scores("sql")
    assert s[0] > s[1]


def test_bm25_empty_collection_does_not_crash():
    assert BM25([]).scores("anything").shape == (0,)
