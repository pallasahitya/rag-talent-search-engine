import json

import pandas as pd
import pytest

from talent_rag.data import (
    Resume,
    dataset_statistics,
    load_resumes,
    parse_dataturks_record,
    parse_years,
    resume_frame,
    split_skill_blob,
    years_of_experience,
)


def test_sample_loads_with_annotations(resumes):
    assert len(resumes) == 45 and len({r.id for r in resumes}) == 45
    assert all(r.entities.get("Name") and r.entities.get("Skills") for r in resumes)
    assert all(r.id.startswith("R") for r in resumes)


def test_every_annotation_span_matches_text_exactly():
    for line in open("data/sample/resumes_sample.json", encoding="utf-8"):
        o = json.loads(line)
        for ann in o["annotation"]:
            for p in ann["points"]:
                assert o["content"][p["start"] : p["end"] + 1] == p["text"]  # inclusive end, like the Kaggle file


def test_dataturks_record_parsing_inclusive_and_exclusive_end():
    text = "Jane Doe\nPython developer"
    inclusive = {"content": text, "annotation": [{"label": ["Name"], "points": [{"start": 0, "end": 7, "text": "Jane Doe"}]}]}
    exclusive = {"content": text, "annotation": [{"label": ["Name"], "points": [{"start": 0, "end": 8, "text": "Jane Doe"}]}]}
    assert parse_dataturks_record(inclusive, 0).entities["Name"] == ["Jane Doe"]
    assert parse_dataturks_record(exclusive, 0).entities["Name"] == ["Jane Doe"]
    assert parse_dataturks_record(inclusive, 0).spans == [("Name", 0, 8)]


def test_parsing_is_robust_to_missing_or_odd_annotations():
    assert parse_dataturks_record({"content": "x y z", "annotation": None}, 3).entities == {}
    r = parse_dataturks_record({"content": "abc", "annotation": [{"label": "Skills", "points": [{"start": 0, "end": 2, "text": "abc"}]}]}, 0)
    assert r.entities == {"Skills": ["abc"]} and r.id == "R0001"
    r = parse_dataturks_record({"content": "abc", "metadata": {"id": "X9"}}, 0)
    assert r.id == "X9"


def test_label_names_are_canonicalised():
    obj = {"content": "Acme Corp", "annotation": [{"label": ["companies worked at"], "points": [{"start": 0, "end": 8, "text": "Acme Corp"}]}]}
    assert "Companies worked at" in parse_dataturks_record(obj, 0).entities


def test_skills_block_keeps_its_layout_so_it_can_be_split(resumes):
    blob = resumes[0].get("Skills")[0]
    assert "\n" in blob
    skills = split_skill_blob(blob)
    assert len(skills) >= 5 and all(len(s.split()) <= 4 for s in skills)


def test_split_skill_blob_handles_separators():
    assert split_skill_blob("• Python\n• SQL, Tableau; Excel | Power BI") == ["Python", "SQL", "Tableau", "Excel", "Power BI"]
    assert split_skill_blob("Languages: Java, C++") == ["Java", "C++"]
    assert split_skill_blob("C#, .NET, Node.js, CI/CD.") == ["C#", ".NET", "Node.js", "CI/CD"]


@pytest.mark.parametrize("text,expected", [("5 years", 5.0), ("2+ years", 2.0), ("18 months", 1.5), ("1 year", 1.0), ("Fresher", 0.0), ("n/a", None), ("120 years", None)])
def test_parse_years(text, expected):
    assert parse_years(text) == expected


def test_years_of_experience_from_annotation_and_from_text():
    r = Resume("a", "x", {"Years of Experience": ["4 years"]})
    assert years_of_experience(r) == 4.0
    r2 = Resume("b", "Senior engineer with 7 years of experience in cloud.")
    assert years_of_experience(r2) == 7.0
    assert years_of_experience(Resume("c", "no info here")) is None


def test_load_csv_folder_and_json_array(tmp_path):
    pd.DataFrame({"Category": ["HR", "IT"], "Resume_str": ["alpha beta " * 5, "gamma delta " * 5]}).to_csv(tmp_path / "r.csv", index=False)
    rs = load_resumes(tmp_path / "r.csv")
    assert [r.category for r in rs] == ["HR", "IT"] and rs[0].title == "HR"
    d = tmp_path / "txt"
    d.mkdir()
    (d / "a.txt").write_text("Resume A text")
    (d / "b.txt").write_text("Resume B text")
    assert [r.id for r in load_resumes(d)] == ["a", "b"]
    arr = tmp_path / "a.json"
    arr.write_text(json.dumps([{"content": "one two three"}, {"content": ""}, {"content": "four five"}]))
    assert len(load_resumes(arr)) == 2  # empty content skipped


def test_duplicate_ids_are_made_unique(tmp_path):
    f = tmp_path / "d.json"
    f.write_text("\n".join(json.dumps({"content": f"resume {i}", "metadata": {"id": "same"}}) for i in range(3)))
    assert len({r.id for r in load_resumes(f)}) == 3


def test_limit_and_errors(tmp_path):
    assert len(load_resumes("data/sample/resumes_sample.json", limit=5)) == 5
    with pytest.raises(FileNotFoundError, match="DATASET"):
        load_resumes("nope.json")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="text column"):
        load_resumes(bad)


def test_statistics_and_frame(resumes):
    st = dataset_statistics(resumes)
    assert st["resumes"] == 45 and st["annotated_resumes"] == 45 and st["entities_by_label"]["Name"] == 45
    df = resume_frame(resumes)
    assert len(df) == 45 and df["years"].notna().all()
