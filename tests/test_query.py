import pytest

from talent_rag.data import Resume
from talent_rag.query import SkillVocabulary, parse_query, prettify, requirement_coverage


@pytest.fixture(scope="module")
def vocab(resumes):
    return SkillVocabulary.from_resumes(resumes)


def test_vocabulary_finds_skills_with_boundaries_and_casing(vocab):
    assert vocab.find("Experienced in sql, TABLEAU and Power BI.") == ["sql".upper(), "Tableau", "Power BI"]
    assert "SQL" not in vocab.find("PostgreSQL and MySQL only")  # 'sql' inside another word is not a match
    assert vocab.find("c++ and c# and node.js") == ["C++", "C#", "Node.js"]


def test_aliases_and_longer_terms(vocab):
    assert "Power BI" in vocab.find("dashboards in PowerBI")
    assert "Kubernetes" in vocab.find("runs k8s clusters")
    found = vocab.find("SAP FICO consultant")
    assert "SAP FICO" in found and "SAP" not in found  # a skill is not reported twice as its own substring


def test_prettify():
    assert prettify("sql") == "SQL" and prettify("power bi") == "Power BI" and prettify("node.js") == "Node.js" and prettify("tableau") == "Tableau"


@pytest.mark.parametrize("query,skills,role,lo,hi", [
    ("Find me a Junior Data Analyst who knows SQL and Tableau", ["SQL", "Tableau"], "data analyst", None, 2.0),
    ("Senior DevOps Engineer with 6+ years of experience in AWS", ["AWS"], "devops engineer", 6.0, None),
    ("HR manager with payroll, 3-5 years", ["Payroll"], "hr manager", 3.0, 5.0),
    ("Java developer with at least 4 years of Spring Boot", ["Java", "Spring Boot"], "java developer", 4.0, None),
    ("someone with less than 2 years in Excel", ["Excel"], None, None, 2.0),
])
def test_parse_query(vocab, query, skills, role, lo, hi):
    spec = parse_query(query, vocab)
    assert spec.skills == skills and spec.role == role and spec.min_years == lo and spec.max_years == hi


def test_requirements_text(vocab):
    reqs = parse_query("Junior Data Analyst who knows SQL", vocab).requirements()
    assert "skill: SQL" in reqs and "role: data analyst" in reqs and any(r.startswith("experience:") for r in reqs)
    assert "experience: up to 2 years" in reqs
    assert "experience: 6+ years" in parse_query("DevOps engineer with 6+ years", vocab).requirements()
    assert "experience: 3-5 years" in parse_query("HR manager, 3-5 years", vocab).requirements()
    assert parse_query("hello there", vocab).requirements() == []


def test_requirement_coverage(vocab):
    spec = parse_query("Junior Data Analyst who knows SQL and Tableau", vocab)
    full = Resume("a", "Junior Data Analyst\nSummary\nJunior Data Analyst with 1 year of experience.\nSkills\nSQL\nTableau", {"Designation": ["Junior Data Analyst"], "Years of Experience": ["1 year"]})
    partial = Resume("b", "Data Analyst with SQL", {"Designation": ["Data Analyst"], "Years of Experience": ["6 years"]})
    wrong = Resume("c", "Mechanical Engineer using AutoCAD", {"Designation": ["Mechanical Engineer"]})
    c1, c2, c3 = (requirement_coverage(spec, r, vocab) for r in (full, partial, wrong))
    assert c1.score == 1.0 and c1.matched_skills == ["SQL", "Tableau"] and c1.years_ok and c1.role_ok
    assert c2.missing_skills == ["Tableau"] and c2.years_ok is False and c2.role_ok is True and 0 < c2.score < 1
    assert c3.score == 0.0 and c3.role_ok is False and c3.years is None and c3.years_ok is None  # unknown years are not counted
    assert c1.score > c2.score > c3.score


def test_no_requirements_means_full_coverage(vocab):
    assert requirement_coverage(parse_query("hello", vocab), Resume("x", "anything"), vocab).score == 1.0
