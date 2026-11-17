from talent_rag.anonymize import anonymize, guess_names
from talent_rag.chunking import is_noise

TEXT = """Arjun Walker
Junior Accountant
Email: arjun.walker97@examplemail.com
Phone: +92 338 2458371
Profile: https://www.linkedin.com/in/arjunwalker and github.com/arjunw

Arjun is an accountant. He enjoys his work and she said Mr. Walker is great. Mrs. Khan agreed.
Skills: Excel, Tally
Date of Birth: 3/4/1995
Marital Status: Single
Nationality: Pakistani
Experience at Halden Software (2022-Present)
"""


def test_personal_identifiers_are_removed():
    out = anonymize(TEXT, ["Arjun Walker"])
    for leaked in ("Arjun", "Walker", "arjun", "examplemail", "338 2458371", "linkedin.com", "github.com/arjunw"):
        assert leaked not in out
    assert "[EMAIL]" in out and "[PHONE]" in out and "[URL]" in out and "[NAME]" in out


def test_protected_characteristic_lines_are_redacted():
    out = anonymize(TEXT, ["Arjun Walker"])
    for leaked in ("3/4/1995", "Single", "Pakistani"):
        assert leaked not in out
    assert out.count("[REDACTED PERSONAL DETAIL]") == 3


def test_pronouns_are_neutralised_and_titles_dropped():
    out = anonymize(TEXT, ["Arjun Walker"])
    assert " He " not in out and " she " not in out and " his " not in out
    assert "They enjoys" in out or "They enjoy" in out
    assert "Mr." not in out and "Mrs." not in out
    assert anonymize("She and him and herself and HIS") == "They and them and themselves and THEIR"


def test_job_relevant_content_is_kept():
    out = anonymize(TEXT, ["Arjun Walker"])
    for kept in ("Junior Accountant", "Excel, Tally", "Halden Software", "(2022-Present)"):
        assert kept in out


def test_word_boundaries_do_not_corrupt_other_words():
    out = anonymize("Sam Hall\nSample sampling and Hallway.\nAge: 30", ["Sam Hall"])
    assert "Sample sampling" in out and "Hallway" in out and "30" not in out


def test_inline_personal_details_are_redacted():
    out = anonymize("Skills: Python | DOB: 3/4/1995 | Gender: Male | Marital Status: Single, Experience: 4 years")
    for leaked in ("1995", "Male", "Single"):
        assert leaked not in out
    assert "Skills: Python" in out and "Experience: 4 years" in out


def test_agile_is_not_mistaken_for_the_age_line():
    assert "Agile" in anonymize("Skills\nAgile\nScrum")


def test_names_guessed_without_annotation():
    names = guess_names("Priya Nair\nData Analyst\npriya.nair@mail.com")
    assert {"Priya", "Nair", "priya", "nair"} <= set(names) or {"Priya", "Nair"} <= set(names)
    assert "Nair" not in anonymize("Priya Nair\nData Analyst\npriya.nair@mail.com")


def test_pure_redaction_text_counts_as_noise():
    assert is_noise("[REDACTED PERSONAL DETAIL]\n[REDACTED PERSONAL DETAIL]") and is_noise("[NAME] [EMAIL]")
    assert not is_noise("[NAME] Python developer")


def test_a_job_title_on_the_first_line_is_not_mistaken_for_a_name():
    out = anonymize("Data Analyst\nExperienced data analyst skilled in SQL.\nEmail: info@firm.com")
    assert "Data Analyst" in out and "data analyst" in out and "[NAME]" not in out
    assert guess_names("Senior Java Developer\nFive years") == []
    assert guess_names("Curriculum Vitae\nJane") == []


def test_annotated_names_take_precedence_over_guessing():
    out = anonymize("Priya Nair\nPython Developer\nPriya builds APIs.", ["Priya Nair"])
    assert "Priya" not in out and "Nair" not in out and "Python Developer" in out
