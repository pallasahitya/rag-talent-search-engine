import pytest

from talent_rag.bias import (
    FEMALE,
    MALE,
    UNKNOWN,
    counterfactual_audit,
    infer_group,
    qualified_selection_parity,
    run_bias_audit,
    swap_gender,
    write_bias_report,
)
from talent_rag.data import Resume


def test_group_inference(resumes):
    counts = {g: sum(infer_group(r) == g for r in resumes) for g in (FEMALE, MALE, UNKNOWN)}
    assert counts[FEMALE] > 0 and counts[MALE] > 0 and counts[UNKNOWN] > 0 and sum(counts.values()) == 45
    assert infer_group(Resume("x", "Priya Nair\nDeveloper", {"Name": ["Priya Nair"]})) == FEMALE
    assert infer_group(Resume("y", "Taylor Smith\nDev", {"Name": ["Taylor Smith"]})) == UNKNOWN


def test_swap_changes_only_name_email_and_pronouns(resumes):
    r = next(x for x in resumes if " He " in x.text and infer_group(x) == MALE)
    c = swap_gender(r)
    changed = [(a, b) for a, b in zip(r.text.splitlines(), c.text.splitlines()) if a != b]
    assert len(changed) == 3 and len(r.text.splitlines()) == len(c.text.splitlines())  # name line, e-mail line, summary line
    first, new = r.name.split()[0], c.name.split()[0]
    assert infer_group(c) == FEMALE and first != new
    assert f"{first.lower()}." in r.text and f"{new.lower()}." in c.text  # e-mail local part follows the name (lower-case)
    assert " She " in c.text and " her team" in c.text and " He " not in c.text
    assert c.text.replace(new, first).replace("She", "He").replace("her team", "his team").replace(new.lower(), first.lower()) == r.text


def test_swap_is_reversible_in_group_and_skips_unknown_names(resumes):
    f = next(x for x in resumes if infer_group(x) == FEMALE)
    assert infer_group(swap_gender(f)) == MALE
    u = next(x for x in resumes if infer_group(x) == UNKNOWN)
    assert swap_gender(u) is None


def test_anonymised_index_is_exactly_invariant_to_the_swap(index, benchmark):
    res = counterfactual_audit(index, [q.text for q in benchmark], k=3, mode="dense")
    assert res.anonymized and res.swapped_resumes == 35
    assert res.max_abs_score_delta == 0.0 and res.topk_changed_queries == 0.0 and res.spearman == pytest.approx(1.0)
    assert res.severity == "none" and not res.flagged()


def test_raw_index_reacts_to_gendered_cues(raw_index, benchmark):
    res = counterfactual_audit(raw_index, [q.text for q in benchmark], k=3, mode="dense")
    assert not res.anonymized and res.max_abs_score_delta > 0 and res.severity in {"scores", "ranking"} and res.flagged()
    assert res.spearman < 1.0 and 0 < res.top_k_overlap <= 1.0


def test_parity_table_counts_add_up(index, benchmark):
    par = qualified_selection_parity(index, benchmark, k=3, mode="hybrid")
    t = par.table.set_index("group")
    assert t["qualified_slots"].sum() == sum(len(q.relevant) for q in benchmark)
    assert (t["selected"] <= t["qualified_slots"]).all() and t["selection_rate"].between(0, 1).all()
    assert par.disparate_impact is None or 0 <= par.disparate_impact <= 1


def test_parity_warns_when_samples_are_too_small(index, benchmark):
    par = qualified_selection_parity(index, benchmark[:1], k=3, mode="hybrid")
    assert par.disparate_impact is None and any("Too few" in n for n in par.notes) and not par.flagged


def test_audit_runs_both_anonymisation_settings_and_writes_reports(resumes, benchmark, tmp_path):
    rep = run_bias_audit(resumes, benchmark[:15], "lsa:48", progress=lambda *_: None)
    assert [r.anonymized for r in rep.counterfactual] == [True, False]
    assert rep.counterfactual[0].severity == "none"
    assert any("Anonymisation OFF" in f for f in rep.flags) and not any("Anonymisation ON" in f for f in rep.flags)
    assert any("lexicon" in w for w in rep.warnings)
    write_bias_report(rep, tmp_path)
    for name in ("bias_report.md", "bias_report.json", "bias_counterfactual.csv", "bias_parity.csv", "bias_report.png"):
        assert (tmp_path / name).exists(), name
    md = (tmp_path / "bias_report.md").read_text()
    assert "Counterfactual" in md and "not legal or ethical clearance" in md
