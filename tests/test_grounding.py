"""The grouping rule and the committed grounding labels (offline)."""

import pytest

from analyze import load_human_v1
from grounding import analyze, check_coverage, group, load, report


def test_supported_label_is_kept_whatever_the_wording():
    assert group("supported", "The context doesn't say, but it was 1999.") == "supported"


def test_answer_that_opens_with_a_refusal_is_refuses_plus_fact():
    for answer in (
        "The context doesn't identify the largest flood. It notes the 1953 North Sea flood.",
        "The context doesn’t specify which racket material is lightest. It notes graphite.",
        "The passages do not say who won.",
    ):
        assert group("not_in_passages", answer) == "refuses_plus_fact"


def test_answer_that_answers_first_is_unsupported():
    assert group("misread", "The Hub.") == "unsupported"
    assert (
        group("not_in_passages", "Ovarian carcinoma is the most common type, but the context doesn't say.")
        == "unsupported"
    )


def test_committed_labels_cover_every_answered_unanswerable_question():
    labels, runs, unanswerable = load()
    a = analyze(labels, runs, unanswerable)
    assert a["v1"]["answered"] == 29 and a["v2"]["answered"] == 12  # the human refusal counts (decision 0007)
    assert sum(a["v1"].get(g, 0) for g in ("supported", "refuses_plus_fact", "unsupported")) == 29


def test_missing_or_extra_label_is_an_error():
    labels, runs, unanswerable = load()
    human_v1 = load_human_v1()
    missing = dict(labels)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing"):
        check_coverage(missing, runs, unanswerable, human_v1)
    with pytest.raises(ValueError, match="unexpected"):
        check_coverage({**labels, "v2_not-a-real-qid": "supported"}, runs, unanswerable, human_v1)


def with_deepeval(runs, labels, strict, status="ok"):
    """Attach a fake DeepEval row to every labelled answer; strict(key) gives its strict score."""
    for k in labels:
        runs[k[:2]][1][k[3:]]["deepeval"] = {
            "status": status,
            "borderline_passes": 1.0,
            "borderline_fails": strict(k),
        }
    return runs


def without_deepeval(runs):
    for _, scores in runs.values():
        for row in scores.values():
            row.pop("deepeval", None)
    return runs


def test_deepeval_comparison_is_absent_without_its_scores():
    labels, runs, unanswerable = load()
    a = analyze(labels, without_deepeval(runs), unanswerable)
    assert "deepeval" not in a and "DeepEval" not in report(a)


def test_deepeval_strict_that_flags_everything_catches_all_unsupported_and_says_so():
    labels, runs, unanswerable = load()
    a = analyze(labels, with_deepeval(runs, labels, lambda k: 0.0), unanswerable)
    d = a["deepeval"]
    assert d["caught_unsupported"]["deepeval_strict"] == d["unsupported"]["n"] == 16
    assert d["supported"]["deepeval_strict"] == 0  # the cost: it calls every supported answer unfaithful too
    assert d["supported"]["deepeval_default"] == d["supported"]["n"]
    # RAGAS judged 13 of 16 unsupported faithful (decision 0013); flag-everything catches those 13
    assert d["caught_unsupported"]["only_deepeval"] == 13 and d["caught_unsupported"]["only_ragas"] == 0
    assert "RAGAS vs DeepEval" in report(a)


def test_failed_deepeval_rows_block_the_comparison_instead_of_counting_as_caught():
    labels, runs, unanswerable = load()
    a = analyze(labels, with_deepeval(runs, labels, lambda k: None, status="error:X"), unanswerable)
    assert len(a["deepeval"]["not_scored"]) == 41
    assert "Not reported" in report(a)


def test_partial_deepeval_run_says_what_is_missing():
    labels, runs, unanswerable = load()
    first = sorted(labels)[0]
    a = analyze(labels, with_deepeval(without_deepeval(runs), {first: labels[first]}, lambda k: 1.0), unanswerable)
    assert len(a["deepeval"]["not_scored"]) == 40
    assert "Not reported" in report(a)


def test_truncated_deepeval_prompts_block_the_comparison(tmp_path, monkeypatch):
    import grounding

    manifest = tmp_path / "m.json"
    manifest.write_text('{"ollama_truncations_total": 2}')
    monkeypatch.setattr(grounding, "DEEPEVAL_MANIFEST", manifest)
    labels, runs, unanswerable = load()
    a = analyze(labels, with_deepeval(runs, labels, lambda k: 1.0), unanswerable)
    assert a["deepeval"] == {"truncations": 2} and "truncated 2" in report(a)


def test_committed_deepeval_result():
    """The published result (decision 0014): a second library on the same judge did no better."""
    labels, runs, unanswerable = load()
    d = analyze(labels, runs, unanswerable)["deepeval"]
    assert [d["unsupported"][k] for k in ("ragas", "deepeval_default", "deepeval_strict")] == [13, 14, 12]
    assert [d["supported"][k] for k in ("ragas", "deepeval_default", "deepeval_strict")] == [17, 18, 18]
    c = d["caught_unsupported"]
    assert (c["ragas"], c["deepeval_strict"], c["only_ragas"], c["only_deepeval"], c["p"]) == (3, 4, 0, 1, 1.0)
