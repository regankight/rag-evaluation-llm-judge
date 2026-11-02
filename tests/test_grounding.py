"""The grouping rule and the committed grounding labels (offline)."""

import pytest

from analyze import load_human_v1
from grounding import analyze, check_coverage, group, load


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
