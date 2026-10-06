import pytest

from known_truth import VARIANTS, build_cases, summarize, unrelated
from scorers import string_match

QS = [
    {"qid": f"q{i}", "question": f"Question {i}?", "gold_answers": [g]}
    for i, g in enumerate(["Paris", "1939", "Mark Rutte", "the radius and the ulna"])
]


def test_every_question_gets_every_variant_with_a_fixed_label():
    cases = build_cases(QS, seed=1, n=4)
    assert len(cases) == 4 * len(VARIANTS)
    for c in cases:
        assert c["is_correct"] == VARIANTS[c["variant"]][1]


def test_swapped_answer_is_truly_unrelated_to_the_gold():
    for c in build_cases(QS, seed=1, n=4):
        if c["variant"] == "swapped":
            assert not string_match(c["response"], c["gold_answers"])


def test_negation_trap_contains_the_gold_words_but_is_wrong():
    for c in build_cases(QS, seed=1, n=4):
        if c["variant"] == "negation_trap":
            assert string_match(c["response"], c["gold_answers"]) and c["is_correct"] is False


def test_build_is_deterministic():
    assert build_cases(QS, seed=1, n=4) == build_cases(list(reversed(QS)), seed=1, n=4)


def test_unrelated():
    assert unrelated("Paris", "1939")
    assert not unrelated("Paris", "Paris, France")  # one contains the other
    assert not unrelated("", "Paris")


def test_no_unrelated_answer_fails_loudly():
    with pytest.raises(ValueError):
        build_cases(
            [
                {"qid": "a", "question": "q", "gold_answers": ["x"]},
                {"qid": "b", "question": "q", "gold_answers": ["x"]},
            ],
            seed=1,
            n=2,
        )


def test_summarize_counts_right_labels_and_failures_hand_checked():
    rows = [
        {"variant": "exact", "is_correct": True, "string_match": True, "answer_accuracy": {"value": 1.0}},
        {"variant": "exact", "is_correct": True, "string_match": True, "answer_accuracy": {"value": 0.25}},
        {"variant": "exact", "is_correct": True, "string_match": False, "answer_accuracy": {"value": None}},
        {"variant": "swapped", "is_correct": False, "string_match": False, "answer_accuracy": {"value": 0.5}},
    ]
    s = summarize(rows)
    assert (
        s["exact"]["n"],
        s["exact"]["string_match_right"],
        s["exact"]["judge_right"],
        s["exact"]["judge_failed"],
    ) == (3, 2, 1, 1)
    assert s["swapped"]["judge_right"] == 0  # 0.5 counts as "judged correct", which is wrong here
