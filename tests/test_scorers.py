import pytest

from scorers import (
    CORRECT,
    CORRECT_WITHOUT_CONTEXT,
    GENERATION_ERROR,
    RETRIEVAL_MISS,
    UNANSWERABLE,
    answer_in_context,
    failure_label,
    normalize,
    reference_sentence,
    string_match,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("The Eiffel Tower!", "eiffel tower"),
        ("  An   apple, a day ", "apple day"),
        ("2.3–2.8 mm", "23 28 mm"),  # en dash becomes a space, '.' is punctuation
        ("１９３９", "1939"),  # full-width digits unified by NFKC
        ("“Quoted”", "quoted"),  # curly quotes are punctuation too
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_match_ignores_case_articles_and_punctuation():
    assert string_match("It was founded by the Brothers Amos and Wilfrid Ayre.", ["Brothers Amos and Wilfrid Ayre"])


def test_match_requires_whole_words():
    assert not string_match("The year 19390 is not it.", ["1939"])
    assert string_match("It began in 1939.", ["1939"])


def test_match_any_of_several_gold_answers():
    assert string_match("Answer: Paris", ["Lyon", "Paris"])


def test_known_weakness_paraphrase_is_missed():
    # The limitation the LLM judge is compared against: correct, but different words.
    assert not string_match("Roughly 1.3 sq ft", ["1.3 square feet"])


def test_known_weakness_containing_is_not_correct():
    # A response can contain the gold string while asserting the wrong thing.
    assert string_match("It was not 1939; it was 1948.", ["1939"])


def test_empty_gold_never_matches():
    assert not string_match("anything", [""])


def test_answer_in_context_checks_text_not_passage_ids():
    assert answer_in_context(["Tokyo"], ["Paris is in France.", "The capital is Tokyo."])
    assert not answer_in_context(["Tokyo"], [])


@pytest.mark.parametrize(
    "answerable,in_context,correct,label",
    [
        (True, True, True, CORRECT),
        (True, False, True, CORRECT_WITHOUT_CONTEXT),
        (True, True, False, GENERATION_ERROR),
        (True, False, False, RETRIEVAL_MISS),
        (False, False, False, UNANSWERABLE),
        (False, True, True, UNANSWERABLE),
    ],
)
def test_failure_label_truth_table(answerable, in_context, correct, label):
    assert failure_label(answerable, in_context, correct) == label


def test_reference_sentence_wraps_fragment_with_question():
    assert (
        reference_sentence("When was it founded?", ["1939"])
        == 'The answer to the question "When was it founded?" is: 1939.'
    )
    assert reference_sentence("Q?", ["A", "B"]).endswith("is: A (also accepted: B).")


def test_reference_answer_keeps_short_form():
    from scorers import reference_answer

    assert reference_answer(["March 2013"]) == "March 2013"
    assert reference_answer(["A", "B", "C"]) == "A (also accepted: B; C)"


from scorers import ANSWERED, REFUSED, classify_response  # noqa: E402

V2 = "v2-not-in-context"


@pytest.mark.parametrize(
    "text,refusal,method",
    [
        ("NOT_IN_CONTEXT", REFUSED, "marker"),
        ("  NOT_IN_CONTEXT\n", REFUSED, "marker"),
        ("NOT_IN_CONTEXT. The passages discuss other topics.", REFUSED, "marker_with_extra_text"),
        ("The answer may be 1964, but NOT_IN_CONTEXT", ANSWERED, "marker_not_at_start"),
        ("Wyoming.", ANSWERED, "marker"),
    ],
)
def test_v2_marker_classification(text, refusal, method):
    assert classify_response(text, V2) == {"refusal": refusal, "method": method}


@pytest.mark.parametrize(
    "text,refusal",
    [
        ("The context doesn’t specify Syracuse’s age.", REFUSED),  # curly apostrophe
        ("The provided context doesn't say whether he has a wife.", REFUSED),
        ("The context does not specify Georgia’s state flower.", REFUSED),
        ("The passages do not mention it.", REFUSED),
        ("G.I. Joe debuted in 1964, so it was 62 years old in 2026.", ANSWERED),
        ("According to the provided context, the prime minister is Mark Rutte.", ANSWERED),
    ],
)
def test_v1_pattern_first_pass(text, refusal):
    r = classify_response(text, "v1-baseline")
    assert r["refusal"] == refusal and r["method"] == "v1_pattern_needs_review"
