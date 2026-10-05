from tydiqa import ANSWERABLE, UNANSWERABLE, label_question, load, slice_bytes

# "é" is 2 bytes and "東京" is 6 bytes in UTF-8, so byte and character offsets differ.
DOC = "Café intro.\nThe capital is 東京 since 1868.\nUnrelated text."


def byte_range(sub: str) -> tuple[int, int]:
    b, s = DOC.encode("utf-8"), sub.encode("utf-8")
    start = b.index(s)
    return start, start + len(s)


P0, P1, P2 = byte_range("Café intro."), byte_range("The capital is 東京 since 1868."), byte_range("Unrelated text.")
SPAN = byte_range("東京")


def row(chosen, spans, question="What is the capital?"):
    starts = [s[0] if s else -1 for s in spans]
    ends = [s[1] if s else -1 for s in spans]
    return {
        "language": "english", "question_text": question, "document_title": "Japan",
        "document_url": "https://en.wikipedia.org/wiki/Japan", "document_plaintext": DOC,
        "passage_answer_candidates": {"plaintext_start_byte": [P0[0], P1[0], P2[0]],
                                      "plaintext_end_byte": [P0[1], P1[1], P2[1]]},
        "annotations": {"passage_answer_candidate_index": chosen,
                        "minimal_answers_start_byte": starts, "minimal_answers_end_byte": ends,
                        "yes_no_answer": ["NONE"] * 3},
    }


def test_slice_bytes_on_non_ascii_text():
    start, end = SPAN
    assert slice_bytes(DOC, start, end) == "東京"
    assert DOC[start:end] != "東京"  # character slicing silently returns the wrong text


def test_majority_passage_with_span_is_answerable():
    q = label_question(row([1, 1, -1], [SPAN, SPAN, None]))
    assert q.label == ANSWERABLE
    assert q.gold_passage_id.endswith(":1")
    assert q.gold_answers == ("東京",)


def test_span_from_annotator_who_chose_another_passage_is_ignored():
    other = byte_range("Unrelated")
    q = label_question(row([1, 1, 2], [SPAN, None, other]))
    assert q.gold_answers == ("東京",)


def test_all_annotators_none_is_unanswerable():
    q = label_question(row([-1, -1, -1], [None, None, None]))
    assert q.label == UNANSWERABLE and q.gold_passage_id is None and q.gold_answers == ()


def test_disagreement_is_excluded():
    assert label_question(row([0, 1, 2], [None, SPAN, None])) is None


def test_majority_without_any_span_is_excluded():
    assert label_question(row([1, 1, -1], [None, None, None])) is None


def test_ids_are_stable_and_distinct_per_question():
    a = label_question(row([1, 1, -1], [SPAN, SPAN, None], question="Q one"))
    b = label_question(row([1, 1, -1], [SPAN, SPAN, None], question="Q one"))
    c = label_question(row([1, 1, -1], [SPAN, SPAN, None], question="Q two"))
    assert a.qid == b.qid != c.qid


def test_load_dedupes_shared_documents_and_keeps_provenance():
    passages, questions = load([row([1, 1, -1], [SPAN, SPAN, None], "Q1"),
                                row([-1, -1, -1], [None] * 3, "Q2")])
    assert len(passages) == 3 and len(questions) == 2
    p = next(p for p in passages if p.passage_index == 1)
    assert p.text == "The capital is 東京 since 1868."
    assert (p.start_byte, p.end_byte) == P1 and p.url.endswith("/Japan") and p.title == "Japan"
