import asyncio
import math

from score_run import deterministic_scores, pending, run_metric


class Result:
    def __init__(self, value):
        self.value = value


def run(coro):
    return asyncio.run(coro)


async def ok(**_):
    return Result(0.75)


async def nan(**_):
    return Result(math.nan)


async def boom(**_):
    raise ValueError("judge returned invalid JSON")


def test_ok_score_kept():
    r = run(run_metric(ok))
    assert r["value"] == 0.75 and r["status"] == "ok"


def test_nan_is_recorded_not_dropped():
    r = run(run_metric(nan))
    assert r["value"] is None and r["status"] == "nan"


def test_exception_is_recorded_with_reason_and_does_not_raise():
    r = run(run_metric(boom))
    assert r["value"] is None and r["status"].startswith("error:ValueError:judge returned invalid JSON")


def test_pending_skips_already_scored_answers():
    answers = [{"qid": "a"}, {"qid": "b"}, {"qid": "c"}]
    assert [a["qid"] for a in pending(answers, [{"qid": "b"}])] == ["a", "c"]


def answer(label, text, gold, contexts, gold_retrieved=False):
    return {"label": label, "answer": text, "gold_answers": gold, "gold_retrieved": gold_retrieved,
            "retrieved": [{"text": t} for t in contexts]}


def test_deterministic_scores_generation_error():
    s = deterministic_scores(answer("answerable", "It was 1948.", ["1939"], ["Founded in 1939."], True))
    assert s["string_match"] is False and s["answer_in_context"] is True
    assert s["failure_label_string_match"] == "generation_error"


def test_deterministic_scores_unanswerable_has_no_correctness():
    s = deterministic_scores(answer("unanswerable", "It was 1948.", [], ["Some text."]))
    assert s["string_match"] is None and s["answer_in_context"] is None
    assert s["failure_label_string_match"] == "unanswerable"
