import asyncio
import math

from score_run import METRICS, deterministic_scores, missing_metrics, pending, run_metric


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


FULL = {m: {} for m in METRICS}


def test_pending_skips_fully_scored_answers():
    answers = [{"qid": "a"}, {"qid": "b"}, {"qid": "c"}]
    assert [a["qid"] for a in pending(answers, [{"qid": "b", **FULL}])] == ["a", "c"]


def test_row_missing_a_newly_added_metric_is_pending_again():
    old_row = {"qid": "b", "faithfulness": {}, "context_recall": {}, "factual_correctness": {}}
    assert missing_metrics(old_row) == ["answer_accuracy"]
    assert [a["qid"] for a in pending([{"qid": "b"}], [old_row])] == ["b"]


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


def test_truncation_warnings_are_counted():
    from score_run import count_truncations
    log = ('time=1 level=WARN msg="truncating input prompt" limit=4098 prompt=8910 keep=4 new=4098\n'
           "slot release: n_tokens = 2052, truncated = 0\n"
           'time=2 level=WARN msg="truncating input prompt" limit=2050 prompt=28910\n')
    assert count_truncations(log) == 2
    assert count_truncations("") == 0
