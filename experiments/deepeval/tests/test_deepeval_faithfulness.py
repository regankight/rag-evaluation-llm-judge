"""No network: a fake Ollama answers DeepEval's real calls through the pinned transport."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deepeval_faithfulness as df  # noqa: E402

from judge_clients import Recorder, http_client  # noqa: E402

CANNED = {
    "Truths": {"truths": ["The tower is 300 m tall."]},
    "Claims": {"claims": ["The tower is 300 m tall.", "It opened in 1889."]},
    "Verdicts": {"verdicts": [{"verdict": "yes", "reason": None}, {"verdict": "borderline", "reason": "not stated"}]},
}


def fake_ollama(received: list):
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        received.append(body)
        name = body["response_format"]["json_schema"]["name"]
        msg = {"role": "assistant", "content": json.dumps(CANNED[name])}
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )

    return handle


def judge_with(handler, recorder=None):
    recorder = recorder or Recorder()
    client = AsyncOpenAI(
        base_url="http://fake/v1",
        api_key="x",
        http_client=http_client(df.PINNED, recorder, httpx.MockTransport(handler)),
    )
    return df.PinnedJudge(client, "qwen3-14b-8k"), recorder


def answer(text="The tower is 300 m tall and opened in 1889."):
    return {
        "qid": "q1",
        "label": "unanswerable",
        "question": "How tall is the tower?",
        "answer": text,
        "retrieved": [{"text": "The tower is 300 m tall."}],
    }


def test_end_to_end_stores_verdicts_and_both_scores():
    received = []
    judge, recorder = judge_with(fake_ollama(received))
    row = asyncio.run(df.score_one(judge, answer()))
    assert row["status"] == "ok"
    assert [v["verdict"] for v in row["verdicts"]] == ["yes", "borderline"]
    assert row["borderline_passes"] == 1.0  # DeepEval's default: not-backed-up claims pass
    assert row["borderline_fails"] == 0.5  # penalize_ambiguous_claims
    # every request DeepEval made went out with the pinned settings, and was recorded
    assert received and all(b["temperature"] == 0 and b["reasoning_effort"] == "none" for b in received)
    assert all(b["max_tokens"] == 2048 for b in received)
    assert recorder.tokens()["prompt_tokens"] == 5 * len(received)


def test_judge_error_is_recorded_not_raised():
    judge, _ = judge_with(lambda request: httpx.Response(200, json={"choices": []}))
    row = asyncio.run(df.score_one(judge, answer()))
    assert row["status"].startswith("error:") and row["borderline_passes"] is None


def test_invalid_json_from_the_judge_is_an_error():
    def bad(request):
        msg = {"role": "assistant", "content": "not json"}
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            },
        )

    row = asyncio.run(df.score_one(judge_with(bad)[0], answer()))
    assert row["status"].startswith("error:")


def test_scores_from_verdicts():
    assert df.scores_from_verdicts(["yes", "no", "borderline", "borderline"]) == {
        "borderline_passes": 0.75,
        "borderline_fails": 0.25,
    }
    assert df.scores_from_verdicts([]) == {"borderline_passes": 1.0, "borderline_fails": 1.0}


def test_cases_are_exactly_the_grounding_labelled_answers():
    cs = df.cases()
    assert len(cs) == 41
    assert sum(c["version"] == "v1" for c in cs) == 29 and all(c["label"] == "unanswerable" for c in cs)


def test_telemetry_is_off():
    import os

    assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "1"
