"""No network: a fake HTTP transport stands in for OpenAI and records the request body."""

import json

import httpx
import pytest

from generation import NO_CONTEXT, SETTINGS, Generator, cost_usd, format_context
from retrieval import Hit
from tydiqa import Passage

SEEN = []


def fake_openai(request: httpx.Request) -> httpx.Response:
    SEEN.append(json.loads(request.content))
    return httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-6-luna",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "  Tokyo. "}}
            ],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010},
        },
    )


@pytest.fixture
def gen(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    SEEN.clear()
    return Generator("gpt-6-luna", transport=httpx.MockTransport(fake_openai))


HIT = Hit(Passage("d:0", "d", "Japan", "https://x", 0, 0, 5, "Tokyo is the capital."), 1.2, 1)


def test_settings_actually_reach_the_request_body(gen):
    g = gen.answer("What is the capital?", [HIT])
    body = SEEN[-1]
    assert body["model"] == "gpt-6-luna"
    assert body["temperature"] == 0 and body["reasoning_effort"] == "none"
    assert g.sent_settings == {"model": "gpt-6-luna", **SETTINGS}


def test_records_answer_usage_and_cost(gen):
    g = gen.answer("q", [HIT])
    assert g.answer == "Tokyo." and g.returned_model == "gpt-6-luna"
    assert (g.prompt_tokens, g.completion_tokens) == (1000, 10)
    assert g.cost_usd == pytest.approx(0.000105)  # 1000*0.10/1e6 + 10*0.50/1e6


def test_context_carries_rank_title_and_text_into_the_prompt(gen):
    gen.answer("q", [HIT])
    assert "[1] Japan: Tokyo is the capital." in SEEN[-1]["messages"][1]["content"]


def test_no_hits_says_so_explicitly_instead_of_empty_context(gen):
    gen.answer("q", [])
    assert NO_CONTEXT in SEEN[-1]["messages"][1]["content"]
    assert format_context([]) == NO_CONTEXT


def test_cost_hand_calculated():
    assert cost_usd("gpt-6-luna", 2_000_000, 1_000_000) == pytest.approx(0.20 + 0.50)


def test_unknown_model_price_fails_loudly():
    with pytest.raises(KeyError):
        cost_usd("some-other-model", 1, 1)


def test_v2_prompt_is_selected_and_carries_the_marker(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    SEEN.clear()
    g = Generator("gpt-6-luna", prompt_version="v2-not-in-context", transport=httpx.MockTransport(fake_openai))
    g.answer("q", [HIT])
    assert "reply with exactly NOT_IN_CONTEXT and nothing else" in SEEN[-1]["messages"][0]["content"]


def test_unknown_prompt_version_fails_before_any_call(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(KeyError):
        Generator("gpt-6-luna", prompt_version="v9-typo")


def test_prompt_versions_have_distinct_hashes():
    from generation import PROMPTS, prompt_sha256

    assert len({prompt_sha256(v) for v in PROMPTS}) == len(PROMPTS)


def test_finish_reason_is_recorded_so_cut_off_answers_are_visible(gen):
    assert gen.answer("q", [HIT]).finish_reason == "stop"
