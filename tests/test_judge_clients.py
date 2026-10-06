"""No network: a fake server records the body it receives."""

import asyncio
import json

import httpx

from judge_clients import Recorder, http_client

RECEIVED = []


def fake_server(request: httpx.Request) -> httpx.Response:
    RECEIVED.append(json.loads(request.content))
    return httpx.Response(200, json={"ok": True, "usage": {"prompt_tokens": 10, "completion_tokens": 2}})


def post(client, *bodies):
    """Send each body in one client session; return the last response."""

    async def go():
        async with client:
            r = None
            for body in bodies:
                r = await client.post("https://api.example/v1/chat/completions", json=body)
            return r

    return asyncio.run(go())


def test_pinned_settings_override_what_a_library_sends():
    RECEIVED.clear()
    rec = Recorder()
    client = http_client({"temperature": 0, "reasoning_effort": "none"}, rec, httpx.MockTransport(fake_server))
    # what RAGAS 0.4.3 sends for a model named gpt-6-*
    r = post(client, {"model": "gpt-6-luna", "temperature": 1.0, "max_completion_tokens": 1024, "messages": []})
    assert r.json()["ok"] is True
    assert RECEIVED[-1]["temperature"] == 0 and RECEIVED[-1]["reasoning_effort"] == "none"
    assert RECEIVED[-1]["max_completion_tokens"] == 1024  # untouched fields pass through


def test_recorder_sees_the_body_as_sent_and_counts_tokens():
    rec = Recorder()
    client = http_client({"temperature": 0}, rec, httpx.MockTransport(fake_server))
    post(
        client,
        {"model": "m", "temperature": 1.0, "messages": [{"role": "user", "content": "hi"}]},
        {"model": "m", "temperature": 1.0, "messages": []},
    )
    assert rec.settings_sent() == ['{"model": "m", "temperature": 0}']
    assert rec.tokens() == {"prompt_tokens": 20, "completion_tokens": 4}


def test_without_pinning_the_body_is_unchanged():
    RECEIVED.clear()
    client = http_client(None, Recorder(), httpx.MockTransport(fake_server))
    post(client, {"model": "m", "temperature": 1.0})
    assert RECEIVED[-1]["temperature"] == 1.0
