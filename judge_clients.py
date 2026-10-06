"""Judge clients with settings pinned at the HTTP boundary (decision record 0010).

RAGAS 0.4.3 decides by model NAME that anything called gpt-5 ... gpt-19 is a reasoning model
and rewrites temperature to 1.0 (and drops top_p). GPT-6 accepts temperature 0 when
reasoning_effort is "none", so the library silently overrides a valid setting. Pinning the
settings in the HTTP transport, the last step before a request leaves, means no library
layer can change them; the request capture then records what was actually sent.
"""

from __future__ import annotations

import json

import httpx


class PinnedSettingsTransport(httpx.AsyncBaseTransport):
    """Rewrites selected JSON body fields on every request, then forwards it."""

    def __init__(self, pinned: dict, inner: httpx.AsyncBaseTransport | None = None):
        self.pinned = pinned
        self.inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        body.update(self.pinned)
        content = json.dumps(body).encode()
        headers = dict(request.headers)
        headers["content-length"] = str(len(content))
        pinned = httpx.Request(request.method, request.url, headers=headers, content=content)
        return await self.inner.handle_async_request(pinned)


class Recorder:
    """Captures what was sent and the token usage that came back."""

    def __init__(self):
        self.sent: list[dict] = []
        self.usage: list[dict] = []

    async def on_request(self, request: httpx.Request) -> None:
        body = json.loads(request.content)
        self.sent.append({k: v for k, v in body.items() if k not in ("messages", "tools", "response_format")})

    async def on_response(self, response: httpx.Response) -> None:
        await response.aread()
        try:
            u = json.loads(response.content).get("usage")
        except ValueError:
            u = None
        if u:
            self.usage.append({k: u.get(k, 0) for k in ("prompt_tokens", "completion_tokens")})

    def settings_sent(self) -> list[str]:
        return sorted({json.dumps(s, sort_keys=True) for s in self.sent})

    def tokens(self) -> dict:
        return {k: sum(u[k] for u in self.usage) for k in ("prompt_tokens", "completion_tokens")}


def http_client(pinned: dict | None, recorder: Recorder, inner: httpx.AsyncBaseTransport | None = None):
    """Pinning first, then recording, so the recorder sees the body exactly as sent. (httpx
    event hooks run before the transport, so they would see the body before pinning.)"""
    transport = inner or httpx.AsyncHTTPTransport()
    if pinned:
        transport = PinnedSettingsTransport(pinned, _RecordingTransport(recorder, transport))
    else:
        transport = _RecordingTransport(recorder, transport)
    return httpx.AsyncClient(transport=transport, timeout=600)


class _RecordingTransport(httpx.AsyncBaseTransport):
    """Records the request exactly as it goes on the wire (after any pinning)."""

    def __init__(self, recorder: Recorder, inner: httpx.AsyncBaseTransport):
        self.recorder, self.inner = recorder, inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await self.recorder.on_request(request)
        response = await self.inner.handle_async_request(request)
        await self.recorder.on_response(response)
        return response
