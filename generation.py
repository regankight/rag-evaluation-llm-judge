"""Answer generation with OpenAI (docs/decisions/0004).

Every call records the settings that were actually SENT in the HTTP request body, not
the ones we intended to send. In the take-home this repo grew from, temperature=0 was
silently dropped by a wrong payload shape and every run used the default temperature.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import httpx
from openai import OpenAI

from retrieval import Hit

# USD per 1M tokens, OpenAI pricing page checked 2026-10-05 (docs/decisions/0004).
PRICES = {"gpt-6-luna": {"input": 0.10, "output": 0.50}}
SETTINGS = {"reasoning_effort": "none", "temperature": 0, "max_completion_tokens": 200}

REFUSAL_MARKER = "NOT_IN_CONTEXT"
USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\nAnswer concisely."
# Only the system prompt differs between versions (decision record 0007).
PROMPTS = {
    "v1-baseline": "You answer questions using the provided context passages.",
    # Wording revised once on dev for format compliance only (record 0007 addendum): the first
    # version produced free-text refusals ("The passages don't say...") instead of the marker.
    "v2-not-in-context": (
        "Answer using only the provided context passages. If the passages do not "
        f"contain the answer, reply with exactly {REFUSAL_MARKER} and nothing else. "
        "Do not explain what the passages are missing."
    ),
}
DEFAULT_PROMPT = "v1-baseline"
NO_CONTEXT = "(no passages were retrieved)"


def prompt_sha256(version: str = DEFAULT_PROMPT) -> str:
    return hashlib.sha256((PROMPTS[version] + "\x1f" + USER_TEMPLATE).encode()).hexdigest()[:16]


def format_context(hits: list[Hit]) -> str:
    if not hits:
        return NO_CONTEXT
    return "\n\n".join(f"[{h.rank}] {h.passage.title}: {h.passage.text}" for h in hits)


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = PRICES[model]
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


@dataclass
class Generation:
    answer: str
    returned_model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    sent_settings: dict
    finish_reason: str  # "length" means the answer was cut off at max_completion_tokens


class Generator:
    def __init__(
        self,
        model: str,
        prompt_version: str = DEFAULT_PROMPT,
        client: OpenAI | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.prompt_version = prompt_version
        self.system_prompt = PROMPTS[prompt_version]  # unknown version fails here, before any call
        self._last_sent: dict = {}
        http = httpx.Client(transport=transport, event_hooks={"request": [self._capture]}, timeout=120)
        self.client = client or OpenAI(http_client=http)

    def _capture(self, request: httpx.Request) -> None:
        body = json.loads(request.content)
        self._last_sent = {k: v for k, v in body.items() if k != "messages"}

    def answer(self, question: str, hits: list[Hit]) -> Generation:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": USER_TEMPLATE.format(context=format_context(hits), question=question)},
        ]
        start = time.perf_counter()
        r = self.client.chat.completions.create(model=self.model, messages=messages, **SETTINGS)
        latency = time.perf_counter() - start
        u = r.usage
        return Generation(
            answer=(r.choices[0].message.content or "").strip(),
            returned_model=r.model,
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
            cost_usd=cost_usd(self.model, u.prompt_tokens, u.completion_tokens),
            latency_s=round(latency, 3),
            sent_settings=dict(self._last_sent),
            finish_reason=r.choices[0].finish_reason,
        )
