# 0004: Generator is gpt-6-luna, reasoning off, temperature 0

**Status:** Accepted, 2026-10-05

## Context

The generator must be a different model family from the judge (`qwen3:14b`, decision record 0002) to avoid self-preference bias. The setup is a hosted API for generation and a local model for judging, which shows "different models for different stages" deliberately. Answers are short and grounded in a passage; comparisons need answers that are as repeatable as possible.

## Options considered (OpenAI pricing page, checked 2026-10-05; USD per 1M tokens, input / output)

| Model | Price | Verdict |
|---|---|---|
| `gpt-6-astra` | $10 / $50 | Flagship; far beyond what short factual answers need |
| `gpt-6.1-sol`, `gpt-6-sol` | $2 / $10 | 20× Luna's price, with no expected gain on this task |
| **`gpt-6-luna`** | **$0.10 / $0.50** | OpenAI's "most efficient model for focused, high-volume tasks" |

## Decision

`gpt-6-luna` via Chat Completions, with `reasoning_effort="none"` and `temperature=0`.

## Why reasoning is off

OpenAI's GPT-6 guide: when reasoning effort is not `none`, `temperature`, `top_p` and `top_logprobs` must be removed. A pinned temperature therefore *requires* reasoning off. Reasoning is also unnecessary for extracting a short answer from a passage, and costs latency and money. (Third-party gateways handle this inconsistently: one gateway's issue tracker reports GPT-6 behind Bedrock rejecting `temperature` at every effort level, including `none`.)

## Consequences

- **A settings test** checks that the request body actually carries `model`, `reasoning_effort` and `temperature` (the lesson from the Ollama payload bug).
- **No dated snapshot is listed for Luna,** so each run logs the `model` value the API returns, plus the run date.
- **`temperature=0` is not a determinism guarantee.** Every raw answer is saved, and scoring runs from saved answers.
- **Cost is logged per run** from the token counts returned by the API.
