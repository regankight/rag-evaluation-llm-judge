# 0010: API judge vs local judge, on known-truth cases only

**Status:** Accepted, 2026-10-06. Setup and expectations written **before** the API judge was run on the 360 cases.

## The question

The client question this answers: *can a free local model be the judge, or do we need an API model?* We know the local judge (`qwen3-14b-8k`) scores 358/360 on known-truth cases (record 0008). The same cases scored by `gpt-6-luna` give a direct comparison of accuracy, cost and speed.

## Why only the known-truth cases

The known-truth responses are **templates** ("The answer is {gold}."), not written by any model, so an OpenAI judge has nothing of its own to favor. The real held-out answers were written by `gpt-6-luna`; scoring them with `gpt-6-luna` would mix judge quality with self-preference bias (decision D4). That's why they're excluded.

## Found while setting it up: RAGAS overrides temperature by model name

A one-call smoke test with the request captured showed RAGAS 0.4.3 sending **`temperature: 1.0`** although `temperature=0` was requested. In `ragas/llms/base.py`, any model whose name matches `gpt-5` … `gpt-19` (or `o1`–`o9`) is treated as a reasoning model: temperature is forced to 1.0 and `top_p` is dropped. That assumption is outdated for GPT-6, which accepts `temperature=0` when `reasoning_effort="none"` (verified in Phase 0, record 0004).

**Fix:** `judge_clients.py` pins `temperature` and `reasoning_effort` in the HTTP transport, the last step before the request leaves, and records the body as sent. Tested with a fake server, then verified on a real call (sent: `temperature: 0`, `reasoning_effort: "none"`; accepted).

This is the third instance of one bug class in this project's history: a setting silently not applied (Ollama ignoring `temperature` in the take-home; Ollama truncating prompts, record 0005; RAGAS rewriting temperature, here). The defense is always the same: record what was actually sent.

## Setup

Same AnswerAccuracy metric, same 360 cases, same ≥ 0.5 threshold. `gpt-6-luna`, `temperature 0`, `reasoning_effort "none"`, 8 concurrent requests. Cost computed from returned token counts at $0.10 / $0.50 per 1M input/output tokens.

## Expectations (written before running)

- Accuracy: both judges at 355+/360. The 3 bad-gold questions (record 0008) should trouble both, since the problem is in the answer key, not the judge.
- Cost: API under $0.10 for all 360; local $0 but ~30 min of machine time. API wall time: a few minutes.
- If the API judge clearly beats the local judge, the README recommends it for clients despite the cost; if they tie, the local judge wins on cost and data privacy.

## Outcome (2026-10-06)

| | Right (≥ 0.5) | Cost | Time | "Based on the passages…" answers scored 1.0 |
|---|---|---|---|---|
| `qwen3-14b-8k` (local) | 358/360 | $0 | ~23 min, one at a time | 48/60 |
| `gpt-6-luna` (API) | 359/360 | $0.055 | 5.5 min, 8 parallel | 57/60 |

Settings sent (recorded at the wire): `temperature 0`, `reasoning_effort "none"`, `max_completion_tokens 1024`. Identical scores on 337/360 cases.

- **Accuracy expectation held:** both 355+/360.
- **The API judge's one "miss" is a flaw in the test, not the judge:** for *"When did John Grisham publish his first book?"* (gold "June 1989"), the swapped answer was "The answer is 1980s.", which isn't actually wrong. The swap was checked for *string* overlap, not *meaning* overlap. A second limitation of constructed cases, alongside the bad-gold one in record 0008.
- **The real difference is robustness to wording:** the local judge marked down a meaning-neutral prefix ("Based on the passages, it is…") in 12/60 cases; the API judge in 3/60.
- **Cost expectation held:** API $0.055 for 360 cases, about 1.5 cents per 100 answers; local $0. **Speed is about parallelism, not the model:** per case the local judge took ~3.5 s, comparable to the API; the API run finished in 5.5 min because 8 calls ran at once, while the local judge on one laptop runs one at a time. (Corrected: an earlier version of this line wrongly said the local judge was ~4× slower even before parallelism.)

**What to tell a client:** on short-answer correctness, a 14B local judge matches an API judge on accuracy. Choose the local judge when data can't leave the network or volume is huge; choose the API judge for speed and consistency across answer styles. Either way, check it with known-truth cases first: that's what found both judges' real weaknesses here.

**Update (2026-10-06, record 0012):** the analysis now flags known-truth cases whose "wrong" answer shares a year or decade with the gold (only the Grisham question's two cases). Excluding them: API judge **358/358**, local judge **356/358** (its two misses are the bad answer keys from record 0008).
