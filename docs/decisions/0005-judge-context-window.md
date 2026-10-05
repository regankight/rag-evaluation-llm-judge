# 0005: Pin the judge's context window; detect truncation from Ollama's log

**Status:** Accepted, 2026-10-05

## What happened

The first scoring run took ~2 minutes per answer, far slower than the Phase 0 spike. Investigating showed Ollama had loaded `qwen3:14b` with a **4,096-token context window** (`ollama ps`), although the model supports 40,960.

Measured with prompts of known length on the OpenAI-compatible endpoint:

| Context window | Prompt sent | `prompt_tokens` reported | Ollama log |
|---|---|---|---|
| default (4,096) | ~8,900 tokens | 2,050 | `truncating input prompt limit=2050 prompt=8910` |
| default (4,096) | ~28,900 tokens | 2,050 | `truncating input prompt limit=2050 prompt=28910` |
| pinned 8,192 | ~6,400 tokens | 6,410 | none |
| pinned 8,192 | ~8,900 tokens | 4,098 | `truncating input prompt limit=4098 prompt=8910` |

When a prompt exceeds the window, Ollama silently drops part of it (down to about half the window) and returns a normal response. **There is no API error, and the reported token count gets *smaller*,** so a check on `prompt_tokens` alone cannot detect it. The only trace is a WARN line in `~/.ollama/logs/server.log`.

**Impact on results so far:** none. The log shows truncation warnings only for the synthetic test prompts above. The largest real judge prompt was 2,462 tokens. The risk was real for held-out answers with longer passages.

## Decision

1. **Pin the window:** `ollama/Modelfile.qwen3-14b-8k` (`FROM qwen3:14b`, `PARAMETER num_ctx 8192`), created with `ollama create qwen3-14b-8k -f ollama/Modelfile.qwen3-14b-8k`. Same weights, no download. 8,192 is about 3× the largest real prompt. 16,384 would add roughly 2.6 GB of KV cache, too tight next to a 9–10 GB model on a 16 GB machine.
2. **Guard:** `score_run.py` counts `truncating input prompt` warnings in the Ollama log before and after each session, records the difference in the scoring manifest, and prints an INVALID warning if it's non-zero.

## Consequences

- `JUDGE_MODEL=qwen3-14b-8k`. Scores from the earlier 4-answer session (`qwen3:14b`, default window) are not reused; all answers are rescored with the pinned model.
- **Judge speed is the real cost:** ~30–70 s per metric with five-passage contexts, ~2 minutes per answer for three metrics. Dev (30 answers) is ~1 hour; held-out (150) is ~5 hours. Long runs go in the user's own terminal, which has no time limit. The scoring script is resumable.
- Same lesson as the take-home's ignored `temperature` and the GPT-6 settings check: verify what the model actually received, not what the configuration says.
