# 0014: Is the faithfulness blind spot RAGAS's, or the judge's? A second library, same judge

**Status:** Accepted, 2026-11-26. Setup and predictions written **before** DeepEval was run on any answer.

## The question

RAGAS faithfulness scored 13 of the 16 answers a human labelled *unsupported* as faithful (≥ 0.5; decision 0013). That could be a RAGAS problem (how its metric is defined or prompted) or a judge problem (`qwen3-14b-8k` can't tell). Running a second library's faithfulness metric with **the same judge, settings, answers and passages** separates the two. If DeepEval catches what RAGAS missed, the library mattered. If it doesn't, swapping libraries won't fix it.

## Found while setting it up

**1. DeepEval doesn't fit in the main lockfile.** Adding `deepeval==4.3.6` to `pyproject.toml` resolved, but it downgraded RAGAS's own dependencies: `huggingface-hub` 2.1 → 1.16, `datasets` 5.1 → 5.0, `click` 8.5 → 8.3, `fsspec`. That would change the environment every committed result was produced in, with no error. The same bug class as 0002. **Fix:** DeepEval lives in its own uv project, [`experiments/deepeval/`](../../experiments/deepeval/), with its own lockfile. The main `uv.lock` is unchanged.

**2. DeepEval's default "faithfulness" means "not contradicted".** Its verdict prompt (4.3.6) says: *"Only use 'no' if retrieval context DIRECTLY CONTRADICTS the claim … Use 'borderline' for claims not backed up by context."* By default borderline **passes**. So an answer stating a fact found in no passage scores faithful unless a passage says the opposite. This is not the question decision 0013 asks. `penalize_ambiguous_claims=True` makes borderline fail. Both scores are computed from the **same stored verdicts**, so the comparison costs no extra judge calls.

## Setup

- **Answers:** the 41 answers to unanswerable questions that have blind grounding labels (v1: 29, v2: 12). The human groups come from `grounding.py`: 18 supported, 7 refuses-plus-fact, 16 unsupported.
- **Judge:** `qwen3-14b-8k` via Ollama, same as RAGAS. Requests go through `judge_clients.http_client` (pinned and recorded at the wire, decision 0010): `temperature 0`, `reasoning_effort "none"`. `max_tokens` is 2048, not RAGAS's 1024, because DeepEval's structured replies (claim lists) are longer and a cut-off reply is invalid JSON. Ollama truncation is checked from its server log, as in 0005.
- **Metric:** `FaithfulnessMetric(eval_mode="llm", include_reason=False)`. Replies are constrained by the JSON schema DeepEval passes. Telemetry is off.
- **Threshold:** ≥ 0.5 counts as faithful, the same as for RAGAS in 0013.
- **Test:** on the 16 unsupported answers, exact McNemar of RAGAS vs DeepEval strict at catching them (< 0.5).

## Predictions (written before running)

1. **DeepEval default catches almost none:** at most 2 of the 16 unsupported. Few of these answers contradict a passage; they add to it or misuse it.
2. **DeepEval strict catches more than RAGAS:** at least 7 of 16, against RAGAS's 3.
3. **The cost of strict:** it flags at most 3 of the 18 supported answers.
4. **Misreads stay hard for both.** A misread answer quotes a real passage fact that doesn't answer *this* question. Every claim is backed, so no faithfulness metric should catch it. (Descriptive only: 0013 notes the misread / not-in-passages split is unreliable.)
5. **Significance is uncertain:** with 16 pairs, p < 0.05 needs at least 6 answers caught by one scorer only, and none the other way. The direction is the prediction; the p-value is reported, not promised.

**What would change the README:**
- If prediction 2 holds without breaking 3, the advice becomes: pick the faithfulness *definition*, not just the library.
- If DeepEval strict also misses most unsupported answers, the judge is the limit, and blind human labels (0013) stay the only reliable check here.

## Outcome

*Pending: to be filled in from `reports/grounding.md` after the run.*

## Next, only if the result is interesting

Answer relevancy, a G-Eval correctness rubric on the 360 known-truth cases (0008), and contextual precision/recall against `answer_in_context`. They are not run now, so this record answers one question.
