# 0002: Primary judge is qwen3:14b via Ollama (Phase 0 spike)

**Status:** Accepted, 2026-10-02

## Context

The judge must be a different model family from the OpenAI generator, to avoid self-preference bias, and runs locally via Ollama (user decision). The machine is an Apple M4 with 16 GB of RAM. Candidates: `qwen3:14b` (9.3 GB), `qwen3:8b` (5.2 GB, fallback), `gemma4:12b` (~8 GB, second judge in Phase 4). `qwen3.8:27b` (18 GB) does not fit.

## Spike

Throwaway script, not part of the repo: RAGAS 0.4.3 `Faithfulness` (from `ragas.metrics.collections`) with `llm_factory("qwen3:14b")` on Ollama's OpenAI-compatible endpoint. Five English TyDi QA questions, three answers each: the correct gold span, a wrong span taken from another question, and the correct span again. The HTTP request bodies were captured to check which settings were actually sent.

## Results

| Check | Result |
|---|---|
| Structured-output failures | **0 / 30 requests** |
| Repeatability (same input twice) | **15 / 15 identical scores** |
| Settings actually sent | `temperature: 0`, `reasoning_effort: "none"`, `model: qwen3:14b`. **RAGAS also adds `top_p: 0.1` and `max_tokens: 1024` on its own.** Record both in the manifest. 1024 tokens could cut off long verdict lists, so watch for it. |
| Wrong answers | **5 / 5 scored 0** |
| "Correct" answers | 2 / 5 scored 1; 3 / 5 scored 0, and **in all 3 the judge was right** (below) |
| Speed | ~10–23 s per faithfulness score (two LLM calls); first call ~73 s while the model loads |
| Memory | Ollama peak ~9.1 GB resident |

**The 3 "failures" were label problems, not judge problems:**
1. **First run, my test bug:** the short answer was taken from one annotator while the passage came from another, so the "correct" answer wasn't in the context. Fixed by taking the short answer only from an annotator who chose the same passage. The loader must do the same.
2. **After the fix, the gold answers themselves were wrong or overstated:**
   - *Surface area of the human cortex?* Gold: "1.3 square feet". The passage says each **hemisphere** is 1.3 sq ft.
   - *When was the Factortame case?* Gold: "1970". That's when the fisheries policy began; the passage doesn't mention the case.
   - *When was Quezon City founded?* Gold: "1948". The passage says it became the **capital** in 1948.

## Decision

`qwen3:14b` is the primary judge, with thinking mode off. Pin it by digest in the manifest (`bdbd181c33f2` per `ollama list`).

Dependency pin found in the spike: **`langchain-community==0.4.1`**. RAGAS 0.4.3 doesn't pin LangChain, and `langchain-community` 0.4.2 (May 2026) removed a module RAGAS imports, so a fresh install crashes on `import ragas`.

## Consequences

- **Judge validation can't treat gold answers as ground truth.** TyDi's short answers are noisy (3 of 5 here; far too small a sample to estimate the overall rate). In Phase 4, every judge-vs-gold disagreement is reviewed by hand (English) and labeled *judge wrong* / *gold wrong* / *ambiguous*. Agreement is reported both before and after review.
- **Candidate for the answerability rule (Phase 1):** require 2 of 3 annotators to agree on the passage, then take the short answer only from an annotator who chose that passage.
- **Run time is the main cost.** Rough estimate: ~15 s per metric per question, so 200 questions × 4 RAGAS metrics ≈ 3+ hours per configuration. Phase 1 must decide the RAGAS sample size and which metrics run on every configuration vs only the final one.
