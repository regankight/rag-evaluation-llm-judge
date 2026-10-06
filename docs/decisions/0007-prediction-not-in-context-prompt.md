# 0007: Prediction for the NOT_IN_CONTEXT prompt (written before running it)

**Status:** Prediction, written 2026-10-06 **before** prompt v2 was run on any question. Results are reported against it whether they confirm it or not (decision D10).

## The change

| | v1-baseline (current) | v2-not-in-context |
|---|---|---|
| System | "You answer questions using the provided context passages." | "Answer using only the provided context passages. If the passages do not contain the answer, reply with exactly: NOT_IN_CONTEXT" |
| User | Context + question + "Answer concisely." | unchanged |

**Wording rule:** v2's wording may be adjusted **once, on dev only**, and only if the model doesn't follow the marker format (e.g. wraps it in an explanation). It is frozen before any held-out question is answered.

## What the dev baseline showed (v1, 30 questions; small, a hint only)

- **Unanswerable (10):** 7 refused; 3 answered anyway (one from memory plus arithmetic, one substituting a related fact from the context, one hedged date).
- **Answerable with the gold passage not retrieved (6):** 4 answered anyway from memory or other passages (several were correct); 2 refused.
- **Answerable with the gold passage retrieved (14):** 1 refused anyway.

## Definitions (fixed now)

Each response is classified as **REFUSED** or **ANSWERED**:
- **v2:** REFUSED if the response begins with `NOT_IN_CONTEXT`.
- **v1** (free text): a first pass by a fixed pattern ("the context does not / doesn't specify / say / mention / give / describe / contain / confirm / state"), then **every v1 classification used below is checked by hand** in Phase 4.
- A response that says the context lacks the answer but **still offers a candidate answer** (e.g. a hedged date) counts as **ANSWERED**. This is conservative: it counts toward made-up answers.

Held-out sample: 100 answerable + 50 unanswerable, the same questions for both prompts (paired).

## Predictions

| # | Measure | v1 predicted | v2 predicted | How it's tested |
|---|---|---|---|---|
| P1 | **Made-up answer rate:** unanswerable questions ANSWERED | 15–35% | ≤ 10% | Paired; exact McNemar test. Predicted: v2 lower, p < 0.05 |
| P2 | **Over-refusal:** answerable, gold passage retrieved, REFUSED | ≤ 5% | 5–15% | Paired counts; reported as the cost of P1 |
| P3 | **Ungrounded answers:** answerable, gold passage *not* retrieved, ANSWERED | 50–75% | ≤ 30% | Paired counts |
| P4 | **Mean answer accuracy** (answerable) | (baseline) | 0.03–0.10 **lower** than v1 | Paired bootstrap 95% CI of the difference |
| P5 | **Faithfulness on refusals** | v1 refusals score 1.0 (as on dev) | ≥ 50% of v2 `NOT_IN_CONTEXT` responses come back blank (NaN) | Status counts; NaNs are reported, never dropped (D12) |

## What would falsify this

- **P1 fails if** v2's made-up rate is above 10%, or the paired difference isn't significant. If v1 itself is below 10%, the baseline was already well grounded and there is little room to improve; that would be reported as the finding.
- **P4 fails if** accuracy doesn't drop, meaning refusing ungrounded answers costs nothing measurable. That would be a better outcome than predicted.
- **P5 fails if** the judge scores `NOT_IN_CONTEXT` like v1's refusals (1.0). Faithfulness would then reward refusals in both versions.

## Why this matters

The headline isn't "the prompt fixed hallucination". It's the trade-off a product owner has to choose: **a grounded system that says "I don't know" more often, vs. a helpful one that answers from memory and is sometimes right for the wrong reason.** And the evaluation has to be able to show both sides.
