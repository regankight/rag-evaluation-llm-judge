# 0011: A regression gate for prompt and model changes

**Status:** Accepted, 2026-10-06

## Context

The experiments compare two prompts once. A client needs the same comparison on *every* prompt or model change, with a yes/no answer a CI pipeline can act on. `eval_gate.py` turns the analysis into that check: offline, on runs already answered and scored, reusing the paired statistics from `analyze.py`.

## Rules

| Check | Fails when | Why |
|---|---|---|
| Same questions | Different frozen sample or question set | Otherwise the comparison is meaningless |
| Measurement health | Any unscored answer, judge error, or prompt truncation | A broken measurement must never produce a pass |
| Made-up answers (unanswerable) | Rate rises **and** exact McNemar p < 0.05 | Hallucination is the risk clients weigh most. Requiring evidence avoids blocking on noise |
| Answer accuracy (answerable) | 95% paired-bootstrap CI of the change is entirely below 0 | A real drop, not run-to-run noise |
| Over-refusal | Never fails; **warns** if it rises | A product trade-off for a human to decide, not a bug |

A rise that isn't statistically supported produces a **warning**, not a failure.

## Behavior on the committed runs (`make gate-demo`, part of `make reproduce` and CI)

- **Adopting v2** (NOT_IN_CONTEXT) over v1: **PASS**, with two warnings: accuracy −0.048 (CI includes 0) and over-refusal 1/52 → 2/52.
- **Reverting** v2 → v1: **FAIL**: made-up answers 12/50 → 26/50 (worse on 15, better on 1; p = 0.00052).

## Known limitation: automatic refusal detection

A CI gate can't wait for human labels, so refusals are classified automatically: the exact marker for v2, a first-pass pattern for v1's free text. On v1 the pattern finds **26/50** made-up answers, while the 150 human labels (Part A) find **29/50**. The two disagree in both directions: the pattern counts some hedged answers ("refuses, then adds a related fact") as refusals where the human counted an answer, and misses a few plain refusals. The gate's verdict is the same either way, but **published rates use the human labels**, and a client deploying this should give the model an explicit refusal marker, as v2 does, so detection is exact.

## Consequences

- `make reproduce` exercises the gate on real artifacts: adopting must pass and reverting must fail, or CI goes red.
- Unit tests change one thing at a time from a passing baseline: a significant vs a small rise in made-up answers, an accuracy drop, over-refusal, different questions, and each kind of broken measurement.
- Natural next step (not built): send each gate result as a trace event via `rag-otel-tracer`, so quality regressions show up next to latency in the same dashboard.
