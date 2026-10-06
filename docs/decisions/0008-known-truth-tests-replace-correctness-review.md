# 0008: Known-truth tests replace the human correctness review

**Status:** Accepted, 2026-10-06. Written **before** the known-truth cases were scored.

## Context

The plan's Phase 4 had two human parts:
- **Part A:** label each v1 answer refused or answered. **Done:** 150 labels, `review/review-results.json`.
- **Part B:** blind review of answer correctness. **Stopped after a few items.**

Judging whether open-domain trivia answers are correct (who founded a shipyard in 1918, what the first soap opera was) needs domain knowledge the reviewer doesn't have. The passages often don't settle it either, so the verdicts would have been guesses. A human reference built on guesses is weaker than no human reference. This is the general reason production teams use subject-matter experts or gold-labeled sets for correctness review.

## Decision

Test the scorers on **answers whose correct label is guaranteed by construction** (`known_truth.py`). 60 held-out answerable questions (seed `20261007`), 6 variants each, 360 cases:

| Variant | Response | Truth |
|---|---|---|
| exact | `{gold}` | correct |
| sentence | `The answer is {gold}.` | correct |
| reworded | `Based on the passages, it is {gold}.` | correct |
| swapped | `The answer is {other}.` | wrong |
| negation_trap | `It is not {gold}; the answer is {other}.` | wrong, but contains the gold words |
| refusal | `NOT_IN_CONTEXT` | not a correct answer |

`{other}` is another sampled question's gold answer, checked to be unrelated (neither contains the other after normalization).

**Scored by:** string matching, and RAGAS AnswerAccuracy with the Qwen judge (same settings as the main runs).

**Fixed before scoring:** AnswerAccuracy ≥ 0.5 counts as "judged correct". The full value distribution is reported too, so the threshold choice is visible. Failed judge calls are counted, not dropped.

## Expectations (written before scoring)

- **String matching** is right on exact, sentence, reworded, swapped and refusal *by construction*, and **wrong on every negation trap** (it contains the gold words). That's the point of the trap: it shows the limit precisely.
- **The judge** should get most cases right. The open questions are the negation trap (does it notice "it is not X"?) and the swapped answers that are long or topically close.

## What this does and doesn't show

- **Does:** exactly where each scorer fails on known inputs, repeatably and with no reviewer.
- **Doesn't:** how often those failure modes occur in *real* model answers. Constructed cases are cleaner than real ones. That side is covered by the held-out disagreement analysis (string match vs judge on real answers, with examples) and the 150 human refusal labels.
