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

## Results (added after scoring, 2026-10-06)

360 cases, 0 failed judge calls, 0 prompt truncations.

| Variant | Truth | String match right | Judge right (≥ 0.5) | Judge values |
|---|---|---|---|---|
| exact | correct | 60/60 | 59/60 | 1.0 ×55, 0.75 ×3, 0.5 ×1, 0.25 ×1 |
| sentence | correct | 60/60 | 60/60 | 1.0 ×57, 0.5 ×3 |
| reworded | correct | 60/60 | 59/60 | 1.0 ×48, 0.75 ×10, 0.5 ×1, 0.25 ×1 |
| swapped | wrong | 60/60 | 60/60 | 0.0 ×60 |
| negation_trap | wrong | **0/60** | **60/60** | 0.0 ×60 |
| refusal | wrong | 60/60 | 60/60 | 0.0 ×60 |
| **Total** | | **300/360** | **358/360** | |

**As expected:** string matching fails every negation trap. "It is not X; the answer is Y" contains the gold words.

**Not expected:** the judge caught all 60 traps (0.0 every time).

**The judge's 2 "misses" are answer-key errors, not judge errors.** Every correct-by-construction case scored below 0.75 traces to 3 questions with a bad or contradictory gold list:
- *"Where did the Meiji Restoration take place?"* The first gold answer is "1868" (a year for a where-question). The judge scoring it low is right.
- *"Who lost the Battle of Chancellorsville?"* The gold list accepts both armies, the loser and the winner.
- *"When do children start to retain memories?"* The gold answers conflict ("three or four" vs "age of one").

So **known-truth construction inherits the answer key's errors**: "correct by construction" is only as correct as the gold answer it's built from. Same lesson as the Phase 0 spike, found a second, independent way.

**Wording sensitivity:** the same correct answer scored 1.0 in 55/60 cases as a bare answer but in only 48/60 when prefixed with "Based on the passages, it is…" (10 dropped to 0.75). The prefix doesn't change the meaning, but it does move the score. Averages of judge scores should be compared only between answers in a similar style.
