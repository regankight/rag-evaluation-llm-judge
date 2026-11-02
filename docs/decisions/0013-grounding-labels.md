# 0013: "Made-up answers" renamed; unsupported answers measured with blind labels

**Status:** Accepted, 2026-11-02

## Problem

P1 counted answers to questions TyDi QA marks unanswerable and called them "made-up answers". The label only means annotators found no answer in the question's own article. An answer can still be supported by a retrieved passage. For example, "Command & Conquer 3 was released on March 29, 2007" matches a passage word for word. So the 58% → 24% result measured *failing to refuse*, not *inventing*.

## Decision

1. **Rename** the measure to "answered an unanswerable question" in the README, the gate output and the report. The numbers don't change.
2. **Measure the invented part directly.** Every answer to an unanswerable question (v1: 29, v2: 12) was labelled by one person (the author) on a page showing only the question, the answer and the five retrieved passages. Prompt version and judge scores were hidden, and the order was shuffled. Labels: *supported* (a passage says it), *misread* (answers with a passage fact that doesn't answer this question), *not in passages* (states a fact in no passage). Raw labels: [`review/grounding-labels.json`](../../review/grounding-labels.json).
3. **Group by a stated rule** ([`grounding.py`](../../grounding.py)). *Misread* and *not in passages* together are **unsupported**. The labelling page had no button for "refuses, then adds a related fact", so answers that open by saying the context lacks the answer are grouped as *refuses plus fact* whatever label they got (7 answers, all v1). *Supported* labels are never changed.

## Result

| Of 50 unanswerable questions | v1 | v2 |
|---|---|---|
| Answered | 29 | 12 |
| ↳ supported | 10 | 8 |
| ↳ refuses plus fact | 7 | 0 |
| ↳ unsupported | 12 | 4 |

Unsupported answers fell from 12/50 to 4/50: v2 fixed 9 and broke 1 (exact McNemar p = 0.021). RAGAS faithfulness scored 13 of the 16 unsupported answers as faithful (≥ 0.5), so it can't stand in for this check.

## Limits

- One labeller, who also wrote the prompts. The labels were blind to prompt version, but not independent.
- The first-pass labels mixed up *misread* and *not in passages*: the same answer was given different labels in two places. Reporting only their union (*unsupported*) avoids depending on that split.
- These labels are judged against the passages, not real-world truth, following decision 0008.
- This measure was added after the results, so it is reported as descriptive, not as a tested prediction.
