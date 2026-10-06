# 0006: Correctness via RAGAS AnswerAccuracy, not FactualCorrectness

**Status:** Accepted, 2026-10-06. Changes D6 (RAGAS metrics) in the plan.

## What happened

The first dev scoring used RAGAS `FactualCorrectness` (mode `recall`, gold answers wrapped in a sentence; record 0005's run). Correct answers scored 0:

| Response | Gold | FactualCorrectness |
|---|---|---|
| "Warframe was first released for Windows in March 2013." | March 2013 | **0.0** |
| "Aramaic was the lingua franca of the Persian Empire's western provinces." | Aramaic | **0.0** |
| "Wyoming." | Wyoming | 0.5 |

Printing the intermediate claims showed why. The judge's claim-splitting step turned the Warframe answer into the single claim *"Warframe was first released for Windows."*, dropping the date, which is the actual answer. The bare answer "March 2013" became *"The month of March occurred in the year 2013."* Also, RAGAS's `recall` mode computes true positives from the **response's** claims checked against the reference, so a lossy split of the response zeroes the score even when the reference fact is present.

This is a combination of a mid-sized local judge and short TyDi gold answers. It's not a claim that the metric is broken in general.

## Options

1. Keep `FactualCorrectness` and report its scores. Rejected: it measurably mis-scores correct answers on this data.
2. Tune the reference wording until it behaves. Rejected: that fits the metric to the dev set, and every correct answer still goes through lossy claim-splitting.
3. **RAGAS `AnswerAccuracy`:** designed for comparing a response with a short reference answer. Two judge prompts each rate the response 0 / 2 / 4, averaged to 0–1.

## Evidence (6 hardest dev cases, tested before switching)

| Case | FactualCorrectness | AnswerAccuracy |
|---|---|---|
| Warframe (correct) | 0.0 | 0.75 |
| Wyoming (correct) | 0.5 | 1.0 |
| Aramaic (correct, narrowed to "western provinces") | 0.0 | 0.5 |
| Synthetic rubber (wrong vs gold) | 0.0 | 0.0 |
| Stop-motion 1899 vs gold 1897 (wrong vs gold) | 0.0 | 0.0 |
| Vertebrate (loose paraphrase) | 0.0 | 0.25 |

Speed: ~4 s vs ~50 s per answer.

## Decision

Metrics are now **faithfulness, context recall, answer accuracy**. Answer accuracy gets the raw gold answer (`reference_answer`: "A (also accepted: B)"); context recall keeps the sentence-wrapped reference (record 0005's reasoning still applies there). `score_run.py` resumes per metric, so existing rows gain the new metric without re-running the slow ones. Old dev rows keep their `factual_correctness` values as evidence for this record.

## Consequences

- The held-out judge run drops from ~3.5 h to ~2.5–3 h.
- README finding: *the obvious correctness metric was unreliable on this data, and here's the trace that showed why.*
- Whether answer accuracy agrees with a human is not assumed; Phase 4's blind review measures it.
