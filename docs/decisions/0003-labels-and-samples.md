# 0003: What counts as answerable, and the frozen samples

**Status:** Accepted, 2026-10-05

## Labeling rules (English dev split, 1,031 questions)

| Group | Rule | Count | Used? |
|---|---|---|---|
| **Answerable** | ≥2 of 3 annotators chose the **same** passage, and ≥1 of *those* annotators marked a short answer span | **456** | Yes |
| **Unanswerable** | All 3 annotators marked no passage | **284** | Yes |
| Disagreement | Fewer than 2 annotators agree on any one passage | 223 | Excluded: labels too unclear to score against |
| Yes/no only | Majority passage, but only YES/NO answers | 66 | Excluded: string matching a yes/no answer isn't a fair test of the scorers |
| Passage only | Majority passage, no short answer | 2 | Excluded |

Short answers are taken **only from annotators who chose the majority passage**. In the Phase 0 judge spike, pairing one annotator's span with another annotator's passage produced "correct" answers that weren't in the context (record 0002).

**Why ≥2 of 3 rather than any annotator:** cleaner labels at the cost of fewer questions. A single annotator's choice is less reliable (the TyDi paper reports per-annotation accuracy below 100%).

## Correction to earlier counts

Records 0001 and the Phase 0 plan said **597** English questions had "≥2 of 3 annotators agreeing". That number came from code that counted questions where ≥2 annotators marked *any* passage, not the *same* passage. Under the rule as written, it is **524** (456 with a span + 66 yes/no + 2 passage-only). The mistake was caught by re-deriving the count from the written rule before freezing samples. The same miscount likely affected the Japanese figure in 0001 (674); it isn't used in this repo.

Corpus size is **46,185 passages** from 929 unique English articles. Record 0001's 53,555 counted each article once per question that referenced it.

## Frozen samples

Seed `20261005`, drawn once by `samples.py` and committed as `samples/samples.json`, together with the dataset checksum. Rerunning refuses to overwrite.

| Split | Answerable | Unanswerable | Purpose |
|---|---|---|---|
| dev | 20 | 10 | Building and tuning (prompts, thresholds) |
| held-out | 100 | 50 | Untouched until final runs, so tuning can't leak into reported numbers |

The splits are disjoint and stratified by label. The draw is independent of input order (pools are sorted before shuffling).
