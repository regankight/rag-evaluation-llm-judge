# 0012: Code review before publishing: 10 findings, 9 fixed

**Status:** Accepted, 2026-10-06

A whole-repository review (focused on anything that could make a reported number wrong) produced 10 findings. Each was reproduced or traced to a line before fixing.

| # | Finding | Fix | Effect on reported numbers |
|---|---|---|---|
| 1 | Gate crashed (divide by zero) when every accuracy score was blank, and silently compared fewer answers when some were | Every answerable answer must have an accuracy score, or the measurement is "broken" → FAIL | None (all committed runs fully scored) |
| 2 | Scoring manifest overwritten per session, so a truncation in an earlier chunk of a resumed run disappeared | Manifest keeps one entry per session, plus a total; the gate sums sessions (old single-session manifests migrated) | None (held-out runs were scored in one session each). The dev v1 manifest predates this and reflects only its last session. |
| 3 | Gate's health check ignored context-recall errors | Errors in any metric fail the gate; blank faithfulness on refusals stays allowed (expected, P5) | None |
| 4 | String match / answer-in-context / refusal fields were stored once and never recomputed | `score_run.py` refreshes them every session; `analyze.py` recomputes them from current code | None today; protects against future scorer fixes |
| 5 | Eval-set-size CI widths came from one random subsample per size | Expected half-width 1.96·sd/√n from all scores | n=10: ±0.237 → **±0.277**; n=25/50/100 essentially unchanged |
| 6 | `normalize()` merged decimals: "2.3" matched gold "23" | Decimal points between digits kept; thousands commas still dropped | None in the committed answers |
| 7 | Known-truth "wrong" answers checked by string, not meaning ("1980s" vs gold "June 1989") | Year/decade overlap flagged; results reported with and without the 2 flagged cases (not rebuilt, since they were already scored) | API judge 359/360 → **358/358** clean; local 358/360 → **356/358** clean |
| 8 | Accuracy cost split by gold-passage retrieval, which undercounts retrieval | Split by "gold passage *or* answer text retrieved" (gold-passage split kept for reference); failure labels use the same signal | Cost where retrieval succeeded **−0.004**, missed **−0.110** (was +0.019 / −0.120). Mechanism unchanged, and cleaner |
| 9 | Bootstrap percentile indices "off by one" | **No change needed** for the 10,000 resamples used: both tails already excluded 250. Code made symmetric for any resample count | None |
| 10 | Generator ignored `finish_reason`, so cut-off answers would be scored silently | Recorded per answer, warned on in runs, checked in the analysis | None: longest answers used 61 and 52 of the 200-token cap |

Tests added for every fix (113 total). `make reproduce` regenerates the updated report.
