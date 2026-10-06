# Testing approach

## The principle

**An evaluation repo is a measuring instrument; if the instrument is wrong, every comparison it produces is wrong, and nothing crashes to tell you.** So the scoring code gets the same rigor as product code, and the *evaluator itself* is tested with inputs whose correct answer is known.

## Layers

| Layer | What it checks | Where | Runs in CI? |
|---|---|---|---|
| **Unit tests** (≈100) | Pure functions with **hand-calculated expected values**: string normalization, failure labels, McNemar p-values, bootstrap and Wilson intervals, labeling rules, sampling, cost | `tests/test_*.py` | Yes |
| **Integration test** (1) | Dataset rows → loader → BM25 → generator → saved records → scoring → refusal classification, with fake OpenAI (HTTP transport) and a fake judge; checks provenance survives the whole trip | `tests/test_integration.py` | Yes |
| **Reproduction check** | Regenerates every reported number from committed artifacts and fails if a single byte differs | `make reproduce` | Yes |
| **Regression-gate check** | On the real committed runs: adopting the new prompt must pass, reverting it must fail | `make gate-demo` | Yes |
| **Known-truth tests of the scorers** | 360 answers with labels guaranteed by construction, including traps; measures where string matching and the judges fail | `known_truth.py`, [decision 0008](docs/decisions/0008-known-truth-tests-replace-correctness-review.md) | No (needs the judge); results committed |
| **Live smoke tests** | One real call before any batch, with the request body captured: do the settings actually arrive? | Run by hand before long runs | No (needs keys/models) |

## Unit vs integration

Mostly unit tests, because the risk in this repo is **logic that's quietly wrong**, not components failing to connect: an off-by-one in a percentile, a definition that doesn't match the written rule, a label taken from the wrong annotator. Pure functions with hand-checked values catch those precisely and run in milliseconds.

One integration test covers the seams, where a change in one module's output shape could break the next without any unit test noticing. It uses fakes at the two network boundaries (OpenAI over HTTP, the judge as a function), so it's deterministic and runs offline.

## Happy and unhappy paths

Most tests here are **unhappy paths**, deliberately. Each came from a real failure or a real near-miss:

| Unhappy path | Test |
|---|---|
| Byte offsets sliced as characters on non-ASCII text (silent garbage) | `test_slice_bytes_on_non_ascii_text`, integration test |
| Short answer taken from an annotator who chose a different passage | `test_span_from_annotator_who_chose_another_passage_is_ignored` |
| BM25 returns arbitrary passages when no word matches | `test_no_matching_words_returns_nothing_not_arbitrary_passages` |
| A setting silently dropped from the API request | `test_settings_actually_reach_the_request_body` (mutation-checked) |
| A library rewriting a setting (RAGAS forcing temperature 1.0) | `test_pinned_settings_override_what_a_library_sends` |
| Judge returns blank or broken output | `test_nan_is_recorded_not_dropped`, `test_exception_is_recorded_with_reason_and_does_not_raise` |
| Silent prompt truncation by the local judge | `test_truncation_warnings_are_counted`, `test_truncations_are_summed_across_scoring_sessions` |
| Gate run on broken measurements (blank scores, errors, different questions) | `test_blank_accuracy_on_answerable_fails_instead_of_crashing_or_shrinking` and siblings |
| Corrupted or partial dataset download | `test_corrupted_download_is_discarded_and_never_promoted` |
| Decimal numbers matching integers ("2.3" vs "23") | `test_numbers_keep_decimals_and_drop_thousands_separators` |
| Mismatched annotation arrays | `zip(strict=True)` on every paired array; fails loudly instead of truncating |
| Passage text closing the review page's `<script>` block | `test_closing_script_tag_in_data_cannot_break_out` |

The known *limits* of a scorer are pinned as tests too (e.g. `test_known_weakness_containing_is_not_correct`), so a change that silently "fixes" or worsens them shows up.

## Do the tests catch what they claim?

Key tests were **mutation-checked**: introduce the bug, confirm the test fails, restore. For example, removing `temperature` from the request settings fails the settings test, and switching the loader to character slicing fails the integration test. A test that passes against broken code isn't protecting anything.

## What isn't tested automatically

- **Live model behavior**: real judge verdicts and real generations. These are measured, not unit-tested: known-truth tests for the judge, saved raw outputs for generations, and repeatability checks (100/100 identical judge scores across runs with identical inputs).
- **Network and provider changes.** Pinned versions plus a guard test for the RAGAS import crash, and a live smoke test before each batch.

## Running

```bash
make test        # unit + integration, offline
make lint        # ruff check + format check
make reproduce   # tests + lint + gate demo + regenerate the report and diff it
```
