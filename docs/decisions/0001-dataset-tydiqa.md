# 0001: Dataset is TyDi QA (primary task, dev split)

**Status:** Accepted, 2026-10-02

## Context

We need a dataset with passage relevance labels (for retrieval metrics), gold answers (for RAGAS context recall and judge validation), English for Stage 1 and Japanese for the Stage 1 preview and Stage 2, a laptop-sized corpus, and a usable license.

## Options considered

| Dataset | Rejected because |
|---|---|
| MIRACL | No gold answers; corpus is EN 32.9M / JA 7.0M passages |
| MKQA | Parallel questions and answers, but no passages, so retrieval can't be measured |
| Mr. TyDi | No gold answers |
| JQaRA | Japanese only (kept as a Stage 2 extra) |

## Decision

TyDi QA primary task, dev split (Apache 2.0). Questions are written natively in each language, not translated.

Source: `google-research-datasets/tydiqa`, `primary_task/validation-00000-of-00001.parquet`, 232,035,124 bytes,
sha256 `2e9557a6ea4bfd65f381c64b91122c90554e4b00f86e15a8b4f1353be1c81510` (downloaded 2026-10-02; not committed).

## Measured (from the downloaded file)

| | English | Japanese |
|---|---|---|
| Dev questions | 1,031 | 1,709 (both match the paper's Table 4) |
| Passages (corpus size) | 53,555 | 99,113 |
| Unique articles | 929 | 1,352 |
| Questions with ≥1 annotator marking an answer passage | 747 | 894 |
| ... with ≥2 of 3 annotators agreeing | 597 | 674 |
| Questions with ≥1 minimal (short) answer | 641 | 693 |

## Consequences

- **Offsets are UTF-8 byte offsets, not character offsets.** On Japanese text, slicing the Python string with them returns garbage silently. Example: for 化学兵器禁止条約はどこで採択された？, byte slicing gives `パリ` (correct) while character slicing gives `かには必ずし`. The loader must slice bytes, with a Japanese test case.
- **Open decision:** what counts as "answerable" and "relevant": any annotator or a majority of the three. Decide in Phase 1 and record it in its own decision record.
- **Closed-world corpus:** only the dev questions' articles, so retrieval is easier than open Wikipedia.
- **Sparse labels:** only annotated passages count as relevant, so scores are conservative.
- **Not parallel:** the English and Japanese questions differ; the comparison is "same design", not "same questions".
- **Stage 2 hook:** TyDi QA's own simplified gold-passage (GoldP) task excludes Japanese and Thai because their lack of spaces breaks standard English QA tooling.
