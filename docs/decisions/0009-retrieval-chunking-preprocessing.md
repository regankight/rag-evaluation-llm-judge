# 0009: Retrieval, chunking and preprocessing kept deliberately simple

**Status:** Accepted, 2026-10-06. Recorded after the fact: these choices were made in planning (plan D2) but only written into the repo now.

## Context

This repo asks *"which RAG evaluation can you trust?"*. Retrieval is the thing being evaluated, not the thing being optimized. Every retrieval choice is held fixed across all comparisons, so differences come from the prompt or the scorer, not from retrieval changing underneath.

## Decisions

| Choice | Decision | Why | What I'd do for a client |
|---|---|---|---|
| **Retrieval** | BM25 (`bm25s`), top 5, over title + passage text | Simple, deterministic, fast (1.5 s to index 46,185 passages), no embedding model. Hybrid/dense retrieval is measured in `es-hybrid-retrieval-benchmark` | Start with BM25 as the baseline, then add dense + RRF hybrid only if an eval set shows it pays for its latency |
| **Chunking** | None: TyDi's own passage boundaries | TyDi passages are the article's natural paragraphs, the unit annotators judged. Re-chunking would break the alignment between relevance labels and passages, which the failure labels depend on | Chunk on document structure (headings, paragraphs) with a token budget, keep a heading trail per chunk, and evaluate chunk size on the client's own questions |
| **Preprocessing** | Byte-offset slicing and whitespace trimming only | TyDi's plaintext is already clean. The one real hazard was offsets: UTF-8 **byte** offsets silently return the wrong text if sliced as characters (record 0001, tested) | Strip navigation, boilerplate and repeated headers/footers; normalize Unicode; deduplicate near-identical chunks; keep the original text for citations |
| **Zero-score results** | Dropped | `bm25s` returns *k* arbitrary passages with score 0 when no query word matches; passing those to the model as "context" is noise (tested) | Same, plus a minimum-score or reranker cut-off tuned on an eval set |
| **Provenance** | Every passage carries document title, URL, passage index and byte range from loader → search → prompt → answer record | Lets every answer be traced to the exact text it saw, and lets failure labels check whether the answer was in context | Same, surfaced to users as citations |

## Consequences

- Retrieval quality is modest: on held-out, the gold passage is in the top 5 for 52 of 100 answerable questions, and the gold answer text appears in some retrieved passage for 59. That's useful here, since evaluating both retrieval misses and generation errors requires both kinds to occur.
- Conclusions are about evaluation and prompting under this retrieval setup. They don't say which retriever to ship.
