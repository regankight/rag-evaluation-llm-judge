"""BM25 search over TyDi passages. Deliberately simple: retrieval is the thing being
evaluated here, not the thing being optimized (retrieval comparisons live in
es-hybrid-retrieval-benchmark)."""

from __future__ import annotations

from dataclasses import dataclass

import bm25s
from bm25s.tokenization import Tokenizer

from tydiqa import Passage


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float
    rank: int


class BM25Index:
    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.tokenizer = Tokenizer(stopwords="en")
        corpus = [f"{p.title}. {p.text}" for p in passages]
        tokens = self.tokenizer.tokenize(corpus, update_vocab=True, show_progress=False)
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokens, show_progress=False)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        tokens = self.tokenizer.tokenize([query], update_vocab=False, show_progress=False)
        ids, scores = self.bm25.retrieve(tokens, k=min(k, len(self.passages)), show_progress=False)
        hits = []
        for i, s in zip(ids[0], scores[0], strict=True):
            # bm25s still returns k "results" when nothing matches, all with score 0.
            # Those are arbitrary passages, not context, so they are dropped.
            if s > 0:
                hits.append(Hit(self.passages[int(i)], float(s), len(hits) + 1))
        return hits
