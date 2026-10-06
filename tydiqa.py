"""Load TyDi QA (primary task) into passages and questions, with provenance.

Labeling rules (docs/decisions/0003):
- answerable: at least 2 of 3 annotators chose the SAME passage, and at least one of
  those annotators marked a short answer span inside it.
- unanswerable: all 3 annotators marked no passage.
- everything else (disagreement, yes/no-only, passage-only) is excluded.

TyDi offsets are UTF-8 BYTE offsets. Slicing the Python str with them silently returns
the wrong text whenever the document contains non-ASCII characters.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

ANSWERABLE = "answerable"
UNANSWERABLE = "unanswerable"


@dataclass(frozen=True)
class Passage:
    passage_id: str  # "<doc_id>:<passage_index>"
    doc_id: str
    title: str
    url: str
    passage_index: int
    start_byte: int
    end_byte: int
    text: str


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    label: str  # ANSWERABLE or UNANSWERABLE
    doc_id: str
    gold_passage_id: str | None = None
    gold_answers: tuple[str, ...] = field(default_factory=tuple)


def short_hash(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]


def slice_bytes(text: str, start: int, end: int) -> str:
    return text.encode("utf-8")[start:end].decode("utf-8")


def doc_passages(row: Mapping) -> list[Passage]:
    doc_id = short_hash(row["document_url"])
    cands = row["passage_answer_candidates"]
    out = []
    for i, (s, e) in enumerate(zip(cands["plaintext_start_byte"], cands["plaintext_end_byte"], strict=True)):
        out.append(
            Passage(
                passage_id=f"{doc_id}:{i}",
                doc_id=doc_id,
                title=row["document_title"],
                url=row["document_url"],
                passage_index=i,
                start_byte=int(s),
                end_byte=int(e),
                text=slice_bytes(row["document_plaintext"], int(s), int(e)).strip(),
            )
        )
    return out


def label_question(row: Mapping) -> Question | None:
    """Apply the decision-0003 rules; return None for excluded questions."""
    a = row["annotations"]
    chosen = [int(i) for i in a["passage_answer_candidate_index"]]
    starts = [int(s) for s in a["minimal_answers_start_byte"]]
    ends = [int(e) for e in a["minimal_answers_end_byte"]]
    doc_id = short_hash(row["document_url"])
    qid = short_hash(row["language"], row["question_text"], row["document_url"])

    picked = [i for i in chosen if i >= 0]
    if not picked:
        return Question(qid, row["question_text"], UNANSWERABLE, doc_id)
    top = max(set(picked), key=picked.count)
    if picked.count(top) < 2:
        return None  # annotators disagree on which passage answers it

    # Short answers only from annotators who chose the majority passage: pairing a span
    # with a different annotator's passage produced "correct" answers missing from the
    # context in the Phase 0 judge spike (docs/decisions/0002).
    spans = []
    for k, s, e in zip(chosen, starts, ends, strict=True):
        if k == top and s >= 0:
            span = slice_bytes(row["document_plaintext"], s, e).strip()
            if span and span not in spans:
                spans.append(span)
    if not spans:
        return None  # yes/no-only or passage-only answers
    return Question(
        qid, row["question_text"], ANSWERABLE, doc_id, gold_passage_id=f"{doc_id}:{top}", gold_answers=tuple(spans)
    )


def load(rows: Iterable[Mapping]) -> tuple[list[Passage], list[Question]]:
    """Return (all passages of every document, labeled non-excluded questions)."""
    passages: dict[str, Passage] = {}
    questions = []
    for row in rows:
        for p in doc_passages(row):
            passages.setdefault(p.passage_id, p)
        q = label_question(row)
        if q is not None:
            questions.append(q)
    return list(passages.values()), questions


def read_parquet_rows(path: str, language: str = "english") -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(path)
    return df[df.language == language].to_dict("records")
