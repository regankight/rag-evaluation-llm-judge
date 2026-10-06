"""Deterministic scorers: string matching and retrieval-vs-generation failure labels.

String matching is the lightweight custom metric: cheap, instant, reproducible. Its known
weakness (paraphrase, extra detail, wrong-but-containing answers) is exactly what the
LLM judge and the human review are compared against.
"""
from __future__ import annotations

import re
import string
import unicodedata

CORRECT = "correct"
RETRIEVAL_MISS = "retrieval_miss"          # answer not in context, and response wrong
GENERATION_ERROR = "generation_error"      # answer was in context, response still wrong
CORRECT_WITHOUT_CONTEXT = "correct_without_context"  # right, but not from the context (memory)
UNANSWERABLE = "unanswerable"              # labeled in Phase 3, once refusals are detectable


def normalize(text: str) -> str:
    """SQuAD-style: lowercase, drop punctuation and English articles, collapse spaces.
    NFKC also unifies variants like full-width digits; dashes become spaces."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[‐-―\-]", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation and unicodedata.category(ch)[0] != "P")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_any(haystack: str, needles: list[str]) -> bool:
    h = f" {normalize(haystack)} "
    return any(n and f" {n} " in h for n in (normalize(x) for x in needles))


def string_match(response: str, gold_answers: list[str]) -> bool:
    """True if any normalized gold answer appears as whole words in the response."""
    return contains_any(response, gold_answers)


def answer_in_context(gold_answers: list[str], context_texts: list[str]) -> bool:
    """Did retrieval put the answer in front of the model? Checks the text, not just the
    labeled passage id, because another passage can contain the answer too."""
    return any(contains_any(t, gold_answers) for t in context_texts)


def failure_label(answerable: bool, in_context: bool, correct: bool) -> str:
    """Attribute an outcome to retrieval or generation. Check the context BEFORE blaming
    the generator: in the take-home, a hit@k proxy credited retrieval for a case where
    none of the required evidence was in context, and generation got the blame."""
    if not answerable:
        return UNANSWERABLE
    if correct:
        return CORRECT if in_context else CORRECT_WITHOUT_CONTEXT
    return GENERATION_ERROR if in_context else RETRIEVAL_MISS


def reference_sentence(question: str, gold_answers: list[str]) -> str:
    """TyDi gold answers are fragments ("1939"); RAGAS decomposes references into claims,
    and a bare fragment isn't a claim. Wrap it in a sentence that carries the question."""
    first, rest = gold_answers[0], gold_answers[1:]
    extra = f" (also accepted: {'; '.join(rest)})" if rest else ""
    return f'The answer to the question "{question}" is: {first}{extra}.'


def reference_answer(gold_answers: list[str]) -> str:
    """Short reference for AnswerAccuracy, which is designed for short answers."""
    first, rest = gold_answers[0], gold_answers[1:]
    return f"{first} (also accepted: {'; '.join(rest)})" if rest else first
