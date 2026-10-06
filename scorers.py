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
RETRIEVAL_MISS = "retrieval_miss"  # answer not in context, and response wrong
GENERATION_ERROR = "generation_error"  # answer was in context, response still wrong
CORRECT_WITHOUT_CONTEXT = "correct_without_context"  # right, but not from the context (memory)
UNANSWERABLE = "unanswerable"
REFUSED, ANSWERED = "refused", "answered"
REFUSAL_MARKER = "NOT_IN_CONTEXT"


_DECIMAL = "\x00"  # placeholder that survives punctuation removal (category Cc, not P)


def normalize(text: str) -> str:
    """SQuAD-style: lowercase, drop punctuation and English articles, collapse spaces.
    NFKC also unifies variants like full-width digits; dashes become spaces."""
    text = unicodedata.normalize("NFKC", text).lower()
    # Numbers: drop thousands separators ("1,000" == "1000") but keep decimal points, which
    # punctuation removal would otherwise merge ("2.3" -> "23" matched a gold "23").
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"(?<=\d)\.(?=\d)", _DECIMAL, text)
    text = re.sub(r"[‐-―\-]", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation and unicodedata.category(ch)[0] != "P")
    text = text.replace(_DECIMAL, ".")
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


# First-pass pattern for v1's free-text refusals. Every v1 classification used in results is
# checked by hand (decision record 0007), so this only needs to be a good first pass.
_V1_REFUSAL = re.compile(
    r"\b(context|passages?)\b[^.]{0,40}?\b(does not|doesn't|doesn\u2019t|do not|don't|don\u2019t|"
    r"is not|isn't|isn\u2019t)\b[^.]{0,20}?\b(specify|say|mention|give|describe|contain|confirm|state|"
    r"provide|include)",
    re.IGNORECASE,
)


def classify_response(answer: str, prompt_version: str) -> dict:
    """REFUSED or ANSWERED, plus how it was decided. v2 uses the exact marker; a response
    that contains the marker but adds other text is flagged, since v2 asks for the marker
    alone (a format-compliance problem, decision record 0007)."""
    text = answer.strip()
    if prompt_version == "v2-not-in-context":
        if text == REFUSAL_MARKER:
            return {"refusal": REFUSED, "method": "marker"}
        if text.startswith(REFUSAL_MARKER):
            return {"refusal": REFUSED, "method": "marker_with_extra_text"}
        if REFUSAL_MARKER in text:
            return {"refusal": ANSWERED, "method": "marker_not_at_start"}
        return {"refusal": ANSWERED, "method": "marker"}
    return {"refusal": REFUSED if _V1_REFUSAL.search(text) else ANSWERED, "method": "v1_pattern_needs_review"}
