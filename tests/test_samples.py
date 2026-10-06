import pytest

from samples import draw
from tydiqa import ANSWERABLE, UNANSWERABLE, Question

QS = [Question(f"a{i}", "q", ANSWERABLE, "d") for i in range(10)] + [
    Question(f"u{i}", "q", UNANSWERABLE, "d") for i in range(6)
]
SIZES = {"dev": {ANSWERABLE: 2, UNANSWERABLE: 1}, "held_out": {ANSWERABLE: 5, UNANSWERABLE: 3}}


def test_sizes_per_label():
    s = draw(QS, seed=1, sizes=SIZES)
    assert sum(q.startswith("a") for q in s["held_out"]) == 5
    assert sum(q.startswith("u") for q in s["held_out"]) == 3
    assert len(s["dev"]) == 3


def test_splits_are_disjoint():
    s = draw(QS, seed=1, sizes=SIZES)
    assert not set(s["dev"]) & set(s["held_out"])


def test_same_seed_same_draw_regardless_of_input_order():
    assert draw(QS, seed=1, sizes=SIZES) == draw(list(reversed(QS)), seed=1, sizes=SIZES)


def test_different_seed_changes_draw():
    assert draw(QS, seed=1, sizes=SIZES) != draw(QS, seed=2, sizes=SIZES)


def test_not_enough_questions_fails_loudly():
    with pytest.raises(ValueError, match="not enough unanswerable"):
        draw(QS, seed=1, sizes={"dev": {ANSWERABLE: 1, UNANSWERABLE: 99}})
