import pytest

from analyze import ci_halfwidth_at, mcnemar_exact, paired_bootstrap_ci, refuses_plus_extra


def test_mcnemar_hand_calculated():
    # b=5, c=0: p = 2 * (1/2)^5 = 0.0625
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625)
    # b=6, c=0: p = 2 * (1/2)^6 = 0.03125
    assert mcnemar_exact(6, 0) == pytest.approx(0.03125)
    # b=3, c=1: tail = (C(4,0)+C(4,1)) / 16 = 5/16; p = 10/16
    assert mcnemar_exact(3, 1) == pytest.approx(0.625)


def test_mcnemar_symmetric_and_bounded():
    assert mcnemar_exact(2, 7) == mcnemar_exact(7, 2)
    assert mcnemar_exact(2, 2) == 1.0
    assert mcnemar_exact(0, 0) == 1.0


def test_bootstrap_constant_differences_have_zero_width():
    assert paired_bootstrap_ci([0.25] * 20, b=500) == (0.25, 0.25, 0.25)


def test_bootstrap_is_reproducible_and_contains_the_mean():
    diffs = [0, 0.25, -0.5, 1, 0, 0, -0.25, 0.5]
    a = paired_bootstrap_ci(diffs, seed=1, b=2000)
    assert a == paired_bootstrap_ci(diffs, seed=1, b=2000)
    assert a[1] <= a[0] <= a[2]


def test_smaller_samples_give_wider_intervals():
    values = [0, 0.25, 0.5, 0.75, 1] * 20
    assert ci_halfwidth_at(values, 10, b=1000) > ci_halfwidth_at(values, 100, b=1000)


@pytest.mark.parametrize("answer,expected", [
    ("The context doesn’t say how much student athletes earn. It notes that only 2% receive scholarships.", True),
    ("The context does not specify Georgia’s state flower.", False),       # plain refusal
    ("G.I. Joe debuted in 1964, so it was 62 years old in 2026.", False),  # plain answer
])
def test_refuses_plus_extra(answer, expected):
    assert refuses_plus_extra(answer) is expected
