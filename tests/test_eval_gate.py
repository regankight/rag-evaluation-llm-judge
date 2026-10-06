"""Gate logic on small hand-built runs: each test changes one thing from a passing baseline."""

import copy

from eval_gate import evaluate

REFUSAL = "NOT_IN_CONTEXT"


def row(label, answer, acc, gold_retrieved=True, status="ok"):
    return {
        "label": label,
        "answer": answer,
        "prompt_version": "v2-not-in-context",
        "gold_retrieved": gold_retrieved,
        "scores": {
            "faithfulness": {"value": 1.0, "status": status},
            "answer_accuracy": {"value": acc, "status": status if acc is not None else "not_applicable"},
        },
    }


def run(name, rows, sha="s", truncations=0):
    return {"name": name, "rows": rows, "samples_sha256": sha, "truncations": truncations}


def baseline():
    rows = {f"a{i}": row("answerable", "Paris.", 1.0) for i in range(20)}
    rows |= {f"u{i}": row("unanswerable", REFUSAL, None, gold_retrieved=False) for i in range(10)}
    return run("base", rows)


def test_identical_runs_pass_without_warnings():
    g = evaluate(baseline(), baseline())
    assert g.passed and not g.warnings


def test_significant_rise_in_made_up_answers_fails():
    cand = baseline()
    for i in range(8):  # 8 of 10 unanswerable now answered: exact McNemar p = 2 * 0.5**8
        cand["rows"][f"u{i}"]["answer"] = "It was 1964."
    g = evaluate(baseline(), cand)
    assert not g.passed and any("made-up answers rose" in f for f in g.failures)


def test_small_rise_in_made_up_answers_only_warns():
    cand = baseline()
    cand["rows"]["u0"]["answer"] = "It was 1964."
    g = evaluate(baseline(), cand)
    assert g.passed and any("not statistically supported" in w for w in g.warnings)


def test_consistent_accuracy_drop_fails():
    cand = baseline()
    for i in range(20):
        cand["rows"][f"a{i}"]["scores"]["answer_accuracy"]["value"] = 0.5
    g = evaluate(baseline(), cand)
    assert not g.passed and any("accuracy dropped" in f for f in g.failures)


def test_more_refusals_with_gold_retrieved_warns_but_passes():
    cand = baseline()
    cand["rows"]["a0"]["answer"] = REFUSAL
    cand["rows"]["a0"]["scores"]["answer_accuracy"]["value"] = 0.0
    g = evaluate(baseline(), cand)
    assert g.passed and any("over-refusal rose" in w for w in g.warnings)


def test_different_questions_fail():
    cand = baseline()
    cand["samples_sha256"] = "other"
    assert not evaluate(baseline(), cand).passed


def test_judge_errors_unscored_answers_or_truncations_fail():
    for breakage in ("error", "unscored", "truncated"):
        cand = copy.deepcopy(baseline())
        if breakage == "error":
            cand["rows"]["a0"]["scores"]["answer_accuracy"]["status"] = "error:ValueError:bad JSON"
        elif breakage == "unscored":
            cand["rows"]["a0"]["scores"] = None
        else:
            cand["truncations"] = 1
        g = evaluate(baseline(), cand)
        assert not g.passed and "measurement is broken" in g.failures[0], breakage
