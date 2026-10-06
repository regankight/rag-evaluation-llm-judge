"""The runs every analysis reads, in one place (so scripts can't drift apart).

Run folders are named <split>-<prompt>-<UTC timestamp> by run_pipeline.py; these are the
committed held-out runs scored for the reported results.
"""
HELD_OUT = {
    "v1-baseline": "results/held_out-v1-baseline-20261006T064508Z",
    "v2-not-in-context": "results/held_out-v2-not-in-context-20261006T064838Z",
}
JUDGE = "qwen3-14b-8k"           # the pinned-context judge (decision 0005)
JUDGE_CORRECT_AT = 0.5           # AnswerAccuracy >= this counts as "judged correct" (decision 0008)


def scores_file(judge: str = JUDGE) -> str:
    return f"scores-{judge.replace(':', '_')}.jsonl"
