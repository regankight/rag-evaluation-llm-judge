"""Known-truth tests for the scorers (decision record 0008).

Instead of asking a human whether real answers are correct, build answers whose correct
label is guaranteed by construction, then measure where string matching and the LLM judge
get them wrong. The same idea as unit-testing code: inputs with known expected outputs.

    uv run python known_truth.py build             # writes known_truth/cases.jsonl
    uv run python known_truth.py score [--max-minutes 8]   # resumable
    uv run python known_truth.py report
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from runs import HELD_OUT, JUDGE, JUDGE_CORRECT_AT
from scorers import normalize, reference_answer, string_match

SEED = 20261007
N_QUESTIONS = 60
CASES = Path("known_truth/cases.jsonl")
SCORES = Path("known_truth/scores-{judge}.jsonl")

# variant -> (template, is_correct). {gold} is this question's answer, {other} another question's.
VARIANTS = {
    "exact":           ("{gold}", True),
    "sentence":        ("The answer is {gold}.", True),
    "reworded":        ("Based on the passages, it is {gold}.", True),
    "swapped":         ("The answer is {other}.", False),
    "negation_trap":   ("It is not {gold}; the answer is {other}.", False),  # contains the gold words
    "refusal":         ("NOT_IN_CONTEXT", False),
}


def unrelated(a: str, b: str) -> bool:
    """True if neither normalized answer contains the other, so a swap is truly wrong."""
    na, nb = normalize(a), normalize(b)
    return bool(na and nb) and na not in nb and nb not in na


def build_cases(questions: list[dict], seed: int = SEED, n: int = N_QUESTIONS) -> list[dict]:
    """questions: dicts with qid, question, gold_answers. Every question gets every variant."""
    rng = random.Random(seed)
    pool = sorted(questions, key=lambda q: q["qid"])
    rng.shuffle(pool)
    chosen = pool[:n]
    cases = []
    for i, q in enumerate(chosen):
        gold = q["gold_answers"][0]
        # deterministic "other" answer: the next chosen question whose answer is unrelated
        others = [c["gold_answers"][0] for c in chosen[i + 1:] + chosen[:i] if unrelated(gold, c["gold_answers"][0])]
        if not others:
            raise ValueError(f"no unrelated answer available for {q['qid']}")
        for name, (template, is_correct) in VARIANTS.items():
            cases.append({"case_id": f"{q['qid']}:{name}", "qid": q["qid"], "variant": name,
                          "question": q["question"], "gold_answers": q["gold_answers"],
                          "response": template.format(gold=gold, other=others[0]), "is_correct": is_correct})
    return cases


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def cmd_build() -> None:
    from samples import load_samples
    run = Path(HELD_OUT["v1-baseline"], "answers.jsonl")
    held_out = set(load_samples()["splits"]["held_out"])
    qs = [r for r in load_jsonl(run) if r["label"] == "answerable" and r["qid"] in held_out]
    cases = build_cases([{k: r[k] for k in ("qid", "question", "gold_answers")} for r in qs])
    CASES.parent.mkdir(exist_ok=True)
    CASES.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    print(f"{len(cases)} cases from {len(cases) // len(VARIANTS)} questions -> {CASES}")


async def cmd_score(max_minutes: float) -> None:
    import httpx
    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerAccuracy

    from score_run import JUDGE_SETTINGS, count_truncations, read_log, run_metric

    load_dotenv(".env")
    judge = os.environ["JUDGE_MODEL"]
    client = AsyncOpenAI(base_url=os.environ["OLLAMA_BASE_URL"], api_key="ollama",
                         http_client=httpx.AsyncClient(timeout=600))
    metric = AnswerAccuracy(llm=llm_factory(judge, client=client, **JUDGE_SETTINGS))
    out = Path(str(SCORES).format(judge=judge.replace(":", "_")))
    done = {r["case_id"] for r in load_jsonl(out)}
    todo = [c for c in load_jsonl(CASES) if c["case_id"] not in done]
    before, deadline = count_truncations(read_log()), time.monotonic() + max_minutes * 60
    print(f"{len(done)} scored, {len(todo)} to go")
    for i, c in enumerate(todo, 1):
        if time.monotonic() > deadline:
            print(f"time budget reached; rerun to continue ({len(todo) - i + 1} left)")
            break
        r = await run_metric(metric.ascore, user_input=c["question"], response=c["response"],
                             reference=reference_answer(c["gold_answers"]))
        row = {"case_id": c["case_id"], "variant": c["variant"], "is_correct": c["is_correct"],
               "string_match": string_match(c["response"], c["gold_answers"]), "answer_accuracy": r}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
    truncated = count_truncations(read_log()) - before
    print(f"truncations this session: {truncated}" + ("  <- scores INVALID" if truncated else ""))


def summarize(rows: list[dict]) -> dict:
    """Per variant: how often each scorer gives the right label, plus judge status counts."""
    out = defaultdict(lambda: {"n": 0, "string_match_right": 0, "judge_right": 0, "judge_failed": 0, "judge_values": []})
    for r in rows:
        s = out[r["variant"]]
        s["n"] += 1
        s["string_match_right"] += r["string_match"] == r["is_correct"]
        v = r["answer_accuracy"]["value"]
        if v is None:
            s["judge_failed"] += 1  # counted, never dropped (decision D12)
        else:
            s["judge_right"] += (v >= JUDGE_CORRECT_AT) == r["is_correct"]
            s["judge_values"].append(v)
    return dict(out)


def cmd_report() -> None:
    judge = os.environ.get("JUDGE_MODEL", JUDGE)
    rows = load_jsonl(Path(str(SCORES).format(judge=judge.replace(":", "_"))))
    print(f"{'variant':15} {'truth':8} {'n':>3} {'string match':>13} {'judge':>8} {'failed':>7}  judge values")
    for name, (_, correct) in VARIANTS.items():
        s = summarize(rows).get(name)
        if not s:
            continue
        dist = {v: s["judge_values"].count(v) for v in sorted(set(s["judge_values"]))}
        print(f"{name:15} {'correct' if correct else 'wrong':8} {s['n']:>3} {s['string_match_right']:>8}/{s['n']:<4}"
              f" {s['judge_right']:>4}/{s['n']:<3} {s['judge_failed']:>7}  {dist}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "score", "report"])
    ap.add_argument("--max-minutes", type=float, default=8.0)
    a = ap.parse_args()
    if a.command == "build":
        cmd_build()
    elif a.command == "score":
        asyncio.run(cmd_score(a.max_minutes))
    else:
        cmd_report()
