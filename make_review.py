"""Build the blind human-review page (decision D11).

    uv run python make_review.py      # writes review/sample.json and review/review.html

Blind by construction: the page is generated only from answers.jsonl files. Judge scores,
string-match results and the v1 refusal pattern are never read, so they can't leak in.

Part A: every v1 held-out answer, labeled REFUSED / ANSWERED (hand check of the v1 pattern).
Part B: a seeded random sample of answers to review in depth. The reviewer judges each
answer from the question and retrieved passages first; only then is the gold answer
revealed, and the gold itself gets a quality label.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from runs import HELD_OUT as RUNS
SEED = 20261006
N_ANSWERABLE_PER_PROMPT = 25
N_UNANSWERABLE_PER_PROMPT = 8
OUT = Path("review")


def load(run_dir: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in Path(run_dir, "answers.jsonl").read_text().splitlines() if line.strip()]
    return {r["qid"]: r for r in rows}


def draw_part_b(answers: dict[str, dict[str, dict]], seed: int = SEED) -> list[dict]:
    """Each question appears under ONE prompt only, so the reviewer never compares the two
    versions of the same answer side by side."""
    rng = random.Random(seed)
    v1 = answers["v1-baseline"]
    items = []
    for label, n in (("answerable", N_ANSWERABLE_PER_PROMPT), ("unanswerable", N_UNANSWERABLE_PER_PROMPT)):
        qids = sorted(q for q, r in v1.items() if r["label"] == label)
        rng.shuffle(qids)
        for i, prompt in enumerate(answers):
            items += [{"qid": q, "prompt": prompt} for q in qids[i * n:(i + 1) * n]]
    rng.shuffle(items)
    return items


def item_id(part: str, i: int) -> str:
    return f"{part}{i:03d}"


def main() -> None:
    answers = {p: load(d) for p, d in RUNS.items()}
    rng = random.Random(SEED + 1)

    part_a = sorted(answers["v1-baseline"].values(), key=lambda r: r["qid"])
    rng.shuffle(part_a)
    a_items = [{"id": item_id("A", i), "qid": r["qid"], "question": r["question"], "answer": r["answer"]}
               for i, r in enumerate(part_a)]

    b_items = []
    for i, pick in enumerate(draw_part_b(answers)):
        r = answers[pick["prompt"]][pick["qid"]]
        b_items.append({
            "id": item_id("B", i), "qid": r["qid"], "prompt": pick["prompt"], "question": r["question"],
            "passages": [{"rank": p["rank"], "title": p["title"], "text": p["text"]} for p in r["retrieved"]],
            "answer": r["answer"],
            # Revealed only after the reviewer locks a verdict. Unanswerable questions have none.
            "gold": r["gold_answers"], "label": r["label"]})

    OUT.mkdir(exist_ok=True)
    # The key that maps opaque item ids back to questions and prompts; not shown in the page.
    (OUT / "sample.json").write_text(json.dumps({
        "seed": SEED, "runs": RUNS,
        "part_a": [{"id": x["id"], "qid": x["qid"]} for x in a_items],
        "part_b": [{"id": x["id"], "qid": x["qid"], "prompt": x["prompt"], "label": x["label"]} for x in b_items],
    }, indent=1) + "\n")
    # The page itself doesn't need qid/prompt; dropping them keeps the review blind to the prompt
    # version (although v2's NOT_IN_CONTEXT marker is visible in the answer text by design).
    page_a = [{k: x[k] for k in ("id", "question", "answer")} for x in a_items]
    page_b = [{k: x[k] for k in ("id", "question", "passages", "answer", "gold", "label")} for x in b_items]
    template = Path("review_template.html").read_text()
    html = template.replace("/*DATA*/", "const PART_A = " + json.dumps(page_a, ensure_ascii=False)
                            + ";\nconst PART_B = " + json.dumps(page_b, ensure_ascii=False) + ";")
    (OUT / "review.html").write_text(html)
    print(f"Part A: {len(a_items)} items, Part B: {len(b_items)} items -> {OUT / 'review.html'}")


if __name__ == "__main__":
    main()
