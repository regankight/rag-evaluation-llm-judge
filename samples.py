"""Freeze the question samples once, before any results are seen (decision D9).

dev:      used for building and tuning (prompts, thresholds).
held_out: untouched until the final runs, so tuning can't leak into reported numbers.
Run `python samples.py` once; the output file is committed.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from tydiqa import ANSWERABLE, UNANSWERABLE, Question

SEED = 20261005
SIZES = {"dev": {ANSWERABLE: 20, UNANSWERABLE: 10},
         "held_out": {ANSWERABLE: 100, UNANSWERABLE: 50}}
SAMPLES_PATH = Path("samples/samples.json")
DATASET_PATH = "data/raw/tydiqa-primary-validation.parquet"


def draw(questions: list[Question], seed: int = SEED, sizes: dict = SIZES) -> dict[str, list[str]]:
    rng = random.Random(seed)
    pools = {label: sorted(q.qid for q in questions if q.label == label)
             for label in (ANSWERABLE, UNANSWERABLE)}
    for pool in pools.values():
        rng.shuffle(pool)
    out: dict[str, list[str]] = {}
    for split, per_label in sizes.items():
        ids = []
        for label, n in per_label.items():
            if len(pools[label]) < n:
                raise ValueError(f"not enough {label} questions for {split}: need {n}")
            ids += pools[label][:n]
            pools[label] = pools[label][n:]  # removing them keeps splits disjoint
        out[split] = sorted(ids)
    return out


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_samples(path: Path = SAMPLES_PATH) -> dict:
    return json.loads(path.read_text())


if __name__ == "__main__":
    from tydiqa import load, read_parquet_rows

    if SAMPLES_PATH.exists():
        raise SystemExit(f"{SAMPLES_PATH} already exists; samples are frozen. Delete it deliberately to redraw.")
    _, questions = load(read_parquet_rows(DATASET_PATH))
    splits = draw(questions)
    SAMPLES_PATH.parent.mkdir(exist_ok=True)
    SAMPLES_PATH.write_text(json.dumps({
        "seed": SEED, "sizes": SIZES, "dataset_sha256": file_sha256(DATASET_PATH),
        "language": "english", "splits": splits}, indent=1) + "\n")
    print({k: len(v) for k, v in splits.items()}, "->", SAMPLES_PATH)
