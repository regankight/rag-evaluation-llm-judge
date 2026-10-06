"""Answer a frozen sample: BM25 top-k -> OpenAI -> results/<run_id>/answers.jsonl + manifest.json.

uv run python run_pipeline.py --split dev
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

import generation
from generation import Generator
from retrieval import BM25Index
from samples import DATASET_PATH, SAMPLES_PATH, file_sha256, load_samples
from tydiqa import load, read_parquet_rows

TOP_K = 5


def git_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def record(q, hits, g) -> dict:
    return {
        "qid": q.qid,
        "question": q.question,
        "label": q.label,
        "gold_passage_id": q.gold_passage_id,
        "gold_answers": list(q.gold_answers),
        "gold_retrieved": any(h.passage.passage_id == q.gold_passage_id for h in hits),
        "retrieved": [{"rank": h.rank, "score": round(h.score, 4), **asdict(h.passage)} for h in hits],
        "answer": g.answer,
        "returned_model": g.returned_model,
        "prompt_tokens": g.prompt_tokens,
        "completion_tokens": g.completion_tokens,
        "cost_usd": g.cost_usd,
        "latency_s": g.latency_s,
        "sent_settings": g.sent_settings,
        "finish_reason": g.finish_reason,
    }


def answer_questions(questions, index, gen, out_path: Path, k: int = TOP_K) -> list[dict]:
    """Search, answer and record each question; one JSON line per answer, written as it goes."""
    rows = []
    with open(out_path, "w") as f:
        for i, q in enumerate(questions, 1):
            hits = index.search(q.question, k=k)
            row = {**record(q, hits, gen.answer(q.question, hits)), "prompt_version": gen.prompt_version}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(
                f"[{i}/{len(questions)}] {q.label:12} gold_retrieved={row['gold_retrieved']!s:5} {row['answer'][:60]!r}"
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "held_out"], required=True)
    ap.add_argument("--prompt", choices=sorted(generation.PROMPTS), default=generation.DEFAULT_PROMPT)
    ap.add_argument("--limit", type=int, help="answer only the first N questions (smoke tests)")
    args = ap.parse_args()

    load_dotenv(".env")
    model = os.environ["GENERATOR_MODEL"]
    samples = load_samples()
    if samples["dataset_sha256"] != file_sha256(DATASET_PATH):
        raise SystemExit("dataset file does not match the checksum the samples were drawn from")

    passages, questions = load(read_parquet_rows(DATASET_PATH))
    wanted = samples["splits"][args.split]
    by_id = {q.qid: q for q in questions}
    chosen = [by_id[qid] for qid in wanted][: args.limit]

    index = BM25Index(passages)
    gen = Generator(model, prompt_version=args.prompt)
    started = datetime.now(UTC)
    run_id = f"{args.split}-{args.prompt}-{started:%Y%m%dT%H%M%SZ}"
    out = Path("results") / run_id
    out.mkdir(parents=True)

    rows = answer_questions(chosen, index, gen, out / "answers.jsonl")
    cut_off = sum(r["finish_reason"] == "length" for r in rows)
    if cut_off:
        print(f"WARNING: {cut_off} answers were cut off at max_completion_tokens; they are scored as given")

    manifest = {
        "run_id": run_id,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "git": git_state(),
        "split": args.split,
        "n_questions": len(rows),
        "limit": args.limit,
        "dataset_sha256": samples["dataset_sha256"],
        "samples_sha256": file_sha256(SAMPLES_PATH),
        "retrieval": {
            "method": "bm25s",
            "top_k": TOP_K,
            "fields": "title + passage text",
            "corpus_passages": len(passages),
        },
        "generator": {
            "requested_model": model,
            "returned_models": sorted({r["returned_model"] for r in rows}),
            "sent_settings": sorted({json.dumps(r["sent_settings"], sort_keys=True) for r in rows}),
            "prompt_version": args.prompt,
            "prompt_sha256": generation.prompt_sha256(args.prompt),
            "system_prompt": generation.PROMPTS[args.prompt],
            "prices_per_1m": generation.PRICES[model],
        },
        "totals": {
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
            "answers_cut_off_at_token_limit": cut_off,
        },
        "versions": {p: version(p) for p in ("openai", "bm25s", "ragas", "langchain-community")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"\n{len(rows)} answers, ${manifest['totals']['cost_usd']:.4f} -> {out}")


if __name__ == "__main__":
    main()
