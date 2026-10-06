"""Score a pipeline run: string match, failure labels, and RAGAS with the local judge.

    uv run python score_run.py results/<run_id> [--max-minutes 8]

Resumable: scores are appended per answer, and answers already scored are skipped, so a
long run can be split into chunks. Failed or blank judge scores are recorded with a
status, never dropped (decision D12).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

from scorers import (answer_in_context, classify_response, failure_label, reference_answer, reference_sentence,
                     string_match)

# RAGAS adds top_p/max_tokens on its own (seen in the Phase 0 spike); these are ours.
JUDGE_SETTINGS = {"temperature": 0, "reasoning_effort": "none"}
# Correctness uses RAGAS AnswerAccuracy (designed for short reference answers). FactualCorrectness
# was tried first and dropped: on this data its claim-splitting step loses the key fact, so
# correct answers scored 0 (docs/decisions/0006). Old rows may still carry its scores.
METRICS = ("faithfulness", "context_recall", "answer_accuracy")
OLLAMA_LOG = Path.home() / ".ollama/logs/server.log"
TRUNCATION_MARKER = "truncating input prompt"


def count_truncations(log_text: str) -> int:
    """Ollama silently truncates prompts longer than its context window; the only trace
    is a WARN line in its server log (docs/decisions/0005). Counted per scoring session."""
    return sum(1 for line in log_text.splitlines() if TRUNCATION_MARKER in line)


def read_log() -> str:
    return OLLAMA_LOG.read_text(errors="replace") if OLLAMA_LOG.exists() else ""


async def run_metric(coro_fn, **kwargs) -> dict:
    """Run one metric; turn NaN and exceptions into explicit statuses."""
    start = time.perf_counter()
    try:
        value = (await coro_fn(**kwargs)).value
        status = "nan" if value is None or (isinstance(value, float) and math.isnan(value)) else "ok"
        value = None if status == "nan" else float(value)
    except Exception as e:  # recorded, not raised: one bad answer must not stop the run
        value, status = None, f"error:{type(e).__name__}:{str(e)[:120]}"
    return {"value": value, "status": status, "seconds": round(time.perf_counter() - start, 1)}


def not_applicable(reason: str) -> dict:
    return {"value": None, "status": f"not_applicable:{reason}", "seconds": 0.0}


def deterministic_scores(a: dict) -> dict:
    contexts = [r["text"] for r in a["retrieved"]]
    answerable = a["label"] == "answerable"
    correct = answerable and string_match(a["answer"], a["gold_answers"])
    in_ctx = answerable and answer_in_context(a["gold_answers"], contexts)
    return {"string_match": correct if answerable else None,
            "answer_in_context": in_ctx if answerable else None,
            "gold_passage_retrieved": a["gold_retrieved"] if answerable else None,
            "failure_label_string_match": failure_label(answerable, in_ctx, correct),
            # Answer records made before prompt versions existed are v1 (record 0007).
            **classify_response(a["answer"], a.get("prompt_version", "v1-baseline"))}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def missing_metrics(row: dict) -> list[str]:
    return [m for m in METRICS if m not in row]


def pending(answers: list[dict], done: list[dict]) -> list[dict]:
    """Answers with no score row yet, or a row missing a metric (e.g. one added later)."""
    by_id = {d["qid"]: d for d in done}
    return [a for a in answers if missing_metrics(by_id.get(a["qid"], {}))]


def write_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)  # atomic: an interrupted write never leaves a half-written file


async def main() -> None:
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerAccuracy, ContextRecall, Faithfulness

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--max-minutes", type=float, default=8.0)
    args = ap.parse_args()
    load_dotenv(".env")
    judge = os.environ["JUDGE_MODEL"]

    sent: list[dict] = []

    async def capture(req: httpx.Request) -> None:
        body = json.loads(req.content)
        sent.append({k: v for k, v in body.items() if k not in ("messages", "tools", "response_format")})

    client = AsyncOpenAI(base_url=os.environ["OLLAMA_BASE_URL"], api_key="ollama",
                         http_client=httpx.AsyncClient(event_hooks={"request": [capture]}, timeout=600))
    llm = llm_factory(judge, client=client, **JUDGE_SETTINGS)
    faith, recall, accuracy = Faithfulness(llm=llm), ContextRecall(llm=llm), AnswerAccuracy(llm=llm)

    answers = load_jsonl(args.run_dir / "answers.jsonl")
    out_path = args.run_dir / f"scores-{judge.replace(':', '_')}.jsonl"
    existing = {r["qid"]: r for r in load_jsonl(out_path)}
    todo = pending(answers, list(existing.values()))
    truncations_before = count_truncations(read_log())
    deadline = time.monotonic() + args.max_minutes * 60
    print(f"{len(answers) - len(todo)} fully scored, {len(todo)} to go")

    for i, a in enumerate(todo, 1):
        if time.monotonic() > deadline:
            print(f"time budget reached; rerun to continue ({len(todo) - i + 1} left)")
            break
        contexts = [r["text"] for r in a["retrieved"]]
        answerable = a["label"] == "answerable"
        row = existing.get(a["qid"]) or {"qid": a["qid"], "label": a["label"], **deterministic_scores(a),
                                         "judge_model": judge}
        for m in missing_metrics(row):
            if m == "faithfulness":
                row[m] = (await run_metric(faith.ascore, user_input=a["question"], response=a["answer"],
                                           retrieved_contexts=contexts) if contexts else not_applicable("no_context"))
            elif not answerable:
                row[m] = not_applicable("no_reference")
            elif m == "context_recall":
                row[m] = (await run_metric(recall.ascore, user_input=a["question"], retrieved_contexts=contexts,
                                           reference=reference_sentence(a["question"], a["gold_answers"]))
                          if contexts else not_applicable("no_context"))
            elif m == "answer_accuracy":
                row[m] = await run_metric(accuracy.ascore, user_input=a["question"], response=a["answer"],
                                          reference=reference_answer(a["gold_answers"]))
        existing[a["qid"]] = row
        write_rows(out_path, [existing[x["qid"]] for x in answers if x["qid"] in existing])
        print(f"[{i}/{len(todo)}] {a['label']:12} sm={row['string_match']!s:5} "
              + " ".join(f"{m}={row[m]['value']}" for m in METRICS))

    truncated = count_truncations(read_log()) - truncations_before
    if truncated:
        print(f"WARNING: Ollama truncated {truncated} judge prompt(s) this session; scores are INVALID")
    rows = load_jsonl(out_path)
    statuses = Counter(f"{m}:{r[m]['status'].split(':')[0]}" for r in rows
                       for m in METRICS if m in r)
    manifest = {
        "answers_run": args.run_dir.name, "scored": len(rows), "of": len(answers),
        "judge_model": judge, "judge_settings_requested": JUDGE_SETTINGS,
        "judge_settings_sent_this_session": sorted({json.dumps(s, sort_keys=True) for s in sent}),
        "metrics": list(METRICS),
        "ollama_truncations_this_session": truncated, "status_counts": dict(sorted(statuses.items())),
        "versions": {p: version(p) for p in ("ragas", "langchain-community", "openai")},
    }
    (args.run_dir / f"scoring-manifest-{judge.replace(':', '_')}.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["status_counts"]))


if __name__ == "__main__":
    asyncio.run(main())
