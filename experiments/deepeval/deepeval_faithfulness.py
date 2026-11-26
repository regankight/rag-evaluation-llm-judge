"""Does DeepEval's faithfulness catch the unsupported answers RAGAS missed? (decision record 0014)

Scores the 41 answers to unanswerable questions that have blind grounding labels (decision 0013)
with DeepEval FaithfulnessMetric, using the same local judge and settings as RAGAS. Run from the
repo root, in this folder's own environment (it keeps DeepEval out of the main lockfile):

    uv run --project experiments/deepeval python experiments/deepeval/deepeval_faithfulness.py --limit 1
    uv run --project experiments/deepeval python experiments/deepeval/deepeval_faithfulness.py

Writes results/<run>/scores-deepeval-faithfulness-<judge>.jsonl and scoring-manifest-<judge>.json here. Resumable: rows
already scored "ok" are skipped; errors are recorded with a status, never dropped, and retried.

DeepEval asks the judge for one verdict per claim: yes, no (the passages contradict it) or
borderline (not backed by the passages). By default borderline PASSES; with
penalize_ambiguous_claims it fails. Both scores come from the same verdicts, so both are stored.
"""

from __future__ import annotations

import os

# Before deepeval is imported: no usage data leaves this machine.
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["DEEPEVAL_UPDATE_WARNING_OPT_IN"] = "0"

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from importlib.metadata import version  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from deepeval.metrics import FaithfulnessMetric  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from judge_clients import Recorder, http_client  # noqa: E402
from runs import HELD_OUT, deepeval_faithfulness_file  # noqa: E402
from score_run import JUDGE_SETTINGS, count_truncations, load_jsonl, read_log, write_rows  # noqa: E402

LABELS = ROOT / "review/grounding-labels.json"
RUNS = {"v1": ROOT / HELD_OUT["v1-baseline"], "v2": ROOT / HELD_OUT["v2-not-in-context"]}
# RAGAS sent max_tokens 1024 by itself; DeepEval's claim lists can be longer, and a cut-off
# reply is invalid JSON, so the cap is raised and recorded. Sampling settings match RAGAS.
PINNED = {**JUDGE_SETTINGS, "max_tokens": 2048}


class PinnedJudge(DeepEvalBaseLLM):
    """DeepEval's judge as our own OpenAI client, so every request goes through the pinned,
    recorded transport from judge_clients.py (decision 0010): no library layer can change the
    sampling settings, and the manifest records what was actually sent."""

    def __init__(self, client: AsyncOpenAI, model_name: str):
        self.client, self.model_name = client, model_name
        super().__init__(model_name)

    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, schema=None):
        raise NotImplementedError("async only: call metric.a_measure")

    async def a_generate(self, prompt: str, schema=None):
        kwargs = {}
        if schema is not None:  # structured output: the server constrains the reply to the schema
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        r = await self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": prompt}], **kwargs
        )
        text = r.choices[0].message.content or ""
        return schema.model_validate_json(text) if schema is not None else text


def scores_from_verdicts(verdicts: list[str]) -> dict:
    """Fraction of claims that pass. No claims scores 1.0, as in DeepEval."""
    n = len(verdicts)
    return {
        "borderline_passes": 1.0 if n == 0 else sum(v in ("yes", "borderline") for v in verdicts) / n,
        "borderline_fails": 1.0 if n == 0 else sum(v == "yes" for v in verdicts) / n,
    }


def cases() -> list[dict]:
    """The labelled answers: label keys are '<v1|v2>_<qid>' (grounding.py)."""
    labels = json.loads(LABELS.read_text())["labels"]
    answers = {v: {r["qid"]: r for r in load_jsonl(p / "answers.jsonl")} for v, p in RUNS.items()}
    return [{"key": k, "version": k[:2], **answers[k[:2]][k[3:]]} for k in sorted(labels)]


async def score_one(judge: DeepEvalBaseLLM, a: dict) -> dict:
    contexts = [r["text"] for r in a["retrieved"]]
    row = {"qid": a["qid"], "label": a["label"], "judge_model": judge.get_model_name()}
    start = time.perf_counter()
    try:
        metric = FaithfulnessMetric(model=judge, eval_mode="llm", include_reason=False, async_mode=True)
        await metric.a_measure(
            LLMTestCase(input=a["question"], actual_output=a["answer"], retrieval_context=contexts),
            _show_indicator=False,
        )
        verdicts = [str(v.verdict) for v in metric.verdicts]
        row.update(
            status="ok",
            claims=metric.claims,
            verdicts=[{"verdict": str(v.verdict), "reason": v.reason} for v in metric.verdicts],
            **scores_from_verdicts(verdicts),
        )
    except Exception as e:  # recorded, not raised: one bad answer must not stop the run
        row.update(status=f"error:{type(e).__name__}:{str(e)[:120]}", borderline_passes=None, borderline_fails=None)
    row["seconds"] = round(time.perf_counter() - start, 1)
    return row


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--limit", type=int, help="score at most N answers (smoke test before the full run)")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    model = os.environ["JUDGE_MODEL"]
    recorder = Recorder()
    client = AsyncOpenAI(
        base_url=os.environ["OLLAMA_BASE_URL"], api_key="ollama", http_client=http_client(PINNED, recorder)
    )
    judge = PinnedJudge(client, model)
    out = {v: p / deepeval_faithfulness_file(model) for v, p in RUNS.items()}
    done = {v: {r["qid"]: r for r in load_jsonl(path)} for v, path in out.items()}
    todo = [a for a in cases() if done[a["version"]].get(a["qid"], {}).get("status") != "ok"][: args.limit]
    truncations_before = count_truncations(read_log())
    deadline = time.monotonic() + args.max_minutes * 60
    print(f"{len(todo)} to score this session")

    scored = 0
    for i, a in enumerate(todo, 1):
        if time.monotonic() > deadline:
            print(f"time budget reached; rerun to continue ({len(todo) - i + 1} left)")
            break
        row = await score_one(judge, a)
        scored += 1
        done[a["version"]][a["qid"]] = row
        write_rows(out[a["version"]], list(done[a["version"]].values()))
        print(f"[{i}/{len(todo)}] {a['key']} {row['status']} passes={row['borderline_passes']}")

    truncated = count_truncations(read_log()) - truncations_before
    if truncated:
        print(f"WARNING: Ollama truncated {truncated} judge prompt(s) this session; scores are INVALID")
    # One manifest for the experiment, one entry per session, never overwritten (as score_run.py).
    manifest_path = HERE / f"scoring-manifest-{model.replace(':', '_')}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sessions": []}
    manifest["sessions"].append(
        {
            "scored_this_session": scored,
            "judge_settings_sent": recorder.settings_sent(),
            "ollama_truncations": truncated,
            "tokens": recorder.tokens(),
        }
    )
    rows = [r for v in out for r in done[v].values()]
    manifest.update(
        {
            "about": "DeepEval FaithfulnessMetric on the grounding-labelled answers (docs/decisions/0014)",
            "score_files": [str(p.relative_to(ROOT)) for p in out.values()],
            "judge_model": model,
            "judge_settings_requested": PINNED,
            "rows": len(rows),
            "scored_ok": sum(r["status"] == "ok" for r in rows),
            "ollama_truncations_total": sum(s["ollama_truncations"] for s in manifest["sessions"]),
            "versions": {p: version(p) for p in ("deepeval", "openai")},
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
