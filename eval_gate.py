"""Regression gate: does a candidate run regress against a baseline? (decision record 0011)

    uv run python eval_gate.py <baseline_run_dir> <candidate_run_dir>   # exit 0 = pass, 1 = fail

Meant to run on every prompt or model change, offline, on runs already answered and scored.
Fails only on evidence: a statistically supported rise in answers to unanswerable questions or drop in accuracy,
or a broken measurement. Trade-offs (more refusals) are reported as warnings for a human.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from analyze import load_jsonl, mcnemar_exact, paired_bootstrap_ci
from runs import JUDGE, scores_file
from scorers import REFUSED, classify_response

ALPHA = 0.05


@dataclass
class GateResult:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[tuple[str, str]] = field(default_factory=list)  # (check, detail)

    @property
    def passed(self) -> bool:
        return not self.failures


def total_truncations(manifest: dict) -> int:
    """Truncations across every scoring session. Older manifests recorded only the last session."""
    if "sessions" in manifest:
        return sum(x.get("ollama_truncations", 0) for x in manifest["sessions"])
    return manifest.get("ollama_truncations_this_session", 0)


def load_run(run_dir: str | Path, judge: str = JUDGE) -> dict:
    run_dir = Path(run_dir)
    answers = {r["qid"]: r for r in load_jsonl(run_dir / "answers.jsonl")}
    scores = {r["qid"]: r for r in load_jsonl(run_dir / scores_file(judge))}
    manifest = json.loads((run_dir / "manifest.json").read_text())
    scoring = run_dir / f"scoring-manifest-{judge.replace(':', '_')}.json"
    truncations = total_truncations(json.loads(scoring.read_text())) if scoring.exists() else 0
    rows = {q: {**answers[q], "scores": scores.get(q)} for q in answers}
    return {
        "name": run_dir.name,
        "rows": rows,
        "samples_sha256": manifest["samples_sha256"],
        "truncations": truncations,
    }


def metric_results(scores: dict) -> list[dict]:
    """Every judge-metric result in a score row (dicts carrying a status), whatever the metric set."""
    return [v for v in scores.values() if isinstance(v, dict) and "status" in v]


def refused(row: dict) -> bool:
    return classify_response(row["answer"], row.get("prompt_version", "v1-baseline"))["refusal"] == REFUSED


def evaluate(base: dict, cand: dict) -> GateResult:
    g = GateResult()

    # 1. comparable runs
    if base["samples_sha256"] != cand["samples_sha256"] or set(base["rows"]) != set(cand["rows"]):
        g.failures.append("runs cover different questions; comparison is meaningless")
        return g
    g.checks.append(("same questions", f"{len(base['rows'])} questions, same frozen sample"))

    # 2. measurement health. Any judge error in any metric is broken measurement, and every
    # answerable answer must have an accuracy value: a blank (NaN) accuracy would otherwise drop
    # out of the paired comparison and let a gate pass on a shrunken set. Blank faithfulness is
    # expected for refusals (decision 0007, P5), so it isn't treated as broken.
    for run in (base, cand):
        rows = run["rows"].items()
        missing = [q for q, r in rows if r["scores"] is None]
        errors = [
            q
            for q, r in rows
            if r["scores"] and any(str(m.get("status", "")).startswith("error") for m in metric_results(r["scores"]))
        ]
        no_accuracy = [
            q
            for q, r in rows
            if r["scores"] and r["label"] == "answerable" and r["scores"]["answer_accuracy"]["value"] is None
        ]
        if missing or errors or no_accuracy or run["truncations"]:
            g.failures.append(
                f"{run['name']}: {len(missing)} unscored, {len(errors)} judge errors, "
                f"{len(no_accuracy)} answerable answers without an accuracy score, "
                f"{run['truncations']} prompt truncations; measurement is broken"
            )
    if g.failures:
        return g
    g.checks.append(("judge health", "every answer scored, no judge errors, no truncations"))

    qids = sorted(base["rows"])
    un = [q for q in qids if base["rows"][q]["label"] == "unanswerable"]
    ans = [q for q in qids if base["rows"][q]["label"] == "answerable"]

    # 3. answers to unanswerable questions, which should have been refused (paired)
    b_made = {q: not refused(base["rows"][q]) for q in un}
    c_made = {q: not refused(cand["rows"][q]) for q in un}
    worse = sum(1 for q in un if c_made[q] and not b_made[q])
    better = sum(1 for q in un if b_made[q] and not c_made[q])
    p = mcnemar_exact(worse, better)
    rate = f"{sum(b_made.values())}/{len(un)} -> {sum(c_made.values())}/{len(un)}"
    detail = f"{rate} (worse on {worse}, better on {better}; exact McNemar p = {p:.2g})"
    if worse > better and p < ALPHA:
        g.failures.append(f"answered-unanswerable rose: {detail}")
    elif worse > better:
        g.warnings.append(f"answered-unanswerable rose, not statistically supported: {detail}")
    g.checks.append(("answered-unanswerable", detail))

    # 4. answer accuracy on answerable questions (paired bootstrap)
    def acc(run, q):
        return run["rows"][q]["scores"]["answer_accuracy"]["value"]

    paired = [q for q in ans if acc(base, q) is not None and acc(cand, q) is not None]
    mean, lo, hi = paired_bootstrap_ci([acc(cand, q) - acc(base, q) for q in paired])
    detail = f"{mean:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] over {len(paired)} paired answers"
    if hi < 0:
        g.failures.append(f"answer accuracy dropped: {detail}")
    elif mean < 0:
        g.warnings.append(f"answer accuracy lower, not statistically supported: {detail}")
    g.checks.append(("answer accuracy (answerable)", detail))

    # 5. over-refusal: a product trade-off, so a warning, never a failure
    ret = [q for q in ans if base["rows"][q]["gold_retrieved"]]
    b_ref = sum(refused(base["rows"][q]) for q in ret)
    c_ref = sum(refused(cand["rows"][q]) for q in ret)
    detail = f"{b_ref}/{len(ret)} -> {c_ref}/{len(ret)} refused although the gold passage was retrieved"
    if c_ref > b_ref:
        g.warnings.append(f"over-refusal rose: {detail}")
    g.checks.append(("over-refusal (answerable)", detail))
    return g


def render(g: GateResult, base: str, cand: str) -> str:
    lines = [f"eval gate: {cand}  vs baseline  {base}", ""]
    lines += [f"  {c:32} {d}" for c, d in g.checks]
    lines += [
        "",
        "Refusals are classified automatically (exact marker, or a first-pass pattern for free-text prompts).",
    ]
    lines += [f"WARN  {w}" for w in g.warnings]
    lines += [f"FAIL  {f}" for f in g.failures]
    lines += ["", f"RESULT: {'PASS' if g.passed else 'FAIL'}"]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    a = ap.parse_args()
    result = evaluate(load_run(a.baseline), load_run(a.candidate))
    print(render(result, Path(a.baseline).name, Path(a.candidate).name))
    sys.exit(0 if result.passed else 1)
