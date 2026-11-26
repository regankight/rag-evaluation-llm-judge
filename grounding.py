"""Are answers to "unanswerable" questions actually made up? Blind human grounding labels.

TyDi QA marks a question unanswerable when annotators found no answer in its article. A model
that answers one has failed to refuse, but the answer is not necessarily invented: it can be
supported by a retrieved passage, a misreading of one, or a refusal that adds a related fact.
This script turns the blind labels in review/grounding-labels.json into those groups, compares
v1 and v2 on unsupported answers (paired, exact McNemar), and checks the judge's faithfulness
score against the human labels. Writes reports/grounding.md. Offline; reads saved artifacts only.

    uv run python grounding.py
"""

import json
import re
from collections import Counter
from pathlib import Path

from analyze import RUNS, load_human_v1, load_jsonl, mcnemar_exact, refusal
from runs import deepeval_faithfulness_file, scores_file
from scorers import ANSWERED

LABELS = Path("review/grounding-labels.json")
DEEPEVAL_MANIFEST = Path("experiments/deepeval/scoring-manifest-qwen3-14b-8k.json")
REPORT = Path("reports/grounding.md")
GROUPS = ("supported", "refuses_plus_fact", "unsupported")

# An answer that opens by saying the context lacks the answer is a refusal that keeps going.
# The labelling page had no button for that case, so these are grouped by this rule instead.
_OPENS_WITH_REFUSAL = re.compile(
    r"^(the )?(context|passages?)\b[^.]{0,40}?\b(doesn|does not|don|do not|isn|is not)\b", re.IGNORECASE
)


def group(human_label: str, answer: str) -> str:
    """supported / refuses_plus_fact / unsupported (misread or not in passages)."""
    if human_label == "supported":
        return "supported"
    if _OPENS_WITH_REFUSAL.search(answer.strip()):
        return "refuses_plus_fact"
    return "unsupported"


def load() -> tuple[dict, dict, list[str]]:
    labels = json.loads(LABELS.read_text())["labels"]
    runs = {}
    for v, path in RUNS.items():
        answers = {r["qid"]: r for r in load_jsonl(Path(path, "answers.jsonl"))}
        scores = {r["qid"]: r for r in load_jsonl(Path(path, scores_file()))}
        # DeepEval scores (decision 0014) are optional: absent until experiments/deepeval/ has run.
        deepeval = Path(path, deepeval_faithfulness_file())
        for r in load_jsonl(deepeval) if deepeval.exists() else []:
            if r["qid"] in scores:  # only next to a RAGAS score, the other side of the comparison
                scores[r["qid"]]["deepeval"] = r
        runs[v] = (answers, scores)
    unanswerable = sorted(q for q, r in runs["v1"][0].items() if r["label"] == "unanswerable")
    check_coverage(labels, runs, unanswerable, load_human_v1())
    return labels, runs, unanswerable


def check_coverage(labels: dict, runs: dict, unanswerable: list[str], human_v1: dict) -> None:
    """The labels must cover exactly the answers to unanswerable questions (v1: human refusal
    labels, v2: the exact marker). An unlabelled answer would otherwise count as not unsupported."""
    expected = {
        f"{v}_{q}"
        for v, (answers, _) in runs.items()
        for q in unanswerable
        if refusal(v, answers[q], human_v1) == ANSWERED
    }
    missing, extra = sorted(expected - labels.keys()), sorted(labels.keys() - expected)
    if missing or extra:
        raise ValueError(f"grounding labels don't match the answered questions: missing {missing}, unexpected {extra}")


def analyze(labels: dict, runs: dict, unanswerable: list[str]) -> dict:
    groups = {k: group(lbl, runs[k[:2]][0][k[3:]]["answer"]) for k, lbl in labels.items()}
    out = {"n_unanswerable": len(unanswerable)}
    for v in ("v1", "v2"):
        out[v] = {
            "answered": sum(k.startswith(v) for k in groups),
            **Counter(g for k, g in groups.items() if k.startswith(v)),
        }
    bad = {v: {q: groups.get(f"{v}_{q}") == "unsupported" for q in unanswerable} for v in ("v1", "v2")}
    fixed = sum(bad["v1"][q] and not bad["v2"][q] for q in unanswerable)
    broke = sum(bad["v2"][q] and not bad["v1"][q] for q in unanswerable)
    out["unsupported"] = {
        "v1": sum(bad["v1"].values()),
        "v2": sum(bad["v2"].values()),
        "fixed": fixed,
        "broke": broke,
        "p": mcnemar_exact(fixed, broke),
    }
    out["judge_faithful"] = {}
    for g in GROUPS:
        faith = [runs[k[:2]][1][k[3:]]["faithfulness"]["value"] for k, x in groups.items() if x == g]
        out["judge_faithful"][g] = {"n": len(faith), "faithful": sum(f is not None and f >= 0.5 for f in faith)}
    if any("deepeval" in runs[k[:2]][1].get(k[3:], {}) for k in groups):
        manifest = json.loads(DEEPEVAL_MANIFEST.read_text()) if DEEPEVAL_MANIFEST.exists() else {}
        out["deepeval"] = deepeval_vs_ragas(groups, runs, manifest.get("ollama_truncations_total", 0))
    return out


def deepeval_vs_ragas(groups: dict, runs: dict, truncations: int = 0) -> dict:
    """Per human group: how many answers each judge calls faithful (>= 0.5). DeepEval twice,
    from the same verdicts: its default (claims not backed by the passages pass) and with
    penalize_ambiguous_claims (they fail). Then, on the unsupported answers, a paired exact
    McNemar test of RAGAS vs strict DeepEval at catching them (decision 0014)."""

    def scores(k):
        r = runs[k[:2]][1][k[3:]]
        return r["faithfulness"]["value"], r.get("deepeval", {"status": "missing"})

    def faithful(x):
        return x is not None and x >= 0.5

    # Truncated prompts make scores invalid (decision 0005), and a failed or missing row would
    # otherwise count as "caught": compare only when every row scored cleanly.
    if truncations:
        return {"truncations": truncations}
    not_scored = sorted(k for k in groups if scores(k)[1]["status"] != "ok")
    if not_scored:
        return {"not_scored": not_scored}
    out = {}
    for g in GROUPS:
        rows = [scores(k) for k, x in groups.items() if x == g]
        out[g] = {
            "n": len(rows),
            "ragas": sum(faithful(f) for f, _ in rows),
            "deepeval_default": sum(faithful(d["borderline_passes"]) for _, d in rows),
            "deepeval_strict": sum(faithful(d["borderline_fails"]) for _, d in rows),
        }
    bad = [scores(k) for k, x in groups.items() if x == "unsupported"]
    only_ragas = sum(not faithful(f) and faithful(d["borderline_fails"]) for f, d in bad)
    only_deepeval = sum(faithful(f) and not faithful(d["borderline_fails"]) for f, d in bad)
    out["caught_unsupported"] = {
        "ragas": sum(not faithful(f) for f, _ in bad),
        "deepeval_strict": sum(not faithful(d["borderline_fails"]) for _, d in bad),
        "only_ragas": only_ragas,
        "only_deepeval": only_deepeval,
        "p": mcnemar_exact(only_ragas, only_deepeval),
    }
    return out


def report(a: dict) -> str:
    n = a["n_unanswerable"]
    u = a["unsupported"]

    def c(v, g):
        return a[v].get(g, 0)

    rows = [
        "# Grounding of answers to unanswerable questions",
        "",
        "Generated by `grounding.py` from `review/grounding-labels.json` and the saved held-out runs.",
        "Blind labels, one labeller, judged against the five retrieved passages only. See decision 0013.",
        "",
        f"| Of {n} unanswerable questions | v1 | v2 |",
        "|---|---|---|",
        f"| Answered (should have refused) | {a['v1']['answered']} | {a['v2']['answered']} |",
        f"| ↳ supported by a retrieved passage | {c('v1', 'supported')} | {c('v2', 'supported')} |",
        f"| ↳ refuses, then adds a related fact | {c('v1', 'refuses_plus_fact')} | {c('v2', 'refuses_plus_fact')} |",
        f"| ↳ **unsupported** (misread, or not in any passage) "
        f"| **{c('v1', 'unsupported')}** | **{c('v2', 'unsupported')}** |",
        "",
        f"Unsupported answers: {u['v1']}/{n} ({100 * u['v1'] / n:.0f}%) → {u['v2']}/{n} ({100 * u['v2'] / n:.0f}%); "
        f"v2 fixed {u['fixed']} and broke {u['broke']} (exact McNemar p = {u['p']:.2g}).",
        "",
        "## Does the judge's faithfulness score catch unsupported answers?",
        "",
        "| Human group | Answers | Judge faithfulness ≥ 0.5 |",
        "|---|---|---|",
    ]
    for g in GROUPS:
        j = a["judge_faithful"][g]
        rows.append(f"| {g.replace('_', ' ')} | {j['n']} | {j['faithful']} |")
    if "deepeval" in a:
        rows += deepeval_report(a["deepeval"])
    return "\n".join(rows) + "\n"


def deepeval_report(d: dict) -> list[str]:
    rows = ["", "## RAGAS vs DeepEval, same judge (decision 0014)", ""]
    if "truncations" in d:
        return rows + [
            f"Not reported: Ollama truncated {d['truncations']} DeepEval prompt(s), so its scores are invalid."
        ]
    if "not_scored" in d:
        return rows + [f"Not reported: DeepEval has no score for {len(d['not_scored'])} answer(s); rerun the scorer."]
    c = d["caught_unsupported"]
    rows += [
        "Answers each scorer calls faithful (≥ 0.5). Same local judge, same answers and passages.",
        "DeepEval *default*: a claim the passages don't back up passes unless they contradict it.",
        "DeepEval *strict*: `penalize_ambiguous_claims`, so it fails. Both come from the same verdicts.",
        "A good scorer calls supported answers faithful and unsupported ones not.",
        "",
        "| Human group | Answers | RAGAS | DeepEval default | DeepEval strict |",
        "|---|---|---|---|---|",
    ]
    for g in GROUPS:
        x = d[g]
        rows.append(
            f"| {g.replace('_', ' ')} | {x['n']} | {x['ragas']} | {x['deepeval_default']} | {x['deepeval_strict']} |"
        )
    n = d["unsupported"]["n"]
    rows += [
        "",
        f"Unsupported answers caught (scored < 0.5): RAGAS {c['ragas']}/{n}, DeepEval strict "
        f"{c['deepeval_strict']}/{n}; only RAGAS {c['only_ragas']}, only DeepEval {c['only_deepeval']} "
        f"(exact McNemar p = {c['p']:.2g}).",
    ]
    return rows


if __name__ == "__main__":
    text = report(analyze(*load()))
    REPORT.write_text(text)
    print(text)
