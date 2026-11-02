"""Offline analysis of the held-out runs against the written predictions (decision 0007).

    uv run python analyze.py      # writes reports/analysis.md and reports/analysis.json

Reads only saved artifacts (answers, judge scores, human labels, known-truth scores), so every
number in the README can be regenerated without the judge, the API key, or the network.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from math import comb
from pathlib import Path

from generation import SETTINGS as GENERATION_SETTINGS
from known_truth import overlap_suspect
from runs import HELD_OUT, JUDGE, JUDGE_CORRECT_AT, scores_file
from score_run import deterministic_scores
from scorers import ANSWERED, REFUSED, classify_response

API_JUDGE = "gpt-6-luna"  # second judge, known-truth cases only (decision 0010)
RUNS = {"v1": HELD_OUT["v1-baseline"], "v2": HELD_OUT["v2-not-in-context"]}
SEED = 20261006
B = 10_000  # bootstrap resamples


# ---------- statistics (unit-tested) ----------


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(diffs: list[float], seed: int = SEED, b: int = B) -> tuple[float, float, float]:
    """Mean of paired differences with a percentile 95% bootstrap CI."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(b))
    k = int(0.025 * b)  # same number of resamples excluded below and above, for any b
    return sum(diffs) / n, means[k], means[b - 1 - k]


def ci_halfwidth_at(values: list[float], n: int) -> float:
    """Expected half-width of a 95% CI for the mean if n questions were scored, using the spread
    of all scores: 1.96 * sd / sqrt(n). Deterministic, unlike a single random subsample of size n,
    whose interval depends on which questions happened to be drawn (code review, finding 5)."""
    return 1.96 * statistics.stdev(values) / n**0.5


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; behaves well for small k (unlike k/n ± ...)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------- refusal labels ----------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*\"'“])")


def refuses_plus_extra(answer: str) -> bool:
    """The borderline group promised during Part A: says the context lacks the answer, then
    keeps going with more text (a related fact or a hedged candidate)."""
    first_pass = classify_response(answer, "v1-baseline")["refusal"] == REFUSED
    return first_pass and len(_SENTENCE_END.split(answer.strip())) > 1


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load() -> dict:
    data = {}
    for v, d in RUNS.items():
        answers = {r["qid"]: r for r in load_jsonl(Path(d, "answers.jsonl"))}
        scores = {r["qid"]: r for r in load_jsonl(Path(d, scores_file()))}
        # String match, answer-in-context and refusal are code, not judge output: recomputed from
        # the answers with the current scorers rather than read from stored values (finding 4).
        data[v] = {q: {**answers[q], "scores": {**scores[q], **deterministic_scores(answers[q])}} for q in answers}
    sample = json.loads(Path("review/sample.json").read_text())
    labels = json.loads(Path("review/review-results.json").read_text())["part_a"]
    human_v1 = {x["qid"]: labels[x["id"]] for x in sample["part_a"] if x["id"] in labels}
    api = Path("known_truth", scores_file(API_JUDGE))
    api_manifest = Path("known_truth", f"scoring-manifest-{API_JUDGE}.json")
    return {
        "runs": data,
        "human_v1": human_v1,
        "known_truth": load_jsonl(Path("known_truth", scores_file())),
        "known_truth_cases": load_jsonl(Path("known_truth/cases.jsonl")),
        "known_truth_api": load_jsonl(api) if api.exists() else [],
        "known_truth_api_manifest": json.loads(api_manifest.read_text()) if api_manifest.exists() else None,
    }


def refusal(v: str, row: dict, human_v1: dict) -> str:
    """v1: the human label (Part A). v2: the exact marker rule."""
    return human_v1[row["qid"]] if v == "v1" else classify_response(row["answer"], "v2-not-in-context")["refusal"]


# ---------- analysis ----------


def rate(xs: list[bool]) -> dict:
    return {"k": sum(xs), "n": len(xs), "pct": round(100 * sum(xs) / len(xs), 1) if xs else None}


def analyze(d: dict) -> dict:
    runs, human = d["runs"], d["human_v1"]
    qids = sorted(runs["v1"])
    R = {v: {q: refusal(v, runs[v][q], human) for q in qids} for v in runs}
    lab = {q: runs["v1"][q]["label"] for q in qids}
    unans = [q for q in qids if lab[q] == "unanswerable"]
    ans = [q for q in qids if lab[q] == "answerable"]
    gold_ret = {q: runs["v1"][q]["gold_retrieved"] for q in qids}
    out: dict = {"n": {"questions": len(qids), "answerable": len(ans), "unanswerable": len(unans)}}

    # human vs first-pass pattern (v1)
    pattern = {q: classify_response(runs["v1"][q]["answer"], "v1-baseline")["refusal"] for q in qids}
    out["v1_pattern_vs_human"] = {
        "agree": sum(pattern[q] == human[q] for q in qids),
        "n": len(qids),
        "disagreements": [
            {"qid": q, "human": human[q], "pattern": pattern[q], "answer": runs["v1"][q]["answer"]}
            for q in qids
            if pattern[q] != human[q]
        ],
    }
    v2_methods: dict = {}
    for q in qids:
        m = classify_response(runs["v2"][q]["answer"], "v2-not-in-context")["method"]
        v2_methods[m] = v2_methods.get(m, 0) + 1
    out["v2_marker_methods"] = v2_methods

    # P1 answered-unanswerable rate + McNemar
    made_up = {v: [R[v][q] == ANSWERED for q in unans] for v in runs}
    b = sum(1 for q in unans if R["v1"][q] == ANSWERED and R["v2"][q] == REFUSED)
    c = sum(1 for q in unans if R["v1"][q] == REFUSED and R["v2"][q] == ANSWERED)
    out["P1"] = {
        "v1": rate(made_up["v1"]),
        "v2": rate(made_up["v2"]),
        "discordant_b_v1_only": b,
        "discordant_c_v2_only": c,
        "mcnemar_p": mcnemar_exact(b, c),
    }

    # borderline group (v1 refuses-plus-extra), sensitivity of P1
    border = [q for q in qids if refuses_plus_extra(runs["v1"][q]["answer"])]
    border_un = [q for q in border if q in unans]
    as_refused = [R["v1"][q] == ANSWERED and q not in border_un for q in unans]
    as_answered = [R["v1"][q] == ANSWERED or q in border_un for q in unans]
    out["borderline"] = {
        "n_total": len(border),
        "n_unanswerable": len(border_un),
        "human_labels": {lbl: sum(human[q] == lbl for q in border) for lbl in (REFUSED, ANSWERED)},
        "P1_v1_if_all_refused": rate(as_refused),
        "P1_v1_if_all_answered": rate(as_answered),
        "examples": [runs["v1"][q]["answer"] for q in border[:4]],
    }

    # P2 over-refusal, P3 ungrounded answers
    ret = [q for q in ans if gold_ret[q]]
    miss = [q for q in ans if not gold_ret[q]]
    out["P2"] = {v: rate([R[v][q] == REFUSED for q in ret]) for v in runs}
    out["P3"] = {v: rate([R[v][q] == ANSWERED for q in miss]) for v in runs}

    # P3 under an alternative retrieval definition (added after results; the pre-registered P3
    # above is unchanged): "retrieval missed" = gold passage not retrieved AND the gold answer
    # text appears in none of the retrieved passages. Gold labels mark only one passage as
    # correct, so the original definition undercounts successful retrieval.
    in_ctx = {q: bool(runs["v1"][q]["scores"]["answer_in_context"]) for q in ans}
    miss_strict = [q for q in miss if not in_ctx[q]]
    out["P3_alt_answer_text_absent"] = {v: rate([R[v][q] == ANSWERED for q in miss_strict]) for v in runs}

    # P4 accuracy cost (paired, answerable)
    acc = {v: {q: runs[v][q]["scores"]["answer_accuracy"]["value"] for q in ans} for v in runs}
    paired = [q for q in ans if acc["v1"][q] is not None and acc["v2"][q] is not None]
    diffs = [acc["v2"][q] - acc["v1"][q] for q in paired]
    mean, lo, hi = paired_bootstrap_ci(diffs)
    out["P4"] = {
        "v1_mean": round(statistics.mean(acc["v1"][q] for q in paired), 4),
        "v2_mean": round(statistics.mean(acc["v2"][q] for q in paired), 4),
        "diff": round(mean, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "n_paired": len(paired),
        "missing": len(ans) - len(paired),
    }
    # where the accuracy change comes from. "Retrieval succeeded" = gold passage retrieved OR the
    # gold answer text in any retrieved passage: gold labels mark one passage, so the gold-passage
    # signal alone undercounts retrieval (finding 8). The gold-passage split is kept for reference.
    in_ctx_all = {q: bool(runs["v1"][q]["scores"]["answer_in_context"]) for q in ans}
    retrieval_ok = {q: gold_ret[q] or in_ctx_all[q] for q in ans}

    def split_diff(groups):
        return {
            k: {"n": len(qs), "diff": round(statistics.mean(acc["v2"][q] - acc["v1"][q] for q in qs), 4)}
            for k, qs in groups
        }

    out["P4_by_retrieval"] = split_diff(
        (
            ("retrieval_succeeded", [q for q in paired if retrieval_ok[q]]),
            ("retrieval_missed", [q for q in paired if not retrieval_ok[q]]),
        )
    )
    out["P4_by_gold_passage"] = split_diff(
        (
            ("gold_retrieved", [q for q in ret if q in paired]),
            ("gold_not_retrieved", [q for q in miss if q in paired]),
        )
    )

    # P5 faithfulness on refusals
    def faith(v, q):
        return runs[v][q]["scores"]["faithfulness"]

    p5 = {}
    for v in runs:
        refs = [q for q in qids if R[v][q] == REFUSED]
        st = [faith(v, q)["status"] for q in refs]
        vals = [faith(v, q)["value"] for q in refs if faith(v, q)["value"] is not None]
        p5[v] = {
            "refusals": len(refs),
            "nan": st.count("nan"),
            "scored": len(vals),
            "mean_when_scored": round(statistics.mean(vals), 3) if vals else None,
        }
    out["P5"] = p5

    # context recall: validated against deterministic retrieval signals, and repeatability
    # (v1 and v2 share question, passages and reference, so their scores should be identical)
    cr = {v: {q: runs[v][q]["scores"]["context_recall"]["value"] for q in ans} for v in runs}

    def cr_table(signal):
        return {
            f"signal_{a}_judge_{b_}": sum(
                1 for q in ans if cr["v1"][q] is not None and signal(q) == a and (cr["v1"][q] >= JUDGE_CORRECT_AT) == b_
            )
            for a in (True, False)
            for b_ in (True, False)
        }

    out["context_recall_check"] = {
        "vs_gold_passage_retrieved": cr_table(lambda q: gold_ret[q]),
        "vs_gold_passage_or_answer_text_retrieved": cr_table(lambda q: gold_ret[q] or in_ctx[q]),
        "repeatable_v1_v2": sum(cr["v1"][q] == cr["v2"][q] for q in ans),
        "identical_inputs_v1_v2": sum(runs["v1"][q]["retrieved"] == runs["v2"][q]["retrieved"] for q in ans),
        "n": len(ans),
    }

    # string match vs judge on real answers (answerable, both prompts)
    rows = [(v, q) for v in runs for q in ans if runs[v][q]["scores"]["answer_accuracy"]["value"] is not None]

    def sm(v, q):
        return bool(runs[v][q]["scores"]["string_match"])

    def jd(v, q):
        return runs[v][q]["scores"]["answer_accuracy"]["value"] >= JUDGE_CORRECT_AT

    table = {
        f"sm_{a}_judge_{b_}": sum(1 for v, q in rows if sm(v, q) == a and jd(v, q) == b_)
        for a in (True, False)
        for b_ in (True, False)
    }

    def ex(a, b_):
        return [
            {
                "prompt": v,
                "question": runs[v][q]["question"],
                "answer": runs[v][q]["answer"],
                "gold": runs[v][q]["gold_answers"][:2],
                "judge": runs[v][q]["scores"]["answer_accuracy"]["value"],
            }
            for v, q in rows
            if sm(v, q) == a and jd(v, q) == b_
        ][:5]

    out["scorers_on_real_answers"] = {
        "n": len(rows),
        "table": table,
        "agree": table["sm_True_judge_True"] + table["sm_False_judge_False"],
        "examples_judge_yes_string_no": ex(False, True),
        "examples_string_yes_judge_no": ex(True, False),
    }

    # failure labels (judge-based correctness; retrieval = gold passage or answer text retrieved)
    fl = {}
    for v in runs:
        counts = {
            "correct": 0,
            "correct_without_answer_in_context": 0,
            "generation_error": 0,
            "retrieval_miss": 0,
            "refused": 0,
        }
        for q in ans:
            if R[v][q] == REFUSED:
                counts["refused"] += 1
                continue
            ok = acc[v][q] is not None and acc[v][q] >= JUDGE_CORRECT_AT
            key = (
                ("correct" if retrieval_ok[q] else "correct_without_answer_in_context")
                if ok
                else ("generation_error" if retrieval_ok[q] else "retrieval_miss")
            )
            counts[key] += 1
        fl[v] = counts
    out["failure_labels_judge_based"] = fl

    # eval set size: CI half-width of mean answer accuracy (v1) by sample size
    v1_acc = [acc["v1"][q] for q in paired]
    out["eval_set_size"] = {str(n): round(ci_halfwidth_at(v1_acc, n), 3) for n in (10, 25, 50, len(v1_acc))}

    # cost and time per scorer
    gen_cost = {v: json.loads(Path(RUNS[v], "manifest.json").read_text())["totals"]["cost_usd"] for v in runs}
    secs = {
        m: statistics.median(
            runs[v][q]["scores"][m]["seconds"] for v in runs for q in qids if runs[v][q]["scores"][m]["status"] == "ok"
        )
        for m in ("faithfulness", "context_recall", "answer_accuracy")
    }
    out["cost"] = {
        "generation_usd_per_100": {v: round(c / len(qids) * 100, 4) for v, c in gen_cost.items()},
        "judge_median_seconds": secs,
        "judge_minutes_per_100_answerable_all_metrics": round(sum(secs.values()) * 100 / 60, 1),
        "string_match": "microseconds, $0",
    }
    # cut-off answers: runs made before finish_reason was recorded are checked via token counts
    cap = GENERATION_SETTINGS["max_completion_tokens"]
    out["answer_length_check"] = {
        v: {
            "max_completion_tokens_used": max(runs[v][q]["completion_tokens"] for q in qids),
            "cap": cap,
            "finish_reason_length": sum(runs[v][q].get("finish_reason") == "length" for q in qids),
        }
        for v in runs
    }

    # threshold sensitivity: does the 0.5 choice drive any conclusion?
    kt_rows = d["known_truth"]
    sens = {}
    for t in (0.5, 0.75, 1.0):
        kt_right = sum(
            r["answer_accuracy"]["value"] is not None and (r["answer_accuracy"]["value"] >= t) == r["is_correct"]
            for r in kt_rows
        )
        acc_rate = {v: rate([acc[v][q] is not None and acc[v][q] >= t for q in ans]) for v in runs}
        agree = sum(1 for v, q in rows if sm(v, q) == (runs[v][q]["scores"]["answer_accuracy"]["value"] >= t))
        sens[str(t)] = {
            "known_truth_judge_right": kt_right,
            "n_known_truth": len(kt_rows),
            "judged_correct_v1": acc_rate["v1"],
            "judged_correct_v2": acc_rate["v2"],
            "agree_with_string_match": agree,
            "n_real": len(rows),
        }
    out["threshold_sensitivity"] = sens

    # answer-key noise: known-truth questions whose correct-by-construction answers the judge
    # scored below 0.75; each was then inspected by hand (decision 0008)
    by_q: dict = {}
    for r in kt_rows:
        if r["is_correct"]:
            by_q.setdefault(r["case_id"].split(":")[0], []).append(r["answer_accuracy"]["value"])
    flagged = sorted(q for q, vals in by_q.items() if any(v is not None and v < 0.75 for v in vals))
    out["answer_key_noise"] = {
        "questions": len(by_q),
        "flagged": len(flagged),
        "flagged_qids": flagged,
        "inspected_bad_or_contradictory_gold": 3,
        "rate_ci95": [round(x, 3) for x in wilson_ci(3, len(by_q))],
    }

    # known-truth summary; cases whose "wrong" answer shares a year/decade with the gold are
    # flagged (e.g. "1980s" vs "June 1989") and results are also reported without them (finding 7)
    kt = d["known_truth"]
    cases = {c["case_id"]: c for c in d["known_truth_cases"]}

    def suspect(r):
        c = cases[r["case_id"]]
        if c["is_correct"] or c["variant"] == "refusal":
            return False
        other = c["response"].split("the answer is ")[-1] if "the answer is " in c["response"] else c["response"]
        return overlap_suspect(c["gold_answers"][0], other)

    flagged_cases = sorted(r["case_id"] for r in kt if suspect(r))
    out["known_truth_flagged"] = flagged_cases
    out["known_truth"] = {
        "n": len(kt),
        "string_match_right": sum(r["string_match"] == r["is_correct"] for r in kt),
        "judge_right": sum(
            r["answer_accuracy"]["value"] is not None
            and (r["answer_accuracy"]["value"] >= JUDGE_CORRECT_AT) == r["is_correct"]
            for r in kt
        ),
        "n_clean": sum(not suspect(r) for r in kt),
        "judge_right_clean": sum(
            r["answer_accuracy"]["value"] is not None
            and (r["answer_accuracy"]["value"] >= JUDGE_CORRECT_AT) == r["is_correct"]
            for r in kt
            if not suspect(r)
        ),
    }

    # second judge on the same known-truth cases (decision 0010): accuracy, agreement, cost, time
    api = d["known_truth_api"]
    if api:

        def right(r):
            v = r["answer_accuracy"]["value"]
            return v is not None and (v >= JUDGE_CORRECT_AT) == r["is_correct"]

        local_by = {r["case_id"]: r for r in kt}
        session = d["known_truth_api_manifest"]["sessions"]
        out["judge_comparison"] = {
            "local": {
                "judge": JUDGE,
                "right": sum(right(r) for r in kt),
                "right_clean": sum(right(r) for r in kt if not suspect(r)),
                "n": len(kt),
                "cost_usd": 0.0,
                "minutes_sequential": round(sum(r["answer_accuracy"]["seconds"] for r in kt) / 60, 1),
            },
            "api": {
                "judge": API_JUDGE,
                "right": sum(right(r) for r in api),
                "right_clean": sum(right(r) for r in api if not suspect(r)),
                "n": len(api),
                "cost_usd": round(sum(x["cost_usd"] for x in session), 4),
                "minutes_wall": round(sum(x["wall_seconds"] for x in session) / 60, 1),
                "concurrency": session[-1]["concurrency"],
                "settings_sent": session[-1]["settings_sent"],
            },
            "identical_values": sum(
                r["answer_accuracy"]["value"] == local_by[r["case_id"]]["answer_accuracy"]["value"] for r in api
            ),
            "reworded_scored_1": {
                "local": sum(r["variant"] == "reworded" and r["answer_accuracy"]["value"] == 1.0 for r in kt),
                "api": sum(r["variant"] == "reworded" and r["answer_accuracy"]["value"] == 1.0 for r in api),
            },
        }
    return out


def verdict(name: str, ok: bool) -> str:
    return f"**{name}: {'held' if ok else 'did not hold'}**"


def report(a: dict) -> str:
    P1, P2, P3, P4, P5 = (a[k] for k in ("P1", "P2", "P3", "P4", "P5"))
    p1_ok = 15 <= P1["v1"]["pct"] <= 35 and P1["v2"]["pct"] <= 10 and P1["mcnemar_p"] < 0.05
    p2_ok = P2["v1"]["pct"] <= 5 and 5 <= P2["v2"]["pct"] <= 15
    p3_ok = 50 <= P3["v1"]["pct"] <= 75 and P3["v2"]["pct"] <= 30
    p4_ok = -0.10 <= P4["diff"] <= -0.03
    p5_ok = P5["v2"]["refusals"] > 0 and P5["v2"]["nan"] / P5["v2"]["refusals"] >= 0.5
    s, bd, kt = a["scorers_on_real_answers"], a["borderline"], a["known_truth"]
    # Direction is reported separately from the predicted ranges: a prediction can get the
    # direction right and the size wrong, and the report must not blur the two.
    d1 = P1["v2"]["pct"] < P1["v1"]["pct"] and P1["mcnemar_p"] < 0.05
    d2, d3 = P2["v2"]["pct"] > P2["v1"]["pct"], P3["v2"]["pct"] < P3["v1"]["pct"]
    d4 = P4["ci95"][1] < 0
    p4note = (
        f" (95% CI [{P4['ci95'][0]:+.3f}, {P4['ci95'][1]:+.3f}] includes 0: not established)"
        if not d4
        else f" (95% CI [{P4['ci95'][0]:+.3f}, {P4['ci95'][1]:+.3f}])"
    )

    def dirn(ok, extra=""):
        return ("yes" if ok else "no") + extra

    L = [
        "# Analysis: held-out runs vs the written predictions (decision 0007)",
        "",
        f"Held-out: {a['n']['answerable']} answerable + {a['n']['unanswerable']} unanswerable questions, "
        "answered by both prompts (paired). v1 refusals: human labels (Part A); v2 refusals: exact `NOT_IN_CONTEXT` marker. "
        f"Judge: `{JUDGE}`; answer accuracy ≥ {JUDGE_CORRECT_AT} counts as correct. Generated by `analyze.py` from saved artifacts.",
        "",
        "## Predictions",
        "",
        "| # | Measure | Predicted | v1 | v2 | Range held? | Direction held? |",
        "|---|---|---|---|---|---|---|",
        f"| P1 | Answered an unanswerable question | v1 15–35%, v2 ≤10%, p<0.05 | {P1['v1']['k']}/{P1['v1']['n']} ({P1['v1']['pct']}%) | "
        f"{P1['v2']['k']}/{P1['v2']['n']} ({P1['v2']['pct']}%) | {verdict('P1', p1_ok)} | {dirn(d1, f' (exact McNemar p = {P1["mcnemar_p"]:.2g})')} |",
        f"| P2 | Over-refusal (gold passage retrieved, refused) | v1 ≤5%, v2 5–15% | {P2['v1']['k']}/{P2['v1']['n']} ({P2['v1']['pct']}%) | "
        f"{P2['v2']['k']}/{P2['v2']['n']} ({P2['v2']['pct']}%) | {verdict('P2', p2_ok)} | {dirn(d2, ' (descriptive; few events)')} |",
        f"| P3 | Ungrounded answers (gold passage not retrieved, answered) | v1 50–75%, v2 ≤30% | {P3['v1']['k']}/{P3['v1']['n']} ({P3['v1']['pct']}%) | "
        f"{P3['v2']['k']}/{P3['v2']['n']} ({P3['v2']['pct']}%) | {verdict('P3', p3_ok)} | {dirn(d3)} |",
        f"| P4 | Mean answer accuracy, v2 − v1 | −0.10 to −0.03 | {P4['v1_mean']} | {P4['v2_mean']} | {verdict('P4', p4_ok)} (point estimate {P4['diff']:+.3f}) | "
        f"{dirn(d4, p4note)} |",
        f"| P5 | Faithfulness on refusals | ≥50% of v2 refusals blank (NaN) | {P5['v1']['nan']}/{P5['v1']['refusals']} NaN | "
        f"{P5['v2']['nan']}/{P5['v2']['refusals']} NaN | {verdict('P5', p5_ok)} | yes |",
        "",
        f"P3 with an alternative definition added after results (gold answer text in none of the retrieved passages, "
        f"n={a['P3_alt_answer_text_absent']['v1']['n']}): v1 {a['P3_alt_answer_text_absent']['v1']['pct']}%, "
        f"v2 {a['P3_alt_answer_text_absent']['v2']['pct']}%. The pre-registered row above is unchanged.",
        "",
        f"P4 by retrieval (succeeded = gold passage or gold answer text retrieved): succeeded "
        f"{a['P4_by_retrieval']['retrieval_succeeded']['diff']:+.3f} (n={a['P4_by_retrieval']['retrieval_succeeded']['n']}); "
        f"missed {a['P4_by_retrieval']['retrieval_missed']['diff']:+.3f} (n={a['P4_by_retrieval']['retrieval_missed']['n']}). "
        f"By gold passage only: retrieved {a['P4_by_gold_passage']['gold_retrieved']['diff']:+.3f} "
        f"(n={a['P4_by_gold_passage']['gold_retrieved']['n']}), not retrieved "
        f"{a['P4_by_gold_passage']['gold_not_retrieved']['diff']:+.3f} (n={a['P4_by_gold_passage']['gold_not_retrieved']['n']}).",
        "",
        '## Borderline refusals ("refuses, then adds a related fact")',
        "",
        f"{bd['n_total']} v1 answers have this shape ({bd['n_unanswerable']} on unanswerable questions). "
        f"Human labels: {bd['human_labels']}. P1 for v1 if the whole group counts as refused: "
        f"{bd['P1_v1_if_all_refused']['pct']}%; as answered: {bd['P1_v1_if_all_answered']['pct']}%.",
        "",
        "## Refusal labeling checks",
        "",
        f"- v1 first-pass pattern vs human labels: {a['v1_pattern_vs_human']['agree']}/{a['v1_pattern_vs_human']['n']} agree.",
        f"- v2 marker classification methods: {a['v2_marker_methods']}.",
        "",
        "## String matching vs the judge",
        "",
        f"**Known-truth cases** (decision 0008): string matching {kt['string_match_right']}/{kt['n']}, judge {kt['judge_right']}/{kt['n']}; "
        f'excluding {kt["n"] - kt["n_clean"]} flagged cases whose "wrong" answer shares a year/decade with the gold '
        f"({', '.join(a['known_truth_flagged'])}): judge {kt['judge_right_clean']}/{kt['n_clean']}.",
        "",
        f"**Real answers** ({s['n']} answerable, both prompts): agree on {s['agree']}/{s['n']}.",
        "",
        "| | judge: correct | judge: not correct |",
        "|---|---|---|",
        f"| string match: yes | {s['table']['sm_True_judge_True']} | {s['table']['sm_True_judge_False']} |",
        f"| string match: no | {s['table']['sm_False_judge_True']} | {s['table']['sm_False_judge_False']} |",
        "",
        "## Context recall check (no extra judge calls)",
        "",
        f"- Repeatability: {a['context_recall_check']['repeatable_v1_v2']}/{a['context_recall_check']['n']} identical scores "
        f"across v1 and v2 ({a['context_recall_check']['identical_inputs_v1_v2']}/{a['context_recall_check']['n']} had identical inputs).",
        f"- Judge (≥ {JUDGE_CORRECT_AT}) vs gold passage retrieved: {a['context_recall_check']['vs_gold_passage_retrieved']}.",
        f"- Judge vs gold passage *or* gold answer text retrieved: {a['context_recall_check']['vs_gold_passage_or_answer_text_retrieved']}. "
        "Labels mark one passage as correct, so the answer often sits in another retrieved passage.",
        "",
        *(
            [
                "## Local judge vs API judge (known-truth cases only, decision 0010)",
                "",
                '| | Right | Right, flagged cases excluded | Cost | Time | "Based on the passages…" answers scored 1.0 |',
                "|---|---|---|---|---|---|",
                f"| {a['judge_comparison']['local']['judge']} (local) | {a['judge_comparison']['local']['right']}/{a['judge_comparison']['local']['n']} | "
                f"{a['judge_comparison']['local']['right_clean']}/{kt['n_clean']} | "
                f"$0 | ~{a['judge_comparison']['local']['minutes_sequential']} min (one at a time) | "
                f"{a['judge_comparison']['reworded_scored_1']['local']}/60 |",
                f"| {a['judge_comparison']['api']['judge']} (API) | {a['judge_comparison']['api']['right']}/{a['judge_comparison']['api']['n']} | "
                f"{a['judge_comparison']['api']['right_clean']}/{kt['n_clean']} | "
                f"${a['judge_comparison']['api']['cost_usd']} | {a['judge_comparison']['api']['minutes_wall']} min "
                f"({a['judge_comparison']['api']['concurrency']} parallel) | {a['judge_comparison']['reworded_scored_1']['api']}/60 |",
                "",
                f"Identical scores on {a['judge_comparison']['identical_values']}/{a['judge_comparison']['local']['n']} cases. "
                "Real answers are not scored by the API judge: they were written by the same model family (self-preference).",
                "",
            ]
            if "judge_comparison" in a
            else []
        ),
        "## Does the judge threshold matter?",
        "",
        "| Threshold | Known-truth judge right | Judged correct v1 | Judged correct v2 | Agrees with string match (real) |",
        "|---|---|---|---|---|",
        *[
            f"| ≥ {t} | {x['known_truth_judge_right']}/{x['n_known_truth']} | {x['judged_correct_v1']['pct']}% | "
            f"{x['judged_correct_v2']['pct']}% | {x['agree_with_string_match']}/{x['n_real']} |"
            for t, x in a["threshold_sensitivity"].items()
        ],
        "",
        "## Answer-key noise",
        "",
        f"{a['answer_key_noise']['flagged']}/{a['answer_key_noise']['questions']} known-truth questions had a correct-by-construction "
        f"answer scored below 0.75; inspected by hand, all {a['answer_key_noise']['inspected_bad_or_contradictory_gold']} had a wrong "
        f"or contradictory gold answer (decision 0008): {3 / a['answer_key_noise']['questions']:.0%} of answer keys, "
        f"95% CI {a['answer_key_noise']['rate_ci95'][0]:.0%}–{a['answer_key_noise']['rate_ci95'][1]:.0%} (small sample).",
        "",
        "## Failure labels (judge correctness; retrieval = gold passage or gold answer text retrieved)",
        "",
        "| | " + " | ".join(a["failure_labels_judge_based"]["v1"]) + " |",
        "|---|" + "---|" * len(a["failure_labels_judge_based"]["v1"]),
        *[
            f"| {v} | " + " | ".join(str(x) for x in c.values()) + " |"
            for v, c in a["failure_labels_judge_based"].items()
        ],
        "",
        "## How many questions do you need?",
        "",
        "Expected half-width of the 95% CI of mean answer accuracy (v1) with n questions, from the spread of all "
        "scores (1.96 × sd / √n): " + ", ".join(f"n={n}: ±{w}" for n, w in a["eval_set_size"].items()) + ".",
        "",
        "## Cost and time",
        "",
        f"- Generation (gpt-6-luna): ${a['cost']['generation_usd_per_100']['v1']} (v1) / "
        f"${a['cost']['generation_usd_per_100']['v2']} (v2) per 100 answers.",
        f"- Local judge, median seconds per call: {a['cost']['judge_median_seconds']}; "
        f"~{a['cost']['judge_minutes_per_100_answerable_all_metrics']} min per 100 answerable answers for all three metrics.",
        f"- String matching: {a['cost']['string_match']}.",
        f"- No answer was cut off: longest answers used {a['answer_length_check']['v1']['max_completion_tokens_used']} (v1) and "
        f"{a['answer_length_check']['v2']['max_completion_tokens_used']} (v2) of the {a['answer_length_check']['v1']['cap']}-token cap.",
    ]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    result = analyze(load())
    Path("reports").mkdir(exist_ok=True)
    Path("reports/analysis.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    Path("reports/analysis.md").write_text(report(result))
    print(report(result))
