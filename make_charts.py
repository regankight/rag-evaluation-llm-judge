"""README chart: string matching vs the LLM judge on known-truth cases (decision 0008).

    uv run --group charts python make_charts.py   # writes docs/img/known-truth-{light,dark}.png

Built from the committed known-truth scores, so it regenerates like every other number.
Palette and mark specs follow the data-viz method used for this repo's charts: two
categorical slots validated for both modes (CVD and normal-vision separation, contrast),
dots with a surface ring, hairline solid gridlines, a legend plus selective direct labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from known_truth import load_jsonl, summarize  # noqa: E402
from runs import JUDGE, scores_file  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "judge": "#2a78d6", "string": "#eb6834"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "grid": "#2c2c2a",
             "axis": "#383835", "judge": "#3987e5", "string": "#d95926"},
}  # fmt: skip

# top to bottom; (variant key, label)
ROWS = [
    ("exact", "Exact answer"),
    ("sentence", "Answer in a sentence"),
    ("reworded", "Reworded answer"),
    ("swapped", "Wrong answer"),
    ("negation_trap", "Trap: “It is not X; it’s Y”"),
    ("refusal", "Refusal"),
]


def shares() -> dict:
    s = summarize(load_jsonl(Path("known_truth", scores_file(JUDGE))))
    return {k: (100 * s[k]["string_match_right"] / s[k]["n"], 100 * s[k]["judge_right"] / s[k]["n"]) for k, _ in ROWS}


def draw(theme: str, data: dict, out: Path) -> None:
    t = THEMES[theme]
    plt.rcParams.update(
        {"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]}
    )
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    n = len(ROWS)
    ys = list(range(n))[::-1]
    off = 0.14  # the two scorers sit just above/below the row line so 100% dots don't hide each other
    for y, (key, _label) in zip(ys, ROWS, strict=True):
        sm, jd = data[key]
        ax.scatter([jd], [y + off], s=70, color=t["judge"], edgecolors=t["surface"], linewidths=2, zorder=3)
        ax.scatter([sm], [y - off], s=70, color=t["string"], edgecolors=t["surface"], linewidths=2, zorder=3)
        if key == "negation_trap":  # the one row the story is about gets direct labels
            ax.annotate(f"{sm:.0f}%", (sm, y - off), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=9, color=t["text"], fontweight="bold")  # fmt: skip
            ax.annotate(f"{jd:.0f}%", (jd, y + off), xytext=(-9, 0), textcoords="offset points",
                        va="center", ha="right", fontsize=9, color=t["text"], fontweight="bold")  # fmt: skip

    # group divider: answers that are correct by construction above, wrong below
    ax.axhline(ys[2] - 0.5, color=t["axis"], linewidth=1)
    ax.text(-2, ys[0] + 0.55, "CORRECT ANSWERS", fontsize=7.5, color=t["text2"], ha="left", va="bottom")
    ax.text(-2, ys[2] - 0.56, "WRONG ANSWERS", fontsize=7.5, color=t["text2"], ha="left", va="top")

    ax.set_yticks(ys, [label for _, label in ROWS], fontsize=9.5, color=t["text"])
    ax.set_xlim(-3, 106)
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], fontsize=8.5, color=t["text2"])
    ax.tick_params(length=0)
    ax.grid(axis="x", color=t["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(t["axis"])
    ax.set_xlabel("Share of the 60 cases scored right", fontsize=8.5, color=t["text2"], labelpad=8)

    fig.text(0.015, 0.955, "String matching can’t read. The LLM judge can.", fontsize=14, fontweight="bold",
             color=t["text"], ha="left", va="top")  # fmt: skip
    fig.text(0.015, 0.885, "360 answers with known labels · judge: local qwen3:14b via RAGAS answer accuracy (≥ 0.5)",
             fontsize=8.5, color=t["text2"], ha="left", va="top")  # fmt: skip
    # legend: swatch dots beside text-colored labels (identity never by text color)
    lx, ly = 0.015, 0.83
    for i, (name, color) in enumerate((("LLM judge", t["judge"]), ("String matching", t["string"]))):
        x = lx + i * 0.16
        fig.text(x, ly, "●", color=color, fontsize=11, ha="left", va="center")
        fig.text(x + 0.022, ly, name, color=t["text"], fontsize=9, ha="left", va="center")

    fig.subplots_adjust(left=0.27, right=0.97, top=0.76, bottom=0.15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    data = shares()
    for theme in THEMES:
        path = Path("docs/img") / f"known-truth-{theme}.png"
        draw(theme, data, path)
        print(path, {k: tuple(round(v) for v in data[k]) for k, _ in ROWS})
