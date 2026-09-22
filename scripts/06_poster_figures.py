#!/usr/bin/env python
"""
Figures for the A1 poster (poster/main.tex).

    python scripts/06_poster_figures.py

Each figure is saved at exactly the width it occupies on the printed page
(one poster column = 269 mm), so LaTeX includes it at scale 1.0 and every
label prints at the size set here. All figure text is >= 24 pt.

Numbers are read from data/analysis and data/chunks; nothing is typed in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

OUT = Path(__file__).resolve().parents[1] / "poster" / "figures"
MM = 1 / 25.4
COL_W = 269 * MM                      # one poster column, see main.tex geometry

INK, INK2, RULE, BG = "#111111", "#444444", "#d9d6cf", "#ffffff"
BLUE, BLUE_LIGHT, BLUE_DARK = "#2a78d6", "#86b6ef", "#104281"
ORANGE = "#c2491a"                    # darker step of #eb6834: 4.9:1 on white, safe for text
SERIES = {"fixed_128": "#2a78d6", "fixed_256": "#eb6834",
          "recursive_256": "#1baf7a", "semantic": "#c98500"}
MARKER = {"fixed_128": "o", "fixed_256": "s", "recursive_256": "^", "semantic": "D"}
LABEL = {"fixed_128": "fixed 128", "fixed_256": "fixed 256",
         "recursive_256": "recursive", "semantic": "semantic"}
ORDER = ["fixed_128", "fixed_256", "recursive_256", "semantic"]

PT = 24.5                              # figure text floor: 24 pt plus rounding headroom


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["FreeSans", "Liberation Sans"],   # TrueType Helvetica clone: clean Type 42 embedding for print
        "font.size": PT, "axes.labelsize": PT, "xtick.labelsize": PT,
        "ytick.labelsize": PT, "legend.fontsize": PT,
        "axes.edgecolor": RULE, "axes.linewidth": 1.2,
        "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.grid": True, "grid.color": RULE, "grid.linewidth": 1.0,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
        "text.color": INK, "legend.frameon": False,
        "pdf.fonttype": 42,
    })


def bare(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=0, pad=8)


def save(fig, name):
    """No bbox_inches='tight': the saved page must stay exactly COL_W wide."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"  {name:<18} {w / MM:.0f} x {h / MM:.0f} mm")


def spread(ys, gap):
    """Nudge label positions apart so no two are closer than `gap`."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for _ in range(50):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            if out[b] - out[a] < gap:
                mid = (out[a] + out[b]) / 2
                out[a], out[b] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    return out


# ---------------------------------------------------------------------------

def fig_funnel(wide, corpus):
    """Where the answers go: three stages of one quantity (sequential blues)."""
    b = wide[wide.budget == 1000]
    stages = ["answer is in the\ngold Wikipedia page",
              "answer reaches the\n1,000-token context",
              "reader outputs it\nexactly (EM)"]
    vals = [corpus["per_split"]["test"]["answer_in_gold_page"],
            b.answer_recall.mean(), b.em.mean()]
    lo = [vals[0], b.answer_recall.min(), b.em.min()]
    hi = [vals[0], b.answer_recall.max(), b.em.max()]

    fig = plt.figure(figsize=(COL_W, 104 * MM))
    ax = fig.add_axes([0.335, 0.225, 0.60, 0.765])
    y = np.arange(3)[::-1]
    for i, c in enumerate([BLUE_LIGHT, BLUE, BLUE_DARK]):
        ax.barh(y[i], vals[i], height=0.60, color=c, zorder=3)
        if hi[i] - lo[i] > 1e-9:                        # range over 8 conditions
            ax.plot([lo[i], hi[i]], [y[i]] * 2, color=INK, lw=2.4, zorder=5)
            for v in (lo[i], hi[i]):
                ax.plot([v, v], [y[i] - .13, y[i] + .13], color=INK, lw=2.4, zorder=5)
        ax.text(max(vals[i], hi[i]) + 0.025, y[i], f"{vals[i] * 100:.0f}%",
                va="center", ha="left", fontsize=30, fontweight="bold", zorder=6)
    drop = round((vals[1] - vals[2]) * 100)
    ax.annotate("", xy=(0.60, y[2] + 0.30), xytext=(0.60, y[1] - 0.30),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=3,
                                mutation_scale=28, shrinkA=0, shrinkB=0))
    ax.text(0.63, (y[1] + y[2]) / 2, f"−{drop} points", color=ORANGE,
            fontsize=PT, fontweight="bold", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(stages, color=INK, linespacing=1.15)
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.set_ylim(-0.55, 2.5)
    ax.yaxis.grid(False)
    bare(ax, left=False)
    save(fig, "fig_funnel")


def fig_interaction(wide):
    """RQ1: if capacity moderated chunking, these lines would fan out."""
    b = wide[wide.budget == 1000].set_index(["chunker", "embedder"])
    fig = plt.figure(figsize=(COL_W, 118 * MM))
    ax = fig.add_axes([0.155, 0.25, 0.53, 0.735])
    ends = {}
    for ck in ORDER:
        ys = [b.loc[(ck, e), "em"] for e in ("weak", "strong")]
        for x, e in zip((0, 1), ("weak", "strong")):
            ax.plot([x, x], [b.loc[(ck, e), "em_ci_low"], b.loc[(ck, e), "em_ci_high"]],
                    color=SERIES[ck], lw=2, alpha=.35, zorder=2)
        ax.plot([0, 1], ys, color=SERIES[ck], lw=3.2, marker=MARKER[ck],
                markersize=15, markeredgecolor=BG, markeredgewidth=2.5, zorder=4)
        ends[ck] = ys[1]
    ylim = (0.33, 0.435)
    ax.set_ylim(*ylim)
    # 24 pt labels need ~0.0105 EM of vertical room at this plot height
    lab = spread([ends[c] for c in ORDER], gap=0.0136)
    for ck, ly in zip(ORDER, lab):
        # text stays in ink (aqua/yellow text on white is under 3:1);
        # the coloured leader line and marker carry the series identity
        ax.plot([1.02, 1.09], [ends[ck], ly], color=SERIES[ck], lw=2, clip_on=False)
        ax.plot(1.14, ly, marker=MARKER[ck], color=SERIES[ck], markersize=15,
                markeredgecolor=BG, markeredgewidth=2, clip_on=False)
        ax.text(1.21, ly, LABEL[ck], color=INK, fontsize=PT, va="center",
                clip_on=False)
    ax.set_xlim(-0.12, 1.02)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["weak\nMiniLM", "strong\nbge-base"], color=INK,
                       linespacing=1.15)
    ax.set_yticks([0.34, 0.36, 0.38, 0.40, 0.42])
    ax.set_yticklabels(["0.34", "0.36", "0.38", "0.40", "0.42"])
    ax.set_ylabel("exact match", labelpad=10)
    ax.xaxis.grid(False)
    bare(ax, bottom=False)
    save(fig, "fig_interaction")


def fig_effects(inter):
    """The pre-specified test, each estimate against its own detectable effect.

    The interaction is a difference of two paired differences, so its MDE is
    larger than a single contrast's; each row gets its own shaded band.
    """
    did = inter["interaction"]
    rows = [("semantic \u2212 fixed 128,\nweak embedder", inter["weak_effect"]),
            ("semantic \u2212 fixed 128,\nstrong embedder", inter["strong_effect"]),
            ("difference\n(interaction)", {"diff": did["difference_in_differences"],
                                          "ci": did["ci"], "p": did["p"],
                                          "mde": did["mde"]})]
    fig = plt.figure(figsize=(COL_W, 98 * MM))
    ax = fig.add_axes([0.39, 0.295, 0.44, 0.695])
    y = np.arange(3)[::-1]
    for i, (_, e) in enumerate(rows):
        ax.fill_between([-e["mde"], e["mde"]], y[i] - 0.36, y[i] + 0.36,
                        color="#dfeafb", zorder=1, linewidth=0)
    ax.axvline(0, color=INK2, lw=1.6, zorder=2)
    for i, (_, e) in enumerate(rows):
        c = INK if i == 2 else BLUE
        ax.plot(e["ci"], [y[i]] * 2, color=c, lw=3.4, solid_capstyle="round", zorder=3)
        ax.plot(e["diff"], y[i], "o", color=c, markersize=17,
                markeredgecolor=BG, markeredgewidth=2.5, zorder=4)
        ax.text(0.104, y[i], f"p = {e['p']:.2f}", va="center", ha="left",
                fontsize=PT, color=INK, fontweight="bold" if i == 2 else "normal",
                clip_on=False)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=INK, linespacing=1.1)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlim(-0.07, 0.10)
    ax.set_xticks([-0.05, 0, 0.05])
    ax.set_xticklabels(["\u22120.05", "0", "+0.05"])
    ax.set_xlabel("difference in exact match", labelpad=8)
    ax.yaxis.grid(False)
    bare(ax, left=False)
    save(fig, "fig_effects")


def fig_budget(wide):
    """Methodology: fixed top-k would hand the reader 2x different context."""
    stats = json.loads((config.CHUNKS_DIR / "chunk_stats.json").read_text(encoding="utf-8"))
    per_chunk = {s["chunkset_id"]: s["budget_tokens_mean"] for s in stats}
    b = wide[wide.budget == 1000].set_index(["chunker", "embedder"])
    rows = [("fixed 128", 5 * per_chunk["fixed_128"],
             b.loc["fixed_128", "n_tokens_used"].mean(), SERIES["fixed_128"]),
            ("fixed 256", 5 * per_chunk["fixed_256"],
             b.loc["fixed_256", "n_tokens_used"].mean(), SERIES["fixed_256"]),
            ("recursive", 5 * per_chunk["recursive_256"],
             b.loc["recursive_256", "n_tokens_used"].mean(), SERIES["recursive_256"]),
            ("semantic, strong", 5 * per_chunk["semantic__strong"],
             b.loc[("semantic", "strong"), "n_tokens_used"], SERIES["semantic"]),
            ("semantic, weak", 5 * per_chunk["semantic__weak"],
             b.loc[("semantic", "weak"), "n_tokens_used"], SERIES["semantic"])]

    fig = plt.figure(figsize=(COL_W, 116 * MM))
    ax = fig.add_axes([0.30, 0.355, 0.65, 0.625])
    y = np.arange(len(rows))[::-1]
    ax.axvline(1000, color=INK, lw=2, zorder=2)
    for yi, (name, topk, fill, c) in zip(y, rows):
        ax.plot([topk, fill], [yi, yi], color=RULE, lw=5, zorder=2)
        ax.plot(topk, yi, "o", markersize=17, markerfacecolor=BG,
                markeredgecolor=c, markeredgewidth=3.2, zorder=3)
        ax.plot(fill, yi, "o", markersize=17, color=c,
                markeredgecolor=BG, markeredgewidth=2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_xlim(560, 1360)
    ax.set_xticks([600, 800, 1000, 1200])
    ax.set_xticklabels(["600", "800", "1,000", "1,200"])
    ax.set_xlabel("tokens handed to the reader", labelpad=8)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.yaxis.grid(False)
    bare(ax, left=False)
    ax.plot([], [], "o", markersize=15, markerfacecolor=BG, markeredgecolor=INK2,
            markeredgewidth=3, label="fixed top-5 (would give)")
    ax.plot([], [], "o", markersize=15, color=INK2, label="our budget (gave)")
    fig.legend(loc="lower left", bbox_to_anchor=(0.005, 0.0), ncols=2,
               handletextpad=0.3, columnspacing=1.0, borderaxespad=0.2)
    save(fig, "fig_budget")
    spread_topk = max(r[1] for r in rows) / min(r[1] for r in rows)
    spread_fill = max(r[2] for r in rows) / min(r[2] for r in rows)
    print(f"    top-5 spread {spread_topk:.2f}x | fill-to-budget spread {spread_fill:.2f}x")


def main():
    style()
    a = config.ANALYSIS_DIR
    wide = pd.read_csv(a / "results_wide.csv")
    contr = pd.read_csv(a / "contrasts.csv")
    inter = json.loads((a / "interaction.json").read_text(encoding="utf-8"))
    corpus = json.loads((config.CORPUS_DIR / "corpus_report.json").read_text(encoding="utf-8"))
    print(f"figures -> {OUT}")
    fig_funnel(wide, corpus)
    fig_interaction(wide)
    fig_effects(inter)
    fig_budget(wide)


if __name__ == "__main__":
    main()
