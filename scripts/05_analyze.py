#!/usr/bin/env python
"""
Step 5: turn the runs into the tables, tests and figures the poster needs.

    python scripts/05_analyze.py
    python scripts/05_analyze.py --contrast-pair semantic fixed_128 --metric em

Outputs, all under data/analysis/:
    results_wide.csv        the 16-cell grid, means + 95% bootstrap CIs
    per_question.csv.gz     every question x run (the table view / raw data)
    contrasts.csv           pairwise chunker tests within each embedder
    interaction.json        the primary RQ, as one number with a CI
    dissociation.csv        retrieval-vs-answer, and the failure taxonomy
    question_types.csv      EM by question type x chunker x embedder
    fig1_interaction.(png|pdf)   headline
    fig2_dissociation.*          secondary A
    fig3_question_types.*        secondary B
    fig4_budget_check.*          the validity check for the methods box
    report.md               numbers written out in sentences

On the inference, two deliberate choices:

* The interaction is tested on a PRE-SPECIFIED pair of chunkers (default
  semantic vs fixed_128), not on the observed best-minus-worst. Max-minus-min
  is selection-inflated -- it picks the pair that looks most different and then
  tests it -- so it is reported as a descriptive spread and never as the
  p-value.
* Every EM comparison ships its minimum detectable effect. At n=800 that is
  around 4 points, so "no significant difference" has to be read as "no effect
  bigger than the MDE was detectable", and the report says so in those words.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from crag import plotting
from crag.metrics import answer_length_bucket, contains_answer, is_abstention, question_type
from crag.stats import (
    bootstrap_ci,
    holm_bonferroni,
    interaction_contrast,
    mcnemar_exact,
    mde_from_pairs,
    mde_mcnemar,
    paired_bootstrap,
)

ANALYSIS = config.ANALYSIS_DIR
METRICS = ["answer_recall", "page_hit", "em", "em_lenient", "f1", "abstained",
           "context_ok_answer_wrong"]
LABELS = {
    "em": "Exact Match",
    "em_lenient": "EM (lenient: prediction contains a gold answer)",
    "f1": "F1",
    "answer_recall": "answer recall@budget",
    "page_hit": "page hit@budget",
    "abstained": "abstention rate",
    "context_ok_answer_wrong": "right context, wrong answer",
}


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def load_runs(runs_dir: Path) -> pd.DataFrame:
    frames = []
    for summary_path in sorted(runs_dir.glob("*.summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records_path = runs_dir / f"{summary['run_id']}.jsonl"
        if not records_path.exists():
            continue
        df = pd.read_json(records_path, lines=True)
        df["run_id"] = summary["run_id"]
        df["chunker"] = summary["chunker"]
        df["embedder"] = summary["embedder"]
        df["budget"] = summary["budget"]
        frames.append(df)
    if not frames:
        raise SystemExit(f"no runs found in {runs_dir} -- run scripts/03_run_pipeline.py first")
    df = pd.concat(frames, ignore_index=True)
    df["question_type"] = df["question"].map(question_type)
    df["answer_length"] = df["answers"].map(answer_length_bucket)

    # Robustness against exact-match brittleness. NQ golds are short spans and
    # EM punishes a prediction that is right but differently shaped: "April 14,
    # 2019" against a gold of "2019", or "George Merrill, Shannon Rubicam"
    # against golds of "George Merrill" and "Shannon Rubicam" -- both scored 0
    # in the probe. em_lenient credits a prediction whose normalised tokens
    # CONTAIN a gold answer. One-directional on purpose: a prediction of "1"
    # must not be credited for a gold of "1994". If a conclusion holds under
    # both EM and em_lenient, brittleness is not what produced it.
    df["em_lenient"] = [
        float(contains_answer(pred, golds))
        if isinstance(pred, str) and pred and not is_abstention(pred) else
        (np.nan if not isinstance(pred, str) else 0.0)
        for pred, golds in zip(df.get("prediction"), df["answers"])
    ]
    for metric in METRICS:
        if metric not in df.columns:
            df[metric] = np.nan
    return df


def aligned(df: pd.DataFrame, metric: str, chunker: str, embedder: str, budget: int) -> pd.Series:
    """One condition's per-question scores, indexed by question_id.

    Indexed, not positional: every paired test downstream depends on the two
    vectors describing the same questions in the same order, and an index join
    is the only way to guarantee that after any filtering.
    """
    sub = df[(df.chunker == chunker) & (df.embedder == embedder) & (df.budget == budget)]
    return sub.set_index("question_id")[metric].sort_index()


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def results_table(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for (chunker, embedder, budget), sub in df.groupby(["chunker", "embedder", "budget"]):
        row = {"chunker": chunker, "embedder": embedder, "budget": budget, "n": len(sub)}
        for metric in METRICS:
            if metric not in sub.columns:
                continue          # e.g. a run predating a metric being added
            values = sub[metric].dropna().to_numpy()
            if not len(values):
                continue
            mean, lo, hi = bootstrap_ci(values, n_resamples=n_boot, seed=config.SEED)
            row[metric] = round(mean, 4)
            row[f"{metric}_ci_low"] = round(lo, 4)
            row[f"{metric}_ci_high"] = round(hi, 4)
        row["n_chunks_used"] = round(float(sub["n_chunks_used"].mean()), 2)
        row["n_tokens_used"] = round(float(sub["n_tokens_used"].mean()), 1)
        row["budget_utilisation"] = round(float(sub["budget_utilisation"].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["budget", "embedder", "chunker"])


def pairwise_contrasts(df: pd.DataFrame, metric: str, n_boot: int) -> pd.DataFrame:
    rows = []
    chunkers = [c for c in plotting.CHUNKER_ORDER if c in set(df.chunker)]
    for budget in sorted(df.budget.unique()):
        for embedder in sorted(df.embedder.unique()):
            for i, a in enumerate(chunkers):
                for b in chunkers[i + 1:]:
                    va = aligned(df, metric, a, embedder, budget).dropna()
                    vb = aligned(df, metric, b, embedder, budget).dropna()
                    common = va.index.intersection(vb.index)
                    va, vb = va.loc[common], vb.loc[common]
                    if not len(common):
                        continue      # e.g. EM at a retrieval-only budget
                    boot = paired_bootstrap(va.to_numpy(), vb.to_numpy(),
                                            n_resamples=n_boot, seed=config.SEED)
                    row = {
                        "budget": budget, "embedder": embedder, "metric": metric,
                        "a": a, "b": b, "mean_a": round(boot.mean_a, 4),
                        "mean_b": round(boot.mean_b, 4), "diff": round(boot.diff, 4),
                        "ci_low": round(boot.ci_low, 4), "ci_high": round(boot.ci_high, 4),
                        "p_bootstrap": round(boot.p_value, 4),
                        "n": boot.n,
                        "mde": round(mde_from_pairs(va.to_numpy(), vb.to_numpy()), 4),
                    }
                    if metric == "em":
                        mc = mcnemar_exact(va.to_numpy(), vb.to_numpy())
                        row["p_mcnemar"] = round(mc.p_value, 4)
                        row["n_discordant"] = mc.n_discordant
                        row["mde"] = round(
                            mde_mcnemar(mc.n, mc.n_discordant / max(mc.n, 1)), 4)
                    rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        pcol = "p_mcnemar" if "p_mcnemar" in out.columns else "p_bootstrap"
        out["reject_holm"] = holm_bonferroni(out[pcol].tolist(), alpha=config.ALPHA)
    return out


def interaction_result(df, metric, pair, budget, n_boot):
    """The primary RQ: is the chunking effect larger under the weak embedder?"""
    a, b = pair
    out = {"metric": metric, "budget": budget, "pair": list(pair)}
    vectors = {}
    for embedder in ("weak", "strong"):
        va = aligned(df, metric, a, embedder, budget).dropna()
        vb = aligned(df, metric, b, embedder, budget).dropna()
        common = va.index.intersection(vb.index)
        if not len(common):
            raise SystemExit(
                f"no {metric} values at budget {budget} for {a}/{b}/{embedder}. "
                f"Generation ran at {list(config.GENERATION_BUDGETS)} -- pass "
                f"--budget {list(config.GENERATION_BUDGETS)[0]}, or "
                f"--metric answer_recall to analyse retrieval at any budget.")
        vectors[embedder] = (va.loc[common].to_numpy(), vb.loc[common].to_numpy())
        eff = paired_bootstrap(*vectors[embedder], n_resamples=n_boot, seed=config.SEED)
        out[f"{embedder}_effect"] = {
            "diff": round(eff.diff, 4), "ci": [round(eff.ci_low, 4), round(eff.ci_high, 4)],
            "p": round(eff.p_value, 4),
            "mde": round(mde_from_pairs(*vectors[embedder]), 4),
        }
    did = interaction_contrast(vectors["weak"][0], vectors["weak"][1],
                               vectors["strong"][0], vectors["strong"][1],
                               n_resamples=n_boot, seed=config.SEED)
    out["interaction"] = {
        "difference_in_differences": round(did.diff, 4),
        "ci": [round(did.ci_low, 4), round(did.ci_high, 4)],
        "p": round(did.p_value, 4),
        "significant": bool(did.significant),
        "n": did.n,
        # The interaction is a difference of two paired differences, so its
        # variance is roughly twice a single contrast's. Its MDE must come from
        # its own per-question vector -- quoting a pairwise-contrast MDE here
        # would overstate how small an interaction the design could detect.
        "mde": round(mde_from_pairs(vectors["weak"][0] - vectors["weak"][1],
                                    vectors["strong"][0] - vectors["strong"][1]), 4),
    }
    # descriptive only -- selection-inflated, never the headline p-value
    spread = {}
    for embedder in sorted(df.embedder.unique()):
        means = df[(df.embedder == embedder) & (df.budget == budget)].groupby("chunker")[metric].mean()
        if len(means):
            spread[embedder] = {
                "best": means.idxmax(), "worst": means.idxmin(),
                "spread": round(float(means.max() - means.min()), 4),
            }
    out["descriptive_spread_max_minus_min"] = spread
    return out


def dissociation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (chunker, embedder, budget), sub in df.groupby(["chunker", "embedder", "budget"]):
        got = sub[sub.answer_recall > 0]
        missed = sub[sub.answer_recall == 0]
        rows.append({
            "chunker": chunker, "embedder": embedder, "budget": budget,
            "answer_recall": round(float(sub.answer_recall.mean()), 4),
            "em": round(float(sub.em.mean()), 4),
            "em_given_context_has_answer": round(float(got.em.mean()), 4) if len(got) else np.nan,
            "em_given_context_lacks_answer": round(float(missed.em.mean()), 4) if len(missed) else np.nan,
            "abstained_given_context_has_answer": round(float(got.abstained.mean()), 4) if len(got) else np.nan,
            "abstained_given_context_lacks_answer": round(float(missed.abstained.mean()), 4) if len(missed) else np.nan,
            "hallucinated_without_context": round(
                float(((missed.abstained == 0) & (missed.em == 0)).mean()), 4) if len(missed) else np.nan,
            "n_context_has_answer": int(len(got)),
        })
    return pd.DataFrame(rows).sort_values(["budget", "embedder", "chunker"])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_interaction(df, metric, budget, inter, path):
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [2, 1]})
    chunkers = [c for c in plotting.CHUNKER_ORDER if c in set(df.chunker)]
    colors = plotting.chunker_colors(chunkers)
    xs = np.arange(len(plotting.EMBEDDER_ORDER))

    for chunker in chunkers:
        means, los, his = [], [], []
        for embedder in plotting.EMBEDDER_ORDER:
            vals = aligned(df, metric, chunker, embedder, budget).to_numpy()
            m, lo, hi = bootstrap_ci(vals, n_resamples=2000, seed=config.SEED)
            means.append(m); los.append(m - lo); his.append(hi - m)
        ax.errorbar(xs, means, yerr=[los, his], color=colors[chunker], marker="o",
                    capsize=3, elinewidth=1.2, markeredgecolor=plotting.SURFACE,
                    markeredgewidth=1.2, label=chunker)
        # direct label: the relief rule for the low-contrast slots
        ax.annotate(chunker, (xs[-1], means[-1]), xytext=(8, 0), textcoords="offset points",
                    color=colors[chunker], fontsize=9, va="center", fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{e}\n({config.EMBEDDERS[e].model_name.split('/')[-1]})"
                        for e in plotting.EMBEDDER_ORDER])
    ax.set_ylabel(LABELS.get(metric, metric))
    ax.set_xlabel(f"embedder capacity  ·  budget {budget} tokens")
    ax.set_title("Chunking effect by embedder capacity", loc="left", pad=34)
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    ax.margins(x=0.30)
    # legend above the plot: with four direct labels inside, a legend box in the
    # data area collides with the very series it is naming
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4,
              borderaxespad=0, columnspacing=1.2, handletextpad=0.5)

    spread = inter["descriptive_spread_max_minus_min"]
    labels = [e for e in plotting.EMBEDDER_ORDER if e in spread]
    values = [spread[e]["spread"] for e in labels]
    bars = ax2.bar(labels, values, color=[plotting.embedder_colors()[e] for e in labels],
                   width=0.55, edgecolor=plotting.SURFACE, linewidth=2)
    for bar, value in zip(bars, values):
        ax2.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                     xytext=(0, 4), textcoords="offset points", ha="center",
                     fontsize=9, color=plotting.INK_PRIMARY)
    did = inter["interaction"]
    ax2.set_title("Spread across chunkers", loc="left", pad=34)
    ax2.set_ylabel(f"{LABELS.get(metric, metric)}: max - min")
    ax2.grid(axis="y", alpha=0.7)
    ax2.set_axisbelow(True)
    ax2.text(0.0, -0.28,
             f"pre-specified contrast {inter['pair'][0]} - {inter['pair'][1]}:\n"
             f"interaction {did['difference_in_differences']:+.3f} "
             f"[{did['ci'][0]:+.3f}, {did['ci'][1]:+.3f}], p={did['p']:.3f}",
             transform=ax2.transAxes, fontsize=8, color=plotting.INK_SECONDARY, va="top")
    plotting.save(fig, path)
    plt.close(fig)


def fig_dissociation(df, budget, path):
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ecolors = plotting.embedder_colors()
    sub = df[df.budget == budget]

    for (chunker, embedder), g in sub.groupby(["chunker", "embedder"]):
        ax.scatter(g.answer_recall.mean(), g.em.mean(),
                   color=ecolors[embedder], marker=plotting.MARKERS.get(chunker, "o"),
                   s=90, edgecolor=plotting.SURFACE, linewidth=1.5, zorder=3)
    ax.plot([0, 1], [0, 1], color=plotting.INK_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.annotate("EM = retrieval\n(every retrieved answer used)", (0.70, 0.70),
                xytext=(-8, 10), textcoords="offset points", fontsize=8,
                color=plotting.INK_MUTED, ha="right")
    ax.set_xlabel(f"answer recall@budget  (retrieval, budget {budget})")
    ax.set_ylabel("Exact Match  (answer)")
    ax.set_title("Retrieval does not become answers", loc="left", pad=46)
    ax.grid(alpha=0.7); ax.set_axisbelow(True)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ecolors[e], marker="o", linestyle="", label=f"{e} embedder")
               for e in plotting.EMBEDDER_ORDER]
    handles += [Line2D([], [], color=plotting.INK_SECONDARY, marker=plotting.MARKERS[c],
                       linestyle="", label=c)
                for c in plotting.CHUNKER_ORDER if c in set(sub.chunker)]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3,
              borderaxespad=0, columnspacing=1.2, handletextpad=0.5)

    got = sub[sub.answer_recall > 0]
    labels = [c for c in plotting.CHUNKER_ORDER if c in set(got.chunker)]
    width = 0.38
    xs = np.arange(len(labels))
    for i, embedder in enumerate(plotting.EMBEDDER_ORDER):
        vals = [float(got[(got.chunker == c) & (got.embedder == embedder)].em.mean())
                if len(got[(got.chunker == c) & (got.embedder == embedder)]) else 0.0
                for c in labels]
        bars = ax2.bar(xs + (i - 0.5) * width, vals, width=width - 0.02,
                       color=ecolors[embedder], label=f"{embedder} embedder",
                       edgecolor=plotting.SURFACE, linewidth=2)
        for bar, value in zip(bars, vals):
            ax2.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value),
                         xytext=(0, 3), textcoords="offset points", ha="center",
                         fontsize=8, color=plotting.INK_PRIMARY)
    ax2.set_xticks(xs); ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("EM | answer was in the context")
    ax2.set_title("When retrieval works, does the model?", loc="left", pad=34)
    ax2.grid(axis="y", alpha=0.7); ax2.set_axisbelow(True)
    ax2.set_ylim(0, max(1e-3, float(got.em.max()) if len(got) else 1.0) * 1.18)
    ax2.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
               borderaxespad=0, columnspacing=1.2, handletextpad=0.5)
    plotting.save(fig, path)
    plt.close(fig)


def fig_question_types(df, metric, budget, path):
    import matplotlib.pyplot as plt

    sub = df[df.budget == budget]
    types = [t for t in ["person", "date", "place", "count", "entity", "reason", "other"]
             if t in set(sub.question_type)]
    chunkers = [c for c in plotting.CHUNKER_ORDER if c in set(sub.chunker)]
    embedders = [e for e in plotting.EMBEDDER_ORDER if e in set(sub.embedder)]
    cmap = plotting.sequential_cmap()
    n_by_type = sub.drop_duplicates("question_id").groupby("question_type").size().to_dict()

    height = max(3.0, 1.6 + 0.62 * len(types))
    fig, axes = plt.subplots(1, len(embedders), figsize=(4.6 * len(embedders), height),
                             squeeze=False)
    grids = []
    for embedder in embedders:
        grid = np.array([[float(sub[(sub.question_type == t) & (sub.chunker == c) &
                                    (sub.embedder == embedder)][metric].mean())
                          for c in chunkers] for t in types])
        grids.append(grid)
    vmax = float(np.nanmax(grids)) if grids else 1.0

    for ax, embedder, grid in zip(axes[0], embedders, grids):
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=max(vmax, 1e-6), aspect="auto")
        ax.set_xticks(range(len(chunkers)))
        ax.set_xticklabels(chunkers, rotation=20, ha="right")
        ax.set_yticks(range(len(types)))
        ax.set_yticklabels(
            [f"{t}  (n={n_by_type.get(t, 0)})" for t in types], fontsize=9)
        ax.set_title(f"{embedder} embedder", loc="left")
        for i in range(len(types)):
            for j in range(len(chunkers)):
                value = grid[i, j]
                if np.isnan(value):
                    continue
                # the number is always printed: colour is never the only encoding
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if value > 0.6 * max(vmax, 1e-6) else plotting.INK_PRIMARY)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f"{LABELS.get(metric, metric)} by question type  (budget {budget})",
                 x=0.01, ha="left", fontsize=11, fontweight="bold",
                 color=plotting.INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plotting.save(fig, path)
    plt.close(fig)


def fig_budget_check(df, path):
    import matplotlib.pyplot as plt

    budgets = sorted(df.budget.unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.8 * len(budgets), 3.8), squeeze=False)
    ecolors = plotting.embedder_colors()
    for ax, budget in zip(axes[0], budgets):
        sub = df[df.budget == budget]
        labels = [c for c in plotting.CHUNKER_ORDER if c in set(sub.chunker)]
        xs = np.arange(len(labels))
        width = 0.38
        for i, embedder in enumerate(plotting.EMBEDDER_ORDER):
            vals = [float(sub[(sub.chunker == c) & (sub.embedder == embedder)].n_tokens_used.mean())
                    for c in labels]
            ax.bar(xs + (i - 0.5) * width, vals, width=width - 0.02, color=ecolors[embedder],
                   label=f"{embedder} embedder", edgecolor=plotting.SURFACE, linewidth=2)
        ax.axhline(budget, color=plotting.INK_MUTED, linestyle="--", linewidth=1)
        ax.annotate(f"budget {budget}", (len(labels) - 0.5, budget), xytext=(0, 4),
                    textcoords="offset points", ha="right", fontsize=8,
                    color=plotting.INK_SECONDARY)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("context tokens delivered")
        ax.set_title(f"Budgets are matched ({budget})", loc="left")
        ax.grid(axis="y", alpha=0.7); ax.set_axisbelow(True)
        ax.legend(loc="lower right")
    plotting.save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def as_table(frame: pd.DataFrame) -> str:
    """Markdown if tabulate is installed, fixed-width text if not."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```\n" + frame.to_string(index=False) + "\n```"


def write_report(df, results, contrasts, inter, dissociation, metric, budget, path):
    did = inter["interaction"]
    lines = [
        "# Results",
        "",
        f"{df.question_id.nunique()} questions x {df.run_id.nunique()} runs. "
        f"Headline metric `{metric}` at budget {budget}.",
        "",
        "## Primary RQ - does embedder capacity moderate the chunking effect?",
        "",
        f"Pre-specified contrast: **{inter['pair'][0]} vs {inter['pair'][1]}**.",
        "",
        f"- weak embedder: {inter['weak_effect']['diff']:+.4f} "
        f"[{inter['weak_effect']['ci'][0]:+.4f}, {inter['weak_effect']['ci'][1]:+.4f}], "
        f"p={inter['weak_effect']['p']:.4f}",
        f"- strong embedder: {inter['strong_effect']['diff']:+.4f} "
        f"[{inter['strong_effect']['ci'][0]:+.4f}, {inter['strong_effect']['ci'][1]:+.4f}], "
        f"p={inter['strong_effect']['p']:.4f}",
        f"- **interaction (weak - strong): {did['difference_in_differences']:+.4f} "
        f"[{did['ci'][0]:+.4f}, {did['ci'][1]:+.4f}], p={did['p']:.4f}"
        f"{' (significant)' if did['significant'] else ' (not significant)'}**",
        "",
        "The max-minus-min spread per embedder is reported in the figure as a "
        "descriptive quantity only: it selects the most extreme pair and would be "
        "inflated if it were tested.",
        "",
        "## Minimum detectable effect",
        "",
        f"The interaction test itself had 80% power to detect a difference-in-differences "
        f"of {did['mde']:.4f} (its own MDE; the simple effects: weak "
        f"{inter['weak_effect']['mde']:.4f}, strong {inter['strong_effect']['mde']:.4f}).",
        "",
    ]
    if len(contrasts):
        worst = contrasts.loc[contrasts["mde"].idxmax()]
        lines += [
            f"Across the pairwise contrasts, the largest MDE is {worst['mde']:.4f} "
            f"({worst['a']} vs {worst['b']}, {worst['embedder']}, budget {worst['budget']}). "
            "A null result here means *no effect larger than that was detectable* -- "
            "it is not evidence that the effect is zero.",
            "",
            "## Pairwise chunker contrasts",
            "",
            as_table(contrasts),
            "",
        ]
    lines += [
        "## Retrieval vs answers",
        "",
        as_table(dissociation),
        "",
        "## The 16-cell grid",
        "",
        as_table(results),
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--metric", default="em", choices=METRICS)
    ap.add_argument("--budget", type=int, default=None, help="headline budget (default: the smaller)")
    ap.add_argument("--contrast-pair", nargs=2, default=["semantic", "fixed_128"],
                    help="the pre-specified chunker pair for the interaction test")
    ap.add_argument("--bootstrap", type=int, default=config.N_BOOTSTRAP)
    ap.add_argument("--out-dir", default=None,
                    help="override data/analysis -- use it for secondary metrics "
                         "(e.g. --metric answer_recall --out-dir data/analysis_recall) "
                         "so the pre-specified em analysis is never overwritten")
    args = ap.parse_args()

    config.ensure_dirs()
    global ANALYSIS
    if args.out_dir:
        ANALYSIS = Path(args.out_dir)
        ANALYSIS.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir) if args.runs_dir else config.RUNS_DIR
    df = load_runs(runs_dir)
    # the headline budget is where the answers actually are
    # int(), not numpy.int64: this value ends up in interaction.json
    answered = sorted(int(b) for b in df.budget.unique() if df[df.budget == b]["em"].notna().any())
    budget = int(args.budget or (answered[0] if answered else min(df.budget.unique())))
    if answered and len(answered) < df.budget.nunique():
        print(f"answers exist at budget(s) {answered}; retrieval metrics at all of "
              f"{sorted(int(b) for b in df.budget.unique())}")
    print(f"{len(df)} rows | {df.run_id.nunique()} runs | {df.question_id.nunique()} questions")

    results = results_table(df, args.bootstrap)
    results.to_csv(ANALYSIS / "results_wide.csv", index=False)

    keep = ["run_id", "chunker", "embedder", "budget", "question_id", "question_type",
            "answer_length", "n_chunks_used", "n_tokens_used", "budget_utilisation",
            "prediction"] + METRICS
    df[[c for c in keep if c in df.columns]].to_csv(
        ANALYSIS / "per_question.csv.gz", index=False, compression="gzip")

    contrasts = pairwise_contrasts(df, args.metric, args.bootstrap)
    contrasts.to_csv(ANALYSIS / "contrasts.csv", index=False)

    pair = tuple(args.contrast_pair)
    inter = interaction_result(df, args.metric, pair, budget, args.bootstrap)
    (ANALYSIS / "interaction.json").write_text(json.dumps(inter, indent=2), encoding="utf-8")

    dissociation = dissociation_table(df)
    dissociation.to_csv(ANALYSIS / "dissociation.csv", index=False)

    qt = (df.groupby(["question_type", "chunker", "embedder", "budget"])[args.metric]
            .agg(["mean", "count"]).reset_index())
    qt.to_csv(ANALYSIS / "question_types.csv", index=False)

    plotting.apply_style()
    fig_interaction(df, args.metric, budget, inter, ANALYSIS / "fig1_interaction")
    fig_dissociation(df, budget, ANALYSIS / "fig2_dissociation")
    fig_question_types(df, args.metric, budget, ANALYSIS / "fig3_question_types")
    fig_budget_check(df, ANALYSIS / "fig4_budget_check")

    write_report(df, results, contrasts, inter, dissociation, args.metric, budget,
                 ANALYSIS / "report.md")

    did = inter["interaction"]
    print(f"\ninteraction ({pair[0]} - {pair[1]}, weak - strong) on {args.metric}: "
          f"{did['difference_in_differences']:+.4f} "
          f"[{did['ci'][0]:+.4f}, {did['ci'][1]:+.4f}] p={did['p']:.4f}")
    print(f"outputs -> {ANALYSIS}")


if __name__ == "__main__":
    main()
