"""
Figure style for the poster.

Print target, light surface, one shared palette. The colour choices here are
not taste: the four-slot categorical order is the validated default palette and
it clears the CVD and normal-vision separation gates on the adjacent pairlist
used by lines and bars. Two of the four slots (aqua, yellow) sit below 3:1
contrast on this surface, which obliges the relief rule -- so every chart that
uses all four also carries direct labels, and 05_analyze.py writes the same
numbers to CSV as a table view.

Scatter plots colour by EMBEDDER (two slots, which clear the all-pairs gate)
and encode the chunker with marker shape instead: four colours would fail the
all-pairs floors, and a scatter has no adjacency to hide behind.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8a86"
GRID = "#e3e2de"

# validated categorical order: blue, orange, aqua, yellow
CATEGORICAL: List[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# one-hue sequential ramp (blue, light -> dark) for the heatmap
SEQUENTIAL: List[str] = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
]

CHUNKER_ORDER = ["fixed_128", "fixed_256", "recursive_256", "semantic"]
EMBEDDER_ORDER = ["weak", "strong"]
MARKERS = {"fixed_128": "o", "fixed_256": "s", "recursive_256": "^", "semantic": "D"}


def chunker_colors(chunkers: Sequence[str]) -> Dict[str, str]:
    """Colour follows the entity, never its rank: a chunker keeps its slot even
    when a chart shows a subset."""
    return {c: CATEGORICAL[CHUNKER_ORDER.index(c) % len(CATEGORICAL)] for c in chunkers}


def embedder_colors() -> Dict[str, str]:
    return {"weak": CATEGORICAL[0], "strong": CATEGORICAL[1]}


def apply_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK_PRIMARY,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("crag_blue", SEQUENTIAL)


def save(fig, path) -> None:
    """PNG for the screen, PDF for the printer."""
    from pathlib import Path
    path = Path(path)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
