"""
Significance testing, sized for the fact that this study is underpowered for
small effects and should say so out loud.

    paired_bootstrap  -- ACL-standard paired bootstrap for F1 / recall
    mcnemar_exact     -- exact binomial McNemar for EM (paired binary)
    mde_*             -- minimum detectable effect at the actual n
    holm_bonferroni   -- family-wise correction across the run matrix

No scipy: the exact test is math.comb, the normal quantiles are
statistics.NormalDist. One less thing to install on a Windows conda box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_NORM = NormalDist()


def _z(p: float) -> float:
    return _NORM.inv_cdf(p)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    n_resamples: int

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d["significant"] = self.significant
        return d


@dataclass
class McNemarResult:
    n: int
    n01: int          # a wrong, b right
    n10: int          # a right, b wrong
    n_discordant: int
    diff: float       # mean(a) - mean(b)
    p_value: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 13,
    alpha: float = 0.05,
    block: int = 1000,
) -> BootstrapResult:
    """Paired bootstrap over per-question scores (same questions, two systems).

    Resamples QUESTIONS, not scores, so the pairing that makes this test
    powerful is preserved. Returns a percentile CI on the mean difference and
    a two-sided bootstrap p-value.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"unpaired inputs: {a_arr.shape} vs {b_arr.shape}")
    n = a_arr.size
    if n == 0:
        raise ValueError("empty inputs")

    d = a_arr - b_arr
    observed = float(d.mean())

    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    done = 0
    while done < n_resamples:
        size = min(block, n_resamples - done)
        idx = rng.integers(0, n, size=(size, n))
        means[done:done + size] = d[idx].mean(axis=1)
        done += size

    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_left = float((means <= 0).mean())
    p_right = float((means >= 0).mean())
    p = min(1.0, 2 * min(p_left, p_right))
    p = max(p, 1.0 / n_resamples)  # a bootstrap cannot resolve below 1/B

    return BootstrapResult(
        mean_a=float(a_arr.mean()),
        mean_b=float(b_arr.mean()),
        diff=observed,
        ci_low=float(lo),
        ci_high=float(hi),
        p_value=p,
        n=n,
        n_resamples=n_resamples,
    )


def bootstrap_ci(
    x: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 13,
    alpha: float = 0.05,
    block: int = 1000,
) -> Tuple[float, float, float]:
    """(mean, ci_low, ci_high) for a single condition."""
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    if n == 0:
        raise ValueError("empty input")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    done = 0
    while done < n_resamples:
        size = min(block, n_resamples - done)
        idx = rng.integers(0, n, size=(size, n))
        means[done:done + size] = arr[idx].mean(axis=1)
        done += size
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# McNemar (exact) for EM
# ---------------------------------------------------------------------------

def mcnemar_exact(a_correct: Sequence[float], b_correct: Sequence[float]) -> McNemarResult:
    """Exact two-sided McNemar on paired binary outcomes (EM).

    Only the discordant pairs carry information: questions both systems get
    right, or both get wrong, say nothing about which system is better.
    """
    a_arr = np.asarray(a_correct, dtype=np.float64)
    b_arr = np.asarray(b_correct, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"unpaired inputs: {a_arr.shape} vs {b_arr.shape}")
    a_bin = a_arr > 0.5
    b_bin = b_arr > 0.5

    n10 = int(np.sum(a_bin & ~b_bin))   # a right, b wrong
    n01 = int(np.sum(~a_bin & b_bin))   # a wrong, b right
    n_disc = n10 + n01

    if n_disc == 0:
        p = 1.0
    else:
        k = min(n10, n01)
        tail = sum(math.comb(n_disc, i) for i in range(k + 1)) / (2 ** n_disc)
        p = min(1.0, 2 * tail)

    return McNemarResult(
        n=int(a_arr.size),
        n01=n01,
        n10=n10,
        n_discordant=n_disc,
        diff=float(a_arr.mean() - b_arr.mean()),
        p_value=float(p),
    )


# ---------------------------------------------------------------------------
# Minimum detectable effect
# ---------------------------------------------------------------------------

def mde_paired_mean(
    sd_diff: float, n: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest true mean difference detectable at the given n (F1, recall)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return (_z(1 - alpha / 2) + _z(power)) * sd_diff / math.sqrt(n)


def mde_from_pairs(
    a: Sequence[float], b: Sequence[float], alpha: float = 0.05, power: float = 0.80
) -> float:
    """MDE using the observed sd of per-question differences."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return mde_paired_mean(float(d.std(ddof=1)), d.size, alpha=alpha, power=power)


def mde_mcnemar(
    n_pairs: int, p_discordant: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """MDE for a difference in paired proportions (EM).

    Normal approximation: for a small true difference, Var(p_a - p_b) is
    approximately p_discordant / n, so the detectable difference is
    (z_alpha/2 + z_power) * sqrt(p_disc / n). Report it next to any null
    result -- "no significant difference" means nothing without it.
    """
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    if not 0 <= p_discordant <= 1:
        raise ValueError("p_discordant must be a proportion")
    return (_z(1 - alpha / 2) + _z(power)) * math.sqrt(p_discordant / n_pairs)


def required_n_paired_mean(
    effect: float, sd_diff: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """How many questions would be needed to detect `effect`."""
    if effect <= 0:
        raise ValueError("effect must be positive")
    n = ((_z(1 - alpha / 2) + _z(power)) * sd_diff / effect) ** 2
    return int(math.ceil(n))


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------

def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Step-down Holm correction. With 16 runs there are a lot of pairwise
    contrasts; the interaction contrast is the pre-registered one, everything
    else is exploratory and should be corrected."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        if p_values[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


# ---------------------------------------------------------------------------
# Interaction contrast: the primary RQ
# ---------------------------------------------------------------------------

def interaction_contrast(
    weak_best: Sequence[float],
    weak_worst: Sequence[float],
    strong_best: Sequence[float],
    strong_worst: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 13,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Difference-in-differences: (chunking spread under the weak embedder)
    minus (chunking spread under the strong embedder).

    This is the primary RQ as a single number. Both spreads are paired within
    embedder, and the two embedders are evaluated on the same questions, so
    the whole contrast is paired and one bootstrap over questions is valid.
    """
    w = np.asarray(weak_best, dtype=np.float64) - np.asarray(weak_worst, dtype=np.float64)
    s = np.asarray(strong_best, dtype=np.float64) - np.asarray(strong_worst, dtype=np.float64)
    return paired_bootstrap(w, s, n_resamples=n_resamples, seed=seed, alpha=alpha)
