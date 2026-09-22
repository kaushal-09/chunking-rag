"""
Fill-to-budget context selection.

This is the fix for the confound in the original design. Fixed top-k hands the
generator a different number of tokens in every condition -- 5 x 128-token
chunks is 640 tokens, 5 x 250-token semantic chunks is 1,250 -- so a "chunking
effect" measured at fixed k is partly a context-length effect. Instead we
retrieve N_CANDIDATES from FAISS and add chunks in rank order until the budget
would overflow, so every condition delivers the same token budget and the only
thing that varies is how that budget is carved up.

Budget tokens are counted with the GENERATOR's tokenizer, because the budget
is a claim about what the LLM reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_SEPARATOR = "\n\n"


@dataclass
class Candidate:
    """One FAISS hit, with its budget-token cost precomputed at index time."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    n_budget_tokens: int
    rank: int = -1
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Selection:
    chunks: List[Candidate]
    context: str
    n_chunks_used: int
    n_tokens_used: int          # sum of chunk costs + separators (bookkeeping)
    n_tokens_actual: int        # tokens in the assembled string (ground truth)
    budget: int
    n_candidates_seen: int
    overflow_stopped_at: Optional[int]  # rank of the chunk that ended selection

    @property
    def doc_ids(self) -> List[str]:
        return [c.doc_id for c in self.chunks]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_chunks_used": self.n_chunks_used,
            "n_tokens_used": self.n_tokens_used,
            "n_tokens_actual": self.n_tokens_actual,
            "budget": self.budget,
            "n_candidates_seen": self.n_candidates_seen,
            "overflow_stopped_at": self.overflow_stopped_at,
            "chunk_ids": [c.chunk_id for c in self.chunks],
            "doc_ids": self.doc_ids,
            "scores": [float(c.score) for c in self.chunks],
        }


def fill_to_budget(
    candidates: Sequence[Candidate],
    budget: int,
    separator: str = DEFAULT_SEPARATOR,
    separator_tokens: int = 1,
    policy: str = "stop",
    count_fn: Optional[Callable[[str], int]] = None,
) -> Selection:
    """Select chunks in rank order under a hard token budget.

    policy="stop" (the locked default) halts at the first chunk that would
        overflow -- literally "add them in rank order until the next would
        overflow the budget".
    policy="skip" keeps walking down the ranking to top the budget up. Better
        budget utilisation, but it reorders relevance, so it is an ablation
        rather than the main design.

    `count_fn` (the generator tokenizer's count) is optional; when given, the
    assembled string is re-counted so n_tokens_actual is exact rather than
    estimated.
    """
    if policy not in {"stop", "skip"}:
        raise ValueError(f"unknown budget policy: {policy!r}")
    if budget <= 0:
        raise ValueError("budget must be positive")

    chosen: List[Candidate] = []
    total = 0
    stopped_at: Optional[int] = None

    for i, cand in enumerate(candidates):
        cost = int(cand.n_budget_tokens) + (separator_tokens if chosen else 0)
        if total + cost > budget:
            if policy == "stop":
                stopped_at = cand.rank if cand.rank >= 0 else i
                break
            continue
        chosen.append(cand)
        total += cost

    context = separator.join(c.text for c in chosen)
    actual = count_fn(context) if (count_fn and context) else total

    return Selection(
        chunks=chosen,
        context=context,
        n_chunks_used=len(chosen),
        n_tokens_used=total,
        n_tokens_actual=int(actual),
        budget=budget,
        n_candidates_seen=len(candidates),
        overflow_stopped_at=stopped_at,
    )


def budget_utilisation(selection: Selection) -> float:
    """Share of the budget actually delivered. Report this per condition: if
    two conditions differ by more than a few points, the budget is not matched
    and the comparison is not clean."""
    return selection.n_tokens_used / selection.budget if selection.budget else 0.0
