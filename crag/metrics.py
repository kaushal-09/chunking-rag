"""
Boundary-invariant metrics.

The original proposal scored MRR@5 / Hit@5 over chunks. That is ill-defined
here: chunk boundaries ARE the independent variable, so each condition has a
different candidate set and there is no consistent "gold chunk" to rank
against. Everything in this module is instead invariant to how the boundaries
were drawn:

    answer_recall  -- does the assembled context contain a gold answer string?
                      (token-sequence containment, DPR-style has_answer)
    page_hit       -- did any retrieved chunk come from a gold wikipedia_id?
                      (KILT provenance; independent of chunking entirely)
    EM / F1        -- standard SQuAD normalisation over the generated answer
    abstention     -- did the model emit the NO_ANSWER token?

Abstention is scored separately and never counted as a correct answer: it is
the thing that lets us tell "declined" apart from "hallucinated".
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Set

NO_ANSWER = "NO_ANSWER"

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# SQuAD / NQ normalisation
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, drop articles, squash whitespace."""
    if s is None:
        return ""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def answer_tokens(s: str) -> List[str]:
    return normalize_answer(s).split()


# ---------------------------------------------------------------------------
# Answer-level metrics
# ---------------------------------------------------------------------------

def is_abstention(prediction: str, no_answer: str = NO_ANSWER) -> bool:
    if prediction is None:
        return False
    stripped = prediction.strip()
    if not stripped:
        return False
    upper = stripped.upper()
    token = no_answer.upper()
    # accept the bare token, or the token as the first thing the model says
    return upper == token or upper.startswith(token) or normalize_answer(
        stripped
    ) == normalize_answer(no_answer.replace("_", " "))


def _clean_prediction(prediction: str, no_answer: str = NO_ANSWER) -> str:
    """An abstention scores 0 on EM/F1 -- it is not a wrong answer, but it is
    not a right one either, and it is reported on its own axis."""
    if prediction is None:
        return ""
    if is_abstention(prediction, no_answer):
        return ""
    return prediction


def exact_match(prediction: str, golds: Sequence[str], no_answer: str = NO_ANSWER) -> float:
    pred = normalize_answer(_clean_prediction(prediction, no_answer))
    if not pred:
        return 0.0
    return float(any(pred == normalize_answer(g) for g in golds))


def f1(prediction: str, golds: Sequence[str], no_answer: str = NO_ANSWER) -> float:
    pred_tokens = answer_tokens(_clean_prediction(prediction, no_answer))
    best = 0.0
    for gold in golds:
        gold_tokens = answer_tokens(gold)
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        precision = n_same / len(pred_tokens)
        recall = n_same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


# ---------------------------------------------------------------------------
# Retrieval-level metrics (boundary-invariant)
# ---------------------------------------------------------------------------

def contains_answer(context: str, golds: Sequence[str]) -> bool:
    """DPR-style has_answer: a gold answer's normalised TOKEN SEQUENCE appears
    contiguously in the normalised context.

    Token-sequence matching, not raw substring matching: substring matching
    fires on "1" inside "1994" and inflates recall for short numeric answers,
    which NQ has a lot of.
    """
    ctx = answer_tokens(context)
    if not ctx:
        return False
    for gold in golds:
        gt = answer_tokens(gold)
        if not gt:
            continue
        n = len(gt)
        first = gt[0]
        for i in range(len(ctx) - n + 1):
            if ctx[i] == first and ctx[i:i + n] == gt:
                return True
    return False


def answer_recall(context: str, golds: Sequence[str]) -> float:
    return float(contains_answer(context, golds))


def page_hit(retrieved_doc_ids: Iterable[str], gold_page_ids: Iterable[str]) -> float:
    gold: Set[str] = {str(g) for g in gold_page_ids}
    if not gold:
        return 0.0
    return float(any(str(d) in gold for d in retrieved_doc_ids))


def page_precision(retrieved_doc_ids: Sequence[str], gold_page_ids: Iterable[str]) -> float:
    """Share of retrieved chunks coming from a gold page. Diagnostic only --
    it says how much of the budget was spent on the right document."""
    ids = list(retrieved_doc_ids)
    if not ids:
        return 0.0
    gold: Set[str] = {str(g) for g in gold_page_ids}
    return sum(1 for d in ids if str(d) in gold) / len(ids)


# ---------------------------------------------------------------------------
# Question typing (secondary RQ: does the best chunker depend on the question?)
# ---------------------------------------------------------------------------

_TYPE_RULES = (
    ("count", re.compile(r"\bhow (many|much)\b")),
    ("date", re.compile(r"\b(when|what year|which year|what date)\b")),
    ("person", re.compile(r"\b(who|whose|whom)\b")),
    ("place", re.compile(r"\b(where)\b")),
    ("reason", re.compile(r"\b(why|how (do|does|did|is|are|was|were|can))\b")),
    ("entity", re.compile(r"\b(which|what)\b")),
)


def question_type(question: str) -> str:
    """Coarse question type from its wh-word.

    NQ ships no question-type label, so this is a heuristic, and it is a
    *surface* one: it reads the wh-word, nothing more. Good enough to break an
    aggregate effect apart and see whether it is an averaging artifact; not
    good enough to build a claim on by itself. Report it as descriptive.
    """
    q = (question or "").lower()
    for name, pattern in _TYPE_RULES:
        if pattern.search(q):
            return name
    return "other"


def answer_length_bucket(answers: Sequence[str]) -> str:
    """Short factoid vs longer answer -- the other axis the report flags."""
    if not answers:
        return "none"
    n = min(len(answer_tokens(a)) for a in answers)
    if n <= 1:
        return "1_token"
    if n <= 3:
        return "2-3_tokens"
    return "4+_tokens"


# ---------------------------------------------------------------------------
# Per-query record scoring
# ---------------------------------------------------------------------------

def score_query(
    prediction: Optional[str],
    golds: Sequence[str],
    context: str,
    retrieved_doc_ids: Sequence[str],
    gold_page_ids: Sequence[str],
    no_answer: str = NO_ANSWER,
) -> dict:
    """All DVs for one question, in one dict, ready to append to a run jsonl."""
    rec = {
        "answer_recall": answer_recall(context, golds),
        "page_hit": page_hit(retrieved_doc_ids, gold_page_ids),
        "page_precision": page_precision(retrieved_doc_ids, gold_page_ids),
    }
    if prediction is not None:
        rec["em"] = exact_match(prediction, golds, no_answer)
        rec["f1"] = f1(prediction, golds, no_answer)
        rec["abstained"] = float(is_abstention(prediction, no_answer))
        # the diagnostic that motivates the secondary RQ: context had the
        # answer, model still got it wrong
        rec["context_ok_answer_wrong"] = float(
            rec["answer_recall"] > 0 and rec["em"] == 0
        )
    return rec
