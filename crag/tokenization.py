"""
Tokenizer interface + HF wrapper + offline stub, plus TokenView.

Everything downstream measures length in TOKENS but reports boundaries in
CHARACTERS, so the one thing this module has to get right is the mapping
between the two. HF fast tokenizers give it to us via offset_mapping; the
stub reproduces the same contract with a regex so the whole chunking layer is
testable with no downloads and no GPU.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TokenSpan = Tuple[int, int]  # (char_start, char_end), end-exclusive


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

class BaseTokenizer(ABC):
    """Minimal contract: text -> character spans of its tokens."""

    name: str = "base"

    @abstractmethod
    def token_spans(self, text: str) -> List[TokenSpan]:
        """Non-overlapping, ascending, whitespace-excluding token spans."""

    def count(self, text: str) -> int:
        return len(self.token_spans(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        spans = self.token_spans(text)
        if len(spans) <= max_tokens:
            return text
        if max_tokens <= 0:
            return ""
        return text[spans[0][0]:spans[max_tokens - 1][1]]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.__class__.__name__}({self.name!r})"


class StubTokenizer(BaseTokenizer):
    """Offline, dependency-free tokenizer for tests and CI.

    Splits into word-ish runs and single punctuation marks. Not wordpiece, but
    the same *shape* of output (ordered, non-overlapping char spans), which is
    all the chunkers depend on. English prose runs ~1.3 wordpieces per word, so
    stub counts are systematically a bit lower than BERT's -- fine for
    structural tests, never used for reported numbers.
    """

    name = "stub"
    _PATTERN = re.compile(r"\w+|[^\w\s]")

    def token_spans(self, text: str) -> List[TokenSpan]:
        return [(m.start(), m.end()) for m in self._PATTERN.finditer(text)]


class HFTokenizer(BaseTokenizer):
    """Wraps a HuggingFace *fast* tokenizer (offset_mapping is required)."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer  # lazy: keeps step-1 import-free

        self.name = model_name
        try:
            self._tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except TypeError:
            # transformers v5 dropped slow tokenizers, and with them the flag
            self._tok = AutoTokenizer.from_pretrained(model_name)
        if not getattr(self._tok, "is_fast", False):
            raise RuntimeError(
                f"{model_name} has no fast tokenizer; offset mapping is required"
            )

    def token_spans(self, text: str) -> List[TokenSpan]:
        if not text:
            return []
        enc = self._tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        out: List[TokenSpan] = []
        for start, end in enc["offset_mapping"]:
            start, end = int(start), int(end)
            if end > start:
                out.append((start, end))
        return out

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tok(text, add_special_tokens=False)["input_ids"])


_TOKENIZER_CACHE: Dict[str, BaseTokenizer] = {}


def get_tokenizer(model_name: str, offline: Optional[bool] = None) -> BaseTokenizer:
    """Factory. `offline=True` (or CRAG_OFFLINE=1) returns the stub."""
    if offline is None:
        offline = os.environ.get("CRAG_OFFLINE", "0") == "1"
    key = "stub" if (offline or model_name in {"stub", "offline"}) else model_name
    if key not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[key] = StubTokenizer() if key == "stub" else HFTokenizer(key)
    return _TOKENIZER_CACHE[key]


# ---------------------------------------------------------------------------
# TokenView: tokenize a document once, then answer range queries in O(log n)
# ---------------------------------------------------------------------------

class TokenView:
    """Precomputed token spans over one document.

    Chunkers ask the same document a lot of overlapping questions ("how many
    tokens between these two characters?", "split this range into <=256-token
    pieces"). Tokenizing per question is quadratic; this tokenizes once and
    answers by binary search, which also guarantees every chunker measures
    length identically.
    """

    __slots__ = ("text", "spans", "_starts", "_ends")

    def __init__(self, text: str, tokenizer: BaseTokenizer) -> None:
        self.text = text
        self.spans: List[TokenSpan] = self._normalise(tokenizer.token_spans(text))
        self._starts = [s for s, _ in self.spans]
        self._ends = [e for _, e in self.spans]

    @staticmethod
    def _normalise(spans: List[TokenSpan]) -> List[TokenSpan]:
        """Force strictly increasing, non-overlapping spans.

        This is not paranoia. `count_range` answers by binary search over the
        start and end arrays, while `split_range` slices by token INDEX -- and
        the two agree only if both arrays are sorted. A fast tokenizer's
        offset_mapping is usually monotonic, but normalisation (BERT lowercases
        and strips accents) can emit an overlapping or out-of-order pair on
        unusual Unicode. When that happens the two methods disagree by a token
        or two, and a chunker that splits by index then measures by bisect
        quietly exceeds its own cap -- which is exactly how a 128-token chunker
        produced a 129-token chunk over 4,331 Wikipedia pages.
        """
        cleaned: List[TokenSpan] = []
        last_end = 0
        for start, end in spans:
            if end <= start:
                continue
            start = max(start, last_end)
            if end <= start:
                continue
            cleaned.append((start, end))
            last_end = end
        return cleaned

    # -- basics ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.spans)

    @property
    def n_tokens(self) -> int:
        return len(self.spans)

    # -- range queries -----------------------------------------------------
    def token_index_range(self, char_start: int, char_end: int) -> Tuple[int, int]:
        """[i, j) token indices fully contained in [char_start, char_end)."""
        i = bisect_left(self._starts, char_start)
        j = bisect_right(self._ends, char_end)
        return (i, max(i, j))

    def count_range(self, char_start: int, char_end: int) -> int:
        i, j = self.token_index_range(char_start, char_end)
        return j - i

    def snap(self, char_start: int, char_end: int) -> Optional[Tuple[int, int]]:
        """Shrink a char range to exactly cover whole tokens.

        Returns None if the range contains no tokens (pure whitespace, say),
        which is the signal to drop the chunk.
        """
        i, j = self.token_index_range(char_start, char_end)
        if j <= i:
            return None
        return (self._starts[i], self._ends[j - 1])

    def split_range(
        self, char_start: int, char_end: int, max_tokens: int, stride: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """Hard-split a char range into <=max_tokens pieces, on token borders.

        `stride` < max_tokens produces overlap; the study runs stride == max
        (overlap 0).
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        stride = max_tokens if stride is None else stride
        if stride <= 0:
            raise ValueError("stride must be positive")
        i, j = self.token_index_range(char_start, char_end)
        out: List[Tuple[int, int]] = []
        k = i
        while k < j:
            end_tok = min(k + max_tokens, j)
            out.append((self._starts[k], self._ends[end_tok - 1]))
            if end_tok >= j:
                break
            k += stride
        return out
