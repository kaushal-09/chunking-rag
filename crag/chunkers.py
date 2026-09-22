"""
The four chunkers (IV1). All return character offsets, all hard-capped.

Contract, identical for every chunker:
    chunk(doc_id, text) -> List[Chunk(doc_id, text, char_start, char_end, n_tokens)]
with the invariants
    * chunk.text == doc_text[chunk.char_start:chunk.char_end]   (exact slice)
    * chunk.n_tokens <= max_tokens                              (hard cap)
    * chunks are ordered and non-overlapping (overlap is 0 in this study)
    * no chunk is empty or whitespace-only

Character offsets are not decoration: they are what makes provenance,
answer-string containment and any later token-level analysis possible after
the boundaries have moved.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .tokenization import BaseTokenizer, TokenView

CharRange = Tuple[int, int]
EmbedFn = Callable[[List[str]], np.ndarray]


# ---------------------------------------------------------------------------
# Chunk record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    doc_id: str
    text: str
    char_start: int
    char_end: int
    n_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Chunk":
        return Chunk(
            doc_id=d["doc_id"],
            text=d["text"],
            char_start=int(d["char_start"]),
            char_end=int(d["char_end"]),
            n_tokens=int(d["n_tokens"]),
        )


# ---------------------------------------------------------------------------
# Sentence splitting (offline, deterministic, offset-preserving)
# ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "eg", "ie",
    "fig", "no", "al", "inc", "ltd", "co", "corp", "dept", "est", "approx",
    "cf", "ca", "vol", "op", "pp", "ed", "rev", "gen", "col", "capt", "sgt",
    "lt", "hon", "pres", "gov", "sen", "rep", "univ", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_SENT_END = re.compile(r'([.!?]+)(["\'’”)\]]*)(\s+)')
_PARAGRAPH = re.compile(r"\n\s*\n")
_LAST_WORD = re.compile(r"(\S+)\s*$")


def split_sentences(text: str) -> List[CharRange]:
    """Sentence char ranges. Regex-based on purpose: no nltk download, fully
    deterministic, and reproducible on a machine with no network."""
    boundaries = set()

    for m in _PARAGRAPH.finditer(text):
        boundaries.add(m.end())

    for m in _SENT_END.finditer(text):
        head = text[:m.start(1)]
        lw = _LAST_WORD.search(head)
        if lw:
            word = lw.group(1).strip(".,;:()[]\"'").lower()
            # "Dr." / "e.g." / initials like "J." are not sentence ends
            if word in _ABBREVIATIONS or len(word) <= 1:
                continue
            if word.replace(".", "") in _ABBREVIATIONS:
                continue
        boundaries.add(m.end())

    cuts = sorted(b for b in boundaries if 0 < b < len(text))
    out: List[CharRange] = []
    prev = 0
    for b in cuts + [len(text)]:
        seg = text[prev:b]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            trail = len(seg) - len(seg.rstrip())
            out.append((prev + lead, b - trail))
        prev = b
    return out


# ---------------------------------------------------------------------------
# Base chunker
# ---------------------------------------------------------------------------

class Chunker(ABC):
    kind: str = "base"

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        max_tokens: int = 256,
        min_chunk_tokens: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_tokens = int(max_tokens)
        self.min_chunk_tokens = int(min_chunk_tokens)

    # -- public API --------------------------------------------------------
    def chunk(self, doc_id: str, text: str) -> List[Chunk]:
        if not text or not text.strip():
            return []
        tv = TokenView(text, self.tokenizer)
        if tv.n_tokens == 0:
            return []
        ranges = self._ranges(tv, doc_id)
        return self._emit(doc_id, tv, ranges)

    @property
    def params(self) -> Dict[str, Any]:
        return {"max_tokens": self.max_tokens, "min_chunk_tokens": self.min_chunk_tokens}

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({self.params})"

    # -- to implement ------------------------------------------------------
    @abstractmethod
    def _ranges(self, tv: TokenView, doc_id: str = "") -> List[CharRange]:
        """Candidate char ranges; the cap and clean-up are applied by _emit."""

    # -- shared post-processing -------------------------------------------
    def _emit(self, doc_id: str, tv: TokenView, ranges: Sequence[CharRange]) -> List[Chunk]:
        # 1. snap to whole tokens, drop empties, enforce the hard cap
        #    Splitting is repeated until every piece measures within the cap:
        #    split_range works in token indices and count_range by bisect, so a
        #    single pass is only enough if they agree exactly. They do now
        #    (TokenView normalises its spans), but the cap is the one invariant
        #    the whole study rests on, so it is enforced by measurement rather
        #    than by trusting the arithmetic.
        capped: List[CharRange] = []
        pending = []
        for start, end in ranges:
            snapped = tv.snap(start, end)
            if snapped is not None:
                pending.append(snapped)

        for _ in range(8):                      # bounded; one pass is normal
            over = [r for r in pending if tv.count_range(*r) > self.max_tokens]
            if not over:
                break
            rebuilt: List[CharRange] = []
            for s, e in pending:
                if tv.count_range(s, e) > self.max_tokens:
                    rebuilt.extend(tv.split_range(s, e, self.max_tokens))
                else:
                    rebuilt.append((s, e))
            if rebuilt == pending:              # not converging: stop cleanly
                break
            pending = rebuilt
        capped = pending

        # 2. absorb runt chunks into the previous one where the cap allows.
        #    Semantic chunkers in particular emit 5-token stragglers; a chunk
        #    too small to carry an answer is dead weight in the index.
        merged: List[CharRange] = []
        for s, e in capped:
            if (
                merged
                and self.min_chunk_tokens > 0
                and tv.count_range(s, e) < self.min_chunk_tokens
                and tv.count_range(merged[-1][0], e) <= self.max_tokens
            ):
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        # 3. materialise, and refuse to emit anything over the cap
        out: List[Chunk] = []
        for s, e in merged:
            n = tv.count_range(s, e)
            if n == 0:
                continue
            body = tv.text[s:e]
            if not body.strip():
                continue
            if n > self.max_tokens:
                raise AssertionError(
                    f"{self.__class__.__name__} produced a {n}-token chunk with a "
                    f"{self.max_tokens}-token cap in doc {doc_id!r} at [{s}:{e}]"
                )
            out.append(Chunk(doc_id=doc_id, text=body, char_start=s, char_end=e, n_tokens=n))
        return out


# ---------------------------------------------------------------------------
# 1 & 2. Fixed-token chunkers  (fixed_128, fixed_256)
# ---------------------------------------------------------------------------

class FixedTokenChunker(Chunker):
    """Fixed-token windows. `overlap_tokens` is 0 throughout this study."""

    kind = "fixed"

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        target_tokens: int = 128,
        overlap_tokens: int = 0,
        max_tokens: Optional[int] = None,
        min_chunk_tokens: int = 0,
    ) -> None:
        target_tokens = int(target_tokens)
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        super().__init__(
            tokenizer,
            max_tokens=min(target_tokens, max_tokens or target_tokens),
            min_chunk_tokens=min_chunk_tokens,
        )
        self.target_tokens = target_tokens
        self.overlap_tokens = int(overlap_tokens)

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "target_tokens": self.target_tokens,
            "overlap_tokens": self.overlap_tokens,
            **super().params,
        }

    def _ranges(self, tv: TokenView, doc_id: str = "") -> List[CharRange]:
        stride = self.target_tokens - self.overlap_tokens
        return tv.split_range(0, len(tv.text), self.target_tokens, stride=stride)


# ---------------------------------------------------------------------------
# 3. Recursive character splitter, measured in tokens  (recursive_256)
# ---------------------------------------------------------------------------

class RecursiveTokenChunker(Chunker):
    """Split on the most semantic separator that fits, then greedily re-merge.

    Same idea as LangChain's RecursiveCharacterTextSplitter, but length is
    measured with the study's tokenizer instead of len(str), and every piece is
    tracked as a char range so offsets survive.

    Free parameter -- `min_fill_ratio`. Pure greedy packing (ratio 1.0, the
    default, and exactly the standard splitter's behaviour) merges across a
    paragraph break whenever the tokens fit, so a chunk can straddle two
    sections. A lower ratio lets the splitter close a chunk at a strong
    separator once it is at least that full, trading packing density for
    structural alignment. 1.0 is in the tuning grid, so if the standard
    behaviour wins on dev, that is what gets reported -- the point is that the
    baseline was given the same tuning courtesy as the semantic chunker.
    """

    kind = "recursive"

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        target_tokens: int = 256,
        separators: Sequence[str] = ("\n\n", "\n", ". ", " ", ""),
        min_chunk_tokens: int = 32,
        min_fill_ratio: float = 1.0,
        strong_separator_level: int = 1,
        max_tokens: Optional[int] = None,
    ) -> None:
        target_tokens = int(target_tokens)
        super().__init__(
            tokenizer,
            max_tokens=min(target_tokens, max_tokens or target_tokens),
            min_chunk_tokens=min_chunk_tokens,
        )
        self.target_tokens = target_tokens
        self.separators = list(separators)
        self.min_fill_ratio = float(min_fill_ratio)
        self.strong_separator_level = int(strong_separator_level)

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "target_tokens": self.target_tokens,
            "separators": self.separators,
            "min_fill_ratio": self.min_fill_ratio,
            "strong_separator_level": self.strong_separator_level,
            **super().params,
        }

    def _ranges(self, tv: TokenView, doc_id: str = "") -> List[CharRange]:
        pieces = self._recurse(tv, 0, len(tv.text), sep_idx=0, inherited_level=len(self.separators))
        return self._merge(tv, pieces)

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _split_on(text: str, start: int, end: int, sep: str) -> List[CharRange]:
        """Contiguous ranges tiling [start, end), cut after each separator."""
        out: List[CharRange] = []
        cur = start
        for m in re.finditer(re.escape(sep), text[start:end]):
            cut = start + m.end()
            if cut > cur:
                out.append((cur, cut))
                cur = cut
        if cur < end:
            out.append((cur, end))
        return out

    def _recurse(
        self, tv: TokenView, start: int, end: int, sep_idx: int, inherited_level: int
    ) -> List[Tuple[int, int, int]]:
        """Pieces as (start, end, boundary_level).

        boundary_level records WHICH separator ends the piece -- 0 is the
        strongest ("\\n\\n"), len(separators) means "no separator, this is
        either a hard token cut or the end of the document". The merge pass
        needs it to tell a paragraph break from a mid-sentence cut.
        """
        if end <= start:
            return []
        if tv.count_range(start, end) <= self.target_tokens:
            return [(start, end, inherited_level)]

        weakest = len(self.separators)
        if sep_idx >= len(self.separators) or self.separators[sep_idx] == "":
            hard = tv.split_range(start, end, self.target_tokens)
            out = [(s, e, weakest) for s, e in hard[:-1]]
            if hard:
                s, e = hard[-1]
                out.append((s, e, inherited_level))
            return out

        parts = self._split_on(tv.text, start, end, self.separators[sep_idx])
        if len(parts) <= 1:
            return self._recurse(tv, start, end, sep_idx + 1, inherited_level)

        out: List[Tuple[int, int, int]] = []
        for i, (s, e) in enumerate(parts):
            is_last = i == len(parts) - 1
            level = inherited_level if is_last else sep_idx
            out.extend(self._recurse(tv, s, e, sep_idx + 1, level))
        return out

    def _merge(self, tv: TokenView, pieces: Sequence[Tuple[int, int, int]]) -> List[CharRange]:
        if not pieces:
            return []
        floor = self.min_fill_ratio * self.target_tokens
        out: List[CharRange] = []
        cur_s, cur_e, cur_level = pieces[0]
        for s, e, level in pieces[1:]:
            at_strong_boundary = cur_level <= self.strong_separator_level
            full_enough = tv.count_range(cur_s, cur_e) >= floor
            if at_strong_boundary and full_enough:
                out.append((cur_s, cur_e))
                cur_s, cur_e, cur_level = s, e, level
            elif tv.count_range(cur_s, e) <= self.target_tokens:
                cur_e, cur_level = e, level
            else:
                out.append((cur_s, cur_e))
                cur_s, cur_e, cur_level = s, e, level
        out.append((cur_s, cur_e))
        return out


# ---------------------------------------------------------------------------
# 4. Semantic chunker  (semantic)
# ---------------------------------------------------------------------------

class SemanticChunker(Chunker):
    """Sentence-split, embed, break where adjacent cosine distance spikes.

    `embed_fn` is injected rather than imported: in the real pipeline it is the
    run's own embedder (locked decision -- a weak embedder draws weak
    boundaries, and that is part of the effect under test); in tests it is a
    deterministic stub.

    The cap matters here more than anywhere else. Percentile thresholding on a
    long topically-uniform page routinely yields one 4,000-token "chunk"; the
    cap hard-splits those, at sentence borders first and inside a sentence only
    when a single sentence is itself over the cap.
    """

    kind = "semantic"

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        embed_fn: EmbedFn,
        percentile: float = 90.0,
        buffer_size: int = 1,
        max_tokens: int = 256,
        min_chunk_tokens: int = 32,
        cache: Optional[Dict[Any, Any]] = None,
    ) -> None:
        super().__init__(tokenizer, max_tokens=max_tokens, min_chunk_tokens=min_chunk_tokens)
        self.embed_fn = embed_fn
        self.percentile = float(percentile)
        self.buffer_size = int(buffer_size)
        # Sentence splits and adjacent distances depend on the document, the
        # embedder and buffer_size -- but NOT on the percentile. Sweeping four
        # percentiles without this cache re-embeds every sentence in the corpus
        # four times for identical vectors; with it, once.
        self.cache = cache

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "percentile": self.percentile,
            "buffer_size": self.buffer_size,
            **super().params,
        }

    def _ranges(self, tv: TokenView, doc_id: str = "") -> List[CharRange]:
        key = (doc_id, self.buffer_size)
        cached = self.cache.get(key) if (self.cache is not None and doc_id) else None
        if cached is None:
            sents = split_sentences(tv.text)
            sents = [(s, e) for (s, e) in sents if tv.count_range(s, e) > 0]
            dists = self._distances(tv, sents) if len(sents) > 1 else np.zeros(0)
            if self.cache is not None and doc_id:
                self.cache[key] = (sents, dists)
        else:
            sents, dists = cached

        if len(sents) <= 1:
            return tv.split_range(0, len(tv.text), self.max_tokens)

        breakpoints = self._threshold(dists)
        groups: List[List[int]] = []
        cur: List[int] = [0]
        for i in range(1, len(sents)):
            if i in breakpoints:
                groups.append(cur)
                cur = [i]
            else:
                cur.append(i)
        groups.append(cur)

        out: List[CharRange] = []
        for group in groups:
            out.extend(self._pack(tv, [sents[i] for i in group]))
        return out

    # -- internals ---------------------------------------------------------
    def _threshold(self, dists: np.ndarray) -> set:
        """Percentile breakpoints over precomputed adjacent distances."""
        if dists.size == 0:
            return set()
        threshold = float(np.percentile(dists, self.percentile))
        # strict ">" matches the standard percentile-breakpoint implementation.
        # Consequence worth knowing: when many distances tie at the threshold
        # (a uniform page, or a stub embedder in the tests) no breakpoint is
        # created and the whole document stays one group -- which is precisely
        # why the hard cap below is not optional.
        return {i + 1 for i, d in enumerate(dists) if float(d) > threshold}

    def _distances(self, tv: TokenView, sents: Sequence[CharRange]) -> np.ndarray:
        buffers: List[str] = []
        for i in range(len(sents)):
            lo = max(0, i - self.buffer_size)
            hi = min(len(sents), i + self.buffer_size + 1)
            buffers.append(" ".join(tv.text[s:e] for s, e in sents[lo:hi]))

        emb = np.asarray(self.embed_fn(buffers), dtype=np.float64)
        if emb.ndim != 2 or emb.shape[0] != len(sents):
            raise ValueError(
                f"embed_fn returned {emb.shape}, expected ({len(sents)}, dim)"
            )
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms

        return 1.0 - np.sum(emb[:-1] * emb[1:], axis=1)

    def _pack(self, tv: TokenView, sents: Sequence[CharRange]) -> List[CharRange]:
        out: List[CharRange] = []
        cur_s: Optional[int] = None
        cur_e: Optional[int] = None
        for s, e in sents:
            n = tv.count_range(s, e)
            if n > self.max_tokens:
                if cur_s is not None:
                    out.append((cur_s, cur_e))
                    cur_s = cur_e = None
                out.extend(tv.split_range(s, e, self.max_tokens))
                continue
            if cur_s is None:
                cur_s, cur_e = s, e
            elif tv.count_range(cur_s, e) <= self.max_tokens:
                cur_e = e
            else:
                out.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        if cur_s is not None:
            out.append((cur_s, cur_e))
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_chunker(
    kind: str,
    tokenizer: BaseTokenizer,
    params: Optional[Dict[str, Any]] = None,
    embed_fn: Optional[EmbedFn] = None,
    global_max_tokens: Optional[int] = 256,
    cache: Optional[Dict[Any, Any]] = None,
) -> Chunker:
    p = dict(params or {})
    if cache is not None and kind == "semantic":
        p["cache"] = cache
    if global_max_tokens is not None:
        p.setdefault("max_tokens", global_max_tokens)
    if kind == "fixed":
        return FixedTokenChunker(tokenizer, **p)
    if kind == "recursive":
        return RecursiveTokenChunker(tokenizer, **p)
    if kind == "semantic":
        if embed_fn is None:
            raise ValueError("the semantic chunker needs an embed_fn")
        return SemanticChunker(tokenizer, embed_fn=embed_fn, **p)
    raise ValueError(f"unknown chunker kind: {kind!r}")


def build_from_spec(
    spec: Any,
    tokenizer: BaseTokenizer,
    embed_fn: Optional[EmbedFn] = None,
    overrides: Optional[Dict[str, Any]] = None,
    global_max_tokens: Optional[int] = 256,
    cache: Optional[Dict[Any, Any]] = None,
) -> Chunker:
    """Build from a config.ChunkerSpec, optionally overriding tuned params."""
    params = dict(spec.params)
    params.update(overrides or {})
    return build_chunker(
        spec.kind, tokenizer, params=params, embed_fn=embed_fn,
        global_max_tokens=global_max_tokens, cache=cache,
    )
