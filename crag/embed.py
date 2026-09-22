"""
Embedder wrappers (IV2), plus an offline stub so the whole pipeline can run
with no model downloaded.

Two things this module exists to get right:

1. **The bge query instruction goes on queries only.** bge-base-en-v1.5 was
   trained with "Represent this sentence for searching relevant passages: "
   prepended to the QUERY side and nothing on the passage side. Putting it on
   passages too (or forgetting it on queries) quietly costs retrieval quality,
   and would show up in the results as "the strong embedder isn't that
   strong" -- a fake finding in exactly the place this study is looking.
2. **Everything is L2-normalised.** The index is `IndexFlatIP`, so inner
   product is only cosine similarity if the vectors are unit length.
"""

from __future__ import annotations

import os
import re
import zlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

_WORD = re.compile(r"\w+")


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class BaseEmbedder(ABC):
    key: str = "base"
    model_name: str = ""
    dim: int = 0
    max_seq_length: int = 0
    query_prefix: str = ""
    passage_prefix: str = ""

    @abstractmethod
    def _encode(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        ...

    def encode_passages(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        prepared = [self.passage_prefix + t for t in texts] if self.passage_prefix else list(texts)
        return l2_normalize(self._encode(prepared, batch_size))

    def encode_queries(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        prepared = [self.query_prefix + t for t in texts] if self.query_prefix else list(texts)
        return l2_normalize(self._encode(prepared, batch_size))

    def chunker_embed_fn(self) -> Callable[[List[str]], np.ndarray]:
        """The callable the semantic chunker uses to score sentence buffers.

        Passage-side encoding: sentence buffers are document text, not
        queries, so they must not get the query instruction.
        """
        return lambda texts: self.encode_passages(texts)

    def describe(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "model_name": self.model_name,
            "dim": self.dim,
            "max_seq_length": self.max_seq_length,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
        }


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, spec: Any, device: Optional[str] = None,
                 fp16: Optional[bool] = None) -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        self.key = spec.key
        self.model_name = spec.model_name
        self.dim = spec.dim
        self.max_seq_length = spec.max_seq_length
        self.query_prefix = spec.query_prefix
        self.passage_prefix = spec.passage_prefix
        self.batch_size = spec.batch_size
        if fp16 is None:
            fp16 = getattr(spec, "fp16", False)
        self.fp16 = bool(fp16)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(self.model_name, device=device)
        # cap the sequence length at the study's shared limit, not the model's
        self.model.max_seq_length = min(self.model.max_seq_length, spec.max_seq_length)
        if self.fp16 and device == "cuda":
            # Off by default: measured 3.1-3.5x SLOWER than fp32 on a
            # tensor-core-less Turing card (see config.EmbedderSpec). Worth
            # turning on again only where a benchmark says so.
            self.model = self.model.half()

        # sentence-transformers 6 renamed this; keep working on both
        getter = (getattr(self.model, "get_embedding_dimension", None)
                  or getattr(self.model, "get_sentence_embedding_dimension"))
        actual = getter()
        if actual != spec.dim:
            raise ValueError(
                f"{self.model_name} has dim {actual}, config says {spec.dim}"
            )

    def _encode(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = self.model.encode(
            list(texts),
            batch_size=batch_size or self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,   # normalised once, in the base class
            show_progress_bar=False,
        )
        return np.asarray(out, dtype=np.float32)


class HashingEmbedder(BaseEmbedder):
    """Deterministic offline stand-in: hashed bag of words, sublinear tf.

    Not a neural embedder, but not noise either -- lexically similar texts get
    similar vectors, so retrieval on the synthetic corpus actually retrieves
    the right page. That makes an offline end-to-end run a real smoke test of
    the pipeline rather than a check that it does not crash.

    CRC32, not Python's hash(): str hashing is salted per process, which would
    make runs irreproducible across restarts.
    """

    def __init__(self, spec: Any = None, dim: int = 256, key: str = "stub") -> None:
        self.key = getattr(spec, "key", key)
        self.model_name = "hashing-stub"
        self.dim = dim
        self.max_seq_length = getattr(spec, "max_seq_length", 256)
        self.query_prefix = getattr(spec, "query_prefix", "")
        self.passage_prefix = getattr(spec, "passage_prefix", "")
        self.batch_size = getattr(spec, "batch_size", 256)

    def _encode(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _WORD.findall(text.lower()):
                out[i, zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
            np.log1p(out[i], out=out[i])
        return out


_EMBEDDER_CACHE: Dict[str, BaseEmbedder] = {}


def get_embedder(
    spec: Any, offline: Optional[bool] = None, device: Optional[str] = None, cache: bool = True
) -> BaseEmbedder:
    if offline is None:
        offline = os.environ.get("CRAG_OFFLINE", "0") == "1"
    cache_key = f"{'stub' if offline else 'st'}:{spec.key}:{device}"
    if cache and cache_key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[cache_key]
    emb: BaseEmbedder = HashingEmbedder(spec) if offline else SentenceTransformerEmbedder(spec, device=device)
    if cache:
        _EMBEDDER_CACHE[cache_key] = emb
    return emb
