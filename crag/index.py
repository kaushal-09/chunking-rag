"""
Vector index: FAISS IndexFlatIP, with a numpy fallback.

Flat, exact, no IVF/PQ. At ~10^5 vectors an approximate index buys nothing but
a second source of variance between conditions -- and "the semantic condition
lost because its index quantised differently" is not a finding anyone wants to
defend on a poster.

The numpy fallback exists so the pipeline runs end to end on a machine where
faiss is not installed. It is exact too, just slower.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False


class BaseIndex:
    backend = "base"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    @property
    def n_vectors(self) -> int:
        raise NotImplementedError

    def add(self, vectors: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError


class NumpyFlatIP(BaseIndex):
    backend = "numpy"

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    @property
    def n_vectors(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._vectors = np.vstack([self._vectors, vectors]) if self.n_vectors else vectors

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if self.n_vectors == 0 or k <= 0:
            return (np.zeros((queries.shape[0], 0), dtype=np.float32),
                    np.zeros((queries.shape[0], 0), dtype=np.int64))
        sims = queries @ self._vectors.T
        k = min(k, self.n_vectors)
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(sims.shape[0])[:, None]
        ordered = idx[rows, np.argsort(-sims[rows, idx], axis=1)]
        return sims[rows, ordered].astype(np.float32), ordered.astype(np.int64)

    def save(self, path: Path) -> None:
        np.save(Path(path).with_suffix(".npy"), self._vectors)

    @classmethod
    def load(cls, path: Path) -> "NumpyFlatIP":
        vectors = np.load(Path(path).with_suffix(".npy"))
        index = cls(vectors.shape[1])
        index.add(vectors)
        return index


class FaissFlatIP(BaseIndex):
    backend = "faiss"

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        import faiss

        self._faiss = faiss
        self.index = faiss.IndexFlatIP(dim)

    @property
    def n_vectors(self) -> int:
        return int(self.index.ntotal)

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = min(k, self.n_vectors)
        return self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)

    def save(self, path: Path) -> None:
        self._faiss.write_index(self.index, str(Path(path).with_suffix(".faiss")))

    @classmethod
    def load(cls, path: Path) -> "FaissFlatIP":
        import faiss

        raw = faiss.read_index(str(Path(path).with_suffix(".faiss")))
        index = cls.__new__(cls)
        index._faiss = faiss
        index.index = raw
        index.dim = raw.d
        return index


def build_index(vectors: np.ndarray, prefer_faiss: bool = True) -> BaseIndex:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    dim = int(vectors.shape[1])
    index: BaseIndex = FaissFlatIP(dim) if (prefer_faiss and faiss_available()) else NumpyFlatIP(dim)
    index.add(vectors)
    return index


# ---------------------------------------------------------------------------
# On-disk bundle: index + the metadata needed to interpret its row numbers
# ---------------------------------------------------------------------------

@dataclass
class IndexMeta:
    chunkset_id: str
    embedder_key: str
    model_name: str
    dim: int
    n_vectors: int
    backend: str
    chunks_path: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def index_paths(index_dir: Path, chunkset_id: str, embedder_key: str) -> Dict[str, Path]:
    stem = f"{chunkset_id}__{embedder_key}"
    base = Path(index_dir) / stem
    return {"stem": base, "meta": base.with_suffix(".meta.json")}


def save_index(
    index: BaseIndex, meta: IndexMeta, index_dir: Path, chunkset_id: str, embedder_key: str
) -> Dict[str, Path]:
    paths = index_paths(index_dir, chunkset_id, embedder_key)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    index.save(paths["stem"])
    paths["meta"].write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    return paths


def load_index(index_dir: Path, chunkset_id: str, embedder_key: str) -> Tuple[BaseIndex, IndexMeta]:
    paths = index_paths(index_dir, chunkset_id, embedder_key)
    if not paths["meta"].exists():
        raise FileNotFoundError(f"no index metadata at {paths['meta']}")
    meta = IndexMeta(**json.loads(paths["meta"].read_text(encoding="utf-8")))
    if meta.backend == "faiss":
        index = FaissFlatIP.load(paths["stem"])
    else:
        index = NumpyFlatIP.load(paths["stem"])
    if index.n_vectors != meta.n_vectors:
        raise ValueError(
            f"index holds {index.n_vectors} vectors, metadata says {meta.n_vectors}"
        )
    return index, meta
