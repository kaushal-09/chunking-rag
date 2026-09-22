#!/usr/bin/env python
"""
Step 2 of the pipeline: chunk the corpus, then embed and index each chunk set.

    # everything the run matrix needs (5 chunk sets, 8 indexes)
    python scripts/02_chunk_and_index.py

    # offline dry run on the synthetic corpus -- no model, no GPU
    set CRAG_OFFLINE=1 && python scripts/02_chunk_and_index.py

    # rebuild one condition after tuning
    python scripts/02_chunk_and_index.py --chunkers semantic --force ^
        --params data/tuning/best_params.json

Five chunk sets, not eight: only the semantic chunker's boundaries depend on
the embedder (locked decision -- it draws them with the same model that
retrieves), so fixed_128 / fixed_256 / recursive_256 are chunked once and
embedded twice.

Chunk lengths are measured with the chunk tokenizer; `n_budget_tokens` is
measured with the GENERATOR's tokenizer and stored per chunk, so fill-to-budget
at query time is exact and costs no tokenisation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config
from crag.chunkers import build_from_spec
from crag.corpus import PageRecord, read_jsonl, write_jsonl
from crag.embed import get_embedder
from crag.index import IndexMeta, build_index, faiss_available, index_paths, save_index
from crag.tokenization import get_tokenizer

PAGES_PATH = config.CORPUS_DIR / "pages.jsonl"
CHUNK_STATS_PATH = config.CHUNKS_DIR / "chunk_stats.json"
INDEX_STATS_PATH = config.INDEX_DIR / "index_stats.json"


def chunks_path(chunkset_id: str) -> Path:
    return config.CHUNKS_DIR / f"{chunkset_id}.jsonl"


def required_chunksets(runs):
    """chunkset_id -> (chunker_key, embedder_key or None)."""
    out = {}
    for r in runs:
        spec = config.CHUNKERS[r.chunker]
        out[r.chunkset_id] = (r.chunker, r.embedder if spec.needs_embedder else None)
    return out


def required_indexes(runs):
    return sorted({(r.chunkset_id, r.embedder) for r in runs})


def load_tuned(params_path, chunkset_id, chunker_key):
    if not params_path:
        return {}
    blob = json.loads(Path(params_path).read_text(encoding="utf-8"))
    return blob.get(chunkset_id, blob.get(chunker_key, {})) or {}


def recorded_params(chunkset_id):
    """What this chunk set was last built with, if it exists."""
    if not CHUNK_STATS_PATH.exists():
        return None
    try:
        for entry in json.loads(CHUNK_STATS_PATH.read_text(encoding="utf-8")):
            if entry.get("chunkset_id") == chunkset_id:
                return entry.get("params")
    except (json.JSONDecodeError, TypeError):
        return None
    return None


def comparable(params):
    return {k: v for k, v in params.items() if k != "separators"}


def progress(iterable, total=None, desc=""):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, unit="it")
    except ImportError:
        return iterable


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------

def build_chunkset(chunkset_id, chunker_key, embedder_key, pages, args):
    out_path = chunks_path(chunkset_id)
    spec = config.CHUNKERS[chunker_key]
    overrides = load_tuned(args.params, chunkset_id, chunker_key)
    chunk_tok = get_tokenizer(config.CHUNK_TOKENIZER)
    budget_tok = get_tokenizer(args.budget_tokenizer or config.GENERATOR_MODEL)

    # lazy: constructing the chunker must not load a 440 MB embedder just so we
    # can decide to skip it
    embed_fn = None
    if spec.needs_embedder:
        def embed_fn(texts, _key=embedder_key):
            return get_embedder(config.EMBEDDERS[_key], device=args.device).encode_passages(texts)

    chunker = build_from_spec(
        spec, chunk_tok, embed_fn=embed_fn, overrides=overrides,
        global_max_tokens=config.MAX_CHUNK_TOKENS,
    )

    if out_path.exists() and not args.force:
        previous = recorded_params(chunkset_id)
        if previous is not None and comparable(previous) == comparable(chunker.params):
            print(f"  {chunkset_id}: exists with identical parameters, skipping")
            return None
        if previous is None:
            print(f"  {chunkset_id}: exists but was built by an older version, rebuilding")
        else:
            changed = {k: (previous.get(k), v) for k, v in comparable(chunker.params).items()
                       if previous.get(k) != v}
            print(f"  {chunkset_id}: parameters changed {changed}, rebuilding")

    print(f"  {chunkset_id}: {chunker.__class__.__name__} {chunker.params}")
    if overrides:
        print(f"    tuned overrides: {overrides}")

    t0 = time.time()
    rows, n_tokens, n_budget = [], [], []
    for page in progress(pages, total=len(pages), desc=chunkset_id):
        for chunk in chunker.chunk(page.doc_id, page.text):
            nb = budget_tok.count(chunk.text)
            rows.append({
                "chunk_id": f"{page.doc_id}:{chunk.char_start}-{chunk.char_end}",
                "doc_id": chunk.doc_id,
                "title": page.title,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "n_tokens": chunk.n_tokens,
                "n_budget_tokens": nb,
                "text": chunk.text,
            })
            n_tokens.append(chunk.n_tokens)
            n_budget.append(nb)

    write_jsonl(out_path, rows)
    arr, barr = np.array(n_tokens), np.array(n_budget)
    stats = {
        "chunkset_id": chunkset_id,
        "chunker": chunker_key,
        "embedder_for_boundaries": embedder_key,
        "params": {k: v for k, v in chunker.params.items() if k != "separators"},
        "n_pages": len(pages),
        "n_chunks": len(rows),
        "chunks_per_page": round(len(rows) / max(1, len(pages)), 2),
        "tokens_mean": round(float(arr.mean()), 1) if len(arr) else 0,
        "tokens_median": float(np.median(arr)) if len(arr) else 0,
        "tokens_p10": float(np.percentile(arr, 10)) if len(arr) else 0,
        "tokens_p90": float(np.percentile(arr, 90)) if len(arr) else 0,
        "tokens_max": int(arr.max()) if len(arr) else 0,
        "budget_tokens_mean": round(float(barr.mean()), 1) if len(barr) else 0,
        "seconds": round(time.time() - t0, 1),
    }
    own_cap = chunker.params.get("max_tokens", config.MAX_CHUNK_TOKENS)
    if stats["tokens_max"] > min(own_cap, config.MAX_CHUNK_TOKENS):
        raise AssertionError(
            f"{chunkset_id} produced a {stats['tokens_max']}-token chunk with a "
            f"{min(own_cap, config.MAX_CHUNK_TOKENS)}-token cap"
        )
    print(f"    {stats['n_chunks']} chunks, mean {stats['tokens_mean']} tok, "
          f"max {stats['tokens_max']}, {stats['seconds']}s")
    return stats


# ---------------------------------------------------------------------------
# indexing
# ---------------------------------------------------------------------------

def build_one_index(chunkset_id, embedder_key, args):
    paths = index_paths(config.INDEX_DIR, chunkset_id, embedder_key)
    src = chunks_path(chunkset_id)
    if not src.exists():
        raise FileNotFoundError(f"missing chunks for {chunkset_id}; run without --only-index")
    if paths["meta"].exists() and not args.force:
        if paths["meta"].stat().st_mtime >= src.stat().st_mtime:
            print(f"  {chunkset_id} x {embedder_key}: up to date, skipping")
            return None
        print(f"  {chunkset_id} x {embedder_key}: chunks are newer than the index, rebuilding")

    rows = list(read_jsonl(src))
    texts = [r["text"] for r in rows]
    spec = config.EMBEDDERS[embedder_key]
    embedder = get_embedder(spec, device=args.device)

    t0 = time.time()
    vectors = np.zeros((len(texts), embedder.dim), dtype=np.float32)
    step = args.encode_batch or spec.batch_size
    for start in progress(range(0, len(texts), step),
                          total=(len(texts) + step - 1) // step,
                          desc=f"{chunkset_id}x{embedder_key}"):
        vectors[start:start + step] = embedder.encode_passages(texts[start:start + step])

    index = build_index(vectors, prefer_faiss=not args.no_faiss)
    meta = IndexMeta(
        chunkset_id=chunkset_id,
        embedder_key=embedder_key,
        model_name=embedder.model_name,
        dim=embedder.dim,
        n_vectors=index.n_vectors,
        backend=index.backend,
        chunks_path=str(src),
    )
    save_index(index, meta, config.INDEX_DIR, chunkset_id, embedder_key)
    elapsed = time.time() - t0
    print(f"    {index.n_vectors} vectors, dim {meta.dim}, {index.backend}, "
          f"{elapsed:.1f}s ({len(texts)/max(elapsed,1e-9):.0f} chunks/s)")
    return {**meta.to_dict(), "seconds": round(elapsed, 1)}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunkers", nargs="*", default=None, choices=list(config.CHUNKERS))
    ap.add_argument("--embedders", nargs="*", default=None, choices=list(config.EMBEDDERS))
    ap.add_argument("--only-chunk", action="store_true")
    ap.add_argument("--only-index", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    ap.add_argument("--params", default=None, help="tuned parameters json (from 04_tune.py)")
    ap.add_argument("--limit-pages", type=int, default=0, help="smoke test on N pages")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--encode-batch", type=int, default=0)
    ap.add_argument("--budget-tokenizer", default=None,
                    help="tokenizer used for n_budget_tokens (default: the generator's)")
    ap.add_argument("--no-faiss", action="store_true", help="force the numpy index")
    args = ap.parse_args()

    config.ensure_dirs()

    default_params = config.TUNING_DIR / "best_params.json"
    if args.params is None and default_params.exists():
        try:
            blob = json.loads(default_params.read_text(encoding="utf-8"))
            if not isinstance(blob, dict):
                raise ValueError("not a mapping")
            args.params = str(default_params)
            print(f"using tuned parameters from {default_params}")
            print(f"  {blob if blob else '(empty - tuning has not run yet)'}")
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"! ignoring unreadable {default_params} ({exc}); using defaults")

    runs = [
        r for r in config.run_matrix()
        if (not args.chunkers or r.chunker in args.chunkers)
        and (not args.embedders or r.embedder in args.embedders)
    ]
    if not runs:
        raise SystemExit("no runs selected")

    pages = [PageRecord(**{k: row[k] for k in ("doc_id", "title", "text")}) for row in read_jsonl(PAGES_PATH)]
    if not pages:
        raise SystemExit(f"no pages at {PAGES_PATH} -- run scripts/01_build_corpus.py first")
    if args.limit_pages:
        pages = pages[:args.limit_pages]
    print(f"corpus: {len(pages)} pages")
    print(f"faiss available: {faiss_available()}")

    chunk_stats = []
    if not args.only_index:
        print("\nchunking")
        for chunkset_id, (chunker_key, embedder_key) in sorted(required_chunksets(runs).items()):
            s = build_chunkset(chunkset_id, chunker_key, embedder_key, pages, args)
            if s:
                chunk_stats.append(s)
        if chunk_stats:
            existing = json.loads(CHUNK_STATS_PATH.read_text()) if CHUNK_STATS_PATH.exists() else []
            keep = [s for s in existing if s["chunkset_id"] not in {x["chunkset_id"] for x in chunk_stats}]
            CHUNK_STATS_PATH.write_text(json.dumps(keep + chunk_stats, indent=2), encoding="utf-8")

    index_stats = []
    if not args.only_chunk:
        print("\nindexing")
        for chunkset_id, embedder_key in required_indexes(runs):
            s = build_one_index(chunkset_id, embedder_key, args)
            if s:
                index_stats.append(s)
        if index_stats:
            existing = json.loads(INDEX_STATS_PATH.read_text()) if INDEX_STATS_PATH.exists() else []
            done = {(x["chunkset_id"], x["embedder_key"]) for x in index_stats}
            keep = [x for x in existing if (x["chunkset_id"], x["embedder_key"]) not in done]
            INDEX_STATS_PATH.write_text(json.dumps(keep + index_stats, indent=2), encoding="utf-8")

    print("\ndone.")
    if chunk_stats:
        print(f"  chunk stats  -> {CHUNK_STATS_PATH}")
    if index_stats:
        print(f"  index stats  -> {INDEX_STATS_PATH}")


if __name__ == "__main__":
    main()
