#!/usr/bin/env python
"""
Step 4: tune every strategy's free parameter on dev, then freeze it.

    python scripts/04_tune.py --dry-run     # what it will evaluate, and how much
    python scripts/04_tune.py               # the sweep
    set CRAG_OFFLINE=1 && python scripts/04_tune.py    # rehearse offline

Why this script exists: the semantic chunker has a free parameter that a
fixed-size chunker does not, and comparing a tuned method against an untuned
baseline is the standard way to manufacture a win. So every parameter that is
genuinely free gets swept here, on the 200-question dev split, and frozen
before anything touches test.

    fixed_128 / fixed_256   no free parameter -- size IS the condition, and the
                            design realises it as two named levels rather than
                            tuning it away. Empty grid by construction.
    recursive_256           min_fill_ratio: how full a chunk must be before it
                            may close at a paragraph break. 1.0 == the standard
                            greedy splitter, and it is in the grid.
    semantic                percentile: the breakpoint threshold.

Objective is answer_recall, not EM: tuning on retrieval keeps the
retrieval-to-answer analysis uncontaminated by the generator.

Scope of a tuned value:
    semantic        tuned per embedder -- its chunk set is already
                    embedder-specific, so there is nothing to share.
    recursive_256   one value shared across embedders (the score is averaged
                    over them), so the chunk set stays identical across the
                    embedder factor and the moderation contrast compares like
                    with like. --per-embedder overrides this.

Nothing is written to data/chunks or data/index: indexes here are built in
memory and thrown away. Only best_params.json and the log survive.
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
from crag.corpus import PageRecord, QuestionRecord, read_jsonl
from crag.embed import get_embedder
from crag.index import build_index
from crag.metrics import answer_recall, page_hit
from crag.retrieve import Candidate, fill_to_budget
from crag.tokenization import get_tokenizer

PAGES_PATH = config.CORPUS_DIR / "pages.jsonl"
QUESTIONS_PATH = config.CORPUS_DIR / "questions.jsonl"
BEST_PATH = config.TUNING_DIR / "best_params.json"
LOG_PATH = config.TUNING_DIR / "tuning_log.json"


def progress(iterable, total=None, desc=""):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, unit="it")
    except ImportError:
        return iterable


def load_pages(limit=0, seed=config.SEED, keep_ids=None):
    """The corpus, optionally subsampled for a faster sweep.

    A subsample always keeps every dev gold page: dropping one would make its
    question unanswerable for every configuration, which adds noise to the
    ranking without making the comparison any fairer. The rest of the budget
    goes to random distractors.
    """
    pages = [PageRecord(**{k: r[k] for k in ("doc_id", "title", "text")})
             for r in read_jsonl(PAGES_PATH)]
    if not limit or limit >= len(pages):
        return pages

    keep_ids = set(keep_ids or ())
    gold = [p for p in pages if p.doc_id in keep_ids]
    others = [p for p in pages if p.doc_id not in keep_ids]
    room = max(0, limit - len(gold))
    if room and others:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(others), size=min(room, len(others)), replace=False)
        others = [others[i] for i in sorted(idx.tolist())]
    else:
        others = []
    kept = gold + others
    print(f"  subsampled: {len(gold)} dev gold pages + {len(others)} distractors "
          f"= {len(kept)} of {len(pages)}")
    return kept


def load_dev():
    rows = [r for r in read_jsonl(QUESTIONS_PATH) if r.get("split") == "dev"]
    rows.sort(key=lambda r: str(r["question_id"]))
    return [QuestionRecord(**r) for r in rows]


def chunk_corpus(chunker, pages, budget_tok, desc=""):
    rows = []
    for page in progress(pages, total=len(pages), desc=desc):
        for chunk in chunker.chunk(page.doc_id, page.text):
            rows.append({
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "n_tokens": chunk.n_tokens,
                "n_budget_tokens": budget_tok.count(chunk.text),
            })
    return rows


def score_chunks(rows, embedder_key, questions, budget_tok, sep_tokens, args):
    """Retrieval-only evaluation of one chunk set under one embedder."""
    embedder = get_embedder(config.EMBEDDERS[embedder_key], device=args.device)
    texts = [r["text"] for r in rows]
    vectors = np.zeros((len(texts), embedder.dim), dtype=np.float32)
    step = config.EMBEDDERS[embedder_key].batch_size
    for start in progress(range(0, len(texts), step),
                          total=(len(texts) + step - 1) // step,
                          desc=f"embed[{embedder_key}]"):
        vectors[start:start + step] = embedder.encode_passages(texts[start:start + step])

    index = build_index(vectors, prefer_faiss=not args.no_faiss)
    qvecs = embedder.encode_queries([q.question for q in questions])
    scores, ids = index.search(qvecs, config.N_CANDIDATES)

    per_budget = {}
    for budget in config.BUDGETS:
        recalls, hits, used = [], [], []
        for qi, q in enumerate(questions):
            cands = [
                Candidate(chunk_id=str(int(cid)), doc_id=rows[int(cid)]["doc_id"],
                          text=rows[int(cid)]["text"], score=float(scores[qi][rank]),
                          n_budget_tokens=int(rows[int(cid)]["n_budget_tokens"]), rank=rank)
                for rank, cid in enumerate(ids[qi])
            ]
            sel = fill_to_budget(cands, budget, separator=config.CONTEXT_SEPARATOR,
                                 separator_tokens=sep_tokens, policy=config.BUDGET_POLICY)
            recalls.append(answer_recall(sel.context, q.answers))
            hits.append(page_hit(sel.doc_ids, q.gold_page_ids))
            used.append(sel.n_tokens_used)
        per_budget[budget] = {
            "answer_recall": round(float(np.mean(recalls)), 4),
            "page_hit": round(float(np.mean(hits)), 4),
            "tokens_used_mean": round(float(np.mean(used)), 1),
        }
    per_budget["mean_answer_recall"] = round(
        float(np.mean([per_budget[b]["answer_recall"] for b in config.BUDGETS])), 4)
    per_budget["n_chunks"] = len(rows)
    return per_budget


def plan(args):
    jobs = []
    for key, spec in config.CHUNKERS.items():
        if not spec.param_grid:
            continue
        for param, values in spec.param_grid.items():
            for value in values:
                if spec.needs_embedder or args.per_embedder:
                    for emb in config.EMBEDDERS:
                        jobs.append((key, param, value, [emb]))
                else:
                    jobs.append((key, param, value, list(config.EMBEDDERS)))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--per-embedder", action="store_true",
                    help="tune embedder-independent chunkers separately per embedder too")
    ap.add_argument("--tune-corpus-pages", type=int, default=0,
                    help="APPROXIMATION: sweep against a random subset of the corpus to save "
                         "hours. The frozen value is then applied to the full corpus. Say so "
                         "on the poster if you use it.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-faiss", action="store_true")
    ap.add_argument("--restart", action="store_true",
                    help="discard tuning_log.json and evaluate every configuration again")
    args = ap.parse_args()

    config.ensure_dirs()
    jobs = plan(args)
    print(f"{len(jobs)} configurations to evaluate on dev:")
    for key, param, value, embs in jobs:
        print(f"  {key:<14} {param}={value:<6} embedders={','.join(embs)}")
    if args.dry_run:
        print("\n(dry run: each configuration chunks the corpus once and embeds it once "
              "per embedder listed)")
        return

    questions = load_dev()
    dev_gold = {p for q in questions for p in q.gold_page_ids}
    pages = load_pages(args.tune_corpus_pages, keep_ids=dev_gold)
    if not questions:
        raise SystemExit("no dev questions -- check splits in questions.jsonl")
    print(f"\ncorpus: {len(pages)} pages | dev: {len(questions)} questions")
    if args.tune_corpus_pages:
        print("  ! tuning against a corpus SUBSET -- an approximation, report it")

    budget_tok = get_tokenizer(config.GENERATOR_MODEL)
    sep_tokens = budget_tok.count(config.CONTEXT_SEPARATOR)

    # Resumable: the sweep is hours long on a laptop GPU, and losing it to a
    # closed lid is not acceptable. Every configuration's result is appended to
    # tuning_log.json as soon as it finishes, and a restart skips what is
    # already there.
    log = []
    if LOG_PATH.exists() and not args.restart:
        try:
            log = json.loads(LOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log = []
    done = {(e["chunker"], e["param"], str(e["value"]), e["target"]) for e in log}
    if done:
        print(f"resuming: {len(done)} configuration(s) already evaluated "
              f"(--restart to redo them)")

    results = {}
    chunk_cache = {}
    # one distance cache per embedder, shared across that embedder's percentile
    # sweep: the sentence embeddings do not depend on the percentile, so this
    # turns four passes over a million sentences into one.
    dist_caches = {}
    def remember(entry):
        best = results.get(entry["target"])
        if best is None or entry["objective_answer_recall"] > best["objective_answer_recall"]:
            results[entry["target"]] = {
                "param": entry["param"], "value": entry["value"],
                "objective_answer_recall": entry["objective_answer_recall"],
            }

    for entry in log:                       # seed the argmax from resumed work
        remember(entry)

    for key, param, value, embs in jobs:
        spec = config.CHUNKERS[key]
        target = f"{key}__{embs[0]}" if (spec.needs_embedder or args.per_embedder) else key
        if (key, param, str(value), target) in done:
            print(f"\n[{key}] {param}={value} -- already evaluated, skipping")
            continue
        t0 = time.time()
        print(f"\n[{key}] {param}={value}")

        scores_by_emb = {}
        for emb in embs:
            cache_key = (key, param, value, emb if spec.needs_embedder else "_shared")
            if cache_key not in chunk_cache:
                embed_fn = None
                if spec.needs_embedder:
                    embed_fn = get_embedder(config.EMBEDDERS[emb], device=args.device).chunker_embed_fn()
                chunker = build_from_spec(spec, get_tokenizer(config.CHUNK_TOKENIZER),
                                          embed_fn=embed_fn, overrides={param: value},
                                          global_max_tokens=config.MAX_CHUNK_TOKENS,
                                          cache=dist_caches.setdefault(emb, {}))
                chunk_cache[cache_key] = chunk_corpus(chunker, pages, budget_tok, desc=f"chunk[{key}]")
            rows = chunk_cache[cache_key]
            scores_by_emb[emb] = score_chunks(rows, emb, questions, budget_tok, sep_tokens, args)
            print(f"  {emb}: recall={scores_by_emb[emb]['mean_answer_recall']} "
                  f"({scores_by_emb[emb]['n_chunks']} chunks)")

        objective = float(np.mean([s["mean_answer_recall"] for s in scores_by_emb.values()]))
        entry = {
            "target": target, "chunker": key, "param": param, "value": value,
            "objective_answer_recall": round(objective, 4),
            "per_embedder": scores_by_emb, "seconds": round(time.time() - t0, 1),
        }
        log.append(entry)
        remember(entry)
        # checkpoint after every configuration, not at the end
        LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
        BEST_PATH.write_text(
            json.dumps({t: {v["param"]: v["value"]} for t, v in results.items()}, indent=2),
            encoding="utf-8")

    BEST_PATH.write_text(
        json.dumps({t: {v["param"]: v["value"]} for t, v in results.items()}, indent=2),
        encoding="utf-8")
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("\nfrozen parameters:")
    for target, chosen in sorted(results.items()):
        print(f"  {target}: {chosen['param']}={chosen['value']} "
              f"(dev answer_recall {chosen['objective_answer_recall']})")
    print(f"\n-> {BEST_PATH}\n-> {LOG_PATH}")
    print("next: python scripts/02_chunk_and_index.py")
    print("      (it picks these up automatically and only rebuilds what changed)")


if __name__ == "__main__":
    main()
