#!/usr/bin/env python
"""
Step 3 of the pipeline: retrieve, fill the budget, generate, score.

    # the reported runs
    python scripts/03_run_pipeline.py --split test

    # retrieval only -- no generator loaded, no GPU time on decoding
    python scripts/03_run_pipeline.py --split dev --no-generate

    # offline end-to-end rehearsal on the synthetic corpus
    set CRAG_OFFLINE=1 && python scripts/03_run_pipeline.py --split test

One run = one (chunker, embedder, budget) cell; 16 in all. The two budgets in
a cell share their retrieval: the FAISS candidate list does not depend on the
budget, only the fill does, so each (chunkset, embedder) pair is searched once
and both budgets are filled from it.

Writes data/runs/<run_id>.jsonl (one record per question, everything needed for
the analysis and for a failure taxonomy) and <run_id>.summary.json.
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
from crag.corpus import QuestionRecord, read_jsonl, write_jsonl
from crag.embed import get_embedder
from crag.generate import clean_answer, get_generator
from crag.index import load_index
from crag.metrics import score_query
from crag.retrieve import Candidate, budget_utilisation, fill_to_budget
from crag.tokenization import get_tokenizer

QUESTIONS_PATH = config.CORPUS_DIR / "questions.jsonl"


def progress(iterable, total=None, desc=""):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, unit="it")
    except ImportError:
        return iterable


def load_split(split, limit=0):
    rows = [r for r in read_jsonl(QUESTIONS_PATH) if split == "all" or r.get("split") == split]
    rows.sort(key=lambda r: str(r["question_id"]))      # deterministic order
    if limit:
        rows = rows[:limit]
    return [QuestionRecord(**r) for r in rows]


def encode_queries(embedder, questions, batch_size):
    vecs = np.zeros((len(questions), embedder.dim), dtype=np.float32)
    texts = [q.question for q in questions]
    for start in range(0, len(texts), batch_size):
        vecs[start:start + batch_size] = embedder.encode_queries(texts[start:start + batch_size])
    return vecs


def summarise(run, records, split, retrieval_seconds, gen_stats, chunkset_id):
    def mean(key):
        vals = [r[key] for r in records if key in r]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "run_id": run.run_id,
        "chunker": run.chunker,
        "chunkset_id": chunkset_id,
        "embedder": run.embedder,
        "budget": run.budget,
        "split": split,
        "n_questions": len(records),
        "generated": any("em" in r for r in records),
        "answer_recall": mean("answer_recall"),
        "page_hit": mean("page_hit"),
        "page_precision": mean("page_precision"),
        "em": mean("em"),
        "f1": mean("f1"),
        "abstention_rate": mean("abstained"),
        "context_ok_answer_wrong": mean("context_ok_answer_wrong"),
        "n_chunks_used_mean": mean("n_chunks_used"),
        "n_tokens_used_mean": mean("n_tokens_used"),
        "n_tokens_actual_mean": mean("n_tokens_actual"),
        "budget_utilisation_mean": mean("budget_utilisation"),
        "retrieval_seconds": round(retrieval_seconds, 1),
        "generation": gen_stats,
        "design": {
            "n_candidates": config.N_CANDIDATES,
            "budget_policy": config.BUDGET_POLICY,
            "max_chunk_tokens": config.MAX_CHUNK_TOKENS,
            "seed": config.SEED,
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "dev", "all"])
    ap.add_argument("--chunkers", nargs="*", default=None, choices=list(config.CHUNKERS))
    ap.add_argument("--embedders", nargs="*", default=None, choices=list(config.EMBEDDERS))
    ap.add_argument("--budgets", nargs="*", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N questions of the split")
    ap.add_argument("--no-generate", action="store_true", help="retrieval metrics only")
    ap.add_argument("--generate-budgets", nargs="*", type=int, default=None,
                    help="override config.GENERATION_BUDGETS "
                         f"(currently {list(config.GENERATION_BUDGETS)})")
    ap.add_argument("--save-context", action="store_true", help="store the assembled context per query")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gen-batch", type=int, default=config.GEN_BATCH_SIZE)
    ap.add_argument("--query-batch", type=int, default=128)
    ap.add_argument("--out-dir", default=None, help="override data/runs (04_tune uses this)")
    args = ap.parse_args()

    config.ensure_dirs()
    out_dir = Path(args.out_dir) if args.out_dir else config.RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        r for r in config.run_matrix()
        if (not args.chunkers or r.chunker in args.chunkers)
        and (not args.embedders or r.embedder in args.embedders)
        and (not args.budgets or r.budget in args.budgets)
    ]
    questions = load_split(args.split, args.limit)
    if not questions:
        raise SystemExit(f"no questions in split {args.split!r} -- run 01_build_corpus.py first")
    print(f"{len(runs)} runs x {len(questions)} questions ({args.split})")

    budget_tok = get_tokenizer(config.GENERATOR_MODEL)
    sep_tokens = budget_tok.count(config.CONTEXT_SEPARATOR)

    # Generation happens only at the budgets the design says to generate at;
    # every other budget still gets its retrieval metrics, which are free.
    gen_budgets = set() if args.no_generate else set(
        args.generate_budgets if args.generate_budgets else config.GENERATION_BUDGETS)
    needs_generator = any(r.budget in gen_budgets for r in runs)
    retrieval_only = sorted({r.budget for r in runs} - gen_budgets)
    if retrieval_only:
        print(f"retrieval-only at budget(s) {retrieval_only} -- no EM/F1 there by "
              f"design (config.GENERATION_BUDGETS)")

    generator = None
    if needs_generator:
        print(f"loading generator {config.GENERATOR_MODEL}")
        generator = get_generator(
            config.GENERATOR_MODEL, config.SYSTEM_PROMPT,
            max_new_tokens=config.GEN_MAX_NEW_TOKENS,
            load_in_4bit=config.GEN_LOAD_IN_4BIT,
            device=args.device, batch_size=args.gen_batch, dtype=config.GEN_DTYPE,
        )
        print(f"  {generator.describe()}")

    groups = {}
    for r in runs:
        groups.setdefault((r.chunkset_id, r.embedder), []).append(r)

    for (chunkset_id, embedder_key), group in sorted(groups.items()):
        pending = [r for r in group
                   if args.force or not (out_dir / f"{r.run_id}.summary.json").exists()]
        if not pending:
            print(f"\n{chunkset_id} x {embedder_key}: all budgets done, skipping")
            continue

        print(f"\n{chunkset_id} x {embedder_key}")
        index, meta = load_index(config.INDEX_DIR, chunkset_id, embedder_key)
        chunks = list(read_jsonl(Path(meta.chunks_path)))
        if len(chunks) != index.n_vectors:
            raise ValueError(
                f"{len(chunks)} chunks but {index.n_vectors} vectors -- rebuild the index"
            )

        embedder = get_embedder(config.EMBEDDERS[embedder_key], device=args.device)
        t0 = time.time()
        qvecs = encode_queries(embedder, questions, args.query_batch)
        scores, ids = index.search(qvecs, config.N_CANDIDATES)
        retrieval_seconds = time.time() - t0
        print(f"  retrieved top-{config.N_CANDIDATES} for {len(questions)} queries "
              f"in {retrieval_seconds:.1f}s")

        for run in sorted(pending, key=lambda r: r.budget):
            selections = []
            for qi in range(len(questions)):
                cands = [
                    Candidate(
                        chunk_id=chunks[int(cid)]["chunk_id"],
                        doc_id=chunks[int(cid)]["doc_id"],
                        text=chunks[int(cid)]["text"],
                        score=float(scores[qi][rank]),
                        n_budget_tokens=int(chunks[int(cid)]["n_budget_tokens"]),
                        rank=rank,
                    )
                    for rank, cid in enumerate(ids[qi])
                ]
                selections.append(fill_to_budget(
                    cands, run.budget,
                    separator=config.CONTEXT_SEPARATOR,
                    separator_tokens=sep_tokens,
                    policy=config.BUDGET_POLICY,
                    count_fn=budget_tok.count,
                ))

            predictions = [None] * len(questions)
            raw_predictions = [None] * len(questions)
            gen_stats = None
            if generator is not None and run.budget in gen_budgets:
                prompts = [
                    config.build_prompt(q.question, sel.context)
                    for q, sel in zip(questions, selections)
                ]
                raw = generator.generate(prompts, batch_size=args.gen_batch)
                raw_predictions = raw
                predictions = [clean_answer(r) for r in raw]
                gen_stats = getattr(generator, "last_stats", None)

            records = []
            for q, sel, pred, raw in zip(questions, selections, predictions, raw_predictions):
                rec = {
                    "question_id": q.question_id,
                    "question": q.question,
                    "answers": q.answers,
                    "gold_page_ids": q.gold_page_ids,
                    "split": q.split,
                    "budget": run.budget,
                    "budget_utilisation": round(budget_utilisation(sel), 4),
                    **sel.to_dict(),
                    "prediction_raw": raw,
                    "prediction": pred,
                    **score_query(pred, q.answers, sel.context, sel.doc_ids, q.gold_page_ids),
                }
                if args.save_context:
                    rec["context"] = sel.context
                records.append(rec)

            write_jsonl(out_dir / f"{run.run_id}.jsonl", records)
            summary = summarise(run, records, args.split, retrieval_seconds, gen_stats, chunkset_id)
            (out_dir / f"{run.run_id}.summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")

            print(f"  {run.run_id}: recall={summary['answer_recall']} "
                  f"page_hit={summary['page_hit']} em={summary['em']} f1={summary['f1']} "
                  f"| {summary['n_chunks_used_mean']} chunks, "
                  f"{summary['n_tokens_used_mean']}/{run.budget} tokens "
                  f"({summary['budget_utilisation_mean']})")
            if gen_stats:
                print(f"    generation: {gen_stats}")

    print(f"\nruns written to {out_dir}")


if __name__ == "__main__":
    main()
