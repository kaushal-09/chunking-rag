#!/usr/bin/env python
"""
Step 1 of the pipeline: build questions.jsonl + pages.jsonl.

    # fetch the questions first and see how many pages the scan has to find
    python scripts/01_build_corpus.py --questions-only

    # the full build (questions + the kilt_wikipedia scan; long, resumable)
    python scripts/01_build_corpus.py

    # resume an interrupted scan -- appends, never restarts
    python scripts/01_build_corpus.py --resume

    # a 60-page stand-in so steps 2-5 can be built and run offline today
    python scripts/01_build_corpus.py --synthetic

The wikipedia scan streams the whole knowledge source to pick out ~3k pages.
Run --questions-only first: it costs a minute and tells you exactly what the
long job is for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from crag.corpus import (
    ScanState,
    assign_splits,
    count_jsonl,
    head_content_length,
    iter_knowledge_source,
    iter_wikipedia,
    load_questions,
    load_scan_state,
    read_jsonl,
    save_scan_state,
    scan_for_pages,
    synthetic_corpus,
    write_jsonl,
)

QUESTIONS_PATH = config.CORPUS_DIR / "questions.jsonl"
PAGES_PATH = config.CORPUS_DIR / "pages.jsonl"
SPLITS_PATH = config.CORPUS_DIR / "splits.json"
STATE_PATH = config.CORPUS_DIR / "scan_state.json"
STATS_PATH = config.CORPUS_DIR / "corpus_stats.json"


def build_questions(cache_dir=None):
    print(f"loading {config.KILT_TASKS_DATASET}/{config.KILT_TASKS_CONFIG} [{config.KILT_SPLIT}]")
    questions = load_questions(
        config.KILT_TASKS_DATASET, config.KILT_TASKS_CONFIG, config.KILT_SPLIT, cache_dir
    )
    print(f"  {len(questions)} questions (spec expects {config.EXPECTED_N_QUESTIONS})")
    if len(questions) != config.EXPECTED_N_QUESTIONS:
        print("  ! count differs from the locked spec -- check the split before continuing")

    splits = config.make_splits([q.question_id for q in questions])
    questions = assign_splits(questions, splits)

    write_jsonl(QUESTIONS_PATH, (q.to_dict() for q in questions))
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    gold = sorted({p for q in questions for p in q.gold_page_ids})
    no_prov = [q.question_id for q in questions if not q.gold_page_ids]
    no_ans = [q.question_id for q in questions if not q.answers]
    print(f"  dev={len(splits['dev'])} test={len(splits['test'])} unused={len(splits['unused'])}")
    print(f"  {len(gold)} unique gold pages to find")
    if no_prov:
        print(f"  ! {len(no_prov)} questions have no provenance (page_hit undefined for them)")
    if no_ans:
        print(f"  ! {len(no_ans)} questions have no answer string")
    return questions, gold


def run_scan(gold_ids, args):
    state = (load_scan_state(STATE_PATH, set(gold_ids), PAGES_PATH) if args.resume
             else ScanState(needed=set(gold_ids)))
    if not args.resume and PAGES_PATH.exists():
        PAGES_PATH.unlink()

    source = args.source or config.KILT_KNOWLEDGE_SOURCE_URL
    print(f"scanning the knowledge source: need {len(state.missing)} more pages")
    print(f"  source: {source}")
    if state.bytes_consumed:
        print(f"  resuming at byte {state.bytes_consumed:,}"
              + (f" of {state.total_bytes:,}" if state.total_bytes else ""))
    print("  checkpointed at line boundaries - Ctrl-C is safe, re-run with --resume")

    if args.via_datasets:
        records = iter_wikipedia(config.KILT_WIKIPEDIA_DATASET, streaming=True, cache_dir=args.cache_dir)
        bar = None
    else:
        records = iter_knowledge_source(source, state, max_bytes=args.limit_bytes)
        bar = None
        try:
            import itertools

            from tqdm import tqdm

            first = next(records)                       # forces headers, fills total_bytes
            bar = tqdm(total=state.total_bytes or None, initial=state.bytes_consumed,
                       unit="B", unit_scale=True, unit_divisor=1024, desc="knowledge source")
            records = _track(itertools.chain([first], records), state, bar)
        except ImportError:
            pass
        except StopIteration:
            records = iter([])

    if args.limit:
        import itertools
        records = itertools.islice(records, args.limit)

    state = scan_for_pages(
        records, state, PAGES_PATH,
        checkpoint_every=args.checkpoint_every,
        state_path=STATE_PATH,
        drop_title_paragraph=not args.keep_title_paragraph,
    )
    if bar is not None:
        bar.close()
    return state


def _track(records, state, bar):
    """Advance the byte-progress bar as the stream reports line boundaries."""
    last = state.bytes_consumed
    for record in records:
        if state.bytes_consumed != last:
            bar.update(state.bytes_consumed - last)
            last = state.bytes_consumed
        yield record


def write_stats(questions, found_ids):
    found = set(found_ids)
    usable = [q for q in questions if set(q.gold_page_ids) & found]
    per_split = {}
    for split in ("dev", "test"):
        in_split = [q for q in questions if q.split == split]
        per_split[split] = {
            "n": len(in_split),
            "n_with_a_gold_page_in_corpus": sum(1 for q in in_split if set(q.gold_page_ids) & found),
        }
    stats = {
        "n_questions": len(questions),
        "n_gold_pages_needed": len({p for q in questions for p in q.gold_page_ids}),
        "n_pages_in_corpus": len(found),
        "n_questions_with_a_gold_page_in_corpus": len(usable),
        "per_split": per_split,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    if stats["n_questions_with_a_gold_page_in_corpus"] < stats["n_questions"]:
        print("  ! some questions have no gold page in the corpus; page_hit is 0 for those by")
        print("    construction. Report the count on the poster rather than dropping them silently.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions-only", action="store_true", help="stop after questions.jsonl")
    ap.add_argument("--synthetic", action="store_true", help="tiny offline stand-in corpus")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted wikipedia scan")
    ap.add_argument("--limit", type=int, default=0, help="stop after N wikipedia records (smoke test)")
    ap.add_argument("--limit-bytes", type=int, default=0, help="stop after N bytes (smoke test)")
    ap.add_argument("--source", default=None,
                    help="knowledge source: a URL, or a path to an already-downloaded copy "
                         f"(default: {config.KILT_KNOWLEDGE_SOURCE_URL})")
    ap.add_argument("--via-datasets", action="store_true",
                    help="legacy: load the knowledge source through `datasets` instead. "
                         "Fails on datasets>=4 - the repo is a loading script with no parquet.")
    ap.add_argument("--head", action="store_true",
                    help="just report the size of the knowledge source and exit")
    ap.add_argument("--checkpoint-every", type=int, default=50_000)
    ap.add_argument("--keep-title-paragraph", action="store_true",
                    help="keep KILT's title line as page text (default: drop it)")
    ap.add_argument("--cache-dir", default=None, help="HuggingFace cache dir (use a big drive)")
    args = ap.parse_args()

    config.ensure_dirs()

    if args.head:
        info = head_content_length(args.source or config.KILT_KNOWLEDGE_SOURCE_URL)
        gb = info["content_length"] / 1024 ** 3
        print(f"url             {info['url']}")
        print(f"content-length  {info['content_length']:,} bytes ({gb:.1f} GiB)")
        print(f"accept-ranges   {info['accept_ranges']}"
              + ("  (resume supported)" if info["accept_ranges"] == "bytes" else
                 "  (no resume - download to a file first and use --source)"))
        return

    if args.synthetic:
        pages, questions = synthetic_corpus()
        splits = {"dev": [], "test": [q.question_id for q in questions], "unused": []}
        write_jsonl(QUESTIONS_PATH, (q.to_dict() for q in questions))
        write_jsonl(PAGES_PATH, (p.to_dict() for p in pages))
        SPLITS_PATH.write_text(json.dumps(splits, indent=2), encoding="utf-8")
        print(f"synthetic corpus: {len(pages)} pages, {len(questions)} questions -> {config.CORPUS_DIR}")
        write_stats(questions, [p.doc_id for p in pages])
        return

    if args.resume and QUESTIONS_PATH.exists():
        from crag.corpus import QuestionRecord
        questions = [QuestionRecord(**row) for row in read_jsonl(QUESTIONS_PATH)]
        gold = sorted({p for q in questions for p in q.gold_page_ids})
        print(f"resuming with {len(questions)} questions, {len(gold)} gold pages")
    else:
        questions, gold = build_questions(args.cache_dir)

    if args.questions_only:
        print(f"\nwrote {QUESTIONS_PATH} and {SPLITS_PATH}")
        print(f"next: python scripts/01_build_corpus.py   # scans for {len(gold)} pages")
        return

    state = run_scan(gold, args)
    print(f"\nseen {state.n_seen} wikipedia records, found {len(state.found)}/{len(state.needed)} pages")
    if state.missing:
        print(f"  {len(state.missing)} still missing -- rerun with --resume")
    print(f"pages.jsonl now holds {count_jsonl(PAGES_PATH)} pages")
    write_stats(questions, state.found)


if __name__ == "__main__":
    main()
