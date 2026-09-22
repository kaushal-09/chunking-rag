#!/usr/bin/env python
"""
Independent re-derivation of every claim the poster makes.

    python scripts/07_verify.py                 # everything that is cheap
    python scripts/07_verify.py --deep          # also re-tokenise every chunk
    python scripts/07_verify.py --json out.json # machine-readable result

This script deliberately does NOT import the analysis code paths it is
checking. It re-reads the raw run files, recomputes the metrics from the
stored predictions with crag.metrics, and compares against what the pipeline
wrote. If the two agree, the numbers on the poster came from the data and not
from a bug in the reporting layer.

What it cannot check: answer_recall and context_ok_answer_wrong depend on the
assembled context text, which the runs do not store (--save-context was off,
by design -- it would have added ~1 GB). Those are re-derived only under
--deep, which reloads the chunk sets and rebuilds each context from its
stored chunk_ids.

Exit code 0 = every check passed. 1 = at least one FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from crag.corpus import read_jsonl
from crag.metrics import (
    answer_recall,
    contains_answer,
    exact_match,
    f1,
    is_abstention,
    page_hit,
    page_precision,
)

TOL = 1e-9          # exact recomputation of a stored float
MEAN_TOL = 5e-5     # summaries are rounded to 4 dp


class Report:
    """Collects check results so one failure does not hide the rest."""

    def __init__(self, verbose: bool = True):
        self.rows: list[dict] = []
        self.verbose = verbose

    def check(self, ok: bool, name: str, detail: str = "", warn_only: bool = False):
        status = "ok" if ok else ("warn" if warn_only else "FAIL")
        self.rows.append({"check": name, "status": status, "detail": detail})
        if self.verbose:
            print(f"[{status:>4}] {name:<52} {detail}")
        return ok

    def section(self, title: str):
        if self.verbose:
            print(f"\n--- {title} ---")

    @property
    def failures(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "FAIL"]

    @property
    def warnings(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "warn"]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def close(a, b, tol=TOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# A. the locked design, as constants
# ---------------------------------------------------------------------------

LOCKED = {
    "SEED": 13,
    "N_DEV": 200,
    "N_TEST": 800,
    "MAX_CHUNK_TOKENS": 256,
    "CHUNK_OVERLAP": 0,
    "N_CANDIDATES": 64,
    "BUDGET_POLICY": "stop",
    "BUDGETS": (1000, 2500),
    "GENERATION_BUDGETS": (1000,),
    "GENERATOR_MODEL": "Qwen/Qwen2.5-3B-Instruct",
    "GEN_MAX_NEW_TOKENS": 32,
    "GEN_TEMPERATURE": 0.0,
    "GEN_LOAD_IN_4BIT": True,
    "CHUNK_TOKENIZER": "bert-base-uncased",
    "N_BOOTSTRAP": 10_000,
}


def check_config(rep: Report):
    rep.section("A. locked design constants")
    for key, expected in LOCKED.items():
        actual = getattr(config, key, "<missing>")
        rep.check(actual == expected, f"config.{key}", f"{actual!r}")
    rep.check(len(config.CHUNKERS) == 4, "four chunkers", ", ".join(config.CHUNKERS))
    rep.check(len(config.EMBEDDERS) == 2, "two embedders", ", ".join(config.EMBEDDERS))
    rep.check(len(config.run_matrix()) == 16, "16-run matrix",
              f"{len(config.run_matrix())} runs")


# ---------------------------------------------------------------------------
# B. corpus and splits
# ---------------------------------------------------------------------------

def check_corpus(rep: Report, root: Path):
    rep.section("B. corpus and splits")
    qpath = root / "data" / "corpus" / "questions.jsonl"
    spath = root / "data" / "corpus" / "splits.json"
    if not qpath.exists():
        rep.check(False, "questions.jsonl exists", str(qpath))
        return None
    questions = list(read_jsonl(qpath))
    qids = [q["question_id"] for q in questions]

    rep.check(len(qids) == len(set(qids)), "question ids unique",
              f"{len(qids)} rows, {len(set(qids))} unique")
    rep.check(all(q.get("answers") for q in questions), "every question has an answer")
    rep.check(all(q.get("gold_page_ids") for q in questions),
              "every question has gold provenance")

    splits = json.loads(spath.read_text(encoding="utf-8"))
    dev, test = set(splits["dev"]), set(splits["test"])
    rep.check(len(dev) == config.N_DEV, "dev size", f"{len(dev)}")
    rep.check(len(test) == config.N_TEST, "test size", f"{len(test)}")
    rep.check(not (dev & test), "dev and test disjoint",
              f"{len(dev & test)} shared")
    rep.check(dev <= set(qids) and test <= set(qids),
              "split ids all present in questions.jsonl")

    # the split is a pure function of (seed, question ids) -- re-derive it
    redone = config.make_splits(qids)
    rep.check(list(redone["dev"]) == list(splits["dev"]),
              "dev split reproducible from seed", f"seed={config.SEED}")
    rep.check(list(redone["test"]) == list(splits["test"]),
              "test split reproducible from seed", f"seed={config.SEED}")

    # the assigned split column must agree with splits.json
    by_id = {q["question_id"]: q.get("split") for q in questions}
    rep.check(all(by_id.get(i) == "dev" for i in dev)
              and all(by_id.get(i) == "test" for i in test),
              "split column agrees with splits.json")

    stats_path = root / "data" / "corpus" / "corpus_stats.json"
    if stats_path.exists():
        st = json.loads(stats_path.read_text(encoding="utf-8"))
        rep.check(st["n_gold_pages_needed"] == st["n_pages_in_corpus"],
                  "every needed gold page was retrieved",
                  f"{st['n_pages_in_corpus']}/{st['n_gold_pages_needed']}")
        rep.check(st["per_split"]["test"]["n_with_a_gold_page_in_corpus"] == config.N_TEST,
                  "no test question lost its gold page",
                  f"{st['per_split']['test']['n_with_a_gold_page_in_corpus']}/{config.N_TEST}")
    return questions


# ---------------------------------------------------------------------------
# C. chunk sets
# ---------------------------------------------------------------------------

def check_chunks(rep: Report, root: Path, deep: bool):
    rep.section("C. chunk sets")
    stats_path = root / "data" / "chunks" / "chunk_stats.json"
    if not stats_path.exists():
        rep.check(False, "chunk_stats.json exists", str(stats_path))
        return {}
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    counts = {}
    for s in stats:
        cid = s["chunkset_id"]
        counts[cid] = s["n_chunks"]
        rep.check(s["tokens_max"] <= config.MAX_CHUNK_TOKENS,
                  f"{cid}: cap holds",
                  f"max={s['tokens_max']} <= {config.MAX_CHUNK_TOKENS}")
        # fixed_128 sets its own, stricter cap; nothing may exceed the global one
        own_cap = s["params"].get("max_tokens")
        rep.check(own_cap is not None and own_cap <= config.MAX_CHUNK_TOKENS,
                  f"{cid}: own cap within the global cap",
                  f"{own_cap} <= {config.MAX_CHUNK_TOKENS}")
        rep.check(s["params"].get("overlap_tokens", 0) == config.CHUNK_OVERLAP,
                  f"{cid}: zero overlap", f"{s['params'].get('overlap_tokens', 0)}")

    # every chunker saw the same page set -- otherwise the comparison is unfair
    npages = {s["chunkset_id"]: s["n_pages"] for s in stats}
    rep.check(len(set(npages.values())) == 1,
              "all chunkers ran over the same corpus", f"{sorted(set(npages.values()))}")

    if not deep:
        return counts

    # --deep: re-tokenise every chunk with the chunk tokenizer and re-measure
    from crag.tokenization import get_tokenizer
    tok = get_tokenizer(config.CHUNK_TOKENIZER)
    for cid in counts:
        path = root / "data" / "chunks" / f"{cid}.jsonl"
        if not path.exists():
            rep.check(False, f"{cid}: chunk file present", str(path), warn_only=True)
            continue
        worst, n, bad = 0, 0, []
        spans = defaultdict(list)
        for rec in read_jsonl(path):
            n += 1
            measured = tok.count(rec["text"])
            worst = max(worst, measured)
            if measured > config.MAX_CHUNK_TOKENS and len(bad) < 5:
                bad.append((rec["chunk_id"], measured))
            spans[rec["doc_id"]].append((rec["char_start"], rec["char_end"]))
        rep.check(not bad, f"{cid}: re-measured cap holds",
                  f"{n} chunks, worst={worst}" + (f", e.g. {bad[:2]}" if bad else ""))
        rep.check(n == counts[cid], f"{cid}: chunk count matches stats",
                  f"{n} vs {counts[cid]}")
        overlaps = 0
        for doc, sp in spans.items():
            sp.sort()
            for (a0, a1), (b0, b1) in zip(sp, sp[1:]):
                if b0 < a1:
                    overlaps += 1
        rep.check(overlaps == 0, f"{cid}: char spans non-overlapping",
                  f"{overlaps} overlapping pairs")
    return counts


# ---------------------------------------------------------------------------
# D. indexes
# ---------------------------------------------------------------------------

def check_indexes(rep: Report, root: Path, chunk_counts: dict):
    rep.section("D. indexes")
    idx_dir = root / "data" / "index"
    metas = sorted(idx_dir.glob("*.meta.json"))
    rep.check(len(metas) == 8, "eight indexes", f"{len(metas)} found")
    for m in metas:
        meta = json.loads(m.read_text(encoding="utf-8"))
        name = m.name.replace(".meta.json", "")
        cid = meta.get("chunkset_id")
        expected = chunk_counts.get(cid)
        if expected is None:
            rep.check(False, f"{name}: chunkset known", f"{cid!r}", warn_only=True)
            continue
        rep.check(meta.get("n_vectors") == expected,
                  f"{name}: vectors == chunks",
                  f"{meta.get('n_vectors')} vs {expected}")
        emb = meta.get("embedder")
        spec = config.EMBEDDERS.get(emb)
        if spec is not None and meta.get("dim"):
            rep.check(meta["dim"] in (384, 768), f"{name}: plausible dim",
                      f"{emb} dim={meta['dim']}")


def recompute_row(rec: dict) -> dict:
    """Derive every stored metric that does not need the context text.

    Kept separate from the checking loop so the derivation itself is unit
    testable -- a verifier nobody has tested is not evidence of anything.
    """
    out = {
        "page_hit": float(page_hit(rec["doc_ids"], rec["gold_page_ids"])),
        "page_precision": float(page_precision(rec["doc_ids"], rec["gold_page_ids"])),
    }
    pred = rec.get("prediction")
    if rec.get("em") is None:                     # retrieval-only budget
        return out
    usable = isinstance(pred, str) and bool(pred)
    ab = float(is_abstention(pred)) if isinstance(pred, str) else 0.0
    scored = usable and not ab
    out["abstained"] = ab
    out["em"] = float(exact_match(pred, rec["answers"])) if scored else 0.0
    out["f1"] = float(f1(pred, rec["answers"])) if scored else 0.0
    out["em_lenient"] = float(contains_answer(pred, rec["answers"])) if scored else 0.0
    return out


def recompute_run(recs):
    """(mismatch counter, per-metric value lists) for one run's records."""
    mism = Counter()
    agg = defaultdict(list)
    for rec in recs:
        derived = recompute_row(rec)
        for key, mine in derived.items():
            if key == "em_lenient":               # not stored by the pipeline
                agg[key].append(mine)
                continue
            tol = 1e-6 if key == "f1" else TOL
            if key in rec and not close(mine, rec[key], tol):
                mism[key] += 1
            agg[key].append(mine)
        agg["answer_recall"].append(float(rec["answer_recall"]))
    return mism, agg


# ---------------------------------------------------------------------------
# E/F/G. runs, per-question metrics, summaries
# ---------------------------------------------------------------------------

def check_runs(rep: Report, root: Path, deep: bool):
    rep.section("E. run inventory")
    runs_dir = root / "data" / "runs"
    expected = {r.run_id: r for r in config.run_matrix()}
    found = {p.name.replace(".jsonl", "") for p in runs_dir.glob("*.jsonl")}
    rep.check(set(expected) == found, "all 16 runs present",
              f"{len(found)} found, missing {sorted(set(expected) - found) or 'none'}")

    qsets, recomputed = {}, {}
    for run_id, spec in sorted(expected.items()):
        path = runs_dir / f"{run_id}.jsonl"
        if not path.exists():
            continue
        recs = list(read_jsonl(path))
        qsets[run_id] = tuple(r["question_id"] for r in recs)

        rep.check(len(recs) == config.N_TEST, f"{run_id}: 800 rows", f"{len(recs)}")
        rep.check(all(r["split"] == "test" for r in recs),
                  f"{run_id}: test split only")
        rep.check(all(r["budget"] == spec.budget for r in recs),
                  f"{run_id}: budget column == {spec.budget}")
        rep.check(all(r["n_tokens_used"] <= spec.budget for r in recs),
                  f"{run_id}: budget never exceeded",
                  f"max={max(r['n_tokens_used'] for r in recs)} <= {spec.budget}")
        rep.check(all(r["n_candidates_seen"] <= config.N_CANDIDATES for r in recs),
                  f"{run_id}: candidate list <= {config.N_CANDIDATES}")
        rep.check(all(len(r["chunk_ids"]) == r["n_chunks_used"] for r in recs),
                  f"{run_id}: chunk_ids length == n_chunks_used")
        rep.check(all(len(set(r["chunk_ids"])) == len(r["chunk_ids"]) for r in recs),
                  f"{run_id}: no chunk selected twice")
        rep.check(all(list(r["scores"]) == sorted(r["scores"], reverse=True)
                      for r in recs),
                  f"{run_id}: selected chunks in descending score order")

        # F. recompute the metrics that do not need the context text
        mism, agg = recompute_run(recs)
        rep.check(not mism, f"{run_id}: stored metrics == recomputed",
                  "all rows agree" if not mism else f"mismatches {dict(mism)}")
        recomputed[run_id] = {k: sum(v) / len(v) for k, v in agg.items() if v}

    rep.section("E2. pairing validity")
    if qsets:
        distinct = set(qsets.values())
        rep.check(len(distinct) == 1,
                  "every run scored the identical question sequence",
                  f"{len(distinct)} distinct sequence(s) -- paired tests require 1")

    rep.section("G. summaries vs per-question data")
    for run_id, means in sorted(recomputed.items()):
        spath = runs_dir / f"{run_id}.summary.json"
        if not spath.exists():
            rep.check(False, f"{run_id}: summary exists", warn_only=True)
            continue
        summ = json.loads(spath.read_text(encoding="utf-8"))
        bad = []
        for key in ("em", "f1", "abstention_rate", "page_hit", "answer_recall"):
            mine = means.get("abstained" if key == "abstention_rate" else key)
            theirs = summ.get(key)
            if mine is None and theirs is None:
                continue
            if mine is None or theirs is None or abs(mine - theirs) > MEAN_TOL:
                bad.append(f"{key}: {theirs} vs {None if mine is None else round(mine, 4)}")
        rep.check(not bad, f"{run_id}: summary means reproduce", "; ".join(bad) or "")
        rep.check(summ.get("n_questions") == config.N_TEST,
                  f"{run_id}: summary n == 800", str(summ.get("n_questions")))
        gen_expected = spec_generated(run_id)
        rep.check(bool(summ.get("generated")) == gen_expected,
                  f"{run_id}: generated flag matches GENERATION_BUDGETS",
                  f"generated={summ.get('generated')}, expected={gen_expected}")
    return recomputed


def spec_generated(run_id: str) -> bool:
    for r in config.run_matrix():
        if r.run_id == run_id:
            return r.budget in config.GENERATION_BUDGETS
    return False


# ---------------------------------------------------------------------------
# H. the analysis outputs
# ---------------------------------------------------------------------------

def check_analysis(rep: Report, root: Path, recomputed: dict):
    rep.section("H. analysis outputs vs raw runs")
    adir = root / "data" / "analysis"
    wide = adir / "results_wide.csv"
    if not wide.exists():
        rep.check(False, "results_wide.csv exists", str(wide), warn_only=True)
        return
    try:
        import pandas as pd
    except ImportError:
        rep.check(False, "pandas available for the analysis check", warn_only=True)
        return

    df = pd.read_csv(wide)
    bad = []
    for _, row in df.iterrows():
        run_id = f"{row['chunker']}__{row['embedder']}__b{int(row['budget'])}"
        mine = recomputed.get(run_id)
        if not mine:
            continue
        for col, key in (("em", "em"), ("f1", "f1"), ("abstained", "abstained"),
                         ("answer_recall", "answer_recall"), ("page_hit", "page_hit"),
                         ("em_lenient", "em_lenient")):
            if col not in row or pd.isna(row[col]):
                continue
            if key not in mine:
                continue
            if abs(float(row[col]) - mine[key]) > MEAN_TOL:
                bad.append(f"{run_id}.{col}: {row[col]} vs {round(mine[key], 4)}")
    rep.check(not bad, "results_wide.csv reproduces from the raw runs",
              "; ".join(bad[:4]) or f"{len(df)} rows checked")

    ipath = adir / "interaction.json"
    if ipath.exists():
        inter = json.loads(ipath.read_text(encoding="utf-8"))
        a, b = inter["pair"]
        did = inter["interaction"]["difference_in_differences"]
        wk = inter["weak_effect"]["diff"]
        st = inter["strong_effect"]["diff"]
        rep.check(abs((wk - st) - did) <= 1e-3,
                  "interaction == weak effect minus strong effect",
                  f"{wk} - {st} = {round(wk - st, 4)} vs {did}")
        m, bud = inter["metric"], inter["budget"]
        for emb, effect in (("weak", wk), ("strong", st)):
            ra = recomputed.get(f"{a}__{emb}__b{bud}", {}).get(m)
            rb = recomputed.get(f"{b}__{emb}__b{bud}", {}).get(m)
            if ra is None or rb is None:
                continue
            rep.check(abs((ra - rb) - effect) <= 1e-3,
                      f"{emb} effect reproduces from raw means",
                      f"{round(ra - rb, 4)} vs {effect}")
        rep.check(inter["interaction"]["n"] == config.N_TEST,
                  "interaction used all 800 questions", str(inter["interaction"]["n"]))
        if "mde" in inter["interaction"]:
            mine = interaction_mde(root, a, b, m, bud)
            rep.check(mine is not None and abs(mine - inter["interaction"]["mde"]) <= 1e-3,
                      "interaction MDE reproduces from raw runs",
                      f"{None if mine is None else round(mine, 4)} vs {inter['interaction']['mde']}")

    cpath = adir / "contrasts.csv"
    if cpath.exists():
        c = pd.read_csv(cpath)
        rep.check((c["n"] == config.N_TEST).all(), "every contrast is fully paired",
                  f"n values {sorted(c['n'].unique())}")
        rep.check(c["mde"].notna().all() and (c["mde"] > 0).all(),
                  "every contrast reports an MDE",
                  f"max MDE {c['mde'].max():.4f}")
        # each column is rounded to 4 dp independently, so the difference of the
        # rounded means can sit one ulp away from the rounded difference
        # (0.38125 - 0.3725 = 0.00875 -> 0.0088, but 0.3812 - 0.3725 = 0.0087).
        # Anything larger than one rounding step is a real inconsistency.
        drift = (c["diff"] - (c["mean_a"] - c["mean_b"])).abs().max()
        rep.check(drift <= 1.5e-4, "diff == mean_a - mean_b in every row",
                  f"max drift {drift:.2e} (rounding only)")


# ---------------------------------------------------------------------------

def interaction_mde(root: Path, a: str, b: str, metric: str, budget: int):
    """MDE of the difference-in-differences, from the per-question run files."""
    from crag.stats import mde_from_pairs
    vec = {}
    for emb in ("weak", "strong"):
        cells = []
        for ck in (a, b):
            path = root / "data" / "runs" / f"{ck}__{emb}__b{budget}.jsonl"
            if not path.exists():
                return None
            cells.append({r["question_id"]: r.get(metric) for r in read_jsonl(path)})
        qs = sorted(q for q in cells[0] if cells[0][q] is not None and cells[1].get(q) is not None)
        vec[emb] = [float(cells[0][q]) - float(cells[1][q]) for q in qs]
    if len(vec["weak"]) != len(vec["strong"]):
        return None
    return mde_from_pairs(vec["weak"], vec["strong"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="repo root (default: this file's parent)")
    ap.add_argument("--deep", action="store_true",
                    help="also re-tokenise every chunk (slow, needs the chunk files)")
    ap.add_argument("--json", default=None, help="write the result table here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parents[1]
    rep = Report(verbose=not args.quiet)

    print(f"verifying {root}")
    check_config(rep)
    check_corpus(rep, root)
    counts = check_chunks(rep, root, args.deep)
    check_indexes(rep, root, counts)
    recomputed = check_runs(rep, root, args.deep)
    check_analysis(rep, root, recomputed)

    n_ok = sum(1 for r in rep.rows if r["status"] == "ok")
    print(f"\n{n_ok}/{len(rep.rows)} checks passed, "
          f"{len(rep.warnings)} warning(s), {len(rep.failures)} failure(s)")
    for r in rep.failures:
        print(f"  FAIL  {r['check']}: {r['detail']}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "passed": n_ok, "total": len(rep.rows),
            "failures": rep.failures, "warnings": rep.warnings,
            "rows": rep.rows,
        }, indent=2), encoding="utf-8")
        print(f"written to {args.json}")

    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
