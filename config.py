"""
config.py -- the locked design spec, as code.

Every number here is a design decision from the project spec. Nothing in
crag/ or scripts/ should hard-code an experimental constant; import it from
here so the poster's methods section and the code can never drift apart.

Design summary
--------------
Primary RQ : does embedding-model capacity moderate the size of the
             chunking effect?
IV1        : chunker (4 levels, all capped at MAX_CHUNK_TOKENS)
IV2        : embedder (weak MiniLM-L6 vs strong bge-base)
IV3        : context budget (1000 / 2500 generator tokens)
Run matrix : 4 x 2 x 2 = 16 runs.
DVs        : answer_recall@budget, page_hit@budget, EM, F1
             (+ abstention rate, n_chunks_used, n_tokens_used)

Two tokenizers, on purpose
--------------------------
CHUNK_TOKENIZER  ("bert-base-uncased") defines chunk size. Both embedders use
    the same BERT wordpiece vocab, so one tokenizer gives chunk sets that are
    comparable across the embedder factor and directly comparable to each
    model's max_seq_length.
BUDGET_TOKENIZER (the generator's own tokenizer) defines the context budget,
    because the budget is a claim about what the LLM actually reads.
Both counts are logged per chunk and per query.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CRAG_DATA_DIR", REPO_ROOT / "data"))

CORPUS_DIR = DATA_DIR / "corpus"      # gold-page union pulled from KILT
CHUNKS_DIR = DATA_DIR / "chunks"      # one .jsonl per (chunker[, embedder])
INDEX_DIR = DATA_DIR / "index"        # one FAISS index per (chunkset, embedder)
RUNS_DIR = DATA_DIR / "runs"          # per-query records, one .jsonl per run
TUNING_DIR = DATA_DIR / "tuning"      # dev-split sweeps (never reported)
ANALYSIS_DIR = DATA_DIR / "analysis"  # tables + figures for the poster

ALL_DIRS: Tuple[Path, ...] = (
    DATA_DIR, CORPUS_DIR, CHUNKS_DIR, INDEX_DIR, RUNS_DIR, TUNING_DIR, ANALYSIS_DIR
)


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Global switches
# ---------------------------------------------------------------------------

SEED = 13

# CRAG_OFFLINE=1 forces the stub tokenizer / stub embedder everywhere, so the
# pure-Python half of the pipeline can be tested with no downloads and no GPU.
OFFLINE = os.environ.get("CRAG_OFFLINE", "0") == "1"


# ---------------------------------------------------------------------------
# Dataset  (KILT-NQ validation: 2,837 questions, with wikipedia_id provenance)
# ---------------------------------------------------------------------------

KILT_TASKS_DATASET = "facebook/kilt_tasks"
KILT_TASKS_CONFIG = "nq"
KILT_SPLIT = "validation"
EXPECTED_N_QUESTIONS = 2837

KILT_WIKIPEDIA_DATASET = "facebook/kilt_wikipedia"

# The knowledge source is fetched from the original file, not through
# `datasets`. facebook/kilt_wikipedia is a loading SCRIPT, and datasets >=4
# refuses scripts ("Dataset scripts are no longer supported"); the repo has no
# parquet conversion, so no newer datasets version will ever load it. The
# script was only ever a wrapper around this JSONL, one page per line, with the
# same fields and the same wikipedia_ids -- so reading it directly changes the
# transport and nothing else. It also streams with byte-range resume, which the
# datasets path did not.
KILT_KNOWLEDGE_SOURCE_URL = (
    "https://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json"
)

# Splits. Dev is for hyperparameter tuning only and is never reported.
N_DEV = 200
N_TEST = 800


def make_splits(question_ids: Sequence[str]) -> Dict[str, List[str]]:
    """Deterministic dev/test split. Order of the input does not matter."""
    ids = sorted(set(question_ids))
    rng = random.Random(SEED)
    rng.shuffle(ids)
    if len(ids) < N_DEV + N_TEST:
        raise ValueError(
            f"need at least {N_DEV + N_TEST} questions, got {len(ids)}"
        )
    return {
        "dev": ids[:N_DEV],
        "test": ids[N_DEV:N_DEV + N_TEST],
        "unused": ids[N_DEV + N_TEST:],
    }


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# 256 = min(max_seq_length) across the two embedders. MiniLM silently truncates
# past 256 wordpieces while bge does not, so anything longer would confound the
# embedder comparison with information loss.
MAX_CHUNK_TOKENS = 256

CHUNK_OVERLAP = 0  # ECIR 2026 found no measurable benefit; ablation, not factor.

CHUNK_TOKENIZER = "bert-base-uncased"

RECURSIVE_SEPARATORS: Tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True)
class ChunkerSpec:
    key: str
    kind: str                      # "fixed" | "recursive" | "semantic"
    params: Dict[str, Any]
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    needs_embedder: bool = False   # semantic boundaries depend on the embedder


# Note on tuning fairness: a fixed-size chunker's only free parameter IS its
# size, and the design realises that parameter as two named conditions
# (fixed_128, fixed_256) rather than tuning it away -- so their grids are empty
# by construction, not by omission. Every parameter that is genuinely free
# (recursive tail-merge threshold, semantic breakpoint percentile) is tuned on
# dev and frozen before test.
CHUNKERS: Dict[str, ChunkerSpec] = {
    "fixed_128": ChunkerSpec(
        key="fixed_128",
        kind="fixed",
        params={"target_tokens": 128, "overlap_tokens": CHUNK_OVERLAP},
        param_grid={},
    ),
    "fixed_256": ChunkerSpec(
        key="fixed_256",
        kind="fixed",
        params={"target_tokens": 256, "overlap_tokens": CHUNK_OVERLAP},
        param_grid={},
    ),
    "recursive_256": ChunkerSpec(
        key="recursive_256",
        kind="recursive",
        params={
            "target_tokens": 256,
            "separators": list(RECURSIVE_SEPARATORS),
            "min_chunk_tokens": 32,
            # 1.0 == the standard greedy recursive splitter; lower values let
            # it close a chunk at a paragraph break once it is that full.
            "min_fill_ratio": 1.0,
        },
        param_grid={"min_fill_ratio": [0.5, 0.75, 1.0]},
    ),
    "semantic": ChunkerSpec(
        key="semantic",
        kind="semantic",
        params={
            "percentile": 90,
            "buffer_size": 1,
            "min_chunk_tokens": 32,
            "max_tokens": MAX_CHUNK_TOKENS,
        },
        param_grid={"percentile": [70, 80, 90, 95]},
        needs_embedder=True,
    ),
}


# ---------------------------------------------------------------------------
# Embedders (IV2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbedderSpec:
    key: str
    model_name: str
    dim: int
    max_seq_length: int
    query_prefix: str = ""
    passage_prefix: str = ""
    batch_size: int = 128       # sized for ~6 GB VRAM; raise on a 16 GB card
    normalize: bool = True      # required: the FAISS index is IndexFlatIP
    fp16: bool = False          # see the note below -- measured, not assumed


# Why fp16 is OFF (measured on the GTX 1660 Ti Max-Q, sm_75):
#
#     weak    fp16   98 t/s  ->  fp32  305 t/s   (batch 128)
#     strong  fp16   13.8    ->  fp32   48.6     (batch 64)
#
# TU116 has no tensor cores, and PyTorch's fp16 GEMM path there is a slow
# fallback -- strong/fp16 measured exactly 13.8 t/s at every batch size from 32
# to 256, the signature of a serialised path. fp32 is 3.1-3.5x faster. On an
# RTX card (sm_75 with tensor cores, or Ampere+) flip fp16 back on and re-run
# the benchmark; the batch sizes below were picked from the same measurement
# (strong collapses to 15 t/s at batch 256, where activations stop fitting).


EMBEDDERS: Dict[str, EmbedderSpec] = {
    "weak": EmbedderSpec(
        key="weak",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        max_seq_length=256,
        batch_size=128,      # measured best: 305 t/s (256 gives 286)
        fp16=False,
    ),
    "strong": EmbedderSpec(
        key="strong",
        model_name="BAAI/bge-base-en-v1.5",
        dim=768,
        max_seq_length=512,
        # bge asks for an instruction on QUERIES ONLY, never on passages.
        query_prefix="Represent this sentence for searching relevant passages: ",
        batch_size=64,       # measured best: 48.6 t/s (256 collapses to 15)
        fp16=False,
    ),
}


# ---------------------------------------------------------------------------
# Retrieval: fill-to-budget, NOT top-k
# ---------------------------------------------------------------------------

N_CANDIDATES = 64          # FAISS top-N before the budget filter
BUDGETS: Tuple[int, ...] = (1000, 2500)

# Budgets at which the GENERATOR runs. Retrieval metrics (answer_recall,
# page_hit) are reported for every budget in BUDGETS -- they cost nothing once
# the index exists -- but EM/F1 exist only where generation ran.
#
# Why this is not both: measured on the target GPU,
# generation runs at 0.12 prompts/s on a 2,500-token prompt at batch 1, because
# 6 GB of VRAM cannot hold a batched fp32 prefill that long. Both budgets would
# be ~21 GPU-hours. Generating at 1,000 only keeps n=800 and the ~4.4-point EM
# minimum detectable effect on the PRIMARY question (chunker x embedder), and
# gives up the secondary context-length comparison rather than statistical
# power. Set this to BUDGETS on a bigger card.
GENERATION_BUDGETS: Tuple[int, ...] = (1000,)
BUDGET_POLICY = "stop"     # "stop" at first overflow (spec) | "skip" and continue
CONTEXT_SEPARATOR = "\n\n"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

GENERATOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"
GEN_MAX_NEW_TOKENS = 32
GEN_TEMPERATURE = 0.0      # greedy: do_sample=False
GEN_LOAD_IN_4BIT = True
# Measured on the target GPU, not assumed: fp32 compute is ~3x faster
# than fp16 on a tensor-core-less Turing card, the same finding as for the
# embedders. Re-measure on any other GPU before trusting this.
GEN_DTYPE = "float32"
GEN_BATCH_SIZE = 4         # left-padded batched generation; batch size measured on the target GPU

NO_ANSWER = "NO_ANSWER"

SYSTEM_PROMPT = (
    "You are a precise extractive question-answering assistant. "
    "You answer only with text taken from the provided context."
)

# One fixed prompt, held constant across all 16 runs.
USER_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Instructions:\n"
    "- Answer with the shortest exact span from the context, a few words at most.\n"
    "- Do not explain, do not write a sentence, do not repeat the question.\n"
    f"- If the context does not contain the answer, reply with exactly {NO_ANSWER}.\n\n"
    "Answer:"
)


def build_prompt(question: str, context: str) -> str:
    return USER_TEMPLATE.format(context=context, question=question)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

N_BOOTSTRAP = 10_000
ALPHA = 0.05
POWER = 0.80


# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    chunker: str
    embedder: str
    budget: int

    @property
    def run_id(self) -> str:
        return f"{self.chunker}__{self.embedder}__b{self.budget}"

    @property
    def chunkset_id(self) -> str:
        """Chunk sets are embedder-specific only when boundaries are.

        Locked decision: the semantic chunker draws its boundaries with the
        SAME embedder used for retrieval in that run, so a weak embedder is
        penalised for both worse boundaries and worse retrieval -- the honest
        end-to-end pipeline, and part of the moderation effect under test.
        """
        if CHUNKERS[self.chunker].needs_embedder:
            return f"{self.chunker}__{self.embedder}"
        return self.chunker


def run_matrix() -> List[RunSpec]:
    return [
        RunSpec(chunker=c, embedder=e, budget=b)
        for c in CHUNKERS
        for e in EMBEDDERS
        for b in BUDGETS
    ]


if __name__ == "__main__":
    runs = run_matrix()
    print(f"{len(runs)} runs")
    for r in runs:
        print(" ", r.run_id, "| chunkset:", r.chunkset_id)
