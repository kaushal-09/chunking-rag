# Methods

Everything a reader needs to reconstruct the experiment, and every decision
that could reasonably be challenged. Numbers here are copied from the run files
and analysis outputs in `data/`; `python scripts/07_verify.py` re-checks them
against the raw runs.

---

## 1. Research question

Does the capacity of the retrieval embedding model **moderate** the effect of
chunking strategy on end-to-end RAG accuracy?

The hypothesis under test is that chunk boundaries matter more when the
retriever is weak — a strong embedder is robust to how text is carved up,
a weak one is not. This is a claim about an **interaction**, not about which
chunker is best, and the design is built to test it as one.

Testing an interaction rather than a main effect is what keeps the study from
being a fifth re-run of "we compared four chunkers": a main-effect comparison
has been done repeatedly and is confounded by context length (see §5).

---

## 2. Dataset

**KILT-NQ validation split**, 2,837 questions — every validation item that
carries both a short answer and Wikipedia provenance.

| | |
|---|---|
| questions | 2,837 |
| dev / test / unused | 200 / 800 / 1,837 |
| gold pages needed | 4,274 |
| gold pages in corpus | 4,274 (100%) |
| corpus size | 82,259,319 chars ≈ 20.6M tokens |
| mean page length | 19,246 chars |

### Corpus construction

The corpus is the **union of the gold Wikipedia pages** for those questions,
not all of Wikipedia. This is a deliberate restriction: it makes the retrieval
problem tractable on one consumer GPU while keeping every question answerable,
and it means a retrieval failure is a *ranking* failure rather than a
*coverage* failure. The cost is that absolute recall is higher than it would
be against full Wikipedia, so the numbers are comparative, not absolute.

Pages were pulled from the KILT knowledge source
(`kilt_knowledgesource.json`, 34.8 GiB) streamed directly over HTTP.

> **Why not `datasets.load_dataset("facebook/kilt_wikipedia")`?**
> `datasets` 5.x removed support for dataset loading scripts, and that
> repository contains only `kilt_wikipedia.py` with no parquet conversion, so
> no version of `datasets` can load it. The loading script was read to recover
> the canonical URL and record schema, and the file is streamed and parsed
> directly. **This changes the transport, not the data** — same file, same
> records, same `wikipedia_id` provenance keys.

The stream is resumable (HTTP byte-range, 12 retries) and checkpoints every
64 MiB, because a 34.8 GiB download does not reliably complete in one
connection. A resume re-reads the window between the last checkpoint and the
crash, which on the first run produced 57 duplicate pages; these were removed
and the resume logic now seeds its found-set from `pages.jsonl`, so a resume
is idempotent.

### Splits

```python
ids = sorted(set(question_ids))     # order of input does not matter
random.Random(13).shuffle(ids)
dev, test, unused = ids[:200], ids[200:1000], ids[1000:]
```

Deterministic and re-derivable from the seed alone; `07_verify.py`
re-runs this and asserts it reproduces `splits.json` exactly. Dev and test are
disjoint. **Every parameter was tuned on dev; test was scored once.**

### Ceiling

The answer string appears somewhere in the gold page(s) for **98.88%** of test
questions (98.0% dev, 97.67% overall). Retrieval recall therefore cannot
exceed ~0.99, and any chunker reaching it has extracted everything the corpus
contains.

---

## 3. Chunking

All four chunkers return `(doc_id, text, char_start, char_end, n_tokens)`, so
provenance and token-level analysis stay possible throughout.

| chunker | rule | free parameter |
|---|---|---|
| `fixed_128` | 128 wordpiece tokens, no overlap | — |
| `fixed_256` | 256 wordpiece tokens, no overlap | — |
| `recursive_256` | recursive separator split (`\n\n`, `\n`, `. `, ` `, `""`), target 256, greedy re-merge | `min_fill_ratio` |
| `semantic` | percentile breakpoints on adjacent-sentence cosine distance | `percentile` |

**Two tokenizers, by design.** Chunk size is measured with
`bert-base-uncased` — the wordpiece vocabulary shared by both embedders — so
"256 tokens" means the same thing to MiniLM and to bge. The context budget
(§5) is measured with the *generator's* tokenizer, because the budget is a
claim about what the LLM reads. Conflating the two would make the budget
unequal across conditions.

**Why the cap is 256.** `MAX_CHUNK_TOKENS = 256` is `min(max_seq_length)`
across the two embedders. MiniLM silently truncates past 256 wordpieces while
bge does not, so any longer chunk would confound the embedder comparison with
information loss — the weak embedder would literally not see part of its own
input. **Overlap is 0 everywhere**, so no answer span is duplicated across
chunks and `answer_recall` is not inflated by redundancy.

**Enforcing the cap.** Semantic chunking routinely emits very long groups
(a topically uniform page can yield one 4,000-token "chunk"), so oversized
groups are hard-split — at sentence borders first, inside a sentence only when
a single sentence exceeds the cap. Every chunk is then re-measured; if any
piece still exceeds the cap it is re-split, up to 8 times, and an
`AssertionError` naming the document is raised if one escapes.

> This check is not defensive boilerplate. Splitting by token *index* and
> measuring by `bisect` over offset arrays disagree when a tokenizer produces
> non-monotonic offsets, and that disagreement produced a **129-token chunk
> from a 128-token chunker**. The fix normalises spans to be strictly
> increasing and non-overlapping, re-splits until every piece *measures*
> within the cap, and asserts against the chunker's own cap rather than the
> global one. Four regression tests cover it.

`recursive_256`'s `min_fill_ratio` controls whether a chunk may close early at
a strong separator (a paragraph break) rather than greedily packing across it.
Ratio 1.0 is exactly LangChain's `RecursiveCharacterTextSplitter` behaviour.

`semantic` uses `buffer_size = 1` (single-sentence context window), a strict
`>` comparison against the percentile threshold, and `min_chunk_tokens = 32`.
**The semantic chunker uses the run's own embedder to draw its boundaries** —
a weak embedder draws weak boundaries, and that is part of the effect under
test. This is the only chunk set that is embedder-specific, which is why there
are five chunk sets for four chunkers.

### Resulting chunk sets

| chunkset | chunks | tokens mean | median | p10 | p90 | max |
|---|---|---|---|---|---|---|
| `fixed_128` | 143,674 | 126.1 | 128 | 128 | 128 | 128 |
| `fixed_256` | 72,910 | 248.5 | 256 | 256 | 256 | 256 |
| `recursive_256` | 92,567 | 195.7 | 202 | 134 | 244 | 256 |
| `semantic__strong` | 134,289 | 134.9 | 121 | 48 | 242 | 256 |
| `semantic__weak` | 93,408 | 193.9 | 229 | 70 | 252 | 256 |

---

## 4. Embedders

| key | model | dim | notes |
|---|---|---|---|
| weak | `sentence-transformers/all-MiniLM-L6-v2` | 384 | no prefix |
| strong | `BAAI/bge-base-en-v1.5` | 768 | instruction prefix on **queries only** |

The bge prefix (`Represent this sentence for searching relevant passages: `)
is applied to queries and never to passages, per the model card. Applying it
to passages as well is a common error that degrades retrieval.

Both run in **fp32**. On the GTX 1660 Ti (Turing TU116) fp16 was measured
3.1–3.5× *slower* than fp32 — weak 305→98 tok/s, strong 48.6→13.8 tok/s —
because TU116 has no tensor cores. This is a throughput decision with no
effect on the results; it is recorded because the obvious assumption
("Turing, so use fp16") is wrong on this specific chip.

Index: **FAISS `IndexFlatIP` over L2-normalised vectors**, i.e. exact cosine
similarity with no approximation — nothing about the retrieval is stochastic
or index-tuned. Eight indexes (5 chunk sets × applicable embedders), 2.0 GB
total; `07_verify.py` asserts vector count equals chunk count for each.

---

## 5. Retrieval: fill-to-budget, not top-k

**This is the methodological core of the study.**

Fixed top-k hands the generator a different number of tokens in every
condition: 5 × 128-token chunks is 640 tokens; 5 × 250-token semantic chunks
is 1,250. A "chunking effect" measured at fixed *k* is therefore partly a
context-length effect, and context length is known to move end-task accuracy
on its own. A comparison that reports only top-k results carries this
confound; Qu et al. (2025), for example, compare chunkers at k ∈ {1, 3, 5, 10}.
Capping the retrieved text at a fixed number of tokens is not new — Chen et
al. (2024) feed the reader the top 100 or 500 retrieved tokens when comparing
passage, sentence and proposition units — and this study follows that idea.

Concretely: retrieve `N_CANDIDATES = 64` from FAISS, then add chunks in rank
order until the next one would overflow the budget, and **stop**
(`BUDGET_POLICY = "stop"`, not "skip" — skipping ahead to a smaller chunk
would reintroduce a rank-order confound). The separator (`\n\n`) is counted
against the budget. Every condition therefore delivers the same token budget
to the generator, and the only thing that varies is how that budget is
carved up.

Budgets are **1,000 and 2,500 generator tokens**. Realised utilisation is
0.84–0.97, and the budget is never exceeded (asserted per-row).

---

## 6. Generation

| | |
|---|---|
| model | `Qwen/Qwen2.5-3B-Instruct` |
| quantisation | 4-bit NF4 (bitsandbytes), fp32 compute |
| decoding | greedy (`do_sample=False`, temperature 0) |
| max new tokens | 32 |
| batch | 4, left-padded |

The prompt asks for the shortest exact span, forbids explanation, and defines
an explicit abstention token:

```
Context:
{context}

Question: {question}

Instructions:
- Answer with the shortest exact span from the context, a few words at most.
- Do not explain, do not write a sentence, do not repeat the question.
- If the context does not contain the answer, reply with exactly NO_ANSWER.

Answer:
```

`NO_ANSWER` is scored separately and **never counted as correct**. It is what
distinguishes "declined" from "hallucinated", which turns out to matter (§9).

### Generation scope — a hardware constraint, stated plainly

**Answers were generated at budget 1,000 only.** Budget 2,500 is scored for
retrieval metrics, which are free, but not generated.

At 2,500 tokens the 6 GB card can only run batch 1, measured at 0.12
prompts/s — **14.8 hours** for the 8 cells, against 5.5 hours at budget 1,000
where batch 4 fits (0.32 prompts/s). Larger batches OOM (batch 4 peaks at
9.01 GB on a 6 GB card).

This preserves full statistical power on the primary RQ, which lives at a
single budget, and costs a secondary robustness check. It is a real limitation
and belongs on the poster. Note that the retrieval data bounds what the
missing runs could have shown: at budget 2,500 the max−min recall spread
across chunkers *shrinks* (weak 0.038→0.016, strong 0.025→0.014) and the
smallest p among those twelve recall contrasts is 0.091. If the chunking effect is mediated
by retrieval, a flatter retrieval profile cannot produce a steeper answer
profile.

---

## 7. Metrics: boundary-invariant only

The original proposal scored MRR@5 and Hit@5 over chunks. **That is
ill-defined here.** Chunk boundaries *are* the independent variable, so each
condition has a different candidate set and there is no consistent "gold
chunk" to rank against — a chunker that splits an answer across two chunks
would be penalised for a boundary choice rather than for a retrieval failure.
Every metric below is invariant to how boundaries were drawn.

| metric | definition |
|---|---|
| `answer_recall` | does the assembled context contain a gold answer string? Token-sequence containment after SQuAD normalisation (DPR-style `has_answer`), not substring matching |
| `page_hit` | did any retrieved chunk come from a gold `wikipedia_id`? Independent of chunking entirely |
| `page_precision` | fraction of retrieved chunks from a gold page |
| `EM` / `F1` | standard SQuAD normalisation (lowercase, strip punctuation, drop articles, squash whitespace) |
| `abstained` | did the model emit `NO_ANSWER`? |
| `context_ok_answer_wrong` | the answer was in the context and the model still got it wrong |
| `em_lenient` | robustness metric: answer-string containment in the prediction rather than exact match |

Token-sequence containment matters: substring matching would count `"1"` as
finding `"1994"`. The implementation checks token sequences, so
`contains_answer("April 14, 2019", ["2019"])` is 1 but
`contains_answer("1", ["1994"])` is 0.

---

## 8. Statistical plan

Pre-specified before the test runs.

- **Paired bootstrap**, 10,000 resamples, seed 13, percentile CIs. Paired
  because every condition scores the identical 800 questions —
  `07_verify.py` asserts all 16 runs share one question sequence.
- **Exact McNemar** (binomial, via `math.comb`, no normal approximation) for
  binary metrics. No scipy dependency anywhere in the stats module.
- **Holm–Bonferroni** at α = 0.05 across the pairwise contrast family.
- **Minimum detectable effect** reported for *every* contrast and for the
  interaction itself (whose MDE is larger, see §9), so a null is reported as
  a bound rather than as evidence of zero.
- **One pre-specified interaction contrast**: `semantic` vs `fixed_128`,
  difference-in-differences across embedders. The max−min spread across all
  four chunkers is reported as a **descriptive** quantity only — it selects
  the most extreme pair post hoc and would be inflated if tested.

### Parameter tuning

Every strategy with a free parameter was tuned **on dev only**, objective =
mean `answer_recall` across budgets (and across embedders, for the
embedder-agnostic `recursive_256`).

| target | parameter | swept | selected | objective |
|---|---|---|---|---|
| `recursive_256` | `min_fill_ratio` | 0.5 / 0.75 / 1.0 | **0.75** | 0.8838 |
| `semantic__weak` | `percentile` | 70 / 80 / 90 / 95 | **95** | 0.8675 |
| `semantic__strong` | `percentile` | 70 / 80 / 90 / 95 | **80** | 0.8875 |

**Caveat worth stating.** Dev is 200 questions, so the selection margins are
small — `recursive_256` won by 0.0025 over the next setting, and the semantic
percentile sweep spans 0.032 (weak) and 0.015 (strong). The tuning is honest
(dev-only, objective fixed in advance, frozen before test) but it is not
powered to distinguish adjacent settings. Each strategy was given its best
available shot; none was given a *reliably* best setting.

---

## 9. What the design can and cannot conclude

**Can:** whether the chunking effect differs between a weak and a strong
embedder, at a fixed context budget, with the context-length confound removed
and boundary-invariant metrics.

**Cannot:**

1. **Whether the null is generator-limited.** Conditional on the answer being
   in the context, EM is only 0.43–0.46. A reader that noisy may be unable to
   express a chunking effect that a stronger reader would show. One generator
   was tested; nothing here distinguishes "chunking doesn't matter" from
   "chunking doesn't matter *to Qwen2.5-3B*". This is the single most
   important limitation.
2. **Whether it holds at other budgets.** Answers exist at one budget (§6).
3. **Whether it holds against full Wikipedia.** The corpus is gold-page-only
   (§2), so absolute recall is optimistic.
4. **That any chunker is best.** After Holm correction, no pairwise contrast
   is significant. The largest MDE across the EM contrasts is 0.0439 — effects
   smaller than ~4.4 EM points were not detectable at n = 800.
5. **A small interaction.** The interaction is a difference of two paired
   differences, so its variance is larger than any single contrast's. Its own
   MDE (80 % power, α = .05) is **0.0556**, computed from the per-question
   difference-in-differences vector and stored in `interaction.json`. The
   honest reading of the primary null is therefore "no moderation larger than
   about 5.6 EM points" — not the 4.4 points that applies to pairwise
   contrasts. The two simple effects have MDEs of 0.0416 (weak) and 0.0407
   (strong).

---

## 10. Reproducing and verifying

From the repository root. The file numbers are not the run order:
`04_tune.py` has to run before `02_chunk_and_index.py`, which reads the tuned
settings from `data/tuning/best_params.json`.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA build first
pip install -r requirements.txt

python scripts/01_build_corpus.py --questions-only   # KILT-NQ questions
python scripts/01_build_corpus.py --resume           # Wikipedia scan; long, resumable
python scripts/04_tune.py                            # dev sweep -> data/tuning
python scripts/02_chunk_and_index.py                 # chunk sets and the 8 indexes
python scripts/03_run_pipeline.py --split test       # the 16 runs; the long GPU job

python scripts/05_analyze.py                         # EM -> data/analysis
python scripts/05_analyze.py --metric em_lenient --out-dir data/analysis_em_lenient
python scripts/05_analyze.py --metric answer_recall --out-dir data/analysis_recall_b1000
python scripts/06_poster_figures.py                  # -> poster/figures
python scripts/07_verify.py                          # 258 checks; exit 0 = all passed
python scripts/07_verify.py --deep                   # also re-tokenises every chunk
```

The last group needs only what is in the repository; `--deep` also needs the
chunk files. The Wikipedia scan streams tens of GB through the Hugging Face
cache, so point `HF_HOME` at a drive with room before running it;
`CRAG_DATA_DIR` moves `data/` elsewhere the same way.

`07_verify.py` deliberately does not import the analysis code it is
checking. It re-reads the raw run files, recomputes the metrics from the
stored predictions using `crag.metrics`, and compares against what the
pipeline wrote — so agreement means the reported numbers came from the data
and not from a bug in the reporting layer. It also re-derives the splits from
the seed, checks the chunk cap, asserts vector counts match chunk counts,
verifies the budget was never exceeded, and confirms `results_wide.csv`,
`contrasts.csv` and `interaction.json` all reproduce from the raw runs.

The runs were produced on Windows 11 with Python 3.12.7 and an NVIDIA GeForce
GTX 1660 Ti Max-Q (6 GB, CUDA 12.1), using torch 2.5.1+cu121, transformers
5.16.1, sentence-transformers 6.0.1, datasets 5.0.1, faiss 1.15.0,
accelerate 1.14.0, bitsandbytes 0.50.2, numpy 1.26.4, pandas 2.2.2 and
matplotlib 3.9.2.
