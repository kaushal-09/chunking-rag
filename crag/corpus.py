"""
Build the study corpus from KILT.

Two artefacts come out of here:

    questions.jsonl  {question_id, question, answers[], gold_page_ids[], split}
    pages.jsonl      {doc_id, title, text}      doc_id == KILT wikipedia_id

`doc_id` is the KILT `wikipedia_id`, which is what makes `page_hit` a
boundary-invariant retrieval ground truth: it is defined by the dataset, not
by where we happened to cut the text.

Corpus scope is the UNION of gold pages over all 2,837 validation questions
(closed corpus). Not gold-pages-per-question -- that would put the answer in a
handful of documents and manufacture a null result -- and not all ~5.9M KILT
articles, which is not indexable three times on one laptop GPU.

The heavy part is finding those pages inside kilt_wikipedia. That scan is
checkpointed: it writes pages as it finds them and records which ids are still
missing, so an interrupted run resumes instead of restarting.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set

# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class QuestionRecord:
    question_id: str
    question: str
    answers: List[str]
    gold_page_ids: List[str]
    split: str = "unassigned"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PageRecord:
    doc_id: str
    title: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# jsonl helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], mode: str = "w") -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl(path: Path) -> int:
    return sum(1 for _ in read_jsonl(path))


# ---------------------------------------------------------------------------
# KILT record parsing  (pure functions -- unit-tested offline)
# ---------------------------------------------------------------------------

def parse_question(record: Dict[str, Any]) -> QuestionRecord:
    """One KILT-NQ example -> QuestionRecord.

    A KILT example carries several `output` entries; each may hold an
    `answer` string, a `provenance` list, or both. Answers are deduplicated
    with order preserved; gold pages are the union of every provenance
    `wikipedia_id` across all outputs.
    """
    answers: List[str] = []
    pages: List[str] = []
    seen_a: Set[str] = set()
    seen_p: Set[str] = set()

    for out in record.get("output") or []:
        ans = (out.get("answer") or "").strip()
        if ans and ans not in seen_a:
            seen_a.add(ans)
            answers.append(ans)
        for prov in out.get("provenance") or []:
            wid = prov.get("wikipedia_id")
            if wid is None:
                continue
            wid = str(wid)
            if wid not in seen_p:
                seen_p.add(wid)
                pages.append(wid)

    return QuestionRecord(
        question_id=str(record.get("id")),
        question=(record.get("input") or "").strip(),
        answers=answers,
        gold_page_ids=pages,
    )


def page_text(record: Dict[str, Any], drop_title_paragraph: bool = True) -> str:
    """One kilt_wikipedia example -> the page's body text.

    KILT stores a page as a list of paragraphs whose first element is the
    title line. That line is dropped by default: it is metadata, it is already
    kept in `title`, and leaving it in gives the first chunk of every page a
    free lexical match on the title.
    """
    text_field = record.get("text")
    if isinstance(text_field, dict):
        paragraphs = list(text_field.get("paragraph") or [])
    elif isinstance(text_field, list):
        paragraphs = list(text_field)
    elif isinstance(text_field, str):
        paragraphs = [text_field]
    else:
        paragraphs = []

    title = (record.get("wikipedia_title") or "").strip()
    if drop_title_paragraph and paragraphs:
        first = paragraphs[0].strip()
        if first == title or (title and first.startswith(title) and len(first) <= len(title) + 3):
            paragraphs = paragraphs[1:]

    cleaned = [p.strip() for p in paragraphs if p and p.strip()]
    return "\n\n".join(cleaned)


def parse_page(record: Dict[str, Any], drop_title_paragraph: bool = True) -> PageRecord:
    return PageRecord(
        doc_id=str(record.get("wikipedia_id")),
        title=(record.get("wikipedia_title") or "").strip(),
        text=page_text(record, drop_title_paragraph=drop_title_paragraph),
    )


# ---------------------------------------------------------------------------
# Loading questions (needs `datasets`)
# ---------------------------------------------------------------------------

def load_questions(
    dataset: str,
    config_name: str,
    split: str,
    cache_dir: Optional[str] = None,
) -> List[QuestionRecord]:
    from datasets import load_dataset

    ds = load_dataset(dataset, config_name, split=split, cache_dir=cache_dir)
    return [parse_question(rec) for rec in ds]


def assign_splits(
    questions: Sequence[QuestionRecord], splits: Dict[str, List[str]]
) -> List[QuestionRecord]:
    lookup: Dict[str, str] = {}
    for name, ids in splits.items():
        for qid in ids:
            lookup[qid] = name
    for q in questions:
        q.split = lookup.get(q.question_id, "unused")
    return list(questions)


# ---------------------------------------------------------------------------
# Scanning kilt_wikipedia for the gold pages
# ---------------------------------------------------------------------------

@dataclass
class ScanState:
    """Checkpoint for the wikipedia scan.

    `bytes_consumed` is what makes an interrupted 30 GB download cheap to
    resume: it is always a line boundary, so the next run asks the server for
    exactly the bytes after it with an HTTP Range request.
    """
    needed: Set[str] = field(default_factory=set)
    found: Set[str] = field(default_factory=set)
    n_seen: int = 0
    bytes_consumed: int = 0
    total_bytes: int = 0

    @property
    def missing(self) -> Set[str]:
        return self.needed - self.found

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_needed": len(self.needed),
            "n_found": len(self.found),
            "n_seen": self.n_seen,
            "bytes_consumed": self.bytes_consumed,
            "total_bytes": self.total_bytes,
            "missing": sorted(self.missing),
        }


def load_scan_state(
    state_path: Path, needed: Set[str], pages_path: Optional[Path] = None
) -> ScanState:
    """Rebuild the checkpoint, trusting the OUTPUT over the state file.

    The state file is written periodically; pages are written immediately. A
    crash between the last checkpoint and the failure therefore leaves pages on
    disk that the checkpoint does not know about, and a naive resume re-reads
    that window and appends them a second time. Seeding `found` from
    pages.jsonl makes resume idempotent: what is already written is never
    written again.
    """
    state = ScanState(needed=set(needed))
    p = Path(state_path)
    if p.exists():
        blob = json.loads(p.read_text(encoding="utf-8"))
        state.found = set(needed) - set(blob.get("missing", []))
        state.n_seen = int(blob.get("n_seen", 0))
        state.bytes_consumed = int(blob.get("bytes_consumed", 0))
        state.total_bytes = int(blob.get("total_bytes", 0))
    if pages_path is not None and Path(pages_path).exists():
        for row in read_jsonl(pages_path):
            doc_id = str(row.get("doc_id"))
            if doc_id in state.needed:
                state.found.add(doc_id)
    return state


def dedupe_pages(pages_path: Path) -> Dict[str, int]:
    """Drop repeated doc_ids from a pages file, keeping the first of each.

    Repairs a corpus written by a resume that predates the fix above. A
    duplicated page becomes duplicated chunks, which then compete against each
    other for the same context budget -- quietly wrong rather than loudly.
    """
    path = Path(pages_path)
    rows = list(read_jsonl(path))
    seen: Set[str] = set()
    kept = []
    for row in rows:
        doc_id = str(row.get("doc_id"))
        if doc_id in seen:
            continue
        seen.add(doc_id)
        kept.append(row)
    if len(kept) != len(rows):
        backup = path.with_suffix(path.suffix + ".withdupes")
        path.replace(backup)
        write_jsonl(path, kept)
    return {"lines_before": len(rows), "lines_after": len(kept),
            "removed": len(rows) - len(kept)}


def save_scan_state(state_path: Path, state: ScanState) -> None:
    p = Path(state_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def head_content_length(url: str, timeout: int = 30) -> Dict[str, Any]:
    """Size and range support of the knowledge source, without downloading it."""
    import requests

    response = requests.head(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return {
        "content_length": int(response.headers.get("Content-Length", 0)),
        "accept_ranges": response.headers.get("Accept-Ranges", "none"),
        "url": response.url,
    }


RETRYABLE_ERRORS: tuple = ()


def _retryable_errors() -> tuple:
    """Network faults worth resuming from rather than failing on.

    A 35 GB HTTP stream WILL be cut. `IncompleteRead` / `ProtocolError` /
    `ChunkedEncodingError` are the server hanging up mid-transfer, and the
    right response is a Range request from the last line boundary, not a
    traceback.
    """
    global RETRYABLE_ERRORS
    if RETRYABLE_ERRORS:
        return RETRYABLE_ERRORS
    errors = [ConnectionError, TimeoutError, OSError]
    try:
        import requests.exceptions as rex
        errors += [rex.ChunkedEncodingError, rex.ConnectionError,
                   rex.ReadTimeout, rex.Timeout]
    except Exception:
        pass
    try:
        import urllib3.exceptions as uex
        errors += [uex.ProtocolError, uex.IncompleteRead, uex.ReadTimeoutError]
    except Exception:
        pass
    RETRYABLE_ERRORS = tuple(dict.fromkeys(errors))
    return RETRYABLE_ERRORS


def _stream_once(
    source: str,
    state: ScanState,
    chunk_size: int,
    timeout: int,
    max_bytes: int,
    origin: int,
) -> Iterator[Dict[str, Any]]:
    """One pass over the source, starting at `state.bytes_consumed`."""
    start = state.bytes_consumed
    path = Path(source)

    if path.exists():
        handle = path.open("rb")
        handle.seek(start)
        if not state.total_bytes:
            state.total_bytes = path.stat().st_size
        chunks: Iterable[bytes] = iter(lambda: handle.read(chunk_size), b"")
        closer = handle.close
    else:
        import requests

        headers = {"Range": "bytes=%d-" % start} if start else {}
        response = requests.get(source, headers=headers, stream=True, timeout=timeout)
        response.raise_for_status()
        if start and response.status_code != 206:
            raise RuntimeError(
                "resume requested at byte %d but the server ignored the Range "
                "header (status %d). Delete the scan state to start over, or "
                "download the file first and pass --source <path>."
                % (start, response.status_code)
            )
        if not state.total_bytes:
            state.total_bytes = int(response.headers.get("Content-Length", 0)) + start
        chunks = response.iter_content(chunk_size=chunk_size)
        closer = response.close

    def decode(raw: bytes, at: int) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "could not parse a record at byte %d. Recorded offsets are always "
                "line boundaries, so this means the checkpoint no longer matches "
                "the source (a stale scan_state.json, or a different copy of the "
                "file). Delete scan_state.json and pages.jsonl to start over. "
                "[%s]" % (at, exc)
            ) from exc

    offset = start
    buffer = b""
    try:
        for chunk in chunks:
            if not chunk:
                continue
            buffer += chunk
            while True:
                newline = buffer.find(b"\n")
                if newline < 0:
                    break
                line = buffer[:newline]
                buffer = buffer[newline + 1:]
                offset += newline + 1
                if line.strip():
                    record = decode(line, offset)
                    state.bytes_consumed = offset
                    yield record
                # checked per line, not per chunk: a 1 MB chunk would otherwise
                # make max_bytes mean "read everything"
                if max_bytes and offset - origin >= max_bytes:
                    return
        if buffer.strip():
            offset += len(buffer)
            record = decode(buffer, offset)
            state.bytes_consumed = offset
            yield record
    finally:
        closer()


def iter_knowledge_source(
    source: str,
    state: Optional[ScanState] = None,
    chunk_size: int = 1 << 20,
    timeout: int = 120,
    max_bytes: int = 0,
    max_retries: int = 12,
    backoff: float = 3.0,
    on_retry: Optional[Any] = None,
) -> Iterator[Dict[str, Any]]:
    """Stream the KILT knowledge source, one page per line, resuming on faults.

    `source` is either the URL or a path to an already-downloaded copy.

    Lines are assembled from raw byte chunks rather than read with
    `iter_lines()` so a byte offset can be recorded at every line boundary.
    That offset is what makes both kinds of interruption cheap: a dropped
    connection is retried in place with `Range: bytes=<offset>-`, and a
    Ctrl-C resumes from the checkpoint on the next run.

    Nothing is written to disk: ~4k pages are kept out of ~5.9M, so there is
    no reason to land the whole knowledge source on a laptop.
    """
    state = state if state is not None else ScanState()
    origin = state.bytes_consumed
    attempt = 0

    while True:
        try:
            for record in _stream_once(source, state, chunk_size, timeout, max_bytes, origin):
                yield record
            return
        except _retryable_errors() as exc:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(
                    "gave up after %d reconnects at byte %d of %d. Re-run with "
                    "--resume to continue from here - nothing already found is lost."
                    % (max_retries, state.bytes_consumed, state.total_bytes)
                ) from exc
            wait = min(backoff * attempt, 60.0)
            message = ("connection lost at %.1f GiB (%s); reconnecting in %.0fs "
                       "[attempt %d/%d]" % (state.bytes_consumed / 1024 ** 3,
                                            type(exc).__name__, wait, attempt, max_retries))
            if on_retry is not None:
                on_retry(message)
            else:
                print("\n  " + message, flush=True)
            time.sleep(wait)


def iter_wikipedia(
    dataset: str,
    split: str = "full",
    streaming: bool = True,
    cache_dir: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Legacy path through `datasets`.

    Kept for the day a parquet conversion of the knowledge source appears.
    Today it raises: facebook/kilt_wikipedia is a loading script, and
    datasets >=4 refuses scripts outright.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, streaming=streaming, cache_dir=cache_dir)
    for record in ds:
        yield record


def scan_for_pages(
    records: Iterable[Dict[str, Any]],
    state: ScanState,
    out_path: Path,
    checkpoint_every: int = 50_000,
    state_path: Optional[Path] = None,
    drop_title_paragraph: bool = True,
    progress: Optional[Any] = None,
    checkpoint_bytes: int = 64 << 20,
) -> ScanState:
    """Walk wikipedia records, append gold pages to `out_path`, checkpoint.

    Appends: an interrupted run keeps what it found. Pages already found in a
    previous run are skipped, so a resumed run only writes new ones.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint_bytes = state.bytes_consumed

    # try/finally, not a bare loop: an interrupted 35 GB pass must leave a
    # checkpoint behind, or the next run starts from zero. The exception still
    # propagates -- it is just not allowed to eat the progress.
    with out_path.open("a", encoding="utf-8") as fh:
        try:
            for record in records:
                state.n_seen += 1
                wid = record.get("wikipedia_id")
                if wid is not None:
                    wid = str(wid)
                    if wid in state.needed and wid not in state.found:
                        page = parse_page(record, drop_title_paragraph=drop_title_paragraph)
                        if page.text.strip():
                            fh.write(json.dumps(page.to_dict(), ensure_ascii=False) + "\n")
                            state.found.add(wid)
                            if progress is not None:
                                progress.update(1)

                due = state.n_seen % checkpoint_every == 0
                if checkpoint_bytes and state.bytes_consumed - last_checkpoint_bytes >= checkpoint_bytes:
                    due = True
                if state_path and due:
                    fh.flush()
                    save_scan_state(state_path, state)
                    last_checkpoint_bytes = state.bytes_consumed
                if not state.missing:
                    break
        finally:
            fh.flush()
            if state_path:
                save_scan_state(state_path, state)
    return state


# ---------------------------------------------------------------------------
# Synthetic corpus: develop steps 3-5 before the real download finishes
# ---------------------------------------------------------------------------

_TOPIC_WORDS = [
    "geology", "opera", "cricket", "botany", "railway", "cinema", "astronomy",
    "cuisine", "heraldry", "linguistics", "volcano", "typography",
]


def synthetic_corpus(
    n_pages: int = 60,
    n_questions: int = 40,
    paragraphs_per_page: int = 8,
    sentences_per_paragraph: int = 6,
    seed: int = 13,
):
    """A miniature stand-in for KILT with the same record shapes.

    Every question's answer is a unique token planted in exactly one page, so
    answer_recall and page_hit are both checkable by construction. Enough to
    exercise chunking, indexing, retrieval, generation and analysis end to end
    before committing to a multi-hour download.
    """
    import random

    rng = random.Random(seed)
    pages: List[PageRecord] = []
    answers: Dict[str, str] = {}

    for i in range(n_pages):
        topic = _TOPIC_WORDS[i % len(_TOPIC_WORDS)]
        answer_token = f"Zyrex{i:04d}"
        answers[str(i)] = answer_token
        paras = []
        for p in range(paragraphs_per_page):
            sents = []
            for s in range(sentences_per_paragraph):
                filler = " ".join(rng.choice(_TOPIC_WORDS) for _ in range(rng.randint(6, 14)))
                sents.append(f"The {topic} record {p}-{s} notes that {filler}.")
            if p == paragraphs_per_page // 2:
                sents.insert(
                    rng.randrange(len(sents)),
                    f"The official designation of {topic} entry {i} is {answer_token}.",
                )
            paras.append(" ".join(sents))
        pages.append(
            PageRecord(doc_id=str(i), title=f"{topic.title()} entry {i}", text="\n\n".join(paras))
        )

    questions: List[QuestionRecord] = []
    for q in range(n_questions):
        page_idx = rng.randrange(n_pages)
        topic = _TOPIC_WORDS[page_idx % len(_TOPIC_WORDS)]
        questions.append(
            QuestionRecord(
                question_id=f"synth-{q}",
                question=f"what is the official designation of {topic} entry {page_idx}?",
                answers=[answers[str(page_idx)]],
                gold_page_ids=[str(page_idx)],
                split="test",
            )
        )
    return pages, questions
