#!/usr/bin/env python3
"""REPL-EMB-2.2 / 2.7 offline retrieval-eval harness (recall legs + statistics). CODE ONLY.

This file starts no process, loads no model on its own and downloads nothing. Dense arms are
reached through caller-owned llama-servers (``--arm-url ARM=URL``) or in-process static models
(``--model-path ARM=DIR``). It reuses, unforked, the shipped ``context_search`` module (BM25Index,
``chunk_sections``, ``ContextSearchIndex``, ``query_text``, ``DenseResult``, the ``rrf_fuse`` call)
through ``repl_emb_arms.load_context_search()`` and the ``repl_emb_arms`` specs/adapters.

Subcommands (all paths are CLI args; formats are fixed here):

  freeze    label queries (``queries-labelled.jsonl``), build the probe45 corpus listing and write
            ``FROZEN-AT-LAUNCH.sha256`` (sha256sum format) + ``inputs.json``; prints every digest
  run       one or more arms -> ``<out>/perquery-<variant>.jsonl`` (resumable), ``armmeta-*.json``,
            ``config-echo.json``, ``invocations.jsonl``; verifies the freeze first
  analyze   per-query files (+ optional speed JSON) -> ``analysis.json`` + ``analysis.md``
  dry-run   everything above on a synthetic bundle with fake embedders, no network (``--dry-run``)

Per-query input rows (JSONL): ``id, stratum, text, code_intent (bool|null), gold {path,
start_line,end_line} | {path, ranges}, bundle`` plus pass-through fields (``answer_named_verbatim``,
``lexical_overlap``, ``follow_kind``...). Variants written per arm id ``X``: ``grep``, ``bm25``,
``X.dense``, ``X.hybrid``, ``X.whybrid`` (static arms only, reported / non-deciding), ``colgrep``.

Speed-block JSON for the selection rule: ``{"arms": {"<arm>.hybrid": {"cpu_s_per_query": f,
"p95_ms": f, "p50_ms": f, "index_build_s": f}}}`` (a bare arm id keys its hybrid variant;
``query_p95_ms`` is accepted for ``p95_ms``, the metric name the speed-block capture uses).

PREDECLARED INTERPRETATIONS (frozen before any arm runs; echoed into the run's config):

@@INTERPRETATIONS@@

Operator ruling: the FP32 sentence-transformers EmbeddingGemma 2 arm is DROPPED (identical to BF16
at half the speed); no ST-FP32 recall arm exists here. Semble is a reported-only diagnostic and is
NOT implemented: the run records ``semble: not_run (tool not available in the isolated venv)``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
# ``python -I`` does not put the script directory on sys.path, and ``ContextSearchIndex.search``
# imports ``src.trace.navigation.rrf_fuse``: both roots are inserted explicitly.
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import repl_emb_arms as arms  # noqa: E402
import repl_emb_stats as stats  # noqa: E402

Pointer = tuple[str, int, int]

# ───────────────────────────────────────────────────────────────────────────── constants

TOP_K = 10
PRIMARY_K = stats.PRIMARY_K
KS = stats.KS
OFFLINE_INDEX_TIMEOUT_S = 1e9
W_DENSE, W_BM25, W_RRF_K = 0.75, 0.25, 60

DECISION_GROUPS = ("probe45", "spill_follow")
TRIVIAL_GROUP = "trivial_head"
REGISTERED_DENSE = ("bge-large", "granite-97m-r2")
REGISTERED_HYBRIDS = tuple(f"{a}.hybrid" for a in REGISTERED_DENSE)
LEXICAL_ARMS = ("grep", "bm25")

CODE_INTENT_REGEX = r"`|\b[a-z]+_[a-z0-9_]+\b|\b[A-Z][a-z0-9]+[A-Z]\w*\b|\w+\(\)|\b\w+\.\w+\("
_CODE_INTENT_RE = re.compile(CODE_INTENT_REGEX)

_STOPWORD_TEXT = """
a about above after again all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has
have having he her here hers him his how i if in into is it its itself just me more most my no
nor not of off on once only or other our out over own same she should so some such than that the
their them then there these they this those through to too under until up very was we were what
when where which while who whom why will with would you your yours
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())
STOPWORDS_SHA256 = hashlib.sha256("\n".join(sorted(STOPWORDS)).encode()).hexdigest()

#: probe45 repo -> pinned checkout directory name and package root inside it.
PROBE_REPOS: dict[str, dict[str, str]] = {
    "mwaskom/seaborn": {"checkout": "seaborn", "package_root": "seaborn"},
    "pylint-dev/pylint": {"checkout": "pylint", "package_root": "pylint"},
    "pytest-dev/pytest": {"checkout": "pytest", "package_root": "src/_pytest"},
}
DEFAULT_PROBE_CHECKOUTS = "/mnt/raid0/llm/tmp/repl-emb-25-20261020"
DEFAULT_ARM_MANIFEST = (
    REPO_ROOT / "data/repl_retrieval_eval/arm-manifest-20261020/arm_manifest.json"
)

COLGREP_BIN = "/mnt/raid0/llm/UTILS/bin/colgrep-1.2.0"
COLGREP_ALPHA = "0.95"
COLGREP_ENV = {"NEXT_PLAID_FORCE_CPU": "1"}
#: Pinned sha256 of the shipped colgrep-1.2.0 binary (src/repl_environment/code_search.py).
COLGREP_EXPECTED_SHA256 = "833e52aa6c40d090142fa132e3c75d3e792a4707474682a2496e3471f646f956"

BGE_REPO = "BAAI/bge-large-en-v1.5"
BGE_GGUF_SHA256 = "3379a0e9cea28fc6d7136df8ea7a88ef99ccce5963b9a6f7af9609997be762e3"
GRANITE_REPO = "ibm-granite/granite-embedding-97m-multilingual-r2"
GRANITE_GGUF_SHA256 = "52e975809b64027c745ddd6815a0514a83a17d462b80dc386f2df689b2b38bed"

EVAL_INTERPRETATIONS: dict[str, str] = {
    "f_pointer_hit": (
        "(f) A chunk pointer is (section, start_line, end_line): start = chunk.line; end = line + the "
        "number of newlines inside chunk.text after removing exactly ONE trailing newline (a chunk is "
        "a run of whole lines, so trailing blank lines stay counted). A pointer hits a query iff it "
        "has the same section (path) and overlaps a gold span by >= 1 line."
    ),
    "g_probe45_bundles": (
        "(g) probe45 bundles, per repo: every *.py under the package root (seaborn/, pylint/, "
        "src/_pytest/) plus the gold files if outside it, read from the pinned checkout (HEAD must "
        "equal the rows' commit); section = repo-relative path; the shipped chunk_sections(..., 800); "
        "the chunk count must be <= 4096 (asserted, recorded)."
    ),
    "h_code_intent": (
        "(h) code_intent (applied only where the input value is null; blind to every arm's output) is "
        "True iff the query text matches the regex in CODE_INTENT_REGEX; the labelled file records "
        "the source ('input' | 'regex') of every label."
    ),
    "i_dense_hybrid": (
        "(i) Dense-only and hybrid share ONE query embedding per query: the vector the shipped "
        "ContextSearchIndex.search embeds (captured by a delegating wrapper). Hybrid = the hits of "
        "idx.search(query, 10, section=None, embedder=adapter); dense-only = "
        "idx._dense_ranking(vector, None)[:10]. A search whose meta mode is not 'hybrid' (dense "
        "failure, invalid or degenerate vector, timeout, build failure) is a MISS for hybrid AND "
        "dense-only and increments n_failed."
    ),
    "j_groups": (
        "(j) Groups: probe45 rows = group probe45; every row of the miner's queries-trace.jsonl = "
        "group spill_follow; an optional trivial-head file = group trivial_head, scored but reported "
        "outside the decision. A row with no gold span is scored:false and excluded everywhere."
    ),
    "k_offline_budget": (
        "(k) The shipped index-build budget is disabled for the offline eval (INDEX_TIMEOUT_MIN_S = "
        "INDEX_TIMEOUT_MAX_S = 1e9) so it cannot censor an embedder; budget exposure belongs to the "
        "speed block. Every cost number produced here is flagged contended_not_for_selection."
    ),
    "l_weighted": (
        "(l) weighted_hybrid (static arms only; reported, non-deciding): score = 0.75/(60+rank_dense) "
        "+ 0.25/(60+rank_bm25) over the same two 64-candidate rankings; ties break on str(id) as "
        "rrf_fuse does; rrf_fuse itself is untouched."
    ),
    "m_colgrep": (
        "(m) ColGREP is a registered arm on probe45 only (code-only, tree-sitter): spill-follow "
        "bundles are .txt, so it is not_applicable there and is excluded from its n. It is reported "
        "against L on probe45 and takes no part in the kill rule."
    ),
}
PREDECLARED: dict[str, str] = {**stats.PREDECLARED_INTERPRETATIONS, **EVAL_INTERPRETATIONS}

__doc__ = (__doc__ or "").replace(
    "@@INTERPRETATIONS@@", "\n".join("  " + t for t in PREDECLARED.values())
)


# ───────────────────────────────────────────────────────────────────────────── small helpers


def _cs() -> Any:
    return arms.load_context_search()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def dumps(obj: Any, **kw: Any) -> str:
    return json.dumps(obj, default=_json_default, ensure_ascii=False, **kw)


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@contextlib.contextmanager
def offline_index_budget() -> Iterator[None]:
    """Interpretation (k): the shipped build budget must not censor an embedder (restored on exit)."""
    cs = _cs()
    prev = (cs.INDEX_TIMEOUT_MIN_S, cs.INDEX_TIMEOUT_MAX_S)
    cs.INDEX_TIMEOUT_MIN_S = cs.INDEX_TIMEOUT_MAX_S = OFFLINE_INDEX_TIMEOUT_S
    try:
        yield
    finally:
        cs.INDEX_TIMEOUT_MIN_S, cs.INDEX_TIMEOUT_MAX_S = prev


# ───────────────────────────────────────────────────────────────────────────── chunks, pointers


def chunk_line_range(chunk: Any) -> tuple[int, int]:
    """Inclusive line range of a shipped ``Chunk`` (interpretation (f))."""
    text = chunk.text
    body = text[:-1] if text.endswith("\n") else text
    return int(chunk.line), int(chunk.line) + body.count("\n")


def chunk_pointer(chunk: Any) -> Pointer:
    s, e = chunk_line_range(chunk)
    return (chunk.section, s, e)


# ───────────────────────────────────────────────────────────────────────────── labelling


def code_intent_label(text: str) -> bool:
    """Interpretation (h): deterministic, blind to arm outputs."""
    return bool(_CODE_INTENT_RE.search(text))


def gold_spans(gold: Mapping[str, Any] | None) -> list[Pointer]:
    """Normalise ``{path,start_line,end_line}`` | ``{path,ranges}`` to ``[(path, s, e), ...]``."""
    if not gold:
        return []
    path = gold.get("path")
    if path is None:
        raise ValueError(f"gold without a path: {gold!r}")
    if gold.get("ranges"):
        ranges = gold["ranges"]
    elif gold.get("start_line") is not None and gold.get("end_line") is not None:
        ranges = [[gold["start_line"], gold["end_line"]]]
    else:
        return []
    out: list[Pointer] = []
    for s, e in ranges:
        s, e = int(s), int(e)
        if e < s:
            raise ValueError(f"inverted gold range {(s, e)} for {path}")
        out.append((str(path), s, e))
    return out


def _labelled_row(row: Mapping[str, Any], group: str, bundle: str) -> dict[str, Any]:
    r = dict(row)
    for req in ("id", "text"):
        if not r.get(req):
            raise ValueError(f"query row without {req!r}: {r!r}")
    spans = gold_spans(r.get("gold"))
    ci = r.get("code_intent")
    if isinstance(ci, bool):
        src = "input"
    elif ci is None:
        ci, src = code_intent_label(r["text"]), "regex"
    else:
        raise ValueError(f"{r['id']}: code_intent must be bool|null, got {ci!r}")
    r.update(
        group=group,
        bundle=bundle,
        code_intent=bool(ci),
        code_intent_source=src,
        gold_spans=[list(s) for s in spans],
        scored=bool(spans),
    )
    r.setdefault("stratum", group)
    return r


def label_queries(
    probe_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    trivial_rows: Sequence[Mapping[str, Any]] = (),
    *,
    probe_repos: Mapping[str, Any] = PROBE_REPOS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The frozen query file's rows (probe45, then spill_follow, then trivial_head) + a summary."""
    out: list[dict[str, Any]] = []
    for r in probe_rows:
        repo = r.get("repo")
        if repo not in probe_repos:
            raise ValueError(f"probe row {r.get('id')!r}: unknown repo {repo!r}")
        out.append(_labelled_row(r, "probe45", f"probe45:{repo}"))
    for group, rows in (("spill_follow", trace_rows), (TRIVIAL_GROUP, trivial_rows)):
        for r in rows:
            bundle = r.get("bundle") or r.get("bundle_id")
            if not bundle:
                raise ValueError(f"{group} row {r.get('id')!r} has no bundle / bundle_id")
            out.append(_labelled_row(r, group, str(bundle)))
    ids = [r["id"] for r in out]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"duplicate query ids: {dup[:5]}")
    summary: dict[str, Any] = {
        "n_rows": len(out),
        "code_intent_regex": CODE_INTENT_REGEX,
        "by_group": {},
    }
    for g in sorted({r["group"] for r in out}):
        rows_g = [r for r in out if r["group"] == g]
        summary["by_group"][g] = {
            "n": len(rows_g),
            "scored": sum(r["scored"] for r in rows_g),
            "unscored": sum(not r["scored"] for r in rows_g),
            "code_intent_true": sum(r["code_intent"] for r in rows_g),
            "code_intent_from_regex": sum(r["code_intent_source"] == "regex" for r in rows_g),
        }
    summary["n_code_intent_true"] = sum(r["code_intent"] for r in out)
    summary["n_labelled_by_regex"] = sum(r["code_intent_source"] == "regex" for r in out)
    return out, summary


def write_labelled(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    text = "".join(dumps(r, sort_keys=True) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return sha256_bytes(text.encode("utf-8"))


# ───────────────────────────────────────────────────────────────────────────── bundles


@dataclasses.dataclass(frozen=True)
class Section:
    """Satisfies the shipped chunker's ``name`` / ``text`` section protocol."""

    name: str
    text: str


@dataclasses.dataclass
class Bundle:
    bundle_id: str
    sections: list[Section]
    files: list[dict[str, Any]]  # {name, path, sha256, bytes}
    commit: str | None = None

    def pairs(self) -> list[tuple[str, str]]:
        return sorted((f["name"], f["sha256"]) for f in self.files)


def _read_text(path: Path) -> tuple[str, str, int]:
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="replace"), sha256_bytes(raw), len(raw)


def git_head(path: str | Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    ).stdout.strip()


def build_probe_bundle(
    repo: str,
    checkouts: str | Path,
    gold_paths: Iterable[str],
    *,
    expected_commit: str | None = None,
    probe_repos: Mapping[str, Mapping[str, str]] = PROBE_REPOS,
    head_fn: Callable[[Path], str] = git_head,
) -> Bundle:
    """Interpretation (g)."""
    cfg = probe_repos[repo]
    root = Path(checkouts) / cfg["checkout"]
    if not root.is_dir():
        raise FileNotFoundError(f"pinned checkout for {repo} missing: {root}")
    commit = None
    if expected_commit is not None:
        commit = head_fn(root)
        if commit != expected_commit:
            raise RuntimeError(f"{repo}: checkout HEAD {commit} != query commit {expected_commit}")
    pkg = root / cfg["package_root"]
    rels = {p.relative_to(root).as_posix() for p in pkg.rglob("*.py") if p.is_file()}
    for g in gold_paths:
        if not (root / g).is_file():
            raise FileNotFoundError(f"{repo}: gold file {g} not found in the checkout")
        rels.add(g)
    sections, files = [], []
    for rel in sorted(rels):
        text, sha, nbytes = _read_text(root / rel)
        sections.append(Section(rel, text))
        files.append({"name": rel, "path": str(root / rel), "sha256": sha, "bytes": nbytes})
    return Bundle(f"probe45:{repo}", sections, files, commit)


def build_spill_bundle(bundle_id: str, entry: Mapping[str, Any]) -> Bundle:
    """A miner bundle: ``{files:[{path,sha256,bytes,chunks}], total_chunks}``; section = path."""
    sections, files = [], []
    for f in entry["files"]:
        text, sha, nbytes = _read_text(Path(f["path"]))
        if f.get("sha256") and sha != f["sha256"]:
            raise RuntimeError(
                f"{bundle_id}: {f['path']} sha256 {sha} != bundles.json {f['sha256']}"
            )
        if f.get("bytes") is not None and nbytes != int(f["bytes"]):
            raise RuntimeError(
                f"{bundle_id}: {f['path']} is {nbytes} bytes, bundles.json says {f['bytes']}"
            )
        sections.append(Section(f["path"], text))
        files.append(
            {
                "name": f["path"],
                "path": f["path"],
                "sha256": sha,
                "bytes": nbytes,
                "miner_chunks": f.get("chunks"),
            }
        )
    return Bundle(bundle_id, sections, files)


def load_bundles_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("bundles"), dict):
        return data["bundles"]
    return data


def corpus_listing(bundles: Iterable[Bundle]) -> str:
    """Sorted ``bundle \\t path \\t sha256`` lines; its sha256 is the corpus digest."""
    lines = sorted(f"{b.bundle_id}\t{n}\t{s}\n" for b in bundles for n, s in b.pairs())
    return "".join(lines)


def validate_gold(rows: Sequence[Mapping[str, Any]], store: "BundleStore") -> dict[str, list[str]]:
    """Every gold path must be a section of its query's bundle (else it can never be hit: error);
    a gold span starting past the end of its file is reported (warning), not an error."""
    errors: list[str] = []
    warnings: list[str] = []
    for q in rows:
        if not q["scored"]:
            continue
        texts = {sec.name: sec.text for sec in store.get(q["bundle"]).sections}
        for path, start, _end in q["gold_spans"]:
            if path not in texts:
                errors.append(
                    f"{q['id']}: gold path {path} is not a section of bundle {q['bundle']}"
                )
                continue
            t = texts[path]
            n_lines = t.count("\n") + (0 if (not t or t.endswith("\n")) else 1)
            if start > n_lines:
                warnings.append(
                    f"{q['id']}: gold starts at line {start} but {path} has {n_lines} lines"
                )
    return {"errors": errors, "warnings": warnings}


class BundleStore:
    """Lazy, cached bundle construction with the <= MAX_DENSE_CHUNKS assertion."""

    def __init__(self, factory: Callable[[str], Bundle]) -> None:
        self._factory = factory
        self._cache: dict[str, Bundle] = {}
        self.summary: dict[str, dict[str, Any]] = {}

    def get(self, bundle_id: str) -> Bundle:
        if bundle_id not in self._cache:
            b = self._factory(bundle_id)
            cs = _cs()
            chunks = cs.chunk_sections(b.sections, cs.CHUNK_MAX_CHARS)
            n = len(chunks)
            assert n <= cs.MAX_DENSE_CHUNKS, (
                f"bundle {bundle_id}: {n} chunks > MAX_DENSE_CHUNKS {cs.MAX_DENSE_CHUNKS}"
            )
            self._cache[bundle_id] = b
            per_file: dict[str, int] = {}
            for c in chunks:
                per_file[c.section] = per_file.get(c.section, 0) + 1
            mismatched = sorted(
                f["name"]
                for f in b.files
                if f.get("miner_chunks") is not None
                and per_file.get(f["name"], 0) != int(f["miner_chunks"])
            )
            self.summary[bundle_id] = {
                "n_files": len(b.files),
                "n_chunks": n,
                "commit": b.commit,
                "corpus_sha256": sha256_bytes(corpus_listing([b]).encode()),
                "chunk_count_mismatch_vs_miner": mismatched,
            }
        return self._cache[bundle_id]


def make_bundle_store(
    queries: Sequence[Mapping[str, Any]],
    *,
    bundles_json: str | Path | None,
    probe_checkouts: str | Path,
    probe_repos: Mapping[str, Mapping[str, str]] = PROBE_REPOS,
    head_fn: Callable[[Path], str] = git_head,
) -> BundleStore:
    gold_paths: dict[str, set[str]] = {}
    commits: dict[str, str] = {}
    for q in queries:
        if q["group"] == "probe45":
            gold_paths.setdefault(q["bundle"], set()).update(g[0] for g in q["gold_spans"])
            if q.get("commit"):
                commits.setdefault(q["bundle"], q["commit"])
    spill = load_bundles_json(bundles_json)

    def factory(bid: str) -> Bundle:
        if bid.startswith("probe45:"):
            return build_probe_bundle(
                bid.split(":", 1)[1],
                probe_checkouts,
                sorted(gold_paths.get(bid, ())),
                expected_commit=commits.get(bid),
                probe_repos=probe_repos,
                head_fn=head_fn,
            )
        if bid not in spill:
            raise KeyError(f"bundle {bid!r} not in bundles.json")
        return build_spill_bundle(bid, spill[bid])

    return BundleStore(factory)


# ───────────────────────────────────────────────────────────────────────────── lexical arms


def grep_terms(query: str) -> list[str]:
    """Distinct lower-case shipped-``tokenize`` tokens of length >= 3, minus the fixed stopwords."""
    cs = _cs()
    return list(dict.fromkeys(t for t in cs.tokenize(query) if len(t) >= 3 and t not in STOPWORDS))


def grep_rank(lowered_chunks: Sequence[str], terms: Sequence[str], k: int = TOP_K) -> list[int]:
    """Chunks scored by the number of distinct terms occurring as substrings; zero-score dropped."""
    scored = []
    for i, text in enumerate(lowered_chunks):
        s = sum(1 for t in terms if t in text)
        if s:
            scored.append((-s, i))
    scored.sort()
    return [i for _, i in scored[:k]]


def bm25_rank(idx: Any, query: str, k: int = TOP_K) -> list[int]:
    return [i for i, _ in idx.bm25.scores(query, None)[:k]]


def weighted_rrf(
    rankings: Sequence[tuple[float, Sequence[Any]]], *, k: int = W_RRF_K, limit: int = TOP_K
) -> list[Any]:
    """Non-deciding weighted RRF: sum of w / (k + rank); ties break on ``str(id)`` like rrf_fuse."""
    scores: dict[Any, float] = {}
    for w, ids in rankings:
        for rank, i in enumerate(ids, start=1):
            scores[i] = scores.get(i, 0.0) + w / (k + rank)
    return sorted(scores, key=lambda i: (-scores[i], str(i)))[:limit]


# ───────────────────────────────────────────────────────────────────────────── dense arms


@dataclasses.dataclass(frozen=True)
class DenseArmDef:
    arm_id: str
    kind: str  # "llama" (caller-owned llama-server) | "static" (in-process model2vec)
    registered: bool
    weighted: bool = False
    autodim: bool = False  # read the real dimension from the first response


DENSE_ARMS: dict[str, DenseArmDef] = {
    d.arm_id: d
    for d in (
        DenseArmDef("bge-large", "llama", True),
        DenseArmDef("granite-97m-r2", "llama", True, autodim=True),
        DenseArmDef("eg2-gguf-bf16-768", "llama", False),
        DenseArmDef("eg2-gguf-bf16-256", "llama", False),
        DenseArmDef("potion-code-16m-v2", "static", False, weighted=True),
        DenseArmDef("potion-base-8m", "static", False, weighted=True),
    )
}


@dataclasses.dataclass(frozen=True)
class MarkerArmSpec(arms.ArmSpec):
    """An ArmSpec whose ``model_id`` carries the model repo name, so the shipped ``query_text``
    recognises ``bge-large-en-v1.5`` in it and adds the BGE query instruction (documents are sent
    unprefixed by the shipped index)."""

    def model_id(self) -> str:
        return f"{self.model_repo}#{self.index_identity()['digest'][:12]}"


def bge_query_prefix() -> str:
    for marker, prefix in _cs()._QUERY_PREFIXES:
        if marker == "bge-large-en-v1.5":
            return prefix
    raise RuntimeError("shipped _QUERY_PREFIXES has no bge-large-en-v1.5 entry")


def eval_specs(llama_runtime: str, m2v_runtime: str) -> dict[str, arms.ArmSpec]:
    """The 2.2/2.7 dense arms: EG2 GGUF BF16 @768/@256 and the two potion arms come from
    ``repl_emb_arms.arm_specs`` unchanged; BGE and Granite are registered here."""
    base = {s.arm_id: s for s in arms.arm_specs("n/a", llama_runtime, m2v_runtime)}
    out: dict[str, arms.ArmSpec] = {
        k: base[k]
        for k in (
            "eg2-gguf-bf16-768",
            "eg2-gguf-bf16-256",
            "potion-code-16m-v2",
            "potion-base-8m",
        )
    }
    out["bge-large"] = MarkerArmSpec(
        arm_id="bge-large",
        family="gguf",
        model_repo=BGE_REPO,
        revision=f"gguf-f16:sha256:{BGE_GGUF_SHA256}",
        dim=1024,
        native_dim=1024,
        dtype="f16",
        runtime=llama_runtime,
        pooling="per GGUF metadata (CLS)",
        normalise="l2",
        notes="query instruction comes from the shipped query_text (model_id marker); documents unprefixed",
    )
    out["granite-97m-r2"] = arms.ArmSpec(
        arm_id="granite-97m-r2",
        family="gguf",
        model_repo=GRANITE_REPO,
        revision=f"gguf-f16:sha256:{GRANITE_GGUF_SHA256}",
        dim=384,
        native_dim=384,
        dtype="f16",
        runtime=llama_runtime,
        pooling="per GGUF metadata",
        normalise="l2",
        notes="no prefixes; the true dimension is read from the first response",
    )
    return {k: out[k] for k in DENSE_ARMS}


def validate_registry(specs: Mapping[str, arms.ArmSpec]) -> None:
    """Prove the prefix mechanics the brief fixes: BGE queries are instructed, nothing else is."""
    cs = _cs()
    bge = specs["bge-large"]
    assert "bge-large-en-v1.5" in bge.model_id().lower()
    assert cs.query_text(bge.model_id(), "q") == bge_query_prefix() + "q"
    for aid, spec in specs.items():
        if aid != "bge-large":
            assert cs.query_text(spec.model_id(), "q") == "q", f"{aid} would get a BGE prefix"
    assert bge.query_prompt == "" and bge.doc_prompt == ""
    assert specs["granite-97m-r2"].query_prompt == specs["granite-97m-r2"].doc_prompt == ""


def runtimes_from_manifest(manifest: Mapping[str, Any]) -> tuple[str, str]:
    by = {a["arm_id"]: a for a in manifest["arms"]}
    return by["eg2-gguf-bf16-768"]["runtime"], by["potion-code-16m-v2"]["runtime"]


def check_against_manifest(
    specs: Mapping[str, arms.ArmSpec], manifest: Mapping[str, Any]
) -> list[str]:
    """Arms the manifest froze must keep their identity (model_id digest)."""
    problems = []
    for a in manifest["arms"]:
        s = specs.get(a["arm_id"])
        if s is not None and s.model_id() != a["dense_embedder_model_id"]:
            problems.append(
                f"{a['arm_id']}: {s.model_id()} != manifest {a['dense_embedder_model_id']}"
            )
    return problems


def resolve_llama_adapter(
    defn: DenseArmDef, spec: arms.ArmSpec, url: str, *, timeout_s: float
) -> arms.LlamaServerAdapter:
    """Probe the served width; Granite takes the real dimension, the others must match the spec."""
    ad = arms.LlamaServerAdapter(spec, url, timeout_s=timeout_s)
    width = int(np.asarray(ad._encode(["dimension probe"])).shape[-1])
    if width == spec.native_dim:
        return ad
    if defn.autodim:
        spec = dataclasses.replace(spec, dim=width, native_dim=width)
        return arms.LlamaServerAdapter(spec, url, timeout_s=timeout_s)
    raise RuntimeError(
        f"{spec.arm_id}: server at {url} returns {width}-d vectors, spec says {spec.native_dim}"
    )


class QueryTap:
    """Delegates to an embedder and keeps the result of the last QUERY call (the vector the
    shipped ``search`` embedded), so dense-only reuses it instead of embedding twice."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.last: tuple[Any, float] | None = None

    def embed(self, texts: Sequence[str], *, timeout_s: float, admission_wait_s: float) -> Any:
        t0 = time.perf_counter()
        res = self.inner.embed(texts, timeout_s=timeout_s, admission_wait_s=admission_wait_s)
        if admission_wait_s == _cs().QUERY_ADMISSION_WAIT_S:
            self.last = (res, (time.perf_counter() - t0) * 1000.0)
        return res


def build_dense(idx: Any, adapter: Any) -> dict[str, Any]:
    """Build the dense side with the shipped ``ensure_dense`` (retry policy included)."""
    cs = _cs()
    no_retry = set(getattr(cs, "_NO_RETRY", ())) | {"too_large", "empty", "disabled"}
    n_batches = math.ceil(max(1, len(idx.chunks)) / cs.DENSE_BATCH_CHUNKS)
    t0, c0 = time.perf_counter(), time.process_time()
    reason, attempts = None, 0
    for _ in range(cs.MAX_DENSE_ATTEMPTS + n_batches + 1):
        reason = idx.ensure_dense(adapter)
        attempts += 1
        if reason is None or reason in no_retry or idx.dense_failures >= cs.MAX_DENSE_ATTEMPTS:
            break
    return {
        "ok": reason is None and idx.vectors is not None,
        "reason": reason,
        "attempts": attempts,
        "n_chunks": len(idx.chunks),
        "dim": int(idx.vectors.shape[1]) if idx.vectors is not None else None,
        "index_build_wall_s": round(time.perf_counter() - t0, 4),
        "index_build_cpu_s": round(time.process_time() - c0, 4),
        "contended_not_for_selection": True,
    }


def _row(
    variant: str, q: Mapping[str, Any], pointers: Sequence[Pointer] | None, ms: float, **extra: Any
) -> dict[str, Any]:
    failed = pointers is None
    return {
        "variant": variant,
        "qid": q["id"],
        "group": q["group"],
        "stratum": q.get("stratum"),
        "bundle": q["bundle"],
        "code_intent": q["code_intent"],
        "pointers": [] if failed else [list(p) for p in pointers],  # type: ignore[union-attr]
        "failed": failed,
        "latency_ms": round(ms, 3),
        **extra,
    }


def dense_query(
    idx: Any, tap: QueryTap, adapter: Any, q: Mapping[str, Any], *, weighted: bool
) -> dict[str, tuple[list[Pointer] | None, float, dict[str, Any]]]:
    """One query on one dense arm -> {'dense'|'hybrid'|'whybrid': (pointers|None, ms, extra)}."""
    cs = _cs()
    query = q["text"]
    tap.last = None
    intent = getattr(adapter, "query_intent", None)
    t0 = time.perf_counter()
    with intent(bool(q["code_intent"])) if intent else contextlib.nullcontext():
        hits, meta = idx.search(query, TOP_K, section=None, embedder=tap)
    hybrid_ms = (time.perf_counter() - t0) * 1000.0
    by_key = {(c.section, c.start): c for c in idx.chunks}
    ok = meta.get("mode") == cs.MODE_HYBRID and tap.last is not None
    extra = {"mode": meta.get("mode"), "reason": meta.get("reason")}
    if not ok:
        fail = {**extra, "reason": extra["reason"] or "no_query_vector"}
        return {v: (None, hybrid_ms, fail) for v in ("dense", "hybrid", "whybrid")}
    hybrid = [chunk_pointer(by_key[(h["section"], h["start"])]) for h in hits]
    res, embed_ms = tap.last
    t1 = time.perf_counter()
    d_rank = [r["id"] for r in idx._dense_ranking(res.vectors[0], None)]
    dense = [chunk_pointer(idx.chunks[i]) for i in d_rank[:TOP_K]]
    out = {
        "dense": (dense, embed_ms + (time.perf_counter() - t1) * 1000.0, extra),
        "hybrid": (hybrid, hybrid_ms, extra),
    }
    if weighted:
        t2 = time.perf_counter()
        bm = [i for i, _ in idx.bm25.scores(query, None)[: cs.CANDIDATES_PER_RANKER]]
        w = weighted_rrf([(W_DENSE, d_rank), (W_BM25, bm)], k=W_RRF_K, limit=TOP_K)
        out["whybrid"] = (
            [chunk_pointer(idx.chunks[i]) for i in w],
            embed_ms + (time.perf_counter() - t2) * 1000.0,
            extra,
        )
    return out


# ───────────────────────────────────────────────────────────────────────────── result store


class ResultStore:
    """``<out>/perquery-<variant>.jsonl``: append-only, one row per (variant, query), last wins."""

    def __init__(self, out: str | Path) -> None:
        self.out = Path(out)

    def path(self, variant: str) -> Path:
        return self.out / f"perquery-{variant}.jsonl"

    def variants(self) -> list[str]:
        return sorted(
            p.name[len("perquery-") : -len(".jsonl")] for p in self.out.glob("perquery-*.jsonl")
        )

    def rows(self, variant: str) -> dict[str, dict[str, Any]]:
        p = self.path(variant)
        out: dict[str, dict[str, Any]] = {}
        if not p.exists():
            return out
        lines = p.read_text(encoding="utf-8").split("\n")
        for n, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if n >= len(lines) - 2:  # a torn final line from an interrupted run
                    continue
                raise
            out[row["qid"]] = row
        return out

    def finished(self, variant: str, retry_failed: bool = False) -> set[str]:
        return {
            qid for qid, r in self.rows(variant).items() if not (retry_failed and r.get("failed"))
        }

    def append(self, variant: str, row: Mapping[str, Any]) -> None:
        p = self.path(variant)
        if p.exists() and p.stat().st_size:
            raw = p.read_bytes()
            if not raw.endswith(
                b"\n"
            ):  # an interrupted write: drop the torn fragment, keep the file valid
                p.write_bytes(raw[: raw.rfind(b"\n") + 1])
        with open(p, "a", encoding="utf-8") as f:
            f.write(dumps(row, sort_keys=True) + "\n")
            f.flush()


class ArmMeta:
    """``<out>/armmeta-<arm>.json``: per-bundle index builds, adapter counters, invocations."""

    def __init__(self, out: Path, arm_id: str) -> None:
        self.path = out / f"armmeta-{arm_id}.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"arm_id": arm_id, "bundles": {}, "invocations": []}

    def save(self) -> None:
        write_json(self.path, self.data)


def _by_bundle(queries: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for q in queries:
        out.setdefault(q["bundle"], []).append(q)
    return out


def run_lexical_arm(
    arm: str,
    queries: Sequence[Mapping[str, Any]],
    bundle_store: BundleStore,
    store: ResultStore,
    *,
    retry_failed: bool = False,
) -> dict[str, Any]:
    assert arm in LEXICAL_ARMS
    cs = _cs()
    done = store.finished(arm, retry_failed)
    n_run = 0
    for bid, qs in _by_bundle(queries).items():
        todo = [q for q in qs if q["id"] not in done]
        if not todo:
            continue
        idx = cs.ContextSearchIndex(bundle_store.get(bid).sections)
        lowered = [c.text.lower() for c in idx.chunks] if arm == "grep" else []
        for q in todo:
            t0 = time.perf_counter()
            ids = (
                grep_rank(lowered, grep_terms(q["text"]), TOP_K)
                if arm == "grep"
                else bm25_rank(idx, q["text"], TOP_K)
            )
            ms = (time.perf_counter() - t0) * 1000.0
            store.append(arm, _row(arm, q, [chunk_pointer(idx.chunks[i]) for i in ids], ms))
            n_run += 1
    return {"arm": arm, "n_run": n_run, "n_skipped": len(queries) - n_run}


def run_dense_arm(
    defn: DenseArmDef,
    adapter: Any,
    queries: Sequence[Mapping[str, Any]],
    bundle_store: BundleStore,
    store: ResultStore,
    meta: ArmMeta,
    *,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """ONE ContextSearchIndex per (bundle, arm); resumable per (variant, query)."""
    cs = _cs()
    aid = defn.arm_id
    variants = {"dense": f"{aid}.dense", "hybrid": f"{aid}.hybrid"}
    if defn.weighted:
        variants["whybrid"] = f"{aid}.whybrid"
    done = {k: store.finished(v, retry_failed) for k, v in variants.items()}
    n_run = n_failed = 0
    with offline_index_budget():
        for bid, qs in _by_bundle(queries).items():
            todo = [q for q in qs if any(q["id"] not in done[k] for k in variants)]
            if not todo:
                continue
            idx = cs.ContextSearchIndex(bundle_store.get(bid).sections)
            tap = QueryTap(adapter)
            build = build_dense(idx, adapter)
            meta.data["bundles"][bid] = build
            meta.save()
            for q in todo:
                if build["ok"]:
                    res = dense_query(idx, tap, adapter, q, weighted=defn.weighted)
                else:
                    fail = {"mode": None, "reason": f"index_build:{build['reason']}"}
                    res = {k: (None, 0.0, fail) for k in variants}
                for k, v in variants.items():
                    if q["id"] in done[k]:
                        continue
                    ptrs, ms, extra = res[k]
                    store.append(v, _row(v, q, ptrs, ms, **extra))
                    n_failed += ptrs is None and k == "hybrid"
                n_run += 1
    calls = getattr(adapter, "calls", [])
    meta.data["invocations"].append(
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_queries_run": n_run,
            "n_failed_hybrid": int(n_failed),
            "adapter_nonfinite_rows": int(getattr(adapter, "nonfinite_rows", 0)),
            "adapter_calls": {
                side: {
                    "calls": sum(1 for c in calls if c["side"] == side),
                    "texts": sum(c["n"] for c in calls if c["side"] == side),
                    "seconds": round(sum(c["s"] for c in calls if c["side"] == side), 3),
                }
                for side in ("document", "query")
            },
            "contended_not_for_selection": True,
        }
    )
    meta.save()
    return {"arm": aid, "n_run": n_run, "n_failed_hybrid": int(n_failed)}


# ───────────────────────────────────────────────────────────────────────────── ColGREP


def colgrep_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    return {**(os.environ if base is None else base), **COLGREP_ENV}


def colgrep_init_cmd(bin_path: str, workdir: str | Path) -> list[str]:
    return [bin_path, "init", "-y", str(workdir)]


def colgrep_search_cmd(
    bin_path: str, query: str, workdir: str | Path, k: int = TOP_K, alpha: str = COLGREP_ALPHA
) -> list[str]:
    """The shipped ``_colgrep_search`` invocation (``-k min(limit, 20) --alpha 0.95 --json PATH``)
    plus ``-y`` so an unattended run never blocks on the large-codebase prompt."""
    return [
        bin_path,
        "search",
        query,
        "-k",
        str(min(k, 20)),
        "--alpha",
        alpha,
        "--json",
        "-y",
        str(workdir),
    ]


def parse_colgrep_json(stdout: str, workdir: str | Path, k: int = TOP_K) -> list[Pointer]:
    """Pointers from ColGREP JSON: ``[{unit:{file,line,end_line},...}]`` (shipped shape) or flat
    ``[{path,start_line,end_line}]``."""
    raw = json.loads(stdout) if stdout.strip() else []
    out: list[Pointer] = []
    for item in raw[:k]:
        unit = item.get("unit") if isinstance(item.get("unit"), dict) else item
        file = unit.get("file") or unit.get("path")
        start = unit.get("line", unit.get("start_line"))
        end = unit.get("end_line")
        if file is None or start is None or end is None:
            continue
        rel = os.path.relpath(file, workdir) if os.path.isabs(file) else os.path.normpath(file)
        out.append((Path(rel).as_posix(), int(start), int(end)))
    return out


def materialize_colgrep_dir(bundle: Bundle, workdir: Path) -> None:
    """Hard-link (copy as fallback) the bundle's files at their repo-relative paths."""
    for f in bundle.files:
        dst = workdir / f["name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if sha256_file(dst) == f["sha256"]:
                continue
            dst.unlink()
        try:
            os.link(f["path"], dst)
        except OSError:
            shutil.copy2(f["path"], dst)


def run_colgrep(
    queries: Sequence[Mapping[str, Any]],
    bundle_store: BundleStore,
    store: ResultStore,
    meta: ArmMeta,
    *,
    workdir_root: str | Path,
    bin_path: str = COLGREP_BIN,
    timeout_s: float = 600.0,
    dry_run: bool = False,
    retry_failed: bool = False,
    runner: Callable[..., Any] = subprocess.run,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """probe45 only. ``dry_run`` prints every command and runs nothing (no files written)."""
    applicable = [q for q in queries if q["group"] == "probe45"]
    na = [q for q in queries if q["group"] != "probe45"]
    meta.data["not_applicable"] = {
        "n_queries": len(na),
        "reason": "ColGREP is code-only (tree-sitter); spill-follow bundles are .txt",
        "groups": sorted({q["group"] for q in na}),
    }
    done = store.finished("colgrep", retry_failed)
    n_run = 0
    if not dry_run and applicable and not Path(bin_path).is_file():
        raise SystemExit(f"ColGREP binary not found: {bin_path}")
    for bid, qs in _by_bundle(applicable).items():
        todo = [q for q in qs if q["id"] not in done]
        if not todo:
            continue
        workdir = Path(workdir_root) / bid.split(":", 1)[1].replace("/", "__")
        init = colgrep_init_cmd(bin_path, workdir)
        if dry_run:
            echo("# env NEXT_PLAID_FORCE_CPU=1")
            echo(" ".join(_q(c) for c in init))
            for q in todo:
                echo(" ".join(_q(c) for c in colgrep_search_cmd(bin_path, q["text"], workdir)))
            continue
        materialize_colgrep_dir(bundle_store.get(bid), workdir)
        t0 = time.perf_counter()
        try:
            proc = runner(
                init,
                env=colgrep_env(),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                stdin=subprocess.DEVNULL,
                check=False,
            )
            init_rc, init_err = proc.returncode, None
        except (OSError, subprocess.TimeoutExpired) as exc:
            init_rc, init_err = None, type(exc).__name__
        meta.data["bundles"][bid] = {
            "ok": init_rc == 0,
            "init_returncode": init_rc,
            "init_error": init_err,
            "index_build_wall_s": round(time.perf_counter() - t0, 3),
            "contended_not_for_selection": True,
        }
        meta.save()
        for q in todo:
            cmd = colgrep_search_cmd(bin_path, q["text"], workdir)
            t1 = time.perf_counter()
            ptrs: list[Pointer] | None
            reason = None
            try:
                p = runner(
                    cmd,
                    env=colgrep_env(),
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    stdin=subprocess.DEVNULL,
                    check=False,
                )
                if p.returncode != 0:
                    ptrs, reason = None, f"exit_{p.returncode}"
                else:
                    ptrs = parse_colgrep_json(p.stdout, workdir)
            except subprocess.TimeoutExpired:
                ptrs, reason = None, "timeout"
            except (OSError, json.JSONDecodeError) as exc:
                ptrs, reason = None, type(exc).__name__
            store.append(
                "colgrep",
                _row("colgrep", q, ptrs, (time.perf_counter() - t1) * 1000.0, reason=reason),
            )
            n_run += 1
    meta.save()
    return {"arm": "colgrep", "n_run": n_run, "dry_run": dry_run, "n_not_applicable": len(na)}


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'" if re.search(r"[^\w@%+=:,./-]", s) else s


def semble_status() -> str:
    import importlib.util

    if importlib.util.find_spec("semble") is None:
        return "not_run (tool not available in the isolated venv)"
    return "available_but_not_run (this harness does not implement the reported-only diagnostic)"


# ───────────────────────────────────────────────────────────────────────────── fake embedder


class HashAdapter(arms._AdapterBase):
    """Deterministic fake ``DenseEmbedder`` for the dry-run and the tests: hashed bag of tokens
    (sha256, never Python's salted ``hash``), optional synonym tables (``concepts`` for both sides,
    ``query_concepts`` for queries only), injectable failures. It goes through the real
    ``_AdapterBase.embed`` path (side detection, prefixing, truncation, finiteness accounting)."""

    def __init__(
        self,
        spec: arms.ArmSpec,
        *,
        concepts: Mapping[str, str] | None = None,
        query_concepts: Mapping[str, str] | None = None,
        fail_docs: bool = False,
        fail_query_containing: str | None = None,
    ) -> None:
        super().__init__(spec)
        self.concepts = dict(concepts or {})
        self.query_concepts = dict(query_concepts or {})
        self.fail_docs = fail_docs
        self.fail_query_containing = fail_query_containing
        self.seen: list[tuple[str, list[str]]] = []
        self.intents: list[bool] = []
        self._side = "document"

    def encode_side(self, texts: Sequence[str], side: str) -> Any:
        self.seen.append((side, self._prefixed(list(texts), side)))  # what the model would see
        self._side = side
        if side == "document" and self.fail_docs:
            raise RuntimeError("injected document-side failure")
        if side == "query":
            self.intents.append(self.code_intent)
            if self.fail_query_containing and any(self.fail_query_containing in t for t in texts):
                raise RuntimeError("injected query failure")
        return super().encode_side(texts, side)

    def _encode(self, texts: list[str]) -> Any:
        cs = _cs()
        used = min(self.spec.dim, self.spec.native_dim)
        out = np.zeros((len(texts), self.spec.native_dim), dtype=np.float32)
        for r, text in enumerate(texts):
            for tok in cs.tokenize(text):
                tok = self.concepts.get(tok, tok)
                if self._side == "query":
                    tok = self.query_concepts.get(tok, tok)
                h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "big")
                out[r, h % used] += 1.0
            if not out[r].any():
                out[r, 0] = 1.0
        return out


# ───────────────────────────────────────────────────────────────────────────── freeze / config


def harness_files() -> list[Path]:
    return [
        HERE / "repl_emb_eval.py",
        HERE / "repl_emb_stats.py",
        HERE / "repl_emb_arms.py",
        REPO_ROOT / "src/repl_environment/context_search.py",
    ]


def predeclared_record() -> dict[str, Any]:
    cs = _cs()
    return {
        "interpretations": PREDECLARED,
        "stats": {
            "k_primary": PRIMARY_K,
            "ks": list(KS),
            "M": stats.M_DEFAULT,
            "X": stats.X_DEFAULT,
            "n_min": stats.N_MIN,
            "resamples": stats.DEFAULT_RESAMPLES,
            "seed": stats.DEFAULT_SEED,
            "ci_level": stats.CI_LEVEL,
            "wilson_z": stats.WILSON_Z,
        },
        "code_intent_regex": CODE_INTENT_REGEX,
        "grep_stopwords_sha256": STOPWORDS_SHA256,
        "grep_stopwords": sorted(STOPWORDS),
        "weighted_hybrid": {"w_dense": W_DENSE, "w_bm25": W_BM25, "k": W_RRF_K},
        "top_k": TOP_K,
        "context_search_constants": {
            n: getattr(cs, n)
            for n in (
                "CHUNK_MAX_CHARS",
                "DENSE_INPUT_MAX_CHARS",
                "CANDIDATES_PER_RANKER",
                "MAX_DENSE_CHUNKS",
                "DENSE_BATCH_CHUNKS",
                "MAX_DENSE_ATTEMPTS",
                "BM25_K1",
                "BM25_B",
                "RRF_K",
                "QUERY_TIMEOUT_S",
                "QUERY_ADMISSION_WAIT_S",
                "INDEX_ADMISSION_WAIT_S",
            )
        },
        "offline_overrides": {
            "INDEX_TIMEOUT_MIN_S": OFFLINE_INDEX_TIMEOUT_S,
            "INDEX_TIMEOUT_MAX_S": OFFLINE_INDEX_TIMEOUT_S,
        },
        "probe_repos": PROBE_REPOS,
        "colgrep": {
            "bin": COLGREP_BIN,
            "alpha": COLGREP_ALPHA,
            "env": COLGREP_ENV,
            "expected_sha256": COLGREP_EXPECTED_SHA256,
        },
    }


def freeze_digests(paths: Sequence[Path]) -> list[tuple[str, Path]]:
    return [(sha256_file(p), p) for p in sorted(set(Path(p).resolve() for p in paths))]


def write_freeze(out: Path, paths: Sequence[Path]) -> list[tuple[str, Path]]:
    rows = freeze_digests(paths)
    (out / "FROZEN-AT-LAUNCH.sha256").write_text(
        "".join(f"{h}  {p}\n" for h, p in rows), encoding="utf-8"
    )
    return rows


def verify_freeze(out: Path) -> list[str]:
    """Problems found re-hashing every file named in ``FROZEN-AT-LAUNCH.sha256`` (empty = intact)."""
    f = out / "FROZEN-AT-LAUNCH.sha256"
    if not f.exists():
        return [f"{f} missing: run `freeze` before any arm"]
    problems = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, path = line.partition("  ")
        if not Path(path).is_file():
            problems.append(f"missing: {path}")
        elif sha256_file(path) != digest:
            problems.append(f"changed since freeze: {path}")
    return problems


def cmd_freeze(args: argparse.Namespace, *, head_fn: Callable[[Path], str] = git_head) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("perquery-*.jsonl")):
        raise SystemExit("refusing to freeze: per-query results already exist in --out")
    freeze_file = out / "FROZEN-AT-LAUNCH.sha256"
    if freeze_file.exists() and not args.refreeze:
        raise SystemExit(
            f"{freeze_file} exists; pass --refreeze to replace it (before any arm runs)"
        )
    probe_rows = read_jsonl(args.probe45)
    trace_rows = read_jsonl(args.trace_queries)
    trivial_rows = read_jsonl(args.trivial_queries) if args.trivial_queries else []
    rows, summary = label_queries(probe_rows, trace_rows, trivial_rows)
    labelled = out / "queries-labelled.jsonl"
    labelled_sha = write_labelled(labelled, rows)

    store = make_bundle_store(
        rows,
        bundles_json=args.bundles_json,
        probe_checkouts=args.probe_checkouts,
        head_fn=head_fn,
    )
    probe_bundles = [
        store.get(b) for b in sorted({q["bundle"] for q in rows if q["group"] == "probe45"})
    ]
    listing = corpus_listing(probe_bundles)
    corpus = out / "corpus-digest-probe45.txt"
    corpus.write_text(listing, encoding="utf-8")
    for bid in sorted({q["bundle"] for q in rows if q["group"] != "probe45"}):
        store.get(bid)  # verifies sha256/bytes of every spill file and the chunk-count ceiling
    gold_check = validate_gold(rows, store)
    write_json(out / "gold-check.json", gold_check)
    if gold_check["errors"]:
        raise SystemExit("gold check failed:\n  " + "\n  ".join(gold_check["errors"][:20]))
    write_json(out / "bundles-summary.json", store.summary)
    write_json(out / "predeclared-interpretations.json", predeclared_record())
    write_json(
        out / "inputs.json",
        {
            "probe45": str(Path(args.probe45).resolve()),
            "trace_queries": str(Path(args.trace_queries).resolve()),
            "trivial_queries": str(Path(args.trivial_queries).resolve())
            if args.trivial_queries
            else None,
            "bundles_json": str(Path(args.bundles_json).resolve()) if args.bundles_json else None,
            "rules_md": str(Path(args.rules_md).resolve()),
            "arm_manifest": str(Path(args.arm_manifest).resolve()),
            "probe_checkouts": str(Path(args.probe_checkouts).resolve()),
        },
    )
    paths = [
        labelled,
        Path(args.trace_queries),
        Path(args.probe45),
        Path(args.rules_md),
        Path(args.arm_manifest),
        corpus,
        out / "bundles-summary.json",
        out / "gold-check.json",
        out / "predeclared-interpretations.json",
        out / "inputs.json",
        *harness_files(),
    ]
    for opt in (args.bundles_json, args.trivial_queries):
        if opt:
            paths.append(Path(opt))
    if args.colgrep_bin and Path(args.colgrep_bin).is_file():
        paths.append(Path(args.colgrep_bin))
        sha = sha256_file(args.colgrep_bin)
        print(
            f"# colgrep binary sha256 {sha} ({'matches' if sha == COLGREP_EXPECTED_SHA256 else 'DIFFERS FROM'} the shipped pin)"
        )
    for w in gold_check["warnings"]:
        print(f"# gold warning: {w}")
    rows_out = write_freeze(out, paths)
    print(f"# queries-labelled.jsonl sha256 {labelled_sha}")
    print(f"# code_intent regex: {CODE_INTENT_REGEX}")
    print(dumps(summary, indent=2, sort_keys=True))
    print(f"# probe45 corpus digest (sha256 of corpus-digest-probe45.txt) {sha256_file(corpus)}")
    for h, p in rows_out:
        print(f"{h}  {p}")
    print(f"# wrote {freeze_file}")
    return 0


def _stable_config(specs: Mapping[str, arms.ArmSpec], out: Path) -> dict[str, Any]:
    rec = predeclared_record()
    return {
        "predeclared": rec,
        "harness_sha256": {p.name: sha256_file(p) for p in harness_files()},
        "context_search_sha256": sha256_file(REPO_ROOT / "src/repl_environment/context_search.py"),
        "arm_identities": {
            a: {"model_id": s.model_id(), "digest": s.index_identity()["digest"]}
            for a, s in specs.items()
        },
        "queries_labelled_sha256": sha256_file(out / "queries-labelled.jsonl"),
        "freeze_sha256": sha256_file(out / "FROZEN-AT-LAUNCH.sha256")
        if (out / "FROZEN-AT-LAUNCH.sha256").exists()
        else None,
    }


def _git_state() -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            stdin=subprocess.DEVNULL,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--"]
            + [str(p) for p in harness_files()],
            capture_output=True,
            text=True,
            check=True,
            stdin=subprocess.DEVNULL,
        ).stdout.strip()
        return {"git_sha": sha, "harness_files_dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"git_sha": None, "error": type(exc).__name__}


def _volatile_config(args: argparse.Namespace, run_arms_: Sequence[str]) -> dict[str, Any]:
    env_keys = (
        "OMP_NUM_THREADS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    )
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "arms": list(run_arms_),
        "endpoints": dict(args.arm_url or []),
        "model_paths": dict(args.model_path or []),
        "declared_server_threads": dict(args.arm_threads or []),
        "env": {k: os.environ.get(k) for k in env_keys},
        "os_cpu_count": os.cpu_count(),
        "sched_affinity_cpus": affinity,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "index_timeout_override_s": OFFLINE_INDEX_TIMEOUT_S,
        "semble": semble_status(),
        "colgrep_dry_run": bool(args.colgrep_dry_run),
        "cost_numbers": "contended_not_for_selection",
        **_git_state(),
    }


# ───────────────────────────────────────────────────────────────────────────── run


def build_specs(
    args: argparse.Namespace, manifest: Mapping[str, Any] | None
) -> dict[str, arms.ArmSpec]:
    llama_rt, m2v_rt = args.llama_runtime, args.m2v_runtime
    if manifest is not None:
        m_llama, m_m2v = runtimes_from_manifest(manifest)
        llama_rt, m2v_rt = llama_rt or m_llama, m2v_rt or m_m2v
    if not (llama_rt and m2v_rt):
        raise SystemExit("--llama-runtime and --m2v-runtime are required without an arm manifest")
    specs = eval_specs(llama_rt, m2v_rt)
    for aid, rt in args.runtime_override or []:
        specs[aid] = dataclasses.replace(specs[aid], runtime=rt)
    validate_registry(specs)
    return specs


def make_adapter(
    defn: DenseArmDef, spec: arms.ArmSpec, args: argparse.Namespace
) -> tuple[Any, arms.ArmSpec]:
    urls, paths = dict(args.arm_url or []), dict(args.model_path or [])
    if defn.kind == "llama":
        if defn.arm_id not in urls:
            raise SystemExit(f"--arm-url {defn.arm_id}=http://127.0.0.1:PORT is required")
        gguf = paths.get(defn.arm_id)
        if gguf and defn.arm_id in ("bge-large", "granite-97m-r2"):
            want = BGE_GGUF_SHA256 if defn.arm_id == "bge-large" else GRANITE_GGUF_SHA256
            if sha256_file(gguf) != want:
                raise SystemExit(f"{defn.arm_id}: {gguf} sha256 does not match the registered GGUF")
        ad = resolve_llama_adapter(defn, spec, urls[defn.arm_id], timeout_s=args.embed_timeout_s)
        return ad, ad.spec
    if defn.arm_id not in paths:
        raise SystemExit(f"--model-path {defn.arm_id}=DIR is required")
    return arms.Model2VecAdapter(spec, paths[defn.arm_id]), spec


def run_arms(
    out: Path,
    queries: Sequence[Mapping[str, Any]],
    bundle_store: BundleStore,
    arm_names: Sequence[str],
    *,
    adapter_for: Callable[[DenseArmDef], tuple[Any, arms.ArmSpec]],
    colgrep: Mapping[str, Any] | None = None,
    retry_failed: bool = False,
    echo: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    store = ResultStore(out)
    reports = []
    for arm in arm_names:
        if arm in LEXICAL_ARMS:
            reports.append(
                run_lexical_arm(arm, queries, bundle_store, store, retry_failed=retry_failed)
            )
        elif arm == "colgrep":
            cfg = dict(colgrep or {})
            reports.append(
                run_colgrep(
                    queries,
                    bundle_store,
                    store,
                    ArmMeta(out, "colgrep"),
                    retry_failed=retry_failed,
                    echo=echo,
                    **cfg,
                )
            )
        elif arm in DENSE_ARMS:
            defn = DENSE_ARMS[arm]
            adapter, spec = adapter_for(defn)
            meta = ArmMeta(out, arm)
            meta.data.update(
                kind=defn.kind,
                registered=defn.registered,
                model_id=adapter.model_id,
                identity_digest=spec.index_identity()["digest"],
                dim=spec.dim,
            )
            reports.append(
                run_dense_arm(
                    defn, adapter, queries, bundle_store, store, meta, retry_failed=retry_failed
                )
            )
        else:
            raise SystemExit(
                f"unknown arm {arm!r}; choose from {sorted([*LEXICAL_ARMS, 'colgrep', *DENSE_ARMS])}"
            )
        _log(f"[run] {reports[-1]}")
    return reports


def cmd_run(
    args: argparse.Namespace,
    *,
    adapter_for: Callable[[DenseArmDef, arms.ArmSpec], tuple[Any, arms.ArmSpec]] | None = None,
    head_fn: Callable[[Path], str] = git_head,
) -> int:
    out = Path(args.out)
    if not args.no_verify_freeze:
        problems = verify_freeze(out)
        if problems:
            raise SystemExit("freeze check failed:\n  " + "\n  ".join(problems))
    inputs = json.loads((out / "inputs.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        Path(args.arm_manifest or inputs["arm_manifest"]).read_text(encoding="utf-8")
    )
    specs = build_specs(args, manifest)
    problems = check_against_manifest(specs, manifest)
    if problems and not args.allow_manifest_mismatch:
        raise SystemExit(
            "arm identity differs from the frozen manifest:\n  " + "\n  ".join(problems)
        )
    rows = read_jsonl(out / "queries-labelled.jsonl")
    groups = set(args.groups.split(",")) if args.groups else None
    queries = [r for r in rows if r["scored"] and (groups is None or r["group"] in groups)]
    bundle_store = make_bundle_store(
        rows,
        bundles_json=args.bundles_json or inputs.get("bundles_json"),
        probe_checkouts=args.probe_checkouts or inputs["probe_checkouts"],
        head_fn=head_fn,
    )
    stable = _stable_config(specs, out)
    echo_path = out / "config-echo.json"
    if echo_path.exists():
        prev = json.loads(echo_path.read_text(encoding="utf-8"))
        if prev["stable"] != json.loads(dumps(stable)):
            raise SystemExit(
                "config-echo.json differs from this invocation's stable config; refusing to resume"
            )
    else:
        write_json(echo_path, {"stable": stable, **_git_state(), "semble": semble_status()})
    volatile = _volatile_config(args, args.arm)
    with open(out / "invocations.jsonl", "a", encoding="utf-8") as f:
        f.write(dumps(volatile, sort_keys=True) + "\n")

    colgrep_cfg = {
        "workdir_root": args.colgrep_workdir or str(out / "colgrep-work"),
        "bin_path": args.colgrep_bin,
        "timeout_s": args.colgrep_timeout_s,
        "dry_run": args.colgrep_dry_run,
    }
    run_arms(
        out,
        queries,
        bundle_store,
        args.arm,
        adapter_for=lambda d: (adapter_for or (lambda dd, sp: make_adapter(dd, sp, args)))(
            d, specs[d.arm_id]
        ),
        colgrep=colgrep_cfg,
        retry_failed=args.retry_failed,
    )
    summary_path = out / "bundles-summary-run.json"
    seen = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    write_json(summary_path, {**seen, **bundle_store.summary})
    return 0


# ───────────────────────────────────────────────────────────────────────────── analysis


def _score_variant(
    rows: Mapping[str, Mapping[str, Any]], qs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Recall vectors over the queries of ``qs`` that HAVE a row; missing rows are not scored."""
    present = [q for q in qs if q["id"] in rows]
    ranked = [
        None if rows[q["id"]]["failed"] else [tuple(p) for p in rows[q["id"]]["pointers"]]
        for q in present
    ]
    gold = [[tuple(g) for g in q["gold_spans"]] for q in present]
    return {
        "qs": present,
        "vec": stats.recall_all_k(ranked, gold) if present else {k: np.zeros(0) for k in KS},
        "n": len(present),
        "n_applicable": len(qs),
        "n_failed": stats.count_failed(ranked),
        "n_missing": len(qs) - len(present),
        "complete": len(present) == len(qs),
    }


def _applicable(variant: str, qs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [q for q in qs if q["group"] == "probe45"] if variant == "colgrep" else list(qs)


def _norm_cost_key(key: str) -> str:
    return key if "." in key or key in LEXICAL_ARMS or key == "colgrep" else f"{key}.hybrid"


def load_speed(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data = data.get("arms", data)
    return {_norm_cost_key(k): v for k, v in data.items()}


def _vecs(sc: Mapping[str, Any]) -> dict[int, list[float]]:
    return {k: v.tolist() for k, v in sc["vec"].items()}


def analyze(
    out: str | Path,
    speed_json: str | Path | None = None,
    *,
    resamples: int = stats.DEFAULT_RESAMPLES,
    seed: int = stats.DEFAULT_SEED,
) -> dict[str, Any]:
    out = Path(out)
    qrows = read_jsonl(out / "queries-labelled.jsonl")
    scored = [q for q in qrows if q["scored"]]
    dec = [q for q in scored if q["group"] in DECISION_GROUPS]
    trivial = [q for q in scored if q["group"] == TRIVIAL_GROUP]
    store = ResultStore(out)
    variants = store.variants()
    rows_of = {v: store.rows(v) for v in variants}
    res: dict[str, Any] = {
        "schema": "repl-emb-analysis/1",
        "n_deciding": len(dec),
        "n_by_group": {
            g: sum(q["group"] == g for q in scored) for g in sorted({q["group"] for q in scored})
        },
        "n_unscored": len(qrows) - len(scored),
        "variants": variants,
        "failures_count_as_misses": True,
        "cost_flag": "contended_not_for_selection",
    }

    sc = {v: _score_variant(rows_of[v], _applicable(v, dec)) for v in variants}
    res["completeness"] = {
        v: {k: sc[v][k] for k in ("n", "n_applicable", "n_failed", "n_missing", "complete")}
        for v in variants
    }

    # per arm & stratum recall, with Wilson intervals
    res["recall"], res["recall_by_stratum_label"] = {}, {}
    for v in variants:
        if sc[v]["n"] == 0:
            continue
        labels = [q["group"] for q in sc[v]["qs"]]
        res["recall"][v] = stats.stratum_report({v: sc[v]["vec"]}, labels)[v]
        res["recall_by_stratum_label"][v] = stats.stratum_report(
            {v: sc[v]["vec"]}, [q.get("stratum") or q["group"] for q in sc[v]["qs"]]
        )[v]

    # lexical baseline L
    L: dict[str, Any] | None = None
    if all(a in sc and sc[a]["complete"] and sc[a]["n"] for a in LEXICAL_ARMS):
        L = stats.lexical_baseline(sc["grep"]["vec"][PRIMARY_K], sc["bm25"]["vec"][PRIMARY_K])
    res["lexical_baseline"] = L

    # lexical vs dense vs hybrid per dense arm (pooled recall@5 and per group)
    res["lexical_vs_dense_vs_hybrid"] = {}
    dense_ids = sorted(
        {v.split(".")[0] for v in variants if v.split(".")[-1] in ("dense", "hybrid", "whybrid")}
    )
    for a in dense_ids:
        entry: dict[str, Any] = {"registered": a in REGISTERED_DENSE}
        for kind in ("dense", "hybrid", "whybrid"):
            v = f"{a}.{kind}"
            if v in sc and sc[v]["n"]:
                entry[kind] = {
                    "pooled_recall5": float(sc[v]["vec"][PRIMARY_K].mean()),
                    "n": sc[v]["n"],
                    "complete": sc[v]["complete"],
                    "n_failed": sc[v]["n_failed"],
                }
        if L:
            entry["lexical_L"] = {
                "arm": L["L"],
                "pooled_recall5": float(sc[L["L"]]["vec"][PRIMARY_K].mean()),
            }
        res["lexical_vs_dense_vs_hybrid"][a] = entry

    # kill rule
    reg = {
        v: sc[v]["vec"][PRIMARY_K]
        for v in REGISTERED_HYBRIDS
        if v in sc and sc[v]["complete"] and sc[v]["n"]
    }
    ext = {
        v: sc[v]["vec"][PRIMARY_K]
        for v in variants
        if v.endswith(".hybrid")
        and v not in REGISTERED_HYBRIDS
        and sc[v]["complete"]
        and sc[v]["n"]
    }
    n_dec = len(dec)
    if L is None:
        kill = {
            "verdict": "INCOMPLETE",
            "n": n_dec,
            "reason": "a lexical leg (grep / bm25) is missing or incomplete",
        }
        verdict_line = f"KILL RULE: INCOMPLETE (n={n_dec})"
    else:
        missing = [v for v in REGISTERED_HYBRIDS if v not in reg]
        kill = stats.kill_decision(
            reg,
            ext,
            sc[L["L"]]["vec"][PRIMARY_K],
            L["L"],
            missing_registered=missing,
            resamples=resamples,
            seed=seed,
        )
        verdict_line = f"KILL RULE: {kill['verdict']} (n={kill['n']})"
    res["kill_rule"] = kill
    lines = [verdict_line]
    if kill.get("changed_by_extension"):
        lines.append(
            f"KILL RULE (incl. extension arms): {kill['verdict_with_extension']} "
            f"[extension_only: {', '.join(kill['extension_only']) or 'none'}] (not a registered-arm result)"
        )

    # bootstrap CIs vs L and vs BGE (recall@5, pooled deciding set / the arm's applicable set)
    res["bootstrap_recall5"] = {}
    for v in variants:
        s = sc[v]
        if not (s["n"] and s["complete"]):
            continue
        ids = [q["id"] for q in s["qs"]]
        entry = {}
        if L and v != L["L"]:
            lv = sc[L["L"]]
            pos = {q["id"]: i for i, q in enumerate(lv["qs"])}
            if all(i in pos for i in ids):
                b = lv["vec"][PRIMARY_K][[pos[i] for i in ids]]
                entry["vs_L"] = {
                    "L": L["L"],
                    **stats.paired_bootstrap(s["vec"][PRIMARY_K], b, resamples, seed),
                }
        kind = v.split(".")[-1] if "." in v else "hybrid"
        ref = f"bge-large.{kind if kind in ('dense', 'hybrid', 'whybrid') else 'hybrid'}"
        if ref in sc and ref != v and sc[ref]["complete"] and sc[ref]["n"]:
            pos = {q["id"]: i for i, q in enumerate(sc[ref]["qs"])}
            if all(i in pos for i in ids):
                b = sc[ref]["vec"][PRIMARY_K][[pos[i] for i in ids]]
                entry["vs_bge"] = {
                    "ref": ref,
                    **stats.paired_bootstrap(s["vec"][PRIMARY_K], b, resamples, seed),
                }
        if entry:
            res["bootstrap_recall5"][v] = entry

    # leakage cut (reported variant)
    res["leakage_cut"] = _leakage_cut(sc, dec, L, resamples, seed)

    # trivial-head stratum: a separate table, outside the decision
    res["trivial_head"] = {}
    if trivial:
        for v in variants:
            t = _score_variant(rows_of[v], _applicable(v, trivial))
            if t["n"]:
                res["trivial_head"][v] = {
                    "n": t["n"],
                    "n_failed": t["n_failed"],
                    "recall": {str(k): float(t["vec"][k].mean()) for k in KS},
                }
        res["trivial_head_note"] = (
            "reported outside the decision; not in n, L, the kill rule or selection"
        )

    # cost (contended) + selection
    res["cost_contended"] = {}
    for v in variants:
        lat = [r["latency_ms"] for r in rows_of[v].values() if not r["failed"]]
        res["cost_contended"][v] = {
            **stats.latency_summary(lat),
            "latencies_ms": lat,
            "flag": "contended_not_for_selection",
        }
    res["index_build_contended"] = {}
    for mp in sorted(out.glob("armmeta-*.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        builds = [b for b in m.get("bundles", {}).values()]
        res["index_build_contended"][m["arm_id"]] = {
            "n_bundles": len(builds),
            "wall_s_total": round(sum(b.get("index_build_wall_s") or 0.0 for b in builds), 4),
            "cpu_s_total": round(sum(b.get("index_build_cpu_s") or 0.0 for b in builds), 4),
            "n_failed_builds": sum(1 for b in builds if not b.get("ok")),
            "adapter_nonfinite_rows": sum(
                i.get("adapter_nonfinite_rows", 0) for i in m.get("invocations", [])
            ),
            "flag": "contended_not_for_selection",
        }
    speed = load_speed(speed_json)
    res["speed_block"] = {"path": str(speed_json) if speed_json else None, "arms": sorted(speed)}
    sel_arms = {
        v: {
            "recall5": float(sc[v]["vec"][PRIMARY_K].mean()),
            "cpu_s_per_query": (speed.get(v) or {}).get("cpu_s_per_query"),
            "p95_ms": (speed.get(v) or {}).get("p95_ms", (speed.get(v) or {}).get("query_p95_ms")),
        }
        for v in variants
        if v.endswith(".hybrid") and sc[v]["complete"] and sc[v]["n"]
    }
    if kill.get("verdict") == "PASS":
        if not speed:
            sel = {
                **stats.selection_rule(sel_arms),
                "status": "pending speed block",
                "selected": None,
            }
            lines.append("SELECTION: pending speed block")
        else:
            sel = stats.selection_rule(sel_arms)
            lines.append(
                f"SELECTION: {sel['selected']}"
                if sel["status"] == "selected"
                else f"SELECTION: pending speed block (no cost row for {', '.join(sel.get('missing_cost', []))})"
            )
    else:
        sel = {"status": "not applicable", "reason": f"kill rule {kill.get('verdict')}"}
        lines.append(f"SELECTION: not applicable (kill rule {kill.get('verdict')})")
    res["selection"] = sel
    res["verdict_lines"] = lines
    res["semble"] = semble_status()
    return res


def _leakage_cut(
    sc: Mapping[str, Any], dec: Sequence[Mapping[str, Any]], L: Any, resamples: int, seed: int
) -> dict[str, Any]:
    flag = {q["id"]: q.get("answer_named_verbatim") for q in dec}
    cuts = {"answer_named_verbatim=true": True, "answer_named_verbatim=false": False}
    out: dict[str, Any] = {
        "note": "reported variant; does not replace the decision",
        "n_unlabelled": sum(1 for v in flag.values() if not isinstance(v, bool)),
        "recall5": {},
    }
    for name, want in cuts.items():
        ids = {i for i, v in flag.items() if v is want}
        out["recall5"][name] = {"n": len(ids)}
        for v, s in sc.items():
            if not (s["complete"] and s["n"]) or v == "colgrep":
                continue
            m = np.array([q["id"] in ids for q in s["qs"]])
            if m.any():
                out["recall5"][name][v] = float(s["vec"][PRIMARY_K][m].mean())
    ids_false = {i for i, v in flag.items() if v is False}
    if L and ids_false:
        mask = {
            v: np.array([q["id"] in ids_false for q in s["qs"]])
            for v, s in sc.items()
            if s["complete"] and s["n"] and v != "colgrep"
        }

        def pick(v: str) -> Any:
            return sc[v]["vec"][PRIMARY_K][mask[v]]

        reg = {v: pick(v) for v in REGISTERED_HYBRIDS if v in mask}
        ext = {v: pick(v) for v in mask if v.endswith(".hybrid") and v not in REGISTERED_HYBRIDS}
        if reg and L["L"] in mask:
            out["kill_rule_on_false_subset"] = stats.kill_decision(
                reg,
                ext,
                pick(L["L"]),
                L["L"],
                resamples=resamples,
                seed=seed,
                missing_registered=[v for v in REGISTERED_HYBRIDS if v not in reg],
            )
    return out


def _pct(x: Any) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render_markdown(res: Mapping[str, Any]) -> str:
    L = ["# REPL-EMB-2.2 offline retrieval eval", ""]
    L += [f"- {line}" for line in res["verdict_lines"]]
    L += [
        "",
        f"deciding n = {res['n_deciding']}  ({res['n_by_group']}; unscored {res['n_unscored']})",
    ]
    L += ["costs here are `contended_not_for_selection`; failures count as misses.", ""]
    lb = res["lexical_baseline"]
    L.append(
        f"**L** = {lb['L'] if lb else 'n/a'}"
        + (
            f" (grep {lb['grep_recall5']:.3f}, bm25 {lb['bm25_recall5']:.3f}, tie={lb['tie']})"
            if lb
            else ""
        )
    )
    L += [
        "",
        "## recall@k per arm and stratum (Wilson 95%)",
        "",
        "| arm | stratum | n | R@1 | R@3 | R@5 | R@10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for v, by in res["recall"].items():
        for g, ks in by.items():
            c = res["completeness"][v]
            tag = "" if c["complete"] else " (INCOMPLETE)"
            cells = " | ".join(
                f"{ks[k]['recall']:.3f} [{_pct(ks[k]['ci_lo'])}, {_pct(ks[k]['ci_hi'])}]"
                for k in KS
            )
            L.append(f"| {v}{tag} | {g} | {ks[1]['n']} | {cells} |")
    kr = res["kill_rule"]
    L += ["", "## kill rule (recall@5, hybrid vs L)", ""]
    if "registered" in kr:
        L += [
            "| hybrid | kind | R@5 | L R@5 | diff | CI lo | CI hi | margin | CI>0 | pass |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for kind in ("registered", "extension"):
            for name, e in kr[kind].items():
                L.append(
                    f"| {name} | {kind}{'' if kind == 'registered' else ' (extension_only if it flips)'} | {e['hybrid_recall5']:.3f} | "
                    f"{e['lexical_recall5']:.3f} | {e['mean_diff']:+.3f} | {e['ci_lo']:+.3f} | {e['ci_hi']:+.3f} | "
                    f"{e['margin_ok']} | {e['ci_ok']} | {e['pass']} |"
                )
        L += [
            "",
            f"M = {kr['M']}, n_min = {kr['n_min']}, bootstrap resamples/seed = {stats.DEFAULT_RESAMPLES}/{stats.DEFAULT_SEED}",
        ]
    else:
        L.append(kr.get("reason", ""))
    L += [
        "",
        "## lexical vs dense vs hybrid (pooled recall@5)",
        "",
        "| dense arm | registered | L | dense | hybrid | weighted hybrid |",
        "|---|---|---|---|---|---|",
    ]
    for a, e in res["lexical_vs_dense_vs_hybrid"].items():
        L.append(
            f"| {a} | {e['registered']} | {_pct(e.get('lexical_L', {}).get('pooled_recall5'))} | "
            f"{_pct(e.get('dense', {}).get('pooled_recall5'))} | {_pct(e.get('hybrid', {}).get('pooled_recall5'))} | "
            f"{_pct(e.get('whybrid', {}).get('pooled_recall5'))} |"
        )
    L += [
        "",
        "## paired bootstrap, recall@5 (diff, 95% CI)",
        "",
        "| arm | vs | diff | CI |",
        "|---|---|---|---|",
    ]
    for v, e in res["bootstrap_recall5"].items():
        for key in ("vs_L", "vs_bge"):
            if key in e:
                b = e[key]
                L.append(
                    f"| {v} | {b.get('L') or b.get('ref')} | {b['mean_diff']:+.3f} | [{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}] |"
                )
    L += [
        "",
        "## leakage cut (reported variant)",
        "",
        f"unlabelled rows: {res['leakage_cut']['n_unlabelled']}",
        "",
    ]
    for name, d in res["leakage_cut"]["recall5"].items():
        L.append(
            f"- {name}: n={d['n']}; " + ", ".join(f"{k} {x:.3f}" for k, x in d.items() if k != "n")
        )
    if res["trivial_head"]:
        L += [
            "",
            "## trivial-head stratum (outside the decision)",
            "",
            "| arm | n | R@1 | R@3 | R@5 | R@10 |",
            "|---|---|---|---|---|---|",
        ]
        for v, t in res["trivial_head"].items():
            L.append(
                f"| {v} | {t['n']} | " + " | ".join(f"{t['recall'][str(k)]:.3f}" for k in KS) + " |"
            )
    L += [
        "",
        "## failures (counted as misses) and contended cost",
        "",
        "| arm | n | failed | missing | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
    ]
    for v, c in res["completeness"].items():
        cost = res["cost_contended"][v]
        L.append(
            f"| {v} | {c['n']} | {c['n_failed']} | {c['n_missing']} | {_pct(cost['p50_ms'])} | {_pct(cost['p95_ms'])} |"
        )
    sel = res["selection"]
    L += ["", "## selection", "", f"status: {sel.get('status')}"]
    for r in sel.get("table", []):
        L.append(
            f"- {r['arm']}: R@5 {r['recall5']:.3f}, cpu_s/q {r['cpu_s_per_query']}, p95 {r['p95_ms']}, candidate={r['candidate']}"
        )
    L += ["", f"semble: {res['semble']}", ""]
    return "\n".join(L)


def cmd_analyze(args: argparse.Namespace) -> int:
    out = Path(args.out)
    res = analyze(out, args.speed_json, resamples=args.resamples, seed=args.seed)
    write_json(out / "analysis.json", res)
    (out / "analysis.md").write_text(render_markdown(res), encoding="utf-8")
    for line in res["verdict_lines"]:
        print(line)
    print(f"# wrote {out / 'analysis.json'} and {out / 'analysis.md'}")
    return 0


# ───────────────────────────────────────────────────────────────────────────── dry run

SYNTH_CONCEPTS = {"crystal": "quartz"}  # both sides: the synonym only dense retrieval can use
SYNTH_QUERY_CONCEPTS = {"marble": "slate"}  # query side only: the fake embedder confuses two rocks


def synthetic_files(ext: str = "py") -> dict[str, str]:
    """Three files x two chunks each (8 lines of exactly 100 chars = one 800-char chunk, so the
    chunks are lines 1-8 and 9-16 of every file)."""
    kws = {
        "alpha": ("quartz", "basalt"),
        "beta": ("granite", "marble"),
        "gamma": ("slate", "gneiss"),
    }
    files = {}
    for name, (k1, k2) in kws.items():
        lines = []
        for kw in (k1, k2):
            for j in range(1, 9):
                lines.append(("# " + f"{kw} {kw} sample sentence number {j}").ljust(99, "-") + "\n")
        files[f"{name}.{ext}"] = "".join(lines)
    return files


def synthetic_queries(path_of: Callable[[str], str], prefix: str) -> list[dict[str, Any]]:
    """Six queries with known gold over ``synthetic_files`` (expected recall@1 in the tests:
    grep/bm25 [1,0,1,0,1,1], dense [1,1,0,0,1,1], hybrid [1,1,1,0,1,1])."""

    def g(name: str, s: int, e: int) -> dict[str, Any]:
        return {"path": path_of(name), "start_line": s, "end_line": e}

    rows = [
        # lexical and dense agree
        ("q1", "where is quartz handled", g("alpha", 1, 8)),
        # no shared token: only the synonym table (dense side) finds it
        ("q2", "where is the crystal handling", g("alpha", 3, 4)),
        # BM25 is right; the fake embedder's query-side confusion sends dense to the slate chunk
        ("q3", "which file talks about marble", g("beta", 9, 16)),
        # gold beyond the end of the file: nothing can hit
        ("q4", "where is slate handled", g("gamma", 17, 20)),
        # gold straddles the chunk boundary (lines 8-9)
        ("q5", "tell me about granite", g("beta", 8, 9)),
        # backticks: code_intent by regex
        ("q6", "what does `basalt` do", g("alpha", 9, 16)),
    ]
    return [
        {
            "id": f"{prefix}{qid}",
            "stratum": "synthetic",
            "text": text,
            "code_intent": None,
            "gold": gold,
            "answer_named_verbatim": qid in ("q1", "q3", "q6"),
        }
        for qid, text, gold in rows
    ]


def make_synthetic_inputs(root: Path) -> dict[str, Any]:
    """A fake pinned checkout, spill files, bundles.json, queries, rules and manifest."""
    root.mkdir(parents=True, exist_ok=True)
    repo = "mwaskom/seaborn"
    co = root / "checkouts" / PROBE_REPOS[repo]["checkout"]
    pkg = co / PROBE_REPOS[repo]["package_root"]
    pkg.mkdir(parents=True, exist_ok=True)
    for name, text in synthetic_files("py").items():
        (pkg / name).write_text(text, encoding="utf-8")
    spill = root / "spill"
    spill.mkdir(exist_ok=True)
    bundle_files = []
    for name, text in synthetic_files("txt").items():
        p = spill / name
        p.write_text(text, encoding="utf-8")
        raw = text.encode()
        bundle_files.append(
            {"path": str(p), "sha256": sha256_bytes(raw), "bytes": len(raw), "chunks": 2}
        )
    (root / "bundles.json").write_text(
        dumps({"spill-b1": {"files": bundle_files, "total_chunks": 6}}, indent=1), encoding="utf-8"
    )
    probe = []
    for q in synthetic_queries(lambda n: f"seaborn/{n}.py", "p-"):
        probe.append({**q, "repo": repo, "commit": "0" * 40, "stratum": "intake1357_probe45"})
    trace = []
    for q in synthetic_queries(lambda n: str(spill / f"{n}.txt"), "s-"):
        g = q["gold"]
        trace.append(
            {
                **q,
                "gold": {"path": g["path"], "ranges": [[g["start_line"], g["end_line"]]]},
                "bundle_id": "spill-b1",
            }
        )
    (root / "probe45.jsonl").write_text("".join(dumps(r) + "\n" for r in probe), encoding="utf-8")
    (root / "queries-trace.jsonl").write_text(
        "".join(dumps(r) + "\n" for r in trace), encoding="utf-8"
    )
    (root / "RULES.md").write_text("synthetic dry-run rules\n", encoding="utf-8")
    llama_rt, m2v_rt = "llama-server dry-run", "model2vec dry-run"
    specs = eval_specs(llama_rt, m2v_rt)
    manifest = {
        "arms": [
            {"arm_id": a, "dense_embedder_model_id": s.model_id(), "runtime": s.runtime}
            for a, s in specs.items()
            if a in ("eg2-gguf-bf16-768", "potion-code-16m-v2")
        ]
    }
    (root / "arm_manifest.json").write_text(dumps(manifest, indent=1), encoding="utf-8")
    return {"root": root, "checkouts": root / "checkouts"}


def run_synthetic(
    out: Path, inputs_root: Path, *, echo: Callable[[str], None] = print
) -> dict[str, Any]:
    """freeze -> run (lexical, every dense arm on a fake embedder, ColGREP commands printed) ->
    analyze, all through the real subcommand entry points. No network, no process, no model."""
    inp = make_synthetic_inputs(inputs_root)
    root = inp["root"]
    out.mkdir(parents=True, exist_ok=True)
    ap = build_parser()
    fake_head = lambda p: "0" * 40  # noqa: E731 -- the synthetic checkout is not a git repo
    cmd_freeze(
        ap.parse_args(
            [
                "freeze",
                "--out",
                str(out),
                "--probe45",
                str(root / "probe45.jsonl"),
                "--trace-queries",
                str(root / "queries-trace.jsonl"),
                "--bundles-json",
                str(root / "bundles.json"),
                "--rules-md",
                str(root / "RULES.md"),
                "--arm-manifest",
                str(root / "arm_manifest.json"),
                "--probe-checkouts",
                str(inp["checkouts"]),
                "--colgrep-bin",
                str(root / "no-such-colgrep"),
                "--refreeze",
            ]
        ),
        head_fn=fake_head,
    )
    assert not verify_freeze(out)

    def adapter_for(d: DenseArmDef, spec: arms.ArmSpec) -> tuple[Any, arms.ArmSpec]:
        return HashAdapter(spec, concepts=SYNTH_CONCEPTS, query_concepts=SYNTH_QUERY_CONCEPTS), spec

    cmd_run(
        ap.parse_args(
            ["run", "--out", str(out), "--colgrep-dry-run"]
            + [x for a in (*LEXICAL_ARMS, *DENSE_ARMS, "colgrep") for x in ("--arm", a)]
        ),
        adapter_for=adapter_for,
        head_fn=fake_head,
    )
    speed = {
        "arms": {
            f"{a}.hybrid": {"cpu_s_per_query": 0.01 * (i + 1), "p95_ms": 5.0 * (i + 1)}
            for i, a in enumerate(DENSE_ARMS)
        }
    }
    write_json(out / "speed-dryrun.json", speed)
    cmd_analyze(
        ap.parse_args(
            ["analyze", "--out", str(out), "--speed-json", str(out / "speed-dryrun.json")]
        )
    )
    return json.loads((out / "analysis.json").read_text(encoding="utf-8"))


def cmd_dry_run(args: argparse.Namespace) -> int:
    ctx = None if args.out else tempfile.TemporaryDirectory(prefix="repl-emb-dry-")
    try:
        base = Path(args.out) if args.out else Path(ctx.name)  # type: ignore[union-attr]
        res = run_synthetic(base / "out", base / "synthetic-inputs")
        print(f"# dry-run ok: {len(res['variants'])} variants; semble: {res['semble']}; out={base}")
    finally:
        if ctx:
            ctx.cleanup()
    return 0


# ───────────────────────────────────────────────────────────────────────────── CLI


def _kv(s: str) -> tuple[str, str]:
    k, sep, v = s.partition("=")
    if not sep or not k or not v:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {s!r}")
    return k, v


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true", help="synthetic end-to-end run, no network")
    ap.add_argument("--out", help="(dry-run) keep the synthetic run in this directory")
    sub = ap.add_subparsers(dest="cmd")

    fz = sub.add_parser("freeze", help="label queries and write FROZEN-AT-LAUNCH.sha256")
    fz.add_argument("--out", required=True)
    fz.add_argument(
        "--probe45",
        default=str(
            REPO_ROOT / "data/repl_retrieval_eval/arm-manifest-20261020/probe45-intake1357.jsonl"
        ),
    )
    fz.add_argument("--trace-queries", required=True, help="the miner's queries-trace.jsonl")
    fz.add_argument(
        "--trivial-queries", help="optional trivial-head query file (reported outside the decision)"
    )
    fz.add_argument("--bundles-json", required=True, help="the miner's trace/bundles.json")
    fz.add_argument("--rules-md", required=True)
    fz.add_argument("--arm-manifest", default=str(DEFAULT_ARM_MANIFEST))
    fz.add_argument("--probe-checkouts", default=DEFAULT_PROBE_CHECKOUTS)
    fz.add_argument("--colgrep-bin", default=COLGREP_BIN)
    fz.add_argument("--refreeze", action="store_true")

    rn = sub.add_parser("run", help="run arms (resumable)")
    rn.add_argument("--out", required=True)
    rn.add_argument(
        "--arm",
        action="append",
        required=True,
        help="grep | bm25 | colgrep | " + " | ".join(DENSE_ARMS),
    )
    rn.add_argument(
        "--arm-url",
        action="append",
        type=_kv,
        help="ARM=http://127.0.0.1:PORT (caller-owned llama-server)",
    )
    rn.add_argument(
        "--model-path",
        action="append",
        type=_kv,
        help="ARM=DIR (static) or ARM=GGUF (sha256 check)",
    )
    rn.add_argument(
        "--arm-threads",
        action="append",
        type=_kv,
        help="ARM=N declared server threads (recorded only)",
    )
    rn.add_argument("--runtime-override", action="append", type=_kv)
    rn.add_argument("--llama-runtime")
    rn.add_argument("--m2v-runtime")
    rn.add_argument("--arm-manifest")
    rn.add_argument("--bundles-json")
    rn.add_argument("--probe-checkouts")
    rn.add_argument("--groups", help="comma list of groups to run (default all)")
    rn.add_argument("--embed-timeout-s", type=float, default=600.0)
    rn.add_argument("--colgrep-workdir")
    rn.add_argument("--colgrep-bin", default=COLGREP_BIN)
    rn.add_argument("--colgrep-timeout-s", type=float, default=600.0)
    rn.add_argument(
        "--colgrep-dry-run", action="store_true", help="print the ColGREP commands, run nothing"
    )
    rn.add_argument("--retry-failed", action="store_true")
    rn.add_argument("--allow-manifest-mismatch", action="store_true")
    rn.add_argument("--no-verify-freeze", action="store_true")

    an = sub.add_parser("analyze", help="write analysis.json / analysis.md")
    an.add_argument("--out", required=True)
    an.add_argument("--speed-json")
    an.add_argument("--resamples", type=int, default=stats.DEFAULT_RESAMPLES)
    an.add_argument("--seed", type=int, default=stats.DEFAULT_SEED)
    dr = sub.add_parser("dry-run", help="synthetic end-to-end run, no network")
    dr.add_argument("--out", help="keep the synthetic run in this directory")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run or args.cmd == "dry-run":
        return cmd_dry_run(args)
    if args.cmd == "freeze":
        return cmd_freeze(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "analyze":
        return cmd_analyze(args)
    build_parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
