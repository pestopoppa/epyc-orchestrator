"""HIL-6 seam S2: MHS-3 eval-leakage guards + frontdoor check moved to ``mutation_guards``.

AutoPilot identity is proven by the PromptForge leakage tests (assertion bodies unchanged; the six
module-attribute monkeypatches retargeted to ``mutation_guards``), a golden digest over the
real question pool, and these: every moved name is the SAME object through PromptForge in both
import contexts, the leakage observer is one global however PromptForge was imported, the
``harness_content`` wrapper still follows PromptForge's own roots, and no test patches a moved
global on PromptForge (where it would silently do nothing).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MOVED_S2 = (
    "CODE_MUTATION_ALLOWLIST", "_SUITE_ALIASES", "_SUITE_TERM_TO_CANONICAL", "_SUITE_TERM_RE",
    "_FRONTDOOR_REQUIRED_MARKERS", "_FRONTDOOR_CORRUPTION_MARKERS", "_suite_mentions",
    "_added_text", "_prompt_integrity_reason", "EVAL_ID_VOCAB_SOURCES_ENV", "_EVAL_ID_KEYS",
    "_LEAKAGE_MIN_ID_LEN", "_LEAKAGE_MIN_NUMERIC_ID_LEN", "_LEAKAGE_MIN_FAMILY_PREFIX",
    "_LEAKAGE_MIN_FAMILY_MEMBERS", "_LEAKAGE_MIN_STEM", "_LEAKAGE_MAX_REPORTED",
    "_LEAKAGE_GENERIC_RE", "_ID_FAMILY_TAIL_RE", "_ID_NATIVE_FAMILY_RE", "_ID_STEM_RE",
    "_default_eval_id_sources", "_iter_eval_rows", "stable_question_qid",
    "_is_identifier_shaped", "_is_word_char", "_CONTENT_NGRAM", "_CONTENT_TEMPLATE_DF",
    "_CONTENT_LONG_ROW_BYTES", "_CONTENT_LONG_EDGE_SHINGLES", "_CONTENT_WINNOW",
    "_CONTENT_CHUNK_BYTES", "_CONTENT_SKETCH_BITS", "_CONTENT_SKETCH_FLUSH", "_CONTENT_RARE_TF",
    "_CONTENT_MAX_ROWS", "_ANSWER_MAX_TOKENS", "_ANSWER_MIN_BYTES", "_ANCHOR_MIN_BYTES",
    "_ANSWER_UNSHAPED_MAX_TF", "_SOURCE_MIN_SHINGLE_HITS", "_CONTENT_MIN_UNANCHORED_RUN",
    "_SOURCE_MIN_SHARED_ANCHORS", "_SOURCE_MAX_QIDS", "_ANSWER_PLACEHOLDER_RE", "_HASH_PRIME",
    "_HASH_MASK", "_HASH_M1", "_HASH_M2", "_HASH_LEN_SALT", "_HASH_SEQ_SALT", "_HASH_CONSTS",
    "_np", "_hash_consts", "_mix64", "_hash_token_buffer", "_content_tokens", "_window_hashes",
    "_sequence_hashes", "_packed_lookup", "_row_label_id", "_anchor_mask", "EvalContentIndex",
    "_EvalContentIndexBuilder", "EvalIdVocabulary", "_EVAL_ID_VOCAB_CACHE",
    "EVAL_ID_VOCAB_NEG_TTL_ENV", "DEFAULT_EVAL_ID_VOCAB_NEG_TTL_S", "_EVAL_ID_VOCAB_NEG_CACHE",
    "_eval_id_vocab_neg_ttl_s", "clear_eval_id_vocabulary_cache", "describe_eval_id_sources",
    "set_eval_leakage_observer", "_notify_eval_leakage_observer", "load_eval_id_vocabulary",
    "_vocabulary_error", "eval_leakage_reason", "_TRACE_SECTION_RE", "_CONTEXT_SECTION_END_RE",
    "_CONTRASTIVE_ENTRY_RE", "_TRACE_IR_JSON_RE", "_TRACE_QID_FIELD_RE", "_TRACE_ID_FIELDS",
    "_trace_ir_blocks", "_trace_blocks", "_HarnessContent", "_HARNESS_CONTENT_CACHE",
    "_text_shingles_tokens", "_in_sorted", "_span_text", "_resolve_trace_qids",
    "eval_content_leakage_reason", "_SUITE_MENTION_MAX_TF", "_SUITE_PHRASE_MAX_TF",
    "_SUITE_MIN_POOL_TOKENS", "_SUITE_HARNESS_VOCAB_FILES", "_SUITE_SHAPED_RE",
    "_SUITE_RULES_CACHE", "_suite_term_pattern", "_suite_rules", "suite_special_casing_reason",
)
# Rebinding globals that must NOT be re-exported (a stale alias), and the wrapper.
NOT_REEXPORTED = ("_EVAL_LEAKAGE_OBSERVER",)


def _run(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_package_context_reexports_are_the_same_objects():
    from scripts.autopilot.species import prompt_forge as pf
    from src.autopilot_core import mutation_guards as mg

    assert [n for n in MOVED_S2 if getattr(pf, n) is not getattr(mg, n)] == []
    assert all(not hasattr(pf, n) for n in NOT_REEXPORTED)
    assert pf.harness_content is not mg.harness_content  # PromptForge keeps a wrapper


def test_bare_autopilot_context_reexports_are_the_same_objects():
    names = ",".join(repr(n) for n in MOVED_S2)
    out = _run(
        "import sys; sys.path.insert(0, 'scripts/autopilot'); sys.path.insert(0, '.');"
        "import species.prompt_forge as pf; import src.autopilot_core.mutation_guards as mg;"
        f"print(sum(getattr(pf, n) is not getattr(mg, n) for n in ({names},)))"
    )
    assert out == "0"


def test_leakage_observer_is_one_global_across_import_contexts():
    # AutoPilot installs its monitor through the BARE module (autopilot.py); a verdict computed
    # through the package-qualified module must reach it (one object, guaranteed by re-export).
    out = _run(
        "import sys; sys.path.insert(0, 'scripts/autopilot'); sys.path.insert(0, '.');"
        "import species.prompt_forge as bare;"
        "from scripts.autopilot.species import prompt_forge as pkg;"
        "import src.autopilot_core.mutation_guards as mg;"
        "seen = []; bare.set_eval_leakage_observer(lambda ok, err: seen.append((ok, err)));"
        "r = pkg.eval_leakage_reason('x', pkg.EvalIdVocabulary(error='probe'));"
        "print(bare is not pkg, mg._EVAL_LEAKAGE_OBSERVER is not None, seen, r)"
    )
    assert out == "True True [(False, 'probe')] eval_leakage_vocabulary_unavailable:probe"


def test_eval_leakage_monitor_keeps_the_bare_module_and_reaches_the_guard():
    out = _run(
        "import sys; sys.path.insert(0, 'scripts/autopilot'); sys.path.insert(0, '.')\n"
        "import eval_leakage_monitor as mon; import species.prompt_forge as pf\n"
        "import src.autopilot_core.mutation_guards as mg\n"
        "class A:\n"
        "    def raise_alarm(self, *a): return True\n"
        "    def clear_alarm(self, *a): return True\n"
        "m = mon.EvalLeakageMonitor(alarm=A()).install();"
        "print(m.pf is sys.modules['species.prompt_forge'], mg._EVAL_LEAKAGE_OBSERVER == m.observe);"
        "m.uninstall(); print(mg._EVAL_LEAKAGE_OBSERVER is None)"
    )
    assert out == "True True\nTrue"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_harness_content_wrapper_follows_promptforge_roots(tmp_path, monkeypatch):
    from scripts.autopilot.species import prompt_forge as pf
    from src.autopilot_core import mutation_guards as mg

    prompts = tmp_path / "prompts"
    _write(prompts / "a.md", "alpha bravo charlie delta echo foxtrot golf hotel india juliet\n")
    _write(tmp_path / "src" / "m.py", "def kilo_lima_mike(): return 'november oscar papa quebec'\n")
    _write(tmp_path / "src" / "tool_policy.py", "ROMEO = 'sierra tango uniform victor whiskey'\n")
    monkeypatch.setattr(pf, "PROMPTS_DIR", prompts)
    monkeypatch.setattr(pf, "NEW_FILE_MUTATION_ROOT", tmp_path / "src")
    monkeypatch.setattr(pf, "PROJECT_ROOT", tmp_path)
    got = pf.harness_content()
    want = mg.harness_content(prompts, code_root=tmp_path / "src", project_root=tmp_path,
                              code_files=pf.CODE_MUTATION_ALLOWLIST)
    assert got is want  # same file identity -> same cache entry
    expected = mg._text_shingles_tokens([
        (prompts / "a.md").read_text(),
        (tmp_path / "src" / "m.py").read_text(),
        (tmp_path / "src" / "tool_policy.py").read_text(),
    ])
    assert np.array_equal(got.tokens, expected.tokens)
    assert np.array_equal(got.shingles, expected.shingles)


def test_harness_content_defaults_agree():
    from scripts.autopilot.species import prompt_forge as pf
    from src.autopilot_core import mutation_guards as mg

    assert mg._ORCH_ROOT == pf.ORCH_ROOT == pf.PROJECT_ROOT
    assert mg._PROMPTS_DIR == pf.PROMPTS_DIR and mg._CODE_ROOT == pf.NEW_FILE_MUTATION_ROOT
    assert pf.harness_content() is mg.harness_content()


def test_default_eval_id_sources_resolve_to_the_same_paths(monkeypatch):
    from scripts.autopilot.species import prompt_forge as pf
    from src.autopilot_core import mutation_guards as mg

    monkeypatch.delenv(mg.EVAL_ID_VOCAB_SOURCES_ENV, raising=False)
    autopilot_dir = Path(pf.__file__).resolve().parents[1]
    sources = mg._default_eval_id_sources()
    assert sources[0][1] is True and sources[0][0].name == "question_pool.jsonl"
    assert [p for p, _ in sources if p.suffix == ".jsonl"][1:] == sorted(
        (pf.ORCH_ROOT / "benchmarks" / "prompts").glob("core_*.jsonl"))
    assert [p for p, _ in sources[-2:]] == [autopilot_dir / "sentinel_questions.yaml",
                                           autopilot_dir / "tool_sentinels.yaml"]


def test_no_test_patches_a_moved_global_on_prompt_forge():
    # A monkeypatch of a moved name on prompt_forge would silently change nothing.
    aliases = r"(?:pf|prompt_forge|prompt_forge_mod|_prompt_forge_module|forge_module)"
    names = "|".join(re.escape(n) for n in (*MOVED_S2, *NOT_REEXPORTED))
    pattern = re.compile(rf"setattr\(\s*{aliases}\s*,\s*[\"']({names})[\"']")
    offenders = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        for m in pattern.finditer(path.read_text(errors="replace")):
            offenders.append(f"{path.relative_to(ROOT)}:{m.group(1)}")
    assert offenders == []


def test_frontdoor_check_is_shared():
    from scripts.autopilot.species import prompt_forge as pf
    from src.autopilot_core import mutation_guards as mg

    assert pf._prompt_integrity_reason is mg._prompt_integrity_reason
    assert mg._prompt_integrity_reason("other.md", "") is None
    assert mg._prompt_integrity_reason("frontdoor.md", "").startswith(
        "frontdoor_missing_required_markers:")


def test_guard_source_is_not_part_of_the_harness_corpus():
    # The guards' own tables moved into src/ with S1/S2; they must not become "text the harness
    # already carries" (e.g. the suite alias table would turn ``debugbench`` into harness
    # vocabulary and silence the suite special-casing refusal for it).
    from src.autopilot_core import mutation_guards as mg

    harness = mg.harness_content()

    def files_with(token: str) -> int:
        return harness.file_count(mg._content_tokens(token)[3][0])

    assert files_with("_GUARD_SOURCE") == 0  # only mutation_guards.py spells it
    assert files_with("LOOP_JOURNAL_MARKER") >= 1  # positive control: substrate.py is counted
    assert files_with("debugbench") < mg._SUITE_HARNESS_VOCAB_FILES
