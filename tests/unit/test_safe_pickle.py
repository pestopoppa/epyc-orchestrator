"""Tests for the hardened checkpoint pickle boundary (D-a).

Layer coverage:
    1. find_class allowlist   — TestAllowlist
    2. HMAC over the blob     — TestHmac
    3. size cap               — TestSizeCap
    4. AST rejects defining serialization dunders — TestAstLayer

See handoffs/active/repl-session-memory-maturity.md D-a.
"""

from __future__ import annotations

import ast
import base64
import pickle

import pytest

from src.repl_environment import safe_pickle
from src.repl_environment.safe_pickle import UnsafePickleError
from src.repl_environment.security import ASTSecurityVisitor


class _Hostile:
    """Mimics what model-authored REPL code could construct."""

    def __reduce__(self):
        return (__import__, ("os",))


class _HostileEval:
    def __reduce__(self):
        return (eval, ("1+1",))


class TestRoundTrip:
    def test_plain_containers(self):
        value = {"a": [1, 2, 3], "b": ("x", "y"), "c": {1, 2}}
        assert safe_pickle.loads(safe_pickle.dumps(value)) == value

    def test_numpy_array(self):
        np = pytest.importorskip("numpy")
        arr = np.arange(12).reshape(3, 4)
        restored = safe_pickle.loads(safe_pickle.dumps(arr))
        assert np.array_equal(restored, arr)
        assert restored.dtype == arr.dtype

    def test_envelope_carries_type_and_size(self):
        env = safe_pickle.dumps([1, 2, 3])
        assert env["type"] == "list"
        assert env["bytes"] > 0
        assert isinstance(env["b64"], str) and isinstance(env["hmac"], str)


class TestAllowlist:
    """Layer 1 — the load-bearing defence."""

    def test_reduce_to_import_is_rejected_at_save(self):
        with pytest.raises(UnsafePickleError, match="not allowlisted"):
            safe_pickle.dumps(_Hostile())

    def test_reduce_to_eval_is_rejected_at_save(self):
        with pytest.raises(UnsafePickleError, match="not allowlisted"):
            safe_pickle.dumps(_HostileEval())

    def test_hostile_blob_smuggled_past_dumps_still_fails_at_load(self):
        """Even a correctly-signed hostile blob must fail closed on load."""
        blob = pickle.dumps(_Hostile(), protocol=safe_pickle.PICKLE_PROTOCOL)
        envelope = {
            "b64": base64.b64encode(blob).decode("ascii"),
            "hmac": safe_pickle._sign(blob),  # valid signature
            "type": "_Hostile",
            "bytes": len(blob),
        }
        with pytest.raises(UnsafePickleError, match="not allowlisted"):
            safe_pickle.loads(envelope)

    def test_repl_defined_classes_cannot_be_reconstructed(self):
        """Model classes live in __main__, which is deliberately not allowlisted."""
        assert not any(mod == "__main__" for mod, _ in safe_pickle.ALLOWED_GLOBALS)

    def test_no_callable_gadgets_on_the_allowlist(self):
        """The allowlist must contain inert data types only."""
        forbidden = {
            ("builtins", "eval"),
            ("builtins", "exec"),
            ("builtins", "__import__"),
            ("builtins", "getattr"),
            ("os", "system"),
            ("subprocess", "Popen"),
            ("operator", "attrgetter"),
            ("operator", "methodcaller"),
            ("functools", "partial"),
            ("copyreg", "_reconstructor"),
        }
        assert not (forbidden & safe_pickle.ALLOWED_GLOBALS)


class TestHmac:
    """Layer 2 — tampering at rest."""

    def test_tampered_payload_is_rejected(self):
        env = safe_pickle.dumps({"k": "v"})
        env["b64"] = base64.b64encode(pickle.dumps({"k": "evil"}, protocol=4)).decode()
        with pytest.raises(UnsafePickleError, match="HMAC mismatch"):
            safe_pickle.loads(env)

    def test_missing_signature_is_rejected(self):
        env = safe_pickle.dumps({"k": "v"})
        del env["hmac"]
        with pytest.raises(UnsafePickleError, match="missing b64 or hmac"):
            safe_pickle.loads(env)

    def test_non_dict_envelope_is_rejected(self):
        with pytest.raises(UnsafePickleError, match="malformed envelope"):
            safe_pickle.loads("not-an-envelope")

    def test_bad_base64_is_rejected(self):
        env = safe_pickle.dumps({"k": "v"})
        env["b64"] = "!!!not base64!!!"
        with pytest.raises(UnsafePickleError, match="bad base64"):
            safe_pickle.loads(env)


class TestSizeCap:
    """Layer 3 — bounded payload."""

    def test_oversize_value_is_rejected_at_save(self):
        big = "x" * (safe_pickle.MAX_PICKLED_BYTES + 1024)
        with pytest.raises(UnsafePickleError, match="too large to save"):
            safe_pickle.dumps(big)

    def test_oversize_payload_is_rejected_at_load_before_hmac(self):
        blob = b"\x00" * (safe_pickle.MAX_PICKLED_BYTES + 1)
        env = {"b64": base64.b64encode(blob).decode(), "hmac": "irrelevant"}
        with pytest.raises(UnsafePickleError, match="exceeds cap"):
            safe_pickle.loads(env)


class TestAstLayer:
    """Layer 4 — refuse the hostile object at construction."""

    @pytest.mark.parametrize(
        "dunder",
        ["__reduce__", "__reduce_ex__", "__getstate__", "__setstate__"],
    )
    def test_defining_a_serialization_dunder_is_flagged(self, dunder):
        code = f"class C:\n    def {dunder}(self):\n        return (print, ())\n"
        visitor = ASTSecurityVisitor()
        visitor.visit(ast.parse(code))
        assert any(dunder in v for v in visitor.violations)

    def test_async_definition_is_also_flagged(self):
        code = "class C:\n    async def __reduce__(self):\n        return None\n"
        visitor = ASTSecurityVisitor()
        visitor.visit(ast.parse(code))
        assert visitor.violations

    @pytest.mark.parametrize(
        "code",
        [
            "class C:\n    __reduce__ = lambda self: (print, ())\n",
            "C = type('C', (), {'__reduce__': lambda s: (print, ())})\n",
        ],
        ids=["class-body-assignment", "type-3-arg"],
    )
    def test_hook_bound_without_a_functiondef_is_flagged(self, code):
        """Binding the hook via assignment or type() must not slip past layer 4."""
        visitor = ASTSecurityVisitor()
        visitor.visit(ast.parse(code))
        assert visitor.violations, f"layer 4 missed: {code!r}"

    def test_ordinary_analysis_code_is_not_flagged(self):
        """The new rule must not create false positives on normal REPL work."""
        code = (
            "class Index:\n"
            "    def __init__(self, docs):\n"
            "        self.docs = docs\n"
            "    def search(self, q):\n"
            "        return [d for d in self.docs if q in d]\n"
            "idx = Index(['a', 'b'])\n"
            "total = sum(len(d) for d in idx.docs)\n"
        )
        visitor = ASTSecurityVisitor()
        visitor.visit(ast.parse(code))
        assert visitor.violations == []


class TestCheckpointIntegration:
    """checkpoint()/restore() must carry pickled values and fail closed."""

    def _env(self):
        from tests.unit.test_repl_state_extended import MockREPLEnvironment

        return MockREPLEnvironment()

    def test_numpy_survives_a_checkpoint_round_trip(self):
        np = pytest.importorskip("numpy")
        env = self._env()
        env._globals = {"artifacts": env.artifacts, "arr": np.arange(6), "n": 3}

        cp = env.checkpoint()
        assert "arr" in cp["pickled_globals"]
        assert cp["user_globals"] == {"n": 3}
        assert "arr" not in cp["skipped_user_globals"]

        fresh = self._env()
        result = fresh.restore(cp)
        assert "arr" in result["restored"]
        assert np.array_equal(fresh._globals["arr"], np.arange(6))

    def test_unpicklable_value_is_reported_with_a_reason(self):
        env = self._env()
        env._globals = {"artifacts": env.artifacts, "fn": lambda x: x}

        cp = env.checkpoint()
        assert "fn" in cp["skipped_user_globals"]
        assert cp["skip_reasons"]["fn"]

        fresh = self._env()
        result = fresh.restore(cp)
        assert "fn" in result["unavailable"]

    def test_tampered_stored_pickle_fails_closed_on_restore(self):
        np = pytest.importorskip("numpy")
        env = self._env()
        env._globals = {"artifacts": env.artifacts, "arr": np.arange(4)}
        cp = env.checkpoint()
        cp["pickled_globals"]["arr"]["hmac"] = "0" * 64

        fresh = self._env()
        result = fresh.restore(cp)
        assert "arr" not in result["restored"]
        assert "arr" not in fresh._globals
        assert "HMAC mismatch" in result["unavailable"]["arr"]


class TestNumpyScalarGuard:
    """multiarray.scalar must never deserialize an object dtype."""

    def test_object_dtype_scalar_is_refused_by_our_guard(self):
        np = pytest.importorskip("numpy")
        guarded = safe_pickle._guarded_numpy_scalar(lambda *a, **k: "REACHED")
        with pytest.raises(UnsafePickleError, match="object-dtype"):
            guarded(np.dtype("O"), b"payload")

    def test_ordinary_dtype_scalar_still_works(self):
        np = pytest.importorskip("numpy")
        guarded = safe_pickle._guarded_numpy_scalar(lambda *a, **k: "REACHED")
        assert guarded(np.dtype("int64"), b"\x00" * 8) == "REACHED"

    def test_numpy_scalar_value_round_trips(self):
        np = pytest.importorskip("numpy")
        assert safe_pickle.loads(safe_pickle.dumps(np.int64(7))) == 7


class TestHmacKeyMaterial:
    """The key must be whole or absent — never a partially-written empty read."""

    def test_key_is_hex_encoded_and_full_length(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ORCHESTRATOR_SESSION_HMAC_KEY", raising=False)
        import src.repl_environment.safe_pickle as sp

        class _Paths:
            sessions_dir = tmp_path

        class _Cfg:
            paths = _Paths()

        monkeypatch.setattr("src.config.get_config", lambda: _Cfg())
        key = sp._resolve_hmac_key()
        assert len(key) == 32
        stored = (tmp_path / sp._HMAC_KEY_FILENAME).read_text().strip()
        assert len(stored) == 64
        assert sp._resolve_hmac_key() == key  # stable across calls

    def test_truncated_key_file_is_not_used_as_a_key(self, tmp_path, monkeypatch):
        """A zero-length or short file must not become an empty HMAC key."""
        monkeypatch.delenv("ORCHESTRATOR_SESSION_HMAC_KEY", raising=False)
        import src.repl_environment.safe_pickle as sp

        class _Paths:
            sessions_dir = tmp_path

        class _Cfg:
            paths = _Paths()

        monkeypatch.setattr("src.config.get_config", lambda: _Cfg())
        (tmp_path / sp._HMAC_KEY_FILENAME).write_text("")
        key = sp._resolve_hmac_key()
        assert key != b""
        assert len(key) == 32


class TestOversizeEnvelope:
    def test_oversize_is_rejected_before_decoding(self):
        huge_b64 = "A" * (safe_pickle.MAX_PICKLED_BYTES * 2)
        with pytest.raises(UnsafePickleError, match="encoded"):
            safe_pickle.loads({"b64": huge_b64, "hmac": "x"})
