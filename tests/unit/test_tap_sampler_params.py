"""SPEC-DTREE-DICE step 1: the tap START event carries the RESOLVED sampler params."""

import json
from types import SimpleNamespace


from src import spec_policy
from src.backends.llama_server import LlamaServerBackend, resolve_sampler_params
from src.inference_tap import TapWriter
from src.llm_primitives.inference import _build_tap_metadata


def _role(accel_temp=None, default_temp=None):
    gen = None if default_temp is None else SimpleNamespace(temperature=default_temp)
    return SimpleNamespace(
        acceleration=SimpleNamespace(temperature=accel_temp), generation_defaults=gen
    )


def _req(**kw):
    base = dict(temperature=None, top_k=None, top_p=None, speculative_n_max=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _start_event(tmp_path, role_config, request):
    meta = _build_tap_metadata(
        request, role="coder", backend_url="u", port=8080, task_id="t", role_config=role_config
    )
    path = str(tmp_path / "tap.log")
    w = TapWriter(path, metadata=meta)
    w.write_header("coder")
    w.write_prompt("hello")
    events = [json.loads(x) for x in (tmp_path / "tap.log.events.jsonl").read_text().splitlines()]
    return next(e for e in events if e["event"] == "start")


def test_request_override_wins(tmp_path):
    start = _start_event(
        tmp_path,
        _role(accel_temp=0.5, default_temp=0.2),
        _req(temperature=0.7, top_k=10, top_p=0.5),
    )
    assert start["sampler"] == {
        "temperature": 0.7,
        "top_k": 10,
        "top_p": 0.5,
        "min_p": None,
        "presence_penalty": None,
        "repeat_penalty": 1.1,
        "speculative_n_max": None,
        "speculative_requested": None,
    }


def test_role_default_when_request_silent(tmp_path):
    start = _start_event(tmp_path, _role(default_temp=0.2), _req())
    assert start["sampler"]["temperature"] == 0.2
    assert (start["sampler"]["top_k"], start["sampler"]["top_p"]) == (40, 0.95)
    # acceleration beats generation_defaults
    assert (
        resolve_sampler_params(_role(accel_temp=0.5, default_temp=0.2), _req())["temperature"]
        == 0.5
    )


def test_no_temperature_anywhere_is_greedy(tmp_path):
    start = _start_event(tmp_path, _role(), _req())
    assert start["sampler"]["temperature"] == 0.0
    assert start["sampler"]["min_p"] is None


def test_speculative_request_recorded():
    assert (
        resolve_sampler_params(_role(), _req(speculative_n_max=0))["speculative_requested"] is False
    )
    r = resolve_sampler_params(_role(), _req(speculative_n_max=4))
    assert (r["speculative_n_max"], r["speculative_requested"]) == (4, True)
    with spec_policy.speculative_n_max_override(0):
        assert resolve_sampler_params(_role(), _req())["speculative_requested"] is False


def test_recorded_values_equal_the_wire():
    """The tap record and the payload come from one resolver, so they cannot drift."""
    for role, req in [(_role(default_temp=0.2), _req()), (_role(), _req(temperature=0.9, top_k=5))]:
        payload: dict = {}
        LlamaServerBackend._apply_deterministic_sampling(None, payload, role, req)  # type: ignore[arg-type]
        s = resolve_sampler_params(role, req)
        assert (payload["temperature"], payload["top_k"], payload["top_p"]) == (
            s["temperature"],
            s["top_k"],
            s["top_p"],
        )


def test_key_absent_without_role_config(tmp_path):
    """Backward compatible: no role_config -> no ``sampler`` key (reads as 'predates the hook')."""
    meta = _build_tap_metadata(_req(), role="coder", backend_url="u", port=1, task_id="t")
    assert "sampler" not in meta


def test_official_sampling_from_generation_defaults():
    """OFFICIAL-SAMPLING-20261124: declared vendor fields reach the resolved sampler and the wire."""
    gen = SimpleNamespace(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                          presence_penalty=1.5, repeat_penalty=1.0)
    role = SimpleNamespace(acceleration=SimpleNamespace(temperature=None), generation_defaults=gen)
    s = resolve_sampler_params(role, _req())
    assert (s["temperature"], s["top_p"], s["top_k"], s["min_p"]) == (1.0, 0.95, 20, 0.0)
    assert (s["presence_penalty"], s["repeat_penalty"]) == (1.5, 1.0)
    assert resolve_sampler_params(role, _req(top_k=5))["top_k"] == 5
    payload: dict = {}
    LlamaServerBackend._apply_deterministic_sampling(None, payload, role, SimpleNamespace(**{**vars(_req()), "seed": None}))  # type: ignore[arg-type]
    assert payload["min_p"] == 0.0 and payload["presence_penalty"] == 1.5 and payload["repeat_penalty"] == 1.0
