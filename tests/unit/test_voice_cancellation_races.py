"""Deterministic transport controls for cancellation across SSE generations."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import httpx

from src.voice.cascade import CascadeBackend
from src.voice.contracts import RetainCancelChoice, VoiceEvent, VoiceTurn
from src.voice.controller import VoiceController
from src.voice.http_clients import VoiceTurnSseClient


class _BufferedTurn(httpx.SyncByteStream):
    def __init__(self, token):
        self.token = token
        self.closed = False

    def __iter__(self):
        # A closed HTTP response can still have complete SSE lines buffered.
        rows = [
            ("answer.delta", {"text": "first"}),
            ("answer.delta", {"text": "tail"}),
            ("done", {"cancel_token": self.token}),
        ]
        yield "".join(
            f"event: {kind}\ndata: {json.dumps({'turn_id': self.token, 'sequence': i, **body})}\n\n"
            for i, (kind, body) in enumerate(rows, 1)
        ).encode()

    def close(self):
        self.closed = True


def _client():
    turns = {}

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json={"status": "ok"})
        token = json.loads(request.content)["cancel_token"]
        stream = turns[token] = _BufferedTurn(token)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream
        )

    client = httpx.Client(transport=httpx.MockTransport(handle))
    return client, VoiceTurnSseClient(client), turns


def _thread(errors, target, name):
    def run():
        try:
            target()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, name=name)
    worker.start()
    return worker


def _join(worker, errors):
    worker.join(3)
    assert not worker.is_alive(), "fixture worker did not drain"
    assert not errors, errors


def test_old_retirement_cannot_erase_typed_controller_cancel_of_replacement():
    client, adapter, turns = _client()
    retired = threading.Event()
    release_retirement = threading.Event()
    errors = []
    lock = threading.Lock()

    class RetirementBarrier:
        def __enter__(self):
            lock.acquire()

        def __exit__(self, *_args):
            retiring = (
                threading.current_thread().name == "old-retirement"
                and not adapter._stream_active
            )
            lock.release()
            if retiring:
                retired.set()
                assert release_retirement.wait(3)

    adapter._state_lock = RetirementBarrier()
    old = adapter.stream_turn("old", session_id="old-session", turn_id="old")
    assert next(old) == VoiceEvent("text_delta", "first")
    worker = _thread(errors, lambda: list(old), "old-retirement")
    events_at_client_boundary = []
    native_stream = adapter.stream_turn

    def observe(*args, **kwargs):
        for event in native_stream(*args, **kwargs):
            events_at_client_boundary.append(event)
            yield event

    adapter.stream_turn = observe
    synthesised = []
    synthesizer = SimpleNamespace(
        health=lambda: True,
        cancel=lambda: None,
        stream_pcm=lambda text, **_kwargs: synthesised.append(text) or iter((b"pcm",)),
    )
    cascade = CascadeBackend(
        SimpleNamespace(health=lambda: True),
        adapter,
        synthesizer,
        tts_text_normalization=False,
        tts_respell=False,
    )
    controller = VoiceController(
        cascade, SimpleNamespace(stop_queued_audio=lambda *_args: None)
    )
    try:
        assert retired.wait(3)
        stream = controller.respond(
            VoiceTurn(
                "replacement",
                "new-session",
                b"",
                transcript="new",
            )
        )
        assert next(stream) == VoiceEvent("text_delta", "first")
        controller.apply_retain_cancel_choice(
            "new-session",
            "replacement",
            RetainCancelChoice.RETAIN,
        )
        assert not turns["replacement"].closed
        controller.apply_retain_cancel_choice(
            "new-session",
            "replacement",
            RetainCancelChoice.CANCEL,
        )
        assert turns["replacement"].closed
        release_retirement.set()
        _join(worker, errors)
        assert list(stream) == []
        assert events_at_client_boundary == [VoiceEvent("text_delta", "first")]
        assert synthesised == []
    finally:
        release_retirement.set()
        worker.join(3)
        client.close()


def test_cancel_signal_and_response_snapshot_cannot_target_different_generations():
    client, adapter, turns = _client()
    signalled = threading.Event()
    replacement_ready = threading.Event()
    errors = []
    replacement = []
    state_lock = adapter._state_lock

    class CancellationBarrier(threading.Event):
        def set(self):
            super().set()
            signalled.set()
            # Let retirement and replacement happen during an unprotected set.
            # With atomic cancellation they cannot enter until its snapshot is
            # captured, so the barrier must not wait while holding that lock.
            if not state_lock.locked():
                assert replacement_ready.wait(3)

    adapter._cancel_requested = CancellationBarrier()
    old = adapter.stream_turn("old", session_id="old-session", turn_id="old")
    assert next(old) == VoiceEvent("text_delta", "first")

    def replace():
        assert signalled.wait(3)
        assert list(old) == []
        new = adapter.stream_turn(
            "new", session_id="new-session", turn_id="replacement"
        )
        replacement.append(new)
        assert next(new) == VoiceEvent("text_delta", "first")
        replacement_ready.set()

    replacer = _thread(errors, replace, "replacement")
    canceller = _thread(errors, adapter.cancel_orchestrator, "cancellation")
    try:
        _join(canceller, errors)
        _join(replacer, errors)
        # The cancel callback owns old's transport. Replacement must remain
        # alive, including its already-buffered tail and natural completion.
        assert turns["old"].closed
        assert not turns["replacement"].closed
        assert list(replacement[0]) == [
            VoiceEvent("text_delta", "tail"),
            VoiceEvent("end"),
        ]
    finally:
        replacement_ready.set()
        canceller.join(3)
        replacer.join(3)
        for stream in replacement:
            stream.close()
        client.close()
