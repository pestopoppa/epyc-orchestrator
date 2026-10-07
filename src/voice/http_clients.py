"""Bounded HTTP adapters for the frozen Whisper.cpp and Qwen TTS APIs.

The defaults name the current local service endpoints.  Clients are injected so
the complete request/response contract can be exercised with ``httpx.MockTransport``
without starting either service or making a network request.
"""

from __future__ import annotations

import json
import argparse
import importlib
import io
import os
import stat
import struct
import threading
import wave
from pathlib import Path
from typing import Iterable, Iterator, TextIO

import httpx

from src.voice.contracts import VoiceBackend, VoiceEvent, VoiceTurn
from src.voice.controller import VoiceController


WHISPER_BASE_URL = "http://127.0.0.1:9000"
QWEN_TTS_BASE_URL = "http://127.0.0.1:9002"
WHISPER_PATH = "/inference"
QWEN_TTS_PATH = "/v1/audio/speech"
MAX_WAV_BYTES = 25 * 1024 * 1024
MAX_TRANSCRIPT_CHARS = 1_000_000
MAX_SPEECH_TEXT_CHARS = 262_144
MAX_PCM_BYTES = 64 * 1024 * 1024
MAX_PCM_CHUNKS = 4096
MAX_SSE_BYTES = 4 * 1024 * 1024
MAX_SSE_LINE_BYTES = 1024 * 1024
PCM_SAMPLE_RATE = 24_000
PCM_CHANNELS = 1
PCM_SAMPLE_FORMAT = "s16le"


class VoiceServiceError(RuntimeError):
    """A configured speech-service request failed its transport contract."""


def _bounded_sse_lines(response: httpx.Response) -> Iterator[str]:
    total = 0
    pending = bytearray()
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > MAX_SSE_BYTES:
            raise VoiceServiceError("voice route SSE response exceeded its byte limit")
        pending.extend(chunk)
        while True:
            newline = pending.find(b"\n")
            if newline < 0:
                if len(pending) > MAX_SSE_LINE_BYTES:
                    raise VoiceServiceError("voice route SSE line exceeded its byte limit")
                break
            if newline > MAX_SSE_LINE_BYTES:
                raise VoiceServiceError("voice route SSE line exceeded its byte limit")
            raw = bytes(pending[:newline])
            del pending[: newline + 1]
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            try:
                yield raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise VoiceServiceError("voice route SSE line was not valid UTF-8") from exc
    if pending:
        raise VoiceServiceError("voice route ended with an incomplete SSE line")


def _require_wav(audio: bytes) -> None:
    if not isinstance(audio, bytes):
        raise TypeError("audio must be bytes")
    if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise ValueError("input must be a RIFF/WAVE file")
    if len(audio) < 44 or len(audio) > MAX_WAV_BYTES:
        raise ValueError("WAV input size is outside the accepted bounds")
    declared_size = struct.unpack_from("<I", audio, 4)[0] + 8
    if declared_size != len(audio):
        raise ValueError("RIFF size does not match the WAV input length")
    offset = 12
    fmt: tuple[int, int, int, int, int, int] | None = None
    data_size: int | None = None
    while offset + 8 <= len(audio):
        chunk_id = audio[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", audio, offset + 4)[0]
        body = offset + 8
        end = body + chunk_size
        padded_end = end + (chunk_size & 1)
        if end > len(audio) or padded_end > len(audio):
            raise ValueError("WAV chunk exceeds the declared RIFF boundary")
        if chunk_id == b"fmt ":
            if chunk_size < 16 or fmt is not None:
                raise ValueError("WAV must contain one valid fmt chunk")
            fmt = struct.unpack_from("<HHIIHH", audio, body)
        elif chunk_id == b"data":
            if data_size is not None:
                raise ValueError("WAV must contain one data chunk")
            data_size = chunk_size
        offset = padded_end
    if offset != len(audio) or fmt is None or data_size is None:
        raise ValueError("WAV requires complete fmt and data chunks")
    format_tag, channels, sample_rate, byte_rate, block_align, bits = fmt
    if (format_tag not in {1, 3} or channels < 1 or sample_rate < 1
            or bits not in ({8, 16, 24, 32} if format_tag == 1 else {32, 64})):
        raise ValueError("WAV format is unsupported")
    expected_align = channels * ((bits + 7) // 8)
    if block_align != expected_align or byte_rate != sample_rate * block_align:
        raise ValueError("WAV format rates are inconsistent")
    if data_size == 0 or data_size % block_align:
        raise ValueError("WAV data chunk is empty or not frame-aligned")


class WhisperHttpTranscriber:
    """Call Whisper.cpp's multipart ``/inference`` JSON endpoint."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        base_url: str = WHISPER_BASE_URL,
        timeout_s: float = 90.0,
    ) -> None:
        if not isinstance(client, httpx.Client):
            raise TypeError("client must be an httpx.Client")
        self._client = client
        self._url = base_url.rstrip("/") + WHISPER_PATH
        self._health_url = base_url.rstrip("/") + "/health"
        self._timeout = httpx.Timeout(timeout_s)

    def transcribe(self, audio: bytes, *, language: str | None = None) -> str:
        _require_wav(audio)
        if language is not None and (not isinstance(language, str) or not language.strip()):
            raise ValueError("language must be None or a non-empty string")
        form = {"response_format": "json"}
        if language is not None:
            form["language"] = language
        try:
            with self._client.stream(
                "POST", self._url,
                files={"file": ("turn.wav", audio, "audio/wav")},
                data=form,
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes():
                    if len(body) + len(chunk) > 2 * 1024 * 1024:
                        raise VoiceServiceError("Whisper response exceeded the JSON byte limit")
                    body.extend(chunk)
                payload = json.loads(body)
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            raise VoiceServiceError("Whisper transcription request failed") from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or len(text) > MAX_TRANSCRIPT_CHARS:
            raise VoiceServiceError("Whisper response did not contain bounded text")
        return text

    def health(self) -> bool:
        try:
            response = self._client.get(self._health_url, timeout=httpx.Timeout(2.0))
            response.raise_for_status()
            return response.json() == {"status": "ok"}
        except (httpx.HTTPError, ValueError):
            return False


class QwenTtsHttpSynthesizer:
    """Stream Qwen TTS's documented mono 24 kHz signed-16 PCM response."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        base_url: str = QWEN_TTS_BASE_URL,
        voice: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        if not isinstance(client, httpx.Client):
            raise TypeError("client must be an httpx.Client")
        if voice is not None and (not isinstance(voice, str) or not voice.strip()):
            raise ValueError("voice must be None or a non-empty string")
        self._client = client
        self._url = base_url.rstrip("/") + QWEN_TTS_PATH
        self._health_url = base_url.rstrip("/") + "/health"
        self._voice = voice
        self._timeout = httpx.Timeout(timeout_s)
        self._lock = threading.Lock()
        self._active_response: httpx.Response | None = None
        self._stream_active = False
        self._cancel_requested = threading.Event()

    def stream_pcm(self, text: str, *, language: str | None = None) -> Iterable[bytes]:
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_SPEECH_TEXT_CHARS:
            raise ValueError("speech text must be non-empty and within the size limit")
        if language is not None:
            raise ValueError("the pinned Qwen TTS HTTP contract has no language field")
        payload: dict[str, str] = {"input": text, "response_format": "pcm"}
        if self._voice is not None:
            payload["voice"] = self._voice
        with self._lock:
            if self._stream_active:
                raise VoiceServiceError("a Qwen TTS stream is already active")
            self._stream_active = True
            self._cancel_requested.clear()
        return _ReservedPcmStream(self, payload)

    def _stream(self, payload: dict[str, str]) -> Iterable[bytes]:
        total = 0
        try:
            if self._cancel_requested.is_set():
                return
            with self._client.stream(
                "POST", self._url, json=payload, timeout=self._timeout
            ) as response:
                with self._lock:
                    self._active_response = response
                    cancelled = self._cancel_requested.is_set()
                if cancelled:
                    return
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type != "audio/pcm":
                    raise VoiceServiceError("Qwen TTS returned a non-PCM response")
                for index, chunk in enumerate(response.iter_bytes()):
                    if self._cancel_requested.is_set():
                        return
                    if index >= MAX_PCM_CHUNKS:
                        raise VoiceServiceError("Qwen TTS exceeded the PCM chunk limit")
                    if not isinstance(chunk, bytes) or not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_PCM_BYTES:
                        raise VoiceServiceError("Qwen TTS exceeded the PCM byte limit")
                    yield chunk
        except httpx.HTTPError as exc:
            if not self._cancel_requested.is_set():
                raise VoiceServiceError("Qwen TTS request failed") from exc
        finally:
            with self._lock:
                self._active_response = None
                self._stream_active = False

    def _abandon_unstarted_stream(self) -> None:
        with self._lock:
            if self._active_response is None:
                self._cancel_requested.set()
                self._stream_active = False

    def cancel(self) -> None:
        self._cancel_requested.set()
        with self._lock:
            response = self._active_response
        if response is not None:
            response.close()

    def health(self) -> bool:
        try:
            response = self._client.get(self._health_url, timeout=httpx.Timeout(2.0))
            response.raise_for_status()
            return response.json() == {"status": "ok"}
        except (httpx.HTTPError, ValueError):
            return False


class _ReservedPcmStream:
    """Own a reserved stream even when callers close it before first iteration."""

    def __init__(self, owner: QwenTtsHttpSynthesizer, payload: dict[str, str]) -> None:
        self._owner = owner
        self._iterator = iter(owner._stream(payload))
        self._started = False
        self._closed = False

    def __iter__(self) -> "_ReservedPcmStream":
        return self

    def __next__(self) -> bytes:
        if self._closed:
            raise StopIteration
        self._started = True
        try:
            return next(self._iterator)
        except StopIteration:
            self._closed = True
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._iterator, "close", None)
        if close is not None:
            close()
        if not self._started:
            self._owner._abandon_unstarted_stream()


class VoiceTurnSseClient:
    """Adapt the first-party ``/v1/voice/turn`` SSE events to VoiceEvent values."""

    _EVENT_MAP = {
        "answer.delta": "text_delta",
        "display": "display",
        "preserve": "preserve",
        "done": "end",
        "error": "error",
    }

    def __init__(
        self,
        client: httpx.Client,
        *,
        base_url: str = "http://127.0.0.1:8000",
        timeout_s: float = 120.0,
    ) -> None:
        if not isinstance(client, httpx.Client):
            raise TypeError("client must be an httpx.Client")
        self._client = client
        self._url = base_url.rstrip("/") + "/v1/voice/turn"
        self._health_url = base_url.rstrip("/") + "/health"
        self._timeout = httpx.Timeout(timeout_s)
        self._state_lock = threading.Lock()
        self._active_response: httpx.Response | None = None
        self._stream_active = False
        self._cancel_requested = threading.Event()

    def stream_turn(
        self,
        transcript: str,
        *,
        session_id: str,
        turn_id: str,
        response_mode: str = "normal",
        must_preserve: tuple[str, ...] = (),
        response_goal: str = "spoken",
        language: str | None = None,
        conversation_context: str | None = None,
    ) -> Iterator[VoiceEvent]:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        if not isinstance(turn_id, str) or not turn_id.strip() or len(turn_id) > 256:
            raise ValueError("turn_id must be non-empty and at most 256 characters")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("transcript must be non-empty")
        if response_goal not in {"spoken", "display"}:
            raise ValueError("response_goal must be spoken or display")
        if response_mode not in {"normal", "verbatim"}:
            raise ValueError("response_mode must be normal or verbatim")
        if conversation_context is not None and not isinstance(conversation_context, str):
            raise TypeError("conversation_context must be a string or None")
        if language is not None and (not isinstance(language, str) or not language.strip()):
            raise ValueError("language must be None or a non-empty string")
        if any(not isinstance(value, str) or not value.strip() for value in must_preserve):
            raise ValueError("must_preserve values must be non-empty strings")
        payload: dict[str, object] = {
            "session_id": session_id,
            "user_request": transcript,
            "response_goal": response_goal,
            "response_mode": response_mode,
            "must_preserve": list(must_preserve),
            "cancel_token": turn_id,
        }
        if conversation_context is not None:
            payload["conversation_context"] = conversation_context
        if language is not None:
            payload["language"] = language
        with self._state_lock:
            if self._stream_active:
                raise VoiceServiceError("a voice route stream is already active")
            self._stream_active = True
            self._cancel_requested.clear()
        seen_turn: str | None = None
        expected_sequence = 1
        terminal = False
        was_cancelled = False
        try:
            with self._client.stream(
                "POST", self._url, json=payload,
                headers={"accept": "text/event-stream", "x-session-id": session_id},
                timeout=self._timeout,
            ) as response:
                with self._state_lock:
                    self._active_response = response
                    cancelled = self._cancel_requested.is_set()
                if cancelled:
                    return
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type != "text/event-stream":
                    raise VoiceServiceError("voice route did not return an SSE response")
                event_name: str | None = None
                data_lines: list[str] = []
                buffered_chars = 0
                for line in _bounded_sse_lines(response):
                    # A transport chunk can contain several complete SSE events.
                    # Closing the response does not discard lines already buffered
                    # from that chunk, so suppress the tail after cancellation too.
                    if self._cancel_requested.is_set():
                        was_cancelled = True
                        return
                    if line == "":
                        if event_name is None and not data_lines:
                            continue
                        if event_name not in self._EVENT_MAP or not data_lines:
                            raise VoiceServiceError("voice route emitted malformed SSE")
                        try:
                            item = json.loads("\n".join(data_lines))
                        except (TypeError, json.JSONDecodeError) as exc:
                            raise VoiceServiceError("voice route emitted invalid SSE JSON") from exc
                        if not isinstance(item, dict):
                            raise VoiceServiceError("voice route emitted a non-object event")
                        event_turn_id = item.get("turn_id")
                        sequence = item.get("sequence")
                        if (not isinstance(event_turn_id, str) or not event_turn_id
                                or not isinstance(sequence, int) or isinstance(sequence, bool)
                                or sequence != expected_sequence):
                            raise VoiceServiceError("voice route event identity/sequence mismatch")
                        if seen_turn is None:
                            seen_turn = event_turn_id
                        elif event_turn_id != seen_turn:
                            raise VoiceServiceError("voice route changed turn identity midstream")
                        kind = self._EVENT_MAP[event_name]
                        if kind == "text_delta":
                            value = item.get("text")
                            if not isinstance(value, str):
                                raise VoiceServiceError("voice route emitted malformed text delta")
                            event = VoiceEvent(kind, value)
                        elif kind == "display":
                            value = item.get("payload")
                            if not isinstance(value, dict):
                                raise VoiceServiceError("voice route emitted malformed display data")
                            event = VoiceEvent(kind, value)
                        elif kind == "preserve":
                            values = item.get("values")
                            if (not isinstance(values, list)
                                    or any(not isinstance(value, str) for value in values)):
                                raise VoiceServiceError("voice route emitted malformed preserve data")
                            event = VoiceEvent(kind, tuple(values))
                        elif kind == "error":
                            code = item.get("code")
                            if not isinstance(code, str) or not code:
                                raise VoiceServiceError("voice route emitted malformed error data")
                            event = VoiceEvent(kind, code)
                        else:
                            event = VoiceEvent(kind)
                        if (kind in {"end", "error"}
                                and item.get("cancel_token") != turn_id):
                            raise VoiceServiceError("voice route cancellation token did not match turn")
                        expected_sequence += 1
                        terminal = kind in {"end", "error"}
                        if self._cancel_requested.is_set():
                            was_cancelled = True
                            return
                        yield event
                        if terminal:
                            return
                        event_name = None
                        data_lines = []
                        buffered_chars = 0
                        continue
                    if line.startswith(":"):
                        continue
                    field, _, value = line.partition(":")
                    if value.startswith(" "):
                        value = value[1:]
                    if field == "event":
                        if event_name is not None:
                            raise VoiceServiceError("voice route emitted duplicate SSE event field")
                        event_name = value
                    elif field == "data":
                        buffered_chars += len(value)
                        if buffered_chars > 1_048_576:
                            raise VoiceServiceError("voice route SSE event exceeded its size limit")
                        data_lines.append(value)
                if event_name is not None or data_lines:
                    raise VoiceServiceError("voice route ended with an incomplete SSE event")
                was_cancelled = self._cancel_requested.is_set()
        except httpx.HTTPError as exc:
            was_cancelled = self._cancel_requested.is_set()
            if not was_cancelled:
                raise VoiceServiceError("voice route request failed") from exc
        finally:
            with self._state_lock:
                self._active_response = None
                self._stream_active = False
            self._cancel_requested.clear()
        if not terminal and not was_cancelled:
            raise VoiceServiceError("voice route ended without a terminal event")

    def inject(self, text: str) -> None:
        raise VoiceServiceError("the first-party voice route has no inject endpoint")

    def cancel_generation(self) -> None:
        self._cancel_requested.set()
        with self._state_lock:
            response = self._active_response
        if response is not None:
            response.close()

    def cancel_orchestrator(self) -> None:
        # This API has no separate remote cancel operation: closing its SSE
        # response is the route's request-disconnect cancellation signal.
        self.cancel_generation()

    def health(self) -> bool:
        try:
            with self._client.stream("GET", self._health_url, timeout=httpx.Timeout(2.0)) as response:
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes():
                    if len(body) + len(chunk) > 1024 * 1024:
                        return False
                    body.extend(chunk)
                payload = json.loads(body)
                return isinstance(payload, dict) and payload.get("status") == "ok"
        except (httpx.HTTPError, ValueError):
            return False


class VoiceEventJsonlRecorder:
    """Write bounded, privacy-minimal event metadata for a WAV/PCM harness.

    Text, transcript, protected literals and audio bytes are intentionally not
    copied into the journal.  It records event order/type and byte/character
    sizes only, so a control harness can inspect completion/error behavior
    without turning the log into a transcript or an audio artifact.
    """

    def __init__(self, sink: TextIO, *, max_events: int = 8192) -> None:
        self._sink = sink
        self._max_events = max_events
        self._sequence = 0

    def record(self, event: object) -> None:
        if not isinstance(event, VoiceEvent):
            raise TypeError("event must be a VoiceEvent")
        if self._sequence >= self._max_events:
            raise VoiceServiceError("voice event journal exceeded its event limit")
        payload = event.payload
        if isinstance(payload, bytes):
            size = len(payload)
        elif isinstance(payload, str):
            size = len(payload)
        elif payload is None:
            size = 0
        else:
            size = None
        if size is not None and size > MAX_PCM_BYTES:
            raise VoiceServiceError("voice event payload exceeded its size limit")
        row = {"sequence": self._sequence, "kind": event.kind, "payload_size": size}
        self._sink.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        self._sequence += 1


class VoiceHttpHarness:
    """Drive an injected voice backend from WAV input and journal PCM events."""

    def __init__(self, backend: VoiceBackend, recorder: VoiceEventJsonlRecorder) -> None:
        self._backend = backend
        self._recorder = recorder

    def stream(self, turn: VoiceTurn) -> Iterable[VoiceEvent]:
        _require_wav(turn.audio_chunk)
        total_pcm = 0
        terminal = False
        for index, event in enumerate(self._backend.respond(turn)):
            if index >= MAX_PCM_CHUNKS:
                raise VoiceServiceError("voice harness exceeded its event limit")
            if not isinstance(event, VoiceEvent):
                raise VoiceServiceError("voice backend emitted a malformed event")
            if event.kind not in {
                "text_delta", "display", "preserve", "audio_chunk", "tool_call", "end", "error"
            }:
                raise VoiceServiceError("voice backend emitted an unknown event kind")
            if event.kind == "audio_chunk":
                if not isinstance(event.payload, bytes):
                    raise VoiceServiceError("voice backend emitted a malformed PCM chunk")
                total_pcm += len(event.payload)
                if total_pcm > MAX_PCM_BYTES:
                    raise VoiceServiceError("voice harness exceeded its PCM byte limit")
            elif event.kind in {"end", "error"}:
                terminal = True
            self._recorder.record(event)
            yield event
            if terminal:
                return
        if not terminal:
            raise VoiceServiceError("voice backend ended without a terminal event")


class VoiceWavFileHarness:
    """Run one WAV turn through the existing controller and emit a WAV plus JSONL."""

    def __init__(
        self,
        controller: VoiceController,
        recorder: VoiceEventJsonlRecorder,
    ) -> None:
        self._controller = controller
        self._recorder = recorder

    def run(
        self,
        wav_input: bytes,
        *,
        session_id: str,
        turn_id: str,
        language: str | None = None,
    ) -> tuple[bytes, bytes]:
        _require_wav(wav_input)
        turn = VoiceTurn(
            turn_id=turn_id,
            session_id=session_id,
            audio_chunk=wav_input,
            language=language,
        )
        text_parts: list[str] = []
        text_chars = 0
        pcm_output = bytearray()
        total_pcm = 0
        terminal = False
        events = iter(self._controller.respond(turn))
        try:
            for index, event in enumerate(events):
                if index >= MAX_PCM_CHUNKS:
                    raise VoiceServiceError("voice file harness exceeded its event limit")
                if not isinstance(event, VoiceEvent):
                    raise VoiceServiceError("voice controller emitted a malformed event")
                self._recorder.record(event)
                if event.kind == "text_delta":
                    if not isinstance(event.payload, str):
                        raise VoiceServiceError("voice controller emitted malformed text")
                    text_chars += len(event.payload)
                    if text_chars > MAX_SPEECH_TEXT_CHARS:
                        raise VoiceServiceError("voice answer exceeded the speech text limit")
                    text_parts.append(event.payload)
                elif event.kind == "error":
                    raise VoiceServiceError("voice controller rejected the turn")
                elif event.kind == "audio_chunk":
                    if not isinstance(event.payload, bytes) or len(event.payload) % 2:
                        raise VoiceServiceError("voice controller emitted malformed s16le PCM")
                    total_pcm += len(event.payload)
                    if total_pcm > MAX_PCM_BYTES:
                        raise VoiceServiceError("voice controller exceeded the PCM byte limit")
                    pcm_output.extend(event.payload)
                elif event.kind == "end":
                    terminal = True
                    break
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                close()
        if not terminal:
            raise VoiceServiceError("voice controller ended without a terminal event")
        answer = "".join(text_parts)
        if not answer.strip():
            raise VoiceServiceError("voice controller produced no spoken text")
        if total_pcm == 0:
            raise VoiceServiceError("voice controller produced no PCM audio")
        wav_output = io.BytesIO()
        with wave.open(wav_output, "wb") as writer:
            writer.setnchannels(PCM_CHANNELS)
            writer.setsampwidth(2)
            writer.setframerate(PCM_SAMPLE_RATE)
            writer.writeframes(bytes(pcm_output))
        return wav_output.getvalue(), bytes(pcm_output)


def _read_wav_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_WAV_BYTES:
            raise ValueError("WAV input must be a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(MAX_WAV_BYTES + 1)
        if len(data) > MAX_WAV_BYTES:
            raise ValueError("WAV input exceeds the size limit")
        _require_wav(data)
        return data
    finally:
        os.close(fd)


def _write_new_file(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(fd)
    except Exception:
        os.close(fd)
        try:
            path.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(fd)


def wav_harness_main(argv: list[str] | None = None) -> int:
    """CLI entry point; a deployment supplies its controller/service factory."""
    parser = argparse.ArgumentParser(description="Run one bounded WAV voice turn")
    parser.add_argument("--factory", required=True, help="Python module:function returning components")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pcm-output", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--turn-id", required=True)
    parser.add_argument("--language")
    args = parser.parse_args(argv)
    targets = [args.input.resolve(), args.output.resolve(), args.pcm_output.resolve(), args.events.resolve()]
    if len(set(targets)) != len(targets):
        parser.error("input, WAV, PCM, and event paths must be distinct")
    module_name, separator, function_name = args.factory.partition(":")
    if not separator or not module_name or not function_name:
        parser.error("--factory must have module:function form")
    factory = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        parser.error("--factory target is not callable")
    components = factory()
    if not isinstance(components, dict):
        parser.error("factory must return a component dictionary")
    try:
        controller = components["controller"]
    except KeyError as exc:
        parser.error(f"factory omitted required component {exc.args[0]!r}")
    wav_input = _read_wav_file(args.input)
    event_fd = os.open(
        args.events,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(event_fd, "w", encoding="utf-8", closefd=False) as event_file:
            harness = VoiceWavFileHarness(
                controller,
                VoiceEventJsonlRecorder(event_file),
            )
            wav_output, pcm_output = harness.run(
                wav_input,
                session_id=args.session_id,
                turn_id=args.turn_id,
                language=args.language,
            )
    finally:
        os.close(event_fd)
    _write_new_file(args.output, wav_output)
    _write_new_file(args.pcm_output, pcm_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(wav_harness_main())
