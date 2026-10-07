#!/usr/bin/env python3
"""CLI shim for the injected, bounded WAV voice harness."""

from src.voice.http_clients import wav_harness_main


if __name__ == "__main__":
    raise SystemExit(wav_harness_main())
