"""KVU-11c: future TTS launch preparation, with no server or inference."""
import hashlib
from pathlib import Path

import pytest

from scripts.server import orchestrator_stack as stack
from scripts.server.stack_manifest import AUX_SERVICES, _MANIFEST
from scripts.voice.check_speech_layout import _threads


def test_current_recipe_unchanged_and_not_opted_in():
    svc = AUX_SERVICES['tts']
    assert stack._tts_thread_flag(svc) is None
    assert svc.env['SHIM_NPROCS'] == '32'
    assert svc.env['LD_PRELOAD'].endswith('/nprocs_shim.so')
    assert '--threads' not in svc.argv


@pytest.mark.parametrize('value', [{}, {'threads': 0, 'binary_sha256': 'a'*64},
                                   {'threads': True, 'binary_sha256': 'a'*64},
                                   {'threads': 16, 'binary_sha256': 'bad'}])
def test_bad_thread_control_refused_at_launch(value, monkeypatch):
    entries = [dict(entry) for entry in _MANIFEST['aux_services']]
    next(entry for entry in entries if entry['name'] == 'tts')['tts_thread_flag'] = value
    monkeypatch.setitem(_MANIFEST, 'aux_services', entries)
    with pytest.raises(ValueError, match='tts_thread_flag'):
        stack._tts_thread_flag(AUX_SERVICES['tts'])


def test_flag_path_pins_binary_and_removes_only_shim(tmp_path, monkeypatch):
    from src.registry import kernel_paths
    binary = tmp_path / 'tts-server'
    binary.write_bytes(b'fake-new-binary --threads <n>')
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    monkeypatch.setattr(kernel_paths, 'server_binary', lambda backend: binary)
    monkeypatch.setattr(kernel_paths, 'backend_dir', lambda backend: tmp_path)
    monkeypatch.setattr(kernel_paths, 'backend_ld_library_path', lambda backend: [str(tmp_path)])
    service = AUX_SERVICES['tts']
    flag = {'threads': 16, 'binary_sha256': digest}
    monkeypatch.setattr(stack, '_tts_thread_flag', lambda service: flag)
    argv, libs, tree = stack._resolve_aux_launch(service)
    assert argv[-2:] == ['--threads', '16'] and tree == tmp_path
    env = stack._build_aux_env(service, libs)
    assert 'LD_PRELOAD' not in env and 'SHIM_NPROCS' not in env
    assert env['GGML_BACKEND'] == 'CPU' and env['HIP_VISIBLE_DEVICES'] == '-1'
    assert env['LD_LIBRARY_PATH'] == str(tmp_path)
    no_shim = service._replace(env={'GGML_BACKEND': 'CPU', 'HIP_VISIBLE_DEVICES': '-1'})
    assert 'LD_PRELOAD' not in stack._build_aux_env(no_shim, libs)
    assert _threads({'tts_thread_flag': flag}, 'tts') == 16
    binary.write_bytes(b'changed --threads')
    with pytest.raises(ValueError, match='pinned future'):
        stack._resolve_aux_launch(service)
    binary.write_bytes(b'old executable without the requested option')
    no_flag_digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    flag['binary_sha256'] = no_flag_digest
    with pytest.raises(ValueError, match='pinned future'):
        stack._resolve_aux_launch(service)


def test_unknown_preload_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(stack, '_tts_thread_flag', lambda service: {'threads': 16, 'binary_sha256': 'a'*64})
    svc = AUX_SERVICES['tts']._replace(env={'LD_PRELOAD': '/tmp/other.so'})
    with pytest.raises(ValueError, match='unexpected TTS preload'):
        stack._build_aux_env(svc, [str(tmp_path)])
