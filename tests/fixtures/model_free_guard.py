"""Explicit model-free CI boundary; production executable/endpoint attempts fail sticky."""
import ipaddress
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import traceback
from urllib.parse import urlparse
from functools import wraps
import pytest

ROOTS = tuple(Path(path).resolve() for path in ('/mnt/raid0/llm/llama.cpp', '/mnt/raid0/llm/kernels', '/mnt/raid0/llm/whisper.cpp', '/mnt/raid0/llm/qwentts.cpp'))
NAMES = frozenset(('llama-embedding', 'llama-server', 'llama-cli', 'llama-completion', 'llama-speculative', 'llama-lookup', 'llama-mtmd-cli', 'llama-qwen2vl-cli', 'whisper-cli', 'whisper-server', 'qwentts-cli', 'qwentts-server'))
_violations = []
_installed = False
_session = None
_model_ports = set(range(8090, 8096))


def _environment(env):
    return {os.fsdecode(key): os.fsdecode(value) for key, value in (os.environ if env is None else env).items()}


def _command_position(tokens, env):
    """A simple command's executable; data arguments are never executable candidates."""
    tokens = list(tokens)
    env = dict(env)
    while tokens and '=' in tokens[0] and tokens[0].split('=', 1)[0].isidentifier():
        key, value = tokens.pop(0).split('=', 1)
        env[key] = value
    if tokens and Path(tokens[0]).name in ('env', 'exec', 'command'):
        wrapper = Path(tokens.pop(0)).name
        if wrapper == 'env':
            while tokens and tokens[0].startswith('-'):
                option = tokens.pop(0)
                if option == '--':
                    break
                if option in ('-i', '--ignore-environment'):
                    env = {}
                elif option in ('-u', '--unset') and tokens:
                    env.pop(tokens.pop(0), None)
                elif option.startswith('--unset='):
                    env.pop(option.split('=', 1)[1], None)
                elif option.startswith('-u') and len(option) > 2:
                    env.pop(option[2:], None)
                else:
                    raise ValueError('unsupported env execution option in model-free profile')
            while tokens and '=' in tokens[0] and tokens[0].split('=', 1)[0].isidentifier():
                key, value = tokens.pop(0).split('=', 1)
                env[key] = value
        return _command_position(tokens, env)
    if tokens and Path(tokens[0]).name in ('sh', 'bash', 'dash'):
        for index, argument in enumerate(tokens[1:], 1):
            if argument.startswith('-') and not argument.startswith('--') and 'c' in argument[1:]:
                if index + 1 < len(tokens):
                    return [(tokens[0], env, [])] + _shell_positions(tokens[index + 1], env)
                break
    return [(tokens[0], env, tokens[1:])] if tokens else []


def _shell_positions(command, env):
    lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|()')
    lexer.whitespace_split = True
    lexer.commenters = '#'
    commands, current = [], []
    for token in lexer:
        if token and all(character in ';&|()' for character in token):
            commands.extend(_command_position(current, env))
            current = []
        else:
            current.append(token)
    return commands + _command_position(current, env)


def forbidden_executable(executable, argv, env=None, cwd=None):
    environment = _environment(env)
    value = os.fsdecode(executable) if executable is not None else None
    arguments = [os.fsdecode(item) for item in argv] if isinstance(argv, (list, tuple)) else []
    candidates = [(value, environment, arguments[1:])] if value is not None else _command_position(arguments, environment)
    if value is not None and Path(value).name in ('sh', 'bash', 'dash'):
        command_options = [index for index, argument in enumerate(arguments[1:], 1)
                           if argument.startswith('-') and not argument.startswith('--') and 'c' in argument[1:]]
        if command_options and command_options[0] + 1 < len(arguments):
            candidates += _shell_positions(arguments[command_options[0] + 1], environment)
    if value is not None and Path(value).name in ('env', 'exec', 'command'):
        candidates += _command_position([value, *arguments[1:]], environment)
    for command, child_env, command_args in candidates:
        directory = Path(os.fsdecode(cwd)).resolve() if cwd is not None else Path.cwd()
        search_path = os.pathsep.join(str(Path(part) if Path(part).is_absolute() else directory / part)
                                      for part in child_env.get('PATH', os.defpath).split(os.pathsep))
        candidate = Path(command)
        if not candidate.is_absolute() and '/' in command:
            candidate = directory / candidate
        resolved = Path(shutil.which(str(candidate), path=search_path) or candidate).resolve()
        if resolved.name.startswith('llama-') or (resolved.name in NAMES and any(resolved.is_relative_to(root) for root in ROOTS)):
            return str(resolved)
        if resolved.name not in ('echo', 'printf', 'sh', 'bash', 'dash', 'env', 'exec', 'command'):
            model_args = [argument for argument in command_args if Path(argument).suffix.lower() == '.gguf']
            if model_args:
                return f'{resolved}: model-file argument {model_args[0]}'
    return None


def forbidden_endpoint(address):
    if not isinstance(address, tuple) or len(address) < 2 or address[1] not in _model_ports:
        return False
    value = address[0]
    if isinstance(value, bytes):
        try:
            value = value.decode('ascii')
        except UnicodeDecodeError:
            return False
    try:
        host = ipaddress.ip_address(value)
        return host.is_loopback or bool(getattr(host, 'ipv4_mapped', None) and host.ipv4_mapped.is_loopback)
    except ValueError:
        return value == 'localhost'


def _refuse(event, target):
    violation = {'event': event, 'target': target if isinstance(target, str) else repr(target), 'test': os.environ.get('PYTEST_CURRENT_TEST'), 'stack': traceback.format_stack()}
    _violations.append(violation)
    if _session is not None:
        _session.shouldstop = 'model-free CI boundary violation; see owned guard receipt'
    directory = Path(os.environ['MODEL_LAUNCH_GUARD_OUTPUT'])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'violation-{os.getpid()}-{len(_violations)}.json').write_text(json.dumps(violation, indent=2) + '\n')
    raise RuntimeError(f'Model-free test boundary refuses: {target}')


def _audit(event, args):
    if os.environ.get('EPYC_MODEL_FREE_CI') != '1':
        return
    if event == 'socket.connect':
        if forbidden_endpoint(args[1]):
            _refuse(event, args[1])
        return
    if event not in ('subprocess.Popen', 'os.posix_spawn', 'os.exec', 'os.system'):
        return
    if event == 'os.system':
        executable, argv, env = '/bin/sh', ['sh', '-c', os.fsdecode(args[0])], None
    else:
        executable, argv = args[:2]
        env = args[3] if event == 'subprocess.Popen' else args[2]
    forbidden = forbidden_executable(executable, argv, env, args[2] if event == 'subprocess.Popen' else None)
    if forbidden is not None:
        _refuse(event, forbidden)


def pytest_configure(config):
    global _installed
    if os.environ.get('EPYC_MODEL_FREE_CI') != '1':
        return
    # Only actual registry serving endpoints; owned ephemeral fixture ports remain usable.
    import yaml
    registry = Path(os.environ['MODEL_FREE_MODEL_REGISTRY'])
    data = yaml.safe_load(registry.read_text())
    for role in data.get('server_mode', {}).values():
        if isinstance(role, dict):
            if isinstance(role.get('port'), int):
                _model_ports.add(role['port'])
            for key in ('url', 'endpoint'):
                parsed = urlparse(role.get(key, ''))
                if parsed.hostname in ('localhost', '127.0.0.1', '::1') and parsed.port:
                    _model_ports.add(parsed.port)
    default = data.get('runtime_defaults', {}).get('server_defaults', {}).get('port')
    if isinstance(default, int):
        _model_ports.add(default)
    if not _installed:
        sys.addaudithook(_audit)
        _installed = True


def pytest_sessionstart(session):
    global _session
    _session = session


def pytest_sessionfinish(session, exitstatus):
    if _violations:
        session.exitstatus = 1


@pytest.fixture(autouse=True)
def model_free_default_embedder():
    """Omitted default construction cannot infer; explicit configs/method mocks survive."""
    if os.environ.get('EPYC_MODEL_FREE_CI') != '1':
        yield
        return
    from orchestration.repl_memory.embedder import EmbeddingConfig, TaskEmbedder
    original_init = TaskEmbedder.__init__
    original_generate = TaskEmbedder._generate_embedding

    @wraps(original_init)
    def safe_init(self, *args, **kwargs):
        default = (not args and kwargs.get('config') is None) or (len(args) == 1 and args[0] is None and 'config' not in kwargs)
        if default:
            args = ()
            kwargs['config'] = EmbeddingConfig(allow_subprocess=False, use_server=False, use_parallel=False, use_fallback=False)
        original_init(self, *args, **kwargs)
        self._model_free_omitted_default = default

    @wraps(original_generate)
    def safe_generate(self, *args, **kwargs):
        if getattr(self, '_model_free_omitted_default', False):
            _refuse('unmocked-default-embedder', 'omitted TaskEmbedder default attempted inference')
        return original_generate(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as isolation:
        isolation.setattr(TaskEmbedder, '__init__', safe_init)
        isolation.setattr(TaskEmbedder, '_generate_embedding', safe_generate)
        yield
