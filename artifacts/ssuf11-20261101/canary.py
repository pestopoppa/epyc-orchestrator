from pathlib import Path
import argparse
import tempfile
import yaml
from src.registry.registry_compiler import compile_lean
from src.registry import registry_loader
from scripts.server import stack_env
from scripts.server import orchestrator_stack as launcher
parser = argparse.ArgumentParser()
parser.add_argument('--master', type=Path, required=True)
args = parser.parse_args()
master = args.master
source = yaml.safe_load(master.read_text())['server_mode']['architect_general']['recipe']
payload = yaml.safe_load(master.read_text())
recipe_roles = {k for k, v in payload['server_mode'].items() if isinstance(v, dict) and v.get('recipe')}
assert recipe_roles == {'architect_general'}, recipe_roles
compiled_registry = compile_lean(master, {'architect_general'})
compiled = compiled_registry['server_mode']
assert compiled['architect_general']['recipe'] == source
assert source['threads'] == 96 and source['cpu_shape'] == 'NUMA_FULL'
assert 'GGML_NOHUGEPAGE_PROCESS' not in source['env']
assert source['env']['GGML_FA_SPLIT_KV'] == '0'
registry_loader.load_server_mode = lambda: compiled
env = stack_env.build_launch_env('architect_general', {'GGML_AMBIENT_TEST': '1'})
preserve = {**stack_env.recipe_env_for('architect_general'), **stack_env._role_env_overrides('architect_general')}
stack_env.strip_ambient_ggml(env, preserve)
for key, expected in source['env'].items():
    assert env[key] == preserve[key]
assert 'GGML_AMBIENT_TEST' not in env
from unittest.mock import patch
with tempfile.TemporaryDirectory() as tmp:
    lean_path = Path(tmp) / 'compiled-registry.yaml'
    lean_path.write_text(yaml.safe_dump(compiled_registry))
    role = registry_loader.RegistryLoader(lean_path, validate_paths=False).get_role('architect_general')
with patch('subprocess.Popen', side_effect=AssertionError('unexpected process launch')), \
     patch.object(launcher, '_verify_artifact_integrity', return_value=[]):
    cmd = launcher.build_server_command(role, 8074, prepare_runtime_dirs=False)
assert cmd[cmd.index('-t') + 1] == str(source['threads'])
assert '--port' in cmd and cmd[cmd.index('--port') + 1] == '8074'
assert 'GGML_NOHUGEPAGE_PROCESS' not in env
print('PASS: source -> lean compiler -> actual default command builder and sanitized env; no process launched')
