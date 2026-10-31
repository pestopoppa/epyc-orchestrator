"""Retained fake-HTTP controls with actual final-synthesis code and JSONL writer."""
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ATTEMPTS = []


def audit(event, args):
    if event in {'os.remove', 'os.rmdir', 'os.rename', 'sqlite3.connect'}:
        ATTEMPTS.append((event, repr(args)))
        raise AssertionError('Forbidden test lifecycle operation: ' + event)


sys.addaudithook(audit)
OWNER = Path('/mnt/raid0/llm/worktrees/mains/refactor-app-code-01a16fc3')
PACKET = Path(__file__).resolve().parent
sys.path.insert(0, str(OWNER))
from src.backends import serving_calls as sc

SOURCE = Path(os.environ.get('LB6_SOURCE', PACKET / 'final/src/api/routes/chat_vision.py'))
spec = importlib.util.spec_from_file_location('lb6_candidate_vision', SOURCE)
vision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vision)
RETAIN_PARENT = Path(os.environ.get('LB6_RETAIN_ROOT', PACKET)).resolve()
if any((ancestor / '.git').exists() for ancestor in (RETAIN_PARENT, *RETAIN_PARENT.parents)):
    raise RuntimeError('LB6_RETAIN_ROOT must be outside Git; no automatic cleanup')
RETAINED = Path(tempfile.mkdtemp(prefix='lb6-vision-retained-', dir=RETAIN_PARENT))


class Client:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return self.response.pop(0) if isinstance(self.response, list) else self.response


def response(status=200, content='Final Answer: synthetic answer'):
    return SimpleNamespace(status_code=status, json=lambda: {
        'choices': [{'finish_reason': 'stop', 'message': {'content': content}}],
        'usage': {'prompt_tokens': 11, 'completion_tokens': 3},
    })


class FinalSynthesisTests(unittest.TestCase):
    def run_case(self, client, *, recorder_failure=False, turns=0):
        log = RETAINED / (self._testMethodName + '.jsonl')
        env = {sc.LOG_ENV: str(log), sc.MAX_MB_ENV: '0', 'ORCHESTRATOR_PATHS_LOG_DIR': str(RETAINED)}
        with patch.dict(os.environ, env), patch.object(vision, '_get_config', return_value=SimpleNamespace(timeouts=SimpleNamespace(vision_inference=1))), patch.object(vision, '_vl_url_for_port', return_value='http://synthetic.invalid:8086'), patch('httpx.AsyncClient', return_value=client):
            if recorder_failure:
                with patch.object(sc, 'record_http_call', side_effect=RuntimeError('synthetic recorder down')):
                    result = asyncio.run(vision._vision_react_mode_answer('question', 'not-real-image', 'image/png', max_turns=turns))
            else:
                result = asyncio.run(vision._vision_react_mode_answer('question', 'not-real-image', 'image/png', max_turns=turns))
        self.assertEqual(len(client.calls), turns + 1)
        url, kwargs = client.calls[-1]
        self.assertEqual(url, 'http://synthetic.invalid:8086/v1/chat/completions')
        self.assertEqual(kwargs['json']['max_tokens'], 2048)
        self.assertEqual(kwargs['json']['temperature'], 0.)
        self.assertEqual(kwargs['json']['stop'], [vision.QWEN_STOP])
        self.assertIn('You have used all available turns.', kwargs['json']['messages'][-1]['content'])
        records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        for record in records:
            self.assertEqual(record['record_sha256'], sc.record_digest(record))
            self.assertIn(record['method'], ['vision_react_turn', 'vision_react_final'])
        self.assertFalse(ATTEMPTS, ATTEMPTS)
        return result, records

    def test_final_answer_and_real_record(self):
        result, records = self.run_case(Client(response()))
        self.assertEqual(result, ('synthetic answer', 0, []))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['finish_reason'], 'stop')
        self.assertEqual(records[0]['notes']['http_status'], 200)
        self.assertEqual(records[0]['result']['tokens_generated'], 3)

    def test_turn_exhaustion_records_both_actual_http_calls_once(self):
        client = Client([response(content='Thought: inspect\nAction: ocr_extract(image_base64="current")'), response()])
        from unittest.mock import AsyncMock
        with patch.object(vision, '_execute_vision_tool', new=AsyncMock(return_value='synthetic observation')):
            result, records = self.run_case(client, turns=1)
        self.assertEqual(result, ('synthetic answer', 1, ['ocr_extract']))
        self.assertEqual([r['method'] for r in records], ['vision_react_turn', 'vision_react_final'])

    def test_http_failure_recorded_without_changed_fallback(self):
        result, records = self.run_case(Client(response(503)))
        self.assertEqual(result, ('[Vision ReAct: no answer produced]', 0, []))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['notes']['http_status'], 503)
        self.assertEqual(records[0]['outcome'], 'failed')

    def test_transport_failure_recorded_without_changed_fallback(self):
        result, records = self.run_case(Client(error=RuntimeError('synthetic transport down')))
        self.assertEqual(result, ('[Vision ReAct: no answer produced]', 0, []))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['outcome'], 'exception')

    def test_recorder_failure_never_changes_answer(self):
        result, records = self.run_case(Client(response()), recorder_failure=True)
        self.assertEqual(result, ('synthetic answer', 0, []))
        self.assertEqual(records, [])


if __name__ == '__main__':
    print('Retained:', RETAINED)
    unittest.main()
