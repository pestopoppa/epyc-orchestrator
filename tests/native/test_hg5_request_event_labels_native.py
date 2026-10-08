"""Prospective actual writer v2 controls; no application/bootstrap/model imports."""
import hashlib,importlib.util,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
SOURCE=Path(__file__).resolve().parents[2]/'src/runtime/hg5_request_event.py'
spec=importlib.util.spec_from_file_location('hg5_labels_actual_writer',SOURCE)
events=importlib.util.module_from_spec(spec);spec.loader.exec_module(events)

class LabelControls(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name);self.directory.chmod(0o700)
  self.context={'commit':'1'*40,'tree':'2'*40,'producer_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'projection':{'category':'CANDIDATE','metric_direction':{k:'lower_better' for k in events.COUNTERS},'protocol_id':''}}
  self.plan=SimpleNamespace(requested='force_architect_general',enabled=True,disabled_reason=None,from_role='frontdoor',final_answer_role='frontdoor',target_role='architect_general',error=None)
 def begin(self,context=None):
  with patch.dict(os.environ,{'HG5_REQUEST_EVENT_DIRECTORY':str(self.directory),'HG5_REQUEST_EVENT_SOURCE_CONTEXT':json.dumps(self.context if context is None else context)}):return events.begin_capture()
 def finish(self,capture):
  capture.update(stage='direct',feature_enabled=True)
  path=events.finish_capture(capture,request_id='synthetic-labelled',plan=self.plan,counters=events.snapshot(None));return json.loads(path.read_bytes())
 def test_default_off_even_valid_labels_do_not_activate_writer(self):
  with patch.dict(os.environ,{'HG5_REQUEST_EVENT_DIRECTORY':'','HG5_REQUEST_EVENT_SOURCE_CONTEXT':json.dumps(self.context)}):self.assertIsNone(events.begin_capture())
  self.assertEqual(list(self.directory.iterdir()),[])
 def test_without_projection_exact_v1_envelope_shape_preserved(self):
  context=dict(self.context);context.pop('projection');body=self.finish(self.begin(context))['body'];self.assertEqual(body['schema'],'hg5.request_intervention.v1');self.assertNotIn('projection',body);self.assertEqual(set(body['source']),{'origin','commit','tree','producer_sha256'})
 def test_valid_explicit_declaration_is_bound_before_window(self):
  capture=self.begin();before=capture['projection_canonical_at_start'];body=self.finish(capture)['body'];self.assertEqual(body['schema'],'hg5.request_intervention.v2');self.assertEqual(body['projection']['declaration'],self.context['projection']);self.assertEqual(body['projection']['declaration_sha256'],hashlib.sha256(before).hexdigest());self.assertEqual(body['projection']['human_ratification'],'not_asserted');self.assertEqual(body['runtime_origin'],'unknown')
 def test_later_context_environment_change_does_not_relabel_capture(self):
  capture=self.begin();self.context['projection']['category']='BASELINE'
  with patch.dict(os.environ,{'HG5_REQUEST_EVENT_SOURCE_CONTEXT':json.dumps(self.context)}):body=self.finish(capture)['body']
  self.assertEqual(body['projection']['declaration']['category'],'CANDIDATE')
 def test_missing_category_direction_protocol_each_refuse_before_write(self):
  for key in ('category','metric_direction','protocol_id'):
   context=json.loads(json.dumps(self.context));context['projection'].pop(key)
   with self.assertRaises(ValueError):self.begin(context)
  self.assertEqual(list(self.directory.iterdir()),[])
 def test_invalid_category_and_direction_are_not_guessed(self):
  context=json.loads(json.dumps(self.context));context['projection']['category']='unknown'
  with self.assertRaises(ValueError):self.begin(context)
  context=json.loads(json.dumps(self.context));context['projection']['metric_direction']['calls']='unknown'
  with self.assertRaises(ValueError):self.begin(context)
 def test_protocol_is_exact_explicit_citation_or_explicit_empty(self):
  self.assertEqual(self.finish(self.begin())['body']['projection']['declaration']['protocol_id'],'')
  self.context['projection']['protocol_id']='synthetic:cost-fixture-v2';self.assertEqual(self.finish(self.begin())['body']['projection']['declaration']['protocol_id'],'synthetic:cost-fixture-v2')
  self.context['projection']['protocol_id']=None
  with self.assertRaises(ValueError):self.begin()
 def test_duplicate_source_context_and_unstable_declaration_types_refuse(self):
  raw=json.dumps(self.context).replace('"commit":','"commit":"0","commit":',1)
  with patch.dict(os.environ,{'HG5_REQUEST_EVENT_DIRECTORY':str(self.directory),'HG5_REQUEST_EVENT_SOURCE_CONTEXT':raw}),self.assertRaises(ValueError):events.begin_capture()
  capture=self.begin();capture['projection_canonical_at_start']={}
  with self.assertRaises(ValueError):self.finish(capture)
 def test_v2_labels_require_full_source_owned_identity(self):
  context={'projection':self.context['projection']}
  with self.assertRaises(ValueError):self.begin(context)
