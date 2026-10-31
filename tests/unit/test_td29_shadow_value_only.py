import json

from src.registry.tool_registry import Tool, ToolCategory, ToolRegistry
from src.typed_decisions.runner import run_typed_decisions
from src.typed_decisions.tool_args import tool_schema_to_questions
from src.typed_decisions.tool_args_integration import (
    propose_shadow_typed_arguments, registry_parameters_to_full_schema,
)


class Canned:
    def __init__(self, response):
        self.response = json.dumps(response)
        self.schemas = []

    def llm_call(self, prompt, **kwargs):
        self.schemas.append(kwargs['json_schema'])
        return self.response


def registry_fixture():
    tool = Tool(name='example', description='example', category=ToolCategory.DATA,
                parameters={'mode': {'type': 'string', 'enum': ['safe', 'fast'], 'required': True},
                            'count': {'type': 'integer', 'minimum': 1, 'maximum': 3, 'required': True},
                            'dry': {'type': 'boolean', 'required': True}})
    registry = ToolRegistry()
    registry.register_tool(tool)
    return registry._tools['example']


def test_registry_shadow_value_only_and_unavailable_confidence():
    tool = registry_fixture()
    schema = registry_parameters_to_full_schema(tool.parameters)
    response = {'answers': {'mode': {'choice': 'safe'}, 'count': {'score': 2}, 'dry': {'noul': True}}}
    fake = Canned(response)
    proposal = propose_shadow_typed_arguments(tool_name=tool.name, full_schema=schema,
        original_kwargs={'mode': 'fast', 'count': 1, 'dry': False}, primitives=fake,
        role='worker', state='state', tool=tool)
    assert proposal.status == 'proposed'
    assert proposal.proposed == {'mode': 'safe', 'count': 2, 'dry': True}
    answer = fake.schemas[0]['properties']['answers']['properties']
    assert all(set(spec['properties']) in ({'choice'}, {'score'}, {'noul'}) for spec in answer.values())
    result = run_typed_decisions(Canned(response), state='state',
        questions=tool_schema_to_questions(tool.name, schema).questions, role='worker', value_only=True)
    assert all(d.confidence is None and d.probabilities == {} for d in result.decisions)


def test_registry_shadow_rejects_extra_confidence_and_invalid_value():
    tool = registry_fixture()
    schema = registry_parameters_to_full_schema(tool.parameters)
    for mode in ({'choice': 'safe', 'confidence': 1}, {'choice': 'invented'}):
        fake = Canned({'answers': {'mode': mode, 'count': {'score': 2}, 'dry': {'noul': True}}})
        proposal = propose_shadow_typed_arguments(tool_name=tool.name, full_schema=schema,
            original_kwargs={'mode': 'fast', 'count': 1, 'dry': False}, primitives=fake,
            role='worker', state='state', tool=tool)
        assert proposal.status == 'abstained' and proposal.proposed is None


def test_default_runner_still_requires_probability_and_confidence():
    tool = registry_fixture()
    schema = registry_parameters_to_full_schema(tool.parameters)
    fake = Canned({'answers': {'mode': {'choice': 'safe'}, 'count': {'score': 2}, 'dry': {'noul': True}}})
    result = run_typed_decisions(fake, state='state', questions=tool_schema_to_questions(tool.name, schema).questions,
                                 role='worker', max_retries=0)
    assert result.failures and not result.decisions
    assert 'confidence' in fake.schemas[0]['properties']['answers']['properties']['mode']['required']


def test_shadow_retry_does_not_request_forbidden_fields():
    tool = registry_fixture()
    schema = registry_parameters_to_full_schema(tool.parameters)
    response = {'answers': {'mode': {'choice': 'safe'}, 'count': {'score': 2}, 'dry': {'noul': True}}}

    class RetryCanned:
        def __init__(self):
            self.prompts = []

        def llm_call(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return 'not JSON' if len(self.prompts) == 1 else json.dumps(response)

    fake = RetryCanned()
    proposal = propose_shadow_typed_arguments(tool_name=tool.name, full_schema=schema,
        original_kwargs={'mode': 'fast', 'count': 1, 'dry': False}, primitives=fake,
        role='worker', state='state', tool=tool)
    assert len(fake.prompts) == 2
    assert 'CORRECTION' in fake.prompts[1]
    assert 'probabilities' not in fake.prompts[1] and 'confidence' not in fake.prompts[1]
    # A recovered retry still abstains under the existing conservative policy.
    assert proposal.status == 'abstained' and proposal.reason == 'typed_failure'
