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


def _validate_first_on():
    from src.features import Features, set_features
    set_features(Features(typed_decisions_validate_first=True))


def test_validate_first_default_off_still_calls_model_for_valid_kwargs():
    from src.features import reset_features
    reset_features()
    tool = registry_fixture()
    schema = registry_parameters_to_full_schema(tool.parameters)
    fake = Canned({'answers': {'mode': {'choice': 'safe'}, 'count': {'score': 2}, 'dry': {'noul': True}}})
    proposal = propose_shadow_typed_arguments(tool_name=tool.name, full_schema=schema,
        original_kwargs={'mode': 'fast', 'count': 1, 'dry': False}, primitives=fake,
        role='worker', state='state', tool=tool)
    assert proposal.status == 'proposed' and fake.schemas


def test_validate_first_valid_kwargs_cost_zero_repair_calls():
    from src.features import reset_features
    _validate_first_on()
    try:
        tool = registry_fixture()
        schema = registry_parameters_to_full_schema(tool.parameters)
        fake = Canned({})
        proposal = propose_shadow_typed_arguments(tool_name=tool.name, full_schema=schema,
            original_kwargs={'mode': 'fast', 'count': 1, 'dry': False}, primitives=fake,
            role='worker', state='state', tool=tool)
        assert proposal.status == 'abstained' and proposal.reason == 'original_valid'
        assert fake.schemas == [] and proposal.calls == ()
    finally:
        reset_features()


def test_validate_first_invalid_closed_set_gets_one_repair_proposal():
    from src.features import reset_features
    _validate_first_on()
    try:
        tool = registry_fixture()
        schema = registry_parameters_to_full_schema(tool.parameters)
        fake = Canned({'answers': {'mode': {'choice': 'safe'}, 'count': {'score': 2}, 'dry': {'noul': True}}})
        proposal = propose_shadow_typed_arguments(tool_name=tool.name, full_schema=schema,
            original_kwargs={'mode': 'turbo', 'count': 1, 'dry': False}, primitives=fake,
            role='worker', state='state', tool=tool)
        assert proposal.status == 'proposed' and len(fake.schemas) == 1
        assert proposal.proposed['mode'] == 'safe'
    finally:
        reset_features()


def test_validate_first_refuses_dropped_constraint_without_model_call():
    from src.features import reset_features
    _validate_first_on()
    try:
        schema = {'type': 'object',
                  'properties': {'mode': {'type': 'string', 'enum': ['a', 'b']},
                                 'ticket': {'type': 'string', 'pattern': '^T-[0-9]+$'}}}
        fake = Canned({})
        proposal = propose_shadow_typed_arguments(tool_name='t', full_schema=schema,
            original_kwargs={'mode': 'a', 'ticket': 'nope'}, primitives=fake,
            role='worker', state='state')
        assert proposal.status == 'abstained'
        assert proposal.reason.startswith('refused_unsupported') and 'pattern' in proposal.reason
        assert fake.schemas == []
    finally:
        reset_features()


def test_validate_first_flag_is_on_autopilot_surface():
    import pathlib
    from scripts.autopilot.config_applicator import HOT_SWAP_FEATURES
    assert 'typed_decisions_validate_first' in HOT_SWAP_FEATURES
    spec = pathlib.Path('orchestration/runtime_flags.spec.yaml').read_text()
    assert 'typed_decisions_validate_first: baseline' in spec


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
