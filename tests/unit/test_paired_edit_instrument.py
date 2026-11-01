"""First-party oracle and custody contract, with no model inference."""
import pytest
from scripts.benchmark import paired_edit_instrument as instrument


def test_pair_has_only_instruction_difference_and_no_hidden_material():
    for fixture in instrument.FIXTURES:
        a = instrument.prompt(fixture, 'descriptive')
        b = instrument.prompt(fixture, 'minimal')
        assert a.replace(fixture.descriptive, '') == b.replace(fixture.minimal, '')
        assert fixture.initial in a and fixture.initial in b
        assert fixture.gold not in a and fixture.mutant not in b
        assert repr(fixture.cases) not in a and repr(fixture.cases) not in b
    with pytest.raises(ValueError):
        instrument.prompt(instrument.FIXTURES[0], 'other')


def test_execution_gold_mutants_and_failures():
    for fixture in instrument.FIXTURES:
        assert instrument.evaluate(fixture, fixture.gold) == 'pass'
        assert instrument.evaluate(fixture, fixture.mutant) == 'fail'
        assert instrument.evaluate(fixture, fixture.initial) == 'fail'
        assert instrument.evaluate(fixture, '') == 'fail'
        assert instrument.evaluate(fixture, 'import sys; sys.exit(0)') == 'fail'
        assert instrument.evaluate(fixture, 'raise ValueError()') == 'fail'
        assert instrument.evaluate(fixture, 'while True: pass', timeout=.3) == 'timeout'


def test_input_mutation_is_rejected():
    fixture = instrument.FIXTURES[1]
    code = 'def unique(items):\n    items[:] = list(dict.fromkeys(items))\n    return items\n'
    assert instrument.evaluate(fixture, code) == 'fail'


def test_refusal_never_passes(monkeypatch):
    def refuse(*args, **kwargs):
        raise instrument.exec_sandbox.ContainmentUnavailable('no kernel support')
    monkeypatch.setattr(instrument.exec_sandbox, 'wrap', refuse)
    assert instrument.evaluate(instrument.FIXTURES[0], instrument.FIXTURES[0].gold) == 'unavailable'
