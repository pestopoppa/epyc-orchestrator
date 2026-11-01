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


def test_candidate_cannot_forge_or_disable_oracle():
    fixture = instrument.FIXTURES[0]
    for code in (
        "import inspect\np=inspect.currentframe().f_back.f_globals['p']\nprint(p['token'])\nraise SystemExit(0)",
        "import inspect\np=inspect.currentframe().f_back.f_globals['p']\np['cases'].clear()\ndef clamp(n, low, high): return n",
    ):
        assert instrument.evaluate(fixture, code) == 'fail'


def test_candidate_cannot_fork():
    fixture = instrument.FIXTURES[0]
    code = "import os\nos.fork()\ndef clamp(n, low, high): return max(low, min(n, high))\n"
    assert instrument.evaluate(fixture, code) == 'fail'


def test_candidate_cannot_read_fixture_source():
    fixture = instrument.FIXTURES[0]
    code = f"open({instrument.__file__!r}).read()\n" + fixture.gold
    assert instrument.evaluate(fixture, code) == 'fail'


def test_extra_output_is_not_a_result():
    fixture = instrument.FIXTURES[0]
    assert instrument.evaluate(fixture, "print('noise')\n" + fixture.gold) == 'fail'


def test_output_size_limit_stops_runaway_output():
    fixture = instrument.FIXTURES[0]
    assert instrument.evaluate(fixture, "while True: print('x' * 4096, flush=True)") == 'fail'


def test_pidfd_setup_failure_closes_stdin_and_reaps_child(monkeypatch):
    children = []
    popen = instrument.subprocess.Popen
    def capture(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child
    def unavailable(_pid):
        raise OSError('injected pidfd failure')
    monkeypatch.setattr(instrument.subprocess, 'Popen', capture)
    monkeypatch.setattr(instrument.os, 'pidfd_open', unavailable)
    assert instrument.evaluate(instrument.FIXTURES[0], instrument.FIXTURES[0].gold) == 'unavailable'
    assert len(children) == 1 and children[0].poll() is not None


def test_timeout_exit_race_still_reaps_child(monkeypatch):
    children = []
    popen = instrument.subprocess.Popen
    send = instrument.signal.pidfd_send_signal
    def capture(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child
    def exit_race(fd, sig):
        send(fd, sig)
        raise ProcessLookupError('injected exit race')
    monkeypatch.setattr(instrument.subprocess, 'Popen', capture)
    monkeypatch.setattr(instrument.signal, 'pidfd_send_signal', exit_race)
    assert instrument.evaluate(instrument.FIXTURES[0], 'while True: pass', timeout=.3) == 'timeout'
    assert len(children) == 1 and children[0].poll() is not None


def test_communication_error_reaps_child(monkeypatch):
    children = []
    popen = instrument.subprocess.Popen
    def capture(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        def fail(*args, **kwargs):
            raise OSError('injected communication error')
        child.communicate = fail
        return child
    monkeypatch.setattr(instrument.subprocess, 'Popen', capture)
    assert instrument.evaluate(instrument.FIXTURES[0], instrument.FIXTURES[0].gold) == 'unavailable'
    assert len(children) == 1 and children[0].poll() is not None
