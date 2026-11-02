from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from src.scheduling.kv_preemption import Attempt, AttemptRegistry, NativeDrainWitness


def attempt(name, *, sequence=10, priority=0, url="server", **kwargs):
    return Attempt(name, f"request-{name}", f"owner-{name}", url, priority, sequence,
                   cancellable=kwargs.pop("cancellable", True),
                   replayable=kwargs.pop("replayable", True),
                   output_uncommitted_verified=kwargs.pop("output_uncommitted_verified", True),
                   **kwargs)


def choose(registry, waiter="waiter", *, url="server", priority=0, sequence=1):
    return registry.select(waiter_id=waiter, server_url=url, priority=priority,
                           queue_sequence=sequence)


def test_default_ineligible_without_explicit_owner_capabilities():
    registry = AttemptRegistry()
    registry.register(Attempt("a", "request-a", "owner-a", "server", 0, 10))
    registry.register(attempt("b", cancellable=False))
    registry.register(attempt("c", replayable=False))
    registry.register(attempt("d", preemptions=1))
    registry.register(attempt("e", url="other"))
    registry.register(attempt("f", sequence=0))
    registry.register(attempt("g", output_uncommitted_verified=False))
    assert choose(registry) is None


def test_priority_then_youngest_and_stable_one_victim_per_waiter():
    registry = AttemptRegistry()
    for candidate in (attempt("a", priority=-1, sequence=5),
                      attempt("b", priority=-1, sequence=7),
                      attempt("c", priority=0, sequence=20)):
        registry.register(candidate)
    victim = choose(registry)
    assert victim.attempt_id == "b"
    registry.finish("b", "owner-b")
    assert choose(registry) == victim
    assert choose(registry, "waiter-2").attempt_id == "a"
    with pytest.raises(ValueError, match="waiter facts changed"):
        choose(registry, priority=1)


def test_only_owner_can_change_attempt_and_commit_excludes_selection():
    registry = AttemptRegistry()
    registry.register(attempt("a"))
    with pytest.raises(ValueError, match="owner"):
        registry.commit_output("a", "wrong-owner")
    assert registry.commit_output("a", "owner-a")
    assert choose(registry) is None
    registry.register(attempt("b"))
    assert choose(registry).attempt_id == "b"
    assert not registry.commit_output("b", "owner-b")
    registry.finish("b", "owner-b")
    assert not registry.commit_output("b", "owner-b")


def test_register_rejects_duplicate_attempt_and_concurrent_request():
    registry = AttemptRegistry()
    a = attempt("a")
    registry.register(a)
    with pytest.raises(ValueError, match="attempt_id"):
        registry.register(a)
    with pytest.raises(ValueError, match="active attempt"):
        registry.register(replace(a, attempt_id="other"))
    registry.finish("a", "owner-a")
    registry.register(replace(a, attempt_id="other"))


def test_concurrent_selectors_do_not_share_victim():
    registry = AttemptRegistry()
    registry.register(attempt("a"))
    start = Barrier(2)

    def select(waiter):
        start.wait()
        return choose(registry, waiter)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(select, ["first", "second"]))
    assert sum(result is not None for result in results) == 1


def test_output_commit_racing_selection_is_exclusive():
    registry = AttemptRegistry()
    registry.register(attempt("a"))
    start = Barrier(2)

    def commit():
        start.wait()
        return registry.commit_output("a", "owner-a")

    def select():
        start.wait()
        return choose(registry)

    with ThreadPoolExecutor(max_workers=2) as executor:
        commit_future, select_future = executor.submit(commit), executor.submit(select)
        committed, victim = commit_future.result(), select_future.result()
    assert committed != (victim is not None)


def test_invalid_identity_and_facts_fail_closed():
    with pytest.raises(ValueError):
        attempt(" ")
    with pytest.raises(ValueError):
        attempt("a", sequence=True)
    registry = AttemptRegistry(max_preemptions=0)
    registry.register(attempt("a"))
    assert choose(registry) is None
    with pytest.raises(ValueError):
        choose(registry, sequence=-1)


def test_selected_request_cannot_reset_preemption_count_on_new_attempt():
    registry = AttemptRegistry(verify_drain=lambda _: True)
    a = attempt("a", server_task_id="task-42")
    registry.register(a)
    victim = choose(registry)
    assert victim.attempt_id == "a"
    command = registry.begin_cancel(victim, "owner-a")
    registry.acknowledge_cancel(command, "owner-a")
    registry.finish("a", "owner-a")
    with pytest.raises(ValueError, match="no verified native drain"):
        registry.register(replace(a, attempt_id="a2", preemptions=1))
    witness = NativeDrainWitness("a", "server", "task-42", command.nonce, "terminal", "a" * 64)
    assert registry.confirm_native_drain(command, "owner-a", witness)
    with pytest.raises(ValueError, match="below request history"):
        registry.register(replace(a, attempt_id="a2", preemptions=0))
    registry.register(replace(a, attempt_id="a2", preemptions=1))
    assert choose(registry, "another-waiter") is None


def test_cancellation_requires_selection_owner_and_native_task_id():
    registry = AttemptRegistry()
    registry.register(attempt("a"))
    victim = choose(registry)
    with pytest.raises(ValueError, match="owner"):
        registry.begin_cancel(victim, "wrong-owner")
    with pytest.raises(ValueError, match="identity missing"):
        registry.begin_cancel(victim, "owner-a")


def test_ack_and_verified_drain_are_bound_and_idempotent():
    verified = []
    registry = AttemptRegistry(verify_drain=lambda receipt: verified.append(receipt) or True)
    registry.register(attempt("a", server_task_id="task-42"))
    victim = choose(registry)
    command = registry.begin_cancel(victim, "owner-a")
    assert registry.begin_cancel(victim, "owner-a") == command
    witness = NativeDrainWitness("a", "server", "task-42", command.nonce, "terminal", "a" * 64)
    assert not registry.confirm_native_drain(command, "owner-a", witness)
    with pytest.raises(ValueError, match="intent"):
        registry.acknowledge_cancel(replace(command, nonce="wrong"), "owner-a")
    registry.acknowledge_cancel(command, "owner-a")
    registry.acknowledge_cancel(command, "owner-a")
    for bad in (replace(witness, server_task_id="other"),
                replace(witness, server_url="other"),
                replace(witness, attempt_id="other"),
                replace(witness, cancel_nonce="other"),
                replace(witness, kind="socket_closed"),
                replace(witness, receipt_sha256="bad")):
        assert not registry.confirm_native_drain(command, "owner-a", bad)
    assert verified == []
    assert registry.confirm_native_drain(command, "owner-a", witness)
    assert registry.confirm_native_drain(command, "owner-a", witness)
    assert verified == [witness]
    assert not registry.confirm_native_drain(command, "owner-a", replace(witness, kind="absent"))


@pytest.mark.parametrize("verifier", [None, lambda _: False, lambda _: 1,
                                      lambda _: (_ for _ in ()).throw(OSError("lost"))])
def test_missing_or_failed_native_verifier_cannot_confirm(verifier):
    registry = AttemptRegistry(verify_drain=verifier)
    registry.register(attempt("a", server_task_id="task-42"))
    command = registry.begin_cancel(choose(registry), "owner-a")
    registry.acknowledge_cancel(command, "owner-a")
    registry.finish("a", "owner-a")  # owner termination is not native drain
    witness = NativeDrainWitness("a", "server", "task-42", command.nonce, "absent", "b" * 64)
    assert not registry.confirm_native_drain(command, "owner-a", witness)
