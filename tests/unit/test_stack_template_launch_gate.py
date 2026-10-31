"""The active stack template must gate normal starts before startup effects."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

from scripts.server import stack_commands


def test_active_profile_resolution_uses_explicit_then_environment_then_default(
    monkeypatch,
) -> None:
    import src.config.stack_templates as templates

    seen: list[str] = []
    monkeypatch.setattr(templates, "load_template", lambda name: seen.append(name) or object())
    monkeypatch.setattr(
        templates,
        "validate_template",
        lambda template, registry_path=None: SimpleNamespace(
            valid=True, errors=[], warnings=[], summary="PASS"
        ),
    )

    monkeypatch.delenv("ORCHESTRATOR_STACK_PROFILE", raising=False)
    assert stack_commands._validate_active_template_for_launch(Namespace())
    monkeypatch.setenv("ORCHESTRATOR_STACK_PROFILE", "custom")
    assert stack_commands._validate_active_template_for_launch(Namespace())
    assert stack_commands._validate_active_template_for_launch(
        Namespace(stack_profile="explicit")
    )

    assert seen == ["default", "custom", "explicit"]


def test_invalid_active_template_refuses_before_prelude_or_launch(monkeypatch, capsys) -> None:
    import src.config.stack_templates as templates

    monkeypatch.setattr(templates, "get_active_profile", lambda: "default")
    monkeypatch.setattr(templates, "load_template", lambda _name: object())
    monkeypatch.setattr(
        templates,
        "validate_template",
        lambda *_args, **_kwargs: SimpleNamespace(
            valid=False,
            errors=["retired role present"],
            warnings=[],
            summary="FAIL, 1 errors",
        ),
    )
    effects: list[str] = []
    monkeypatch.setattr(
        stack_commands,
        "_capacity_launch_gate",
        lambda: effects.append("capacity") or True,
    )
    monkeypatch.setattr(
        stack_commands,
        "_start_registry_prelude",
        lambda *_a, **_kw: effects.append("registry-prelude") or 0,
    )
    monkeypatch.setattr(
        stack_commands,
        "_run_stack_change_launch_gate",
        lambda _args: effects.append("stack-change-gate") or True,
    )
    monkeypatch.setattr(
        stack_commands,
        "start_server",
        lambda *_a, **_kw: effects.append("launch"),
    )

    assert stack_commands.cmd_start(Namespace(dev=False)) == 1
    assert effects == []
    assert "retired role present" in capsys.readouterr().out


def test_missing_or_malformed_active_template_refuses_before_prelude(monkeypatch) -> None:
    import src.config.stack_templates as templates

    effects: list[str] = []
    monkeypatch.setattr(templates, "get_active_profile", lambda: "broken")

    def missing(_name):
        raise FileNotFoundError("template missing")

    monkeypatch.setattr(templates, "load_template", missing)
    monkeypatch.setattr(
        stack_commands,
        "_start_registry_prelude",
        lambda *_a, **_kw: effects.append("registry-prelude") or 0,
    )
    assert stack_commands.cmd_start(Namespace(dev=False)) == 1
    assert effects == []

    def malformed(_name):
        raise ValueError("malformed template")

    monkeypatch.setattr(templates, "load_template", malformed)
    assert stack_commands.cmd_start(Namespace(dev=False)) == 1
    assert effects == []


def test_valid_active_template_reaches_normal_start_gates(monkeypatch) -> None:
    import src.config.stack_templates as templates

    monkeypatch.setattr(templates, "get_active_profile", lambda: "default")
    monkeypatch.setattr(templates, "load_template", lambda _name: object())
    monkeypatch.setattr(
        templates,
        "validate_template",
        lambda *_args, **_kwargs: SimpleNamespace(
            valid=True, errors=[], warnings=[], summary="PASS"
        ),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        stack_commands,
        "_capacity_launch_gate",
        lambda: calls.append("capacity") or False,
    )
    monkeypatch.setattr(
        stack_commands,
        "_start_registry_prelude",
        lambda *_a, **_kw: calls.append("registry-prelude") or 0,
    )

    assert stack_commands.cmd_start(Namespace(dev=False)) == 2
    assert calls == ["capacity"]


def test_migration_path_does_not_gate_on_unrelated_active_profile(monkeypatch) -> None:
    import src.config.stack_templates as templates
    import src.config.stack_migration as migration

    def unexpected(*_args, **_kwargs):
        raise AssertionError("migration should validate its target, not active profile")

    monkeypatch.setattr(templates, "load_template", unexpected)
    monkeypatch.setattr(stack_commands, "_capacity_launch_gate", lambda: True)
    monkeypatch.setattr(stack_commands, "_start_registry_prelude", lambda *_a, **_kw: 0)
    monkeypatch.setattr(stack_commands, "_run_stack_change_launch_gate", lambda _args: True)
    monkeypatch.setattr(
        migration,
        "migrate_to_template",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True, summary=lambda: "migration ok"),
    )

    assert stack_commands.cmd_start(
        Namespace(dev=False, migrate_to="target", dry_run=True)
    ) == 0
