"""Smoke tests for DSPy signature definitions (AP-18).

Verifies imports, field definitions, and module creation.
Does NOT require a running LM backend.
"""

import pytest


def test_import_signatures():
    """All three signatures are importable."""
    from src.dspy_signatures import FrontdoorClassifier, EscalationDecider, ModeSelector
    assert FrontdoorClassifier is not None
    assert EscalationDecider is not None
    assert ModeSelector is not None


def test_frontdoor_fields():
    """FrontdoorClassifier has expected input/output fields."""
    from src.dspy_signatures.frontdoor import FrontdoorClassifier
    import dspy

    sig = FrontdoorClassifier
    # Check input fields
    assert "user_prompt" in sig.input_fields
    assert "available_roles" in sig.input_fields
    # Check output fields
    assert "task_type" in sig.output_fields
    assert "primary_role" in sig.output_fields
    assert "priority" in sig.output_fields
    assert "reasoning" in sig.output_fields


def test_escalation_fields():
    """EscalationDecider has expected input/output fields."""
    from src.dspy_signatures.escalation import EscalationDecider

    sig = EscalationDecider
    assert "current_role" in sig.input_fields
    assert "error_category" in sig.input_fields
    assert "failure_count" in sig.input_fields
    assert "action" in sig.output_fields
    assert "target_role" in sig.output_fields


def test_mode_selector_fields():
    """ModeSelector has expected input/output fields."""
    from src.dspy_signatures.mode_selector import ModeSelector

    sig = ModeSelector
    assert "prompt" in sig.input_fields
    assert "has_image" in sig.input_fields
    assert "mode" in sig.output_fields
    assert "role" in sig.output_fields
    assert "confidence" in sig.output_fields


def test_escalation_context_adapter():
    """from_escalation_context converts dataclass to dict."""
    from src.dspy_signatures.escalation import from_escalation_context
    from dataclasses import dataclass

    @dataclass
    class MockContext:
        current_role: str = "coder"
        error_category: str = "timeout"
        error_message: str = "Request timed out"
        failure_count: int = 2
        escalation_count: int = 1

    result = from_escalation_context(MockContext())
    assert result["current_role"] == "coder"
    assert result["failure_count"] == 2
    assert result["escalation_count"] == 1


def test_create_modules():
    """Module factories return valid DSPy modules."""
    from src.dspy_signatures.frontdoor import create_module as create_frontdoor
    from src.dspy_signatures.escalation import create_module as create_escalation
    from src.dspy_signatures.mode_selector import create_module as create_mode
    import dspy

    # Predict mode
    assert isinstance(create_frontdoor(), dspy.Predict)
    assert isinstance(create_escalation(), dspy.Predict)
    assert isinstance(create_mode(), dspy.Predict)

    # ChainOfThought mode
    assert isinstance(create_frontdoor(use_cot=True), dspy.ChainOfThought)


def test_config_test_connection():
    """test_connection returns False when no server is running."""
    from src.dspy_signatures.config import test_connection
    # No local server expected in test environment
    assert test_connection("http://localhost:99999/v1") is False
