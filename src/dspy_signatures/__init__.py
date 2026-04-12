"""DSPy Signatures for orchestrator routing prompts (AP-18).

Wraps the three core routing decision points as DSPy Signatures,
enabling future optimization via GEPA, BootstrapFewShot, or MIPROv2.

Usage:
    from src.dspy_signatures import FrontdoorClassifier, EscalationDecider, ModeSelector
    from src.dspy_signatures.config import configure_local_lm

    configure_local_lm()  # Point DSPy at local llama-server
    classifier = dspy.Predict(FrontdoorClassifier)
    result = classifier(user_prompt="Write a Fibonacci function", available_roles="...")
"""

from .frontdoor import FrontdoorClassifier
from .escalation import EscalationDecider
from .mode_selector import ModeSelector

__all__ = ["FrontdoorClassifier", "EscalationDecider", "ModeSelector"]
