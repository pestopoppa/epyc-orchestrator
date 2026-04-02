_DIRECT_ANSWER_ROLES = frozenset({"worker_explore", "frontdoor"})

_DIRECT_ANSWER_PREFIX = "Answer with ONLY the answer. No explanation.\n\n"


def get_direct_answer_prefix(role: str) -> str:
    """Return a concise-answer directive for roles that need bare output.

    Used by _try_cheap_first in chat.py to prepend a formatting directive
    so the model returns exact answers (e.g. '145', 'red, blue, yellow')
    instead of verbose explanations.
    """
    if role in _DIRECT_ANSWER_ROLES:
        return _DIRECT_ANSWER_PREFIX
    return ""