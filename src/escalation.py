bash
cat >> src/escalation.py << 'PYEOF'


# ---------------------------------------------------------------------------
# Suite-aware cheap-first bypass
# ---------------------------------------------------------------------------

import re as _re

_SKIP_CHEAP_PATTERNS = [
    # Physics / science reasoning — 7B models have ~0% pass rate
    _re.compile(
        r"\b(physic|kinemati|torque|angular momentum|conservation of energy|"
        r"thermodynamic|electrostatic|magnetic field|quantum|relativity|"
        r"newton['\u2019]?s law|friction coefficien|inclined plane|projectile|"
        r"free.?body diagram|circuit|resistor|capacitor|inductor)\b",
        _re.IGNORECASE,
    ),
    # Competition math (AIME, AMC, Olympiad)
    _re.compile(
        r"\b(AIME|AMC\s*(?:10|12)|USAMO|IMO|Putnam|olympiad|competition math)\b",
        _re.IGNORECASE,
    ),
    # USACO / competitive programming
    _re.compile(
        r"\b(USACO|competitive programming|codeforces|leetcode hard|"
        r"dynamic programming|segment tree|binary indexed tree)\b",
        _re.IGNORECASE,
    ),
    # Web research / multi-step reasoning
    _re.compile(
        r"\b(search the web|look up|find online|current price of|"
        r"latest news about|web.?search|browse)\b",
        _re.IGNORECASE,
    ),
]


def should_skip_cheap_first(prompt: str, difficulty_band: str = "easy") -> bool:
    """Determine if cheap-first routing should be skipped for this prompt.

    Returns True when the prompt matches a category where 7B models have
    near-zero pass rate, making the cheap-first attempt a waste of latency.
    """
    if difficulty_band == "hard":
        return True
    for pattern in _SKIP_CHEAP_PATTERNS:
        if pattern.search(prompt):
            return True
    return False
PYEOF