"""PII / identifier regex seeds for the advisory skill lint (handoff M-27 b).

Path, IP, URL-credential and identifier patterns are COPIED, with attribution, from
NVIDIA SkillEvaluator ``src/skillevaluator/config/pii_patterns.yaml`` at
15adf98f7941dc54ad59d30538c85049c515e2cf
(https://github.com/NVIDIA/SkillEvaluator, Apache-2.0,
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES).
The tool was neither installed nor run; only the YAML was read. Per-pattern
``exceptions`` lists are reproduced where the pattern needs them.

Local additions (not from NVIDIA): model-name, port and column-name classes live in
``literal_lint`` (ROLE_MODEL, PORT, COL). ADVISORY ONLY, like the rest of the lint.
"""

from __future__ import annotations

import re

# class -> list of (compiled regex, exceptions tuple)
SEED_PATTERNS: dict[str, list[tuple[re.Pattern[str], tuple[str, ...]]]] = {
    "PII_PATH": [
        (re.compile(r"/Users/[^/\s]+/"),
         ("/users/:", "/users/{", "/Users/:", "/Users/{", "GET /users", "POST /users",
          "PUT /users", "PATCH /users", "DELETE /users", "api/users")),
        (re.compile(r"C:\\Users\\[^\\\s]+"), ()),
        (re.compile(r"/home/([^/\s]+)/"), ()),  # upstream flags only author-matching; here any
    ],
    "PII_IP": [
        (re.compile(
            r"\b(?!10\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[01]\.|192\.168\.|127\.|0\.)"
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), ()),
    ],
    "PII_URL": [
        (re.compile(r"(postgres|postgresql|mysql|mariadb|mongodb|redis)://[^:]+:[^@]+@[^/\s]+"), ()),
        (re.compile(r"(jdbc:[a-z]+://[^:]+:[0-9]+/[a-z0-9_-]+)"), ()),
        (re.compile(r"https://hooks\.slack\.com/services/T[A-Z0-9]{8}/B[A-Z0-9]{8}/[A-Za-z0-9]{24}"), ()),
        (re.compile(r"https://discord\.com/api/webhooks/[0-9]{18}/[A-Za-z0-9_-]{68}"), ()),
    ],
    "PII_ID": [
        (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), ()),
        (re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b"), ()),
        (re.compile(r"arn:aws:[a-z0-9-]+:[a-z0-9-]*:[0-9]{12}:[a-z0-9-/_]+"), ()),
        (re.compile(r"\bghp_[A-Za-z0-9]{36}\b"), ()),
        (re.compile(r"\bghs_[A-Za-z0-9]{36}\b"), ()),
        (re.compile(r"\b0x[a-fA-F0-9]{40}\b"), ()),
        (re.compile(r"\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"), ()),
        (re.compile(r"[A-Za-z0-9._%+-]+@(?!([a-z0-9-]+\.)*example\.com|([a-z0-9-]+\.)*example\.org|"
                    r"([a-z0-9-]+\.)*test\.com|odata\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
         ("Copyright", "copyright", "LICENSE", "license", "(c)", "Authors", "://", "@odata.")),
    ],
}


def extract_pii_seed(text: str) -> dict[str, list[str]]:
    """Return {class: sorted unique matches}; a match is dropped if its line holds an exception."""
    out: dict[str, set[str]] = {}
    for cls, pats in SEED_PATTERNS.items():
        for rx, exc in pats:
            for m in rx.finditer(text):
                ls = text.rfind("\n", 0, m.start()) + 1
                le = text.find("\n", m.end())
                line = text[ls : le if le != -1 else len(text)]
                if any(e in line for e in exc):
                    continue
                out.setdefault(cls, set()).add(m.group(0))
    return {k: sorted(v) for k, v in out.items()}
