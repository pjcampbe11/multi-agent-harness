"""
Robust JSON extraction from LLM output.

Generator models wrap JSON in prose, fences, or trailing commentary. This module
pulls the first well-formed JSON value out of a response and, on failure, asks the
model to repair it once. Keeping this in one place is what makes the structured
agents (planner/verifier) reliable enough to run in a loop unattended.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _candidates(text: str) -> List[str]:
    out: List[str] = []
    out += _FENCE_RE.findall(text)
    # Greedy braces/brackets spans as fallbacks.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            out.append(text[start : end + 1])
    out.append(text)
    return out


def try_parse(text: str) -> Optional[Any]:
    for cand in _candidates(text):
        cand = cand.strip()
        if not cand:
            continue
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def parse_or_repair(text: str, repair_fn=None) -> Any:
    """Parse `text` as JSON; if it fails and `repair_fn` is given, call it once with
    the raw text to get a corrected string, then parse that.

    `repair_fn(raw: str) -> str` is typically a small LLM call.
    Raises ValueError if everything fails.
    """
    parsed = try_parse(text)
    if parsed is not None:
        return parsed
    if repair_fn is not None:
        repaired = repair_fn(text)
        parsed = try_parse(repaired)
        if parsed is not None:
            return parsed
    raise ValueError("Could not extract valid JSON from model output.")
