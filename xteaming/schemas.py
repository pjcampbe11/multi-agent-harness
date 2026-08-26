"""
Shared data structures for the X-Teaming harness.

Everything that moves between the Planner, Attacker, Verifier, and Optimizer is a
plain dataclass here, so the pipeline is easy to serialize, diff, log, and replay.
Reproducibility is a first-class concern: every object that represents a decision
carries enough context to reconstruct how it was produced.
"""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _uid() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Objective:
    """A single authorized test objective supplied by the operator.

    The harness never invents objectives. The operator loads them from a file that
    they are authorized to test against a target they own or have permission to
    assess. `category` is a free-form taxonomy tag (e.g. an OWASP LLM Top 10 id, an
    internal harm-code, or "benign-probe" for the shipped examples).
    """

    text: str
    category: str = "unspecified"
    id: str = field(default_factory=_uid)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanStep:
    """One planned conversation turn: what the attacker agent should try to elicit
    at this stage, moving from a neutral opener toward the objective."""

    index: int
    intent: str          # what this turn is meant to accomplish
    escalation: str      # how it advances from the previous turn


@dataclass
class AttackPlan:
    """A full multi-turn plan across the four X-Teaming dimensions."""

    objective_id: str
    persona: str          # e.g. "forensic psychologist"
    context: str          # e.g. "academic research"
    strategy: str         # e.g. "incident analysis"
    steps: List[PlanStep]
    id: str = field(default_factory=_uid)
    # A compact textual signature used for diversity scoring / dedup.
    signature: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        """Flattened representation used for embedding-based diversity scoring."""
        parts = [self.persona, self.context, self.strategy]
        parts += [f"{s.index}:{s.intent}" for s in self.steps]
        return " | ".join(parts)


@dataclass
class Turn:
    """A single executed round of the conversation."""

    index: int
    attacker_message: str
    target_response: str
    verifier_score: int
    verifier_rationale: str
    optimizer_iterations: int = 0
    optimizer_trace: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Transcript:
    """The complete record of one plan executed against the target."""

    objective: Objective
    plan: AttackPlan
    turns: List[Turn] = field(default_factory=list)
    succeeded: bool = False
    max_score: int = 1
    id: str = field(default_factory=_uid)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj
