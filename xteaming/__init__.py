"""
X-Teaming harness — a faithful, model-agnostic reimplementation of the multi-agent
multi-turn red-teaming loop from Rahman et al., "X-Teaming: Multi-Turn Jailbreaks
and Defenses with Adaptive Multi-Agents" (COLM 2025, arXiv:2504.13203v2).

For authorized red teaming and safety-data generation only. See README.

Four core agents:
    Planner            — diverse multi-turn plans across persona/context/strategy/steps
    Attacker           — executes a plan turn by turn against the target
    Verifier           — grades each target response 1-5
    TextGradOptimizer  — textual-gradient rewrite loop when the score drops
"""

from .schemas import Objective, AttackPlan, PlanStep, Turn, Transcript
from .llm import LLMClient, GenerationConfig
from .diversity import DiversityScorer
from .planner import Planner, PlannerConfig
from .attacker import Attacker, AttackerConfig
from .verifier import Verifier, VerifierConfig
from .optimizer import TextGradOptimizer, OptimizerConfig
from .orchestrator import Orchestrator, RunConfig, make_target_fn

__version__ = "0.1.0"

__all__ = [
    "Objective", "AttackPlan", "PlanStep", "Turn", "Transcript",
    "LLMClient", "GenerationConfig", "DiversityScorer",
    "Planner", "PlannerConfig",
    "Attacker", "AttackerConfig",
    "Verifier", "VerifierConfig",
    "TextGradOptimizer", "OptimizerConfig",
    "Orchestrator", "RunConfig", "make_target_fn",
]
