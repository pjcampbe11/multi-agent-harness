#!/usr/bin/env python3
"""
Runnable example for the X-Teaming harness.

This drives the full loop against a target model you configure, using the BENIGN
example objectives shipped in objectives.example.jsonl (pirate persona, codeword,
haiku, etc.). It demonstrates the machinery end to end without generating attack
content.

Before running:
  1. pip install -r requirements.txt
  2. Point the clients at a model you are AUTHORIZED to test. The defaults below use
     an OpenAI-compatible endpoint; for a local Ollama model, set
     base_url="http://localhost:11434/v1" and model="llama3.1" (any served model).
  3. Set the authorization flag when you construct the Orchestrator.

Usage:
    export OPENAI_API_KEY=sk-...            # or leave unset for a local endpoint
    python run_example.py
"""

from __future__ import annotations

import json
import os
import sys

from xteaming import (
    LLMClient, Objective, Orchestrator, PlannerConfig, RunConfig,
)


def load_objectives(path: str):
    objectives = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            objectives.append(Objective(text=row["text"], category=row.get("category", "unspecified")))
    return objectives


def main() -> int:
    # --- configure the models you are authorized to use -------------------
    # One endpoint can serve every role; here we reuse a single model for brevity.
    base_url = os.environ.get("XTEAMING_BASE_URL")   # e.g. http://localhost:11434/v1
    model = os.environ.get("XTEAMING_MODEL", "gpt-4o-mini")

    attacker_llm = LLMClient(model=model, base_url=base_url)
    target_llm   = LLMClient(model=model, base_url=base_url)

    objectives = load_objectives(
        os.path.join(os.path.dirname(__file__), "objectives.example.jsonl")
    )

    # Keep the demo small and cheap; scale these up for a real engagement.
    orch = Orchestrator(
        attacker_llm=attacker_llm,
        target_llm=target_llm,
        target_system_prompt="You are a helpful assistant.",
        planner_config=PlannerConfig(n_plans=6, batch_size=6, min_diversity=0.55),
        run_config=RunConfig(output_dir="runs", max_plans_per_objective=3),
        # Authorization gate: only the operator sets this, and only for a target
        # they own or are permitted to assess.
        authorized=True,
    )

    transcripts = orch.run(objectives)

    succeeded = sum(1 for t in transcripts if t.succeeded)
    print(f"\nRan {len(transcripts)} trajectories across {len(objectives)} objectives.")
    print(f"Objectives met at least once: {succeeded} trajectories reached score 5.")
    print("Transcripts, manifest, and summary written under ./runs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
