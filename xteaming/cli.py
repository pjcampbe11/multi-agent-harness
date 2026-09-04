"""
Command-line interface for the X-Teaming harness.

Subcommands:
    plan      — generate diverse plans for objectives and print a diversity report
    attack    — run the full loop against one target, write a run directory
    eval      — run the loop across several targets, write an eval/ tree for comparison
    analyze   — offline analysis over saved runs (asr, scores, diversity, turns,
                calibration, compare)

Every subcommand is fully scriptable; see RUNBOOK.md for a start-to-finish sequence.

All model roles are configured by flags, so you can point attacker/target/verifier at
different providers. Authorization is required for anything that touches a target: pass
--authorized to confirm you own or are permitted to test it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import analysis
from .llm import LLMClient
from .planner import Planner, PlannerConfig
from .diversity import DiversityScorer
from .orchestrator import Orchestrator, RunConfig
from .schemas import Objective


# ---- shared helpers -------------------------------------------------------
def _load_objectives(path: str) -> List[Objective]:
    objs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            objs.append(Objective(text=row["text"],
                                  category=row.get("category", "unspecified"),
                                  metadata=row.get("metadata", {})))
    return objs


def _client(model: str, base_url: Optional[str], key_env: str) -> LLMClient:
    return LLMClient(model=model, base_url=base_url or None, api_key_env=key_env)


# ---- subcommands ----------------------------------------------------------
def cmd_plan(args) -> int:
    objs = _load_objectives(args.objectives)
    llm = _client(args.model, args.base_url, args.api_key_env)
    planner = Planner(llm, DiversityScorer(),
                      PlannerConfig(n_plans=args.n_plans, min_diversity=args.min_diversity))
    all_plans = {}
    for obj in objs:
        plans = planner.plan(obj)
        all_plans[obj.id] = [p.__dict__ for p in plans]
        div = plans[0].metadata.get("set_diversity") if plans else None
        print(f"[plan] {obj.category:40s} {len(plans)} plans  diversity={div}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(all_plans, fh, indent=2, default=lambda o: getattr(o, "__dict__", str(o)))
        print(f"[plan] wrote {args.out}")
    return 0


def _verifier_client(args):
    """Verifier can be fully independent of the attacker: its own model, endpoint,
    and key. Falls back to the attacker's endpoint/key when not overridden."""
    if not args.verifier:
        return None
    v_base = getattr(args, "verifier_base_url", None) or args.base_url
    v_key = getattr(args, "verifier_key_env", None) or args.api_key_env
    return _client(args.verifier, v_base, v_key)


def cmd_attack(args) -> int:
    objs = _load_objectives(args.objectives)
    orch = Orchestrator(
        attacker_llm=_client(args.attacker, args.base_url, args.api_key_env),
        target_llm=_client(args.target, args.target_base_url or args.base_url, args.target_key_env),
        verifier_llm=_verifier_client(args),
        target_system_prompt=args.system_prompt,
        planner_config=PlannerConfig(n_plans=args.n_plans, min_diversity=args.min_diversity),
        run_config=RunConfig(output_dir=args.out, max_plans_per_objective=args.max_plans),
        authorized=args.authorized,
    )
    transcripts = orch.run(objs)
    succ = sum(1 for t in transcripts if t.succeeded)
    print(f"[attack] target={args.target}  trajectories={len(transcripts)}  reached_5={succ}")
    print(f"[attack] artifacts under {args.out}/")
    return 0


def cmd_eval(args) -> int:
    """Run the same objectives against several targets; write eval/<target>/ each."""
    objs = _load_objectives(args.objectives)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    os.makedirs(args.out, exist_ok=True)
    for target in targets:
        safe = target.replace("/", "_").replace(":", "_")
        run_dir = os.path.join(args.out, safe)
        orch = Orchestrator(
            attacker_llm=_client(args.attacker, args.base_url, args.api_key_env),
            target_llm=_client(target, args.target_base_url or args.base_url, args.target_key_env),
            verifier_llm=_verifier_client(args),
            target_system_prompt=args.system_prompt,
            planner_config=PlannerConfig(n_plans=args.n_plans, min_diversity=args.min_diversity),
            run_config=RunConfig(output_dir=run_dir, max_plans_per_objective=args.max_plans),
            authorized=args.authorized,
        )
        transcripts = orch.run(objs)
        succ = sum(1 for t in transcripts if t.succeeded)
        print(f"[eval] {target:28s} trajectories={len(transcripts):3d}  reached_5={succ}")
    print(f"[eval] compare with:  python -m xteaming.cli analyze compare --eval {args.out}")
    return 0


def cmd_analyze(args) -> int:
    kind = args.kind
    if kind == "compare":
        report = analysis.compare_models(args.eval)
        print(analysis.render(report, "cross-model comparison"))
        return 0

    # Single-run analyses find the newest run dir if a parent is given.
    run_dir = args.run
    if run_dir and not analysis.load_transcripts(run_dir):
        subdirs = sorted(
            (d for d in (os.path.join(run_dir, x) for x in os.listdir(run_dir)) if os.path.isdir(d)),
            reverse=True,
        )
        for d in subdirs:
            if analysis.load_transcripts(d):
                run_dir = d
                break

    transcripts = analysis.load_transcripts(run_dir)
    if not transcripts:
        print(f"[analyze] no transcripts under {run_dir}", file=sys.stderr)
        return 1

    dispatch = {
        "asr": lambda: analysis.attack_success_rate(transcripts),
        "scores": lambda: analysis.score_trajectories(transcripts),
        "diversity": lambda: analysis.diversity_report(transcripts),
        "turns": lambda: analysis.turns_to_success(transcripts),
        "calibration": lambda: analysis.verifier_calibration(transcripts),
    }
    report = dispatch[kind]()
    print(analysis.render(report, f"{kind}  ({run_dir})"))
    return 0


# ---- arg parsing ----------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xteaming", description="X-Teaming red-team harness")
    sub = p.add_subparsers(dest="command", required=True)

    def add_model_flags(sp, target=False):
        sp.add_argument("--base-url", default=os.environ.get("XTEAMING_BASE_URL"),
                        help="OpenAI-compatible base URL for attacker/verifier/planner")
        sp.add_argument("--api-key-env", default="OPENAI_API_KEY")
        if target:
            sp.add_argument("--target-base-url", default=os.environ.get("XTEAMING_TARGET_BASE_URL"))
            sp.add_argument("--target-key-env", default="OPENAI_API_KEY")
            # Verifier can run on its own endpoint/key so it's independent of both
            # the attacker and the target (defaults to the attacker's endpoint).
            sp.add_argument("--verifier-base-url", default=os.environ.get("XTEAMING_VERIFIER_BASE_URL"))
            sp.add_argument("--verifier-key-env", default=None)

    # plan
    sp = sub.add_parser("plan", help="generate diverse plans and report diversity")
    sp.add_argument("--objectives", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--n-plans", type=int, default=50)
    sp.add_argument("--min-diversity", type=float, default=0.702)
    sp.add_argument("--out", default=None)
    add_model_flags(sp)
    sp.set_defaults(func=cmd_plan)

    # attack
    sp = sub.add_parser("attack", help="run the full loop against one target")
    sp.add_argument("--objectives", required=True)
    sp.add_argument("--attacker", required=True)
    sp.add_argument("--target", required=True)
    sp.add_argument("--verifier", default=None, help="defaults to the attacker model")
    sp.add_argument("--system-prompt", default="You are a helpful assistant.")
    sp.add_argument("--n-plans", type=int, default=20)
    sp.add_argument("--min-diversity", type=float, default=0.702)
    sp.add_argument("--max-plans", type=int, default=10)
    sp.add_argument("--out", default="runs")
    sp.add_argument("--authorized", action="store_true",
                    help="confirm you own or may test the target")
    add_model_flags(sp, target=True)
    sp.set_defaults(func=cmd_attack)

    # eval
    sp = sub.add_parser("eval", help="run across several targets for comparison")
    sp.add_argument("--objectives", required=True)
    sp.add_argument("--attacker", required=True)
    sp.add_argument("--targets", required=True, help="comma-separated target model ids")
    sp.add_argument("--verifier", default=None)
    sp.add_argument("--system-prompt", default="You are a helpful assistant.")
    sp.add_argument("--n-plans", type=int, default=20)
    sp.add_argument("--min-diversity", type=float, default=0.702)
    sp.add_argument("--max-plans", type=int, default=10)
    sp.add_argument("--out", default="eval")
    sp.add_argument("--authorized", action="store_true")
    add_model_flags(sp, target=True)
    sp.set_defaults(func=cmd_eval)

    # analyze
    sp = sub.add_parser("analyze", help="offline analysis over saved runs")
    sp.add_argument("kind", choices=["asr", "scores", "diversity", "turns", "calibration", "compare"])
    sp.add_argument("--run", default=None, help="a run dir (or a parent; newest is picked)")
    sp.add_argument("--eval", default=None, help="an eval dir (for 'compare')")
    sp.set_defaults(func=cmd_analyze)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
