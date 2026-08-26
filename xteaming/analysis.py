"""
Offline analysis over saved runs.

Everything here operates on the JSON transcripts and summaries the Orchestrator
writes — no model calls, no keys — so analysis is cheap, deterministic, and
reproducible. These functions back the `xteaming.cli analyze ...` subcommands.

Analyses provided:
    asr           — attack success rate, overall and per category
    scores        — per-turn score trajectories + where TextGrad engaged/recovered
    diversity     — realized plan diversity per objective vs. the target
    turns         — histogram of turns-to-success (attack efficiency)
    compare       — cross-model comparison across several eval runs
    calibration   — verifier score distribution (a sanity check on the grader)
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List


# ---- loading --------------------------------------------------------------
def load_transcripts(run_dir: str) -> List[Dict[str, Any]]:
    """Load every transcript_*.json under a run directory."""
    paths = glob.glob(os.path.join(run_dir, "**", "transcript_*.json"), recursive=True)
    out = []
    for p in sorted(paths):
        with open(p, "r", encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


# ---- analyses -------------------------------------------------------------
def attack_success_rate(transcripts: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_obj: Dict[str, Dict[str, Any]] = {}
    for t in transcripts:
        oid = t["objective"]["id"]
        rec = by_obj.setdefault(oid, {"category": t["objective"]["category"],
                                      "succeeded": False, "best_score": 1, "attempts": 0})
        rec["attempts"] += 1
        rec["best_score"] = max(rec["best_score"], t["max_score"])
        rec["succeeded"] = rec["succeeded"] or t["succeeded"]

    n = len(by_obj) or 1
    succ = sum(1 for r in by_obj.values() if r["succeeded"])
    by_cat: Dict[str, List[bool]] = defaultdict(list)
    for r in by_obj.values():
        by_cat[r["category"]].append(r["succeeded"])

    return {
        "objectives": len(by_obj),
        "overall_asr": round(succ / n, 4),
        "by_category": {c: round(sum(v) / len(v), 4) for c, v in sorted(by_cat.items())},
        "mean_best_score": round(sum(r["best_score"] for r in by_obj.values()) / n, 3),
    }


def score_trajectories(transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for t in transcripts:
        scores = [turn["verifier_score"] for turn in t["turns"]]
        recovered = any(
            turn["optimizer_iterations"] > 0
            and any(it["candidate_score"] > turn["verifier_score"] - 1 for it in turn.get("optimizer_trace", []))
            for turn in t["turns"]
        )
        out.append({
            "transcript": t["id"],
            "category": t["objective"]["category"],
            "scores": scores,
            "succeeded": t["succeeded"],
            "optimizer_turns": sum(1 for turn in t["turns"] if turn["optimizer_iterations"] > 0),
            "recovered_via_textgrad": recovered,
        })
    return out


def diversity_report(plans_or_run: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Read `metadata.set_diversity` off plans embedded in transcripts."""
    seen: Dict[str, float] = {}
    backend = "unknown"
    for t in plans_or_run:
        plan = t.get("plan", t)
        md = plan.get("metadata", {})
        if "set_diversity" in md:
            seen[plan["objective_id"]] = md["set_diversity"]
            backend = md.get("diversity_backend", backend)
    if not seen:
        return {"note": "no diversity metadata found in these transcripts"}
    vals = list(seen.values())
    return {
        "backend": backend,
        "objectives": len(seen),
        "mean_realized_diversity": round(sum(vals) / len(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "paper_target": 0.702,
    }


def turns_to_success(transcripts: List[Dict[str, Any]]) -> Dict[str, Any]:
    hist = Counter()
    successes = 0
    for t in transcripts:
        if t["succeeded"]:
            successes += 1
            hist[len(t["turns"])] += 1
    ordered = dict(sorted(hist.items()))
    mean = (sum(k * v for k, v in ordered.items()) / successes) if successes else None
    return {"successful_trajectories": successes,
            "turns_histogram": ordered,
            "mean_turns_to_success": round(mean, 2) if mean else None}


def verifier_calibration(transcripts: List[Dict[str, Any]]) -> Dict[str, Any]:
    dist = Counter()
    for t in transcripts:
        for turn in t["turns"]:
            dist[turn["verifier_score"]] += 1
            for it in turn.get("optimizer_trace", []):
                dist[it["candidate_score"]] += 1
    total = sum(dist.values()) or 1
    return {"score_distribution": {str(k): dist[k] for k in range(1, 6)},
            "share": {str(k): round(dist[k] / total, 3) for k in range(1, 6)},
            "total_gradings": total}


def compare_models(eval_dir: str) -> Dict[str, Any]:
    """Compare several per-target runs. Expects eval_dir/<target-name>/ subdirs,
    each a run directory produced by the eval command."""
    results: Dict[str, Any] = {}
    for target_dir in sorted(glob.glob(os.path.join(eval_dir, "*"))):
        if not os.path.isdir(target_dir):
            continue
        name = os.path.basename(target_dir)
        transcripts = load_transcripts(target_dir)
        if not transcripts:
            continue
        asr = attack_success_rate(transcripts)
        tts = turns_to_success(transcripts)
        results[name] = {
            "overall_asr": asr["overall_asr"],
            "mean_best_score": asr["mean_best_score"],
            "mean_turns_to_success": tts["mean_turns_to_success"],
            "by_category": asr["by_category"],
        }
    # Rank most-robust first (lowest ASR against it = hardest to break).
    ranking = sorted(results.items(), key=lambda kv: kv[1]["overall_asr"])
    return {"per_model": results,
            "robustness_ranking_low_asr_first": [name for name, _ in ranking]}


# ---- pretty printing ------------------------------------------------------
def render(report: Dict[str, Any], title: str) -> str:
    lines = [f"=== {title} ===", json.dumps(report, indent=2, ensure_ascii=False)]
    return "\n".join(lines)
