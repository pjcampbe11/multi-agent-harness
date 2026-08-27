#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xteam_seeds.py — Seed-set builder for the X-Teaming defensive red-team harness.

WHAT THIS IS
============
A command-line utility that turns *public, peer-reviewed* AI-safety harm
benchmarks into the `objectives.jsonl` seed format consumed by the X-Teaming
harness (Rahman et al., "X-Teaming", COLM 2025, arXiv:2504.13203).

The X-Teaming harness ships with NO harmful objectives. To run a real
evaluation you supply your own objectives from a sanctioned test set. Each
objective is one line of JSON describing a *behavior to probe for*, so a
defender can measure whether a target model can be steered into that behavior
across a multi-turn conversation, and then export refusal-substituted training
records (the "XGuard-style" defense loop). This tool automates the tedious,
error-prone step of converting six canonical benchmarks — each with its own
column names, hosting, and licensing — into one consistent schema.

Nothing in THIS file is harmful content. It contains no attack strings; it is a
schema/ETL converter. The harmful *behavior descriptions* live inside the public
academic datasets it downloads, which are already published for exactly this
defensive purpose and (for the gated ones) require you to accept a research-use
license on Hugging Face first.

OUTPUT SCHEMA (one JSON object per line — JSONL)
================================================
    {"text": "<behavior / forbidden prompt>", "category": "<taxonomy label>", "source": "<dataset id>"}

  * text     — the behavior the harness will try to elicit (the harness reads this).
  * category — a taxonomy bucket, carried through from the source dataset so you
               can slice results by hazard class (the harness reads this too).
  * source   — provenance tag added by this tool. It is an EXTRA key; the harness
               only needs `text`/`category`, and ignores unknown keys. Keeping it
               makes the merged/deduplicated file auditable ("which benchmark did
               this line come from?"). Drop it with --minimal for a strict
               two-key file identical to the wiki's example.

THE SIX BENCHMARKS (see `describe` subcommand for details)
==========================================================
    key            source                              access
    -------------  ----------------------------------  --------------------------
    advbench       github: llm-attacks/llm-attacks      open (raw CSV)
    harmbench      github: centerforaisafety/HarmBench   open (raw CSV)
    jailbreakbench hf: JailbreakBench/JBB-Behaviors      open (HF, no gate)
    strongreject   github: alexandrasouly/strongreject   open (raw CSV)
    xguard         hf: marslabucla/XGuard-Train          GATED — accept license + `auth`
    ailuminate     hf: mlcommons/ailuminate              GATED — accept license + `auth`

TYPICAL USAGE
=============
    # 0. (once) authenticate for the gated Hugging Face datasets
    python xteam_seeds.py auth --token hf_xxx        # or: export HF_TOKEN=...

    # 1. build one dataset -> out/objectives.advbench.jsonl
    python xteam_seeds.py build --dataset advbench

    # 2. build every dataset you can reach, plus a merged, de-duplicated
    #    out/objectives.all.jsonl containing every UNIQUE behavior across sets
    python xteam_seeds.py build --dataset all --merge

    # 3. inspect what each key does, and its license/authorization notes
    python xteam_seeds.py describe
    python xteam_seeds.py list

AUTHORIZATION
=============
Use only against models you own or have written permission to assess, and only
with objectives from a sanctioned engagement. The gated datasets require you to
accept their research-use license on Hugging Face before download — this tool
will not (and cannot) bypass that gate; it just fails with a clear message
telling you which license to accept.

Author: generated for Patrick's X-Teaming harness workflow.
License of THIS script: do whatever you like; the *datasets* keep their own.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import requests

# `datasets` / `huggingface_hub` are only needed for the HF-hosted benchmarks
# (jailbreakbench, xguard, ailuminate). Import lazily so the GitHub-only
# converters still work in an environment where those libs are missing.
try:  # pragma: no cover - trivial import guard
    from datasets import load_dataset  # type: ignore
    _HAVE_DATASETS = True
except Exception:  # noqa: BLE001 - any import failure means "unavailable"
    _HAVE_DATASETS = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTDIR = "out"
HTTP_TIMEOUT = 60  # seconds; the raw CSVs are small but HF/GitHub can be slow
USER_AGENT = "xteam-seeds/1.0 (+defensive safety eval; authorized use only)"

# GitHub raw endpoints for the two CSV-hosted benchmarks. Pinned to `main`.
ADVBENCH_CSV = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)
HARMBENCH_CSV = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "main/data/behavior_datasets/harmbench_behaviors_text_{split}.csv"
)  # split in {all, val, test}
STRONGREJECT_CSV = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/"
    "main/strongreject_dataset/strongreject_{variant}dataset.csv"
)  # variant in {"", "small_"}

# AILuminate public DEMO prompt set (CC-BY-4.0), hosted openly on GitHub — no HF
# gate needed. 1,290 prompts. Columns: release_prompt_id, prompt_text, hazard,
# persona, locale, prompt_hash.
AILUMINATE_DEMO_CSV = (
    "https://raw.githubusercontent.com/mlcommons/ailuminate/"
    "main/airr_official_1.0_demo_en_us_prompt_set_release.csv"
)

# AILuminate hazard codes -> readable category (MLCommons AILuminate v1.0 taxonomy).
AILUMINATE_HAZARDS = {
    "vcr": "violent_crimes", "src": "sex_related_crimes", "cse": "child_sexual_exploitation",
    "ssh": "suicide_self_harm", "iwp": "indiscriminate_weapons_cbrne", "ipv": "intellectual_property",
    "dfm": "defamation", "ncr": "nonviolent_crimes", "hte": "hate", "prv": "privacy",
    "spc_ele": "specialized_advice_elections", "spc_fin": "specialized_advice_financial",
    "spc_hlt": "specialized_advice_health", "spc": "specialized_advice", "sxc_prn": "sexual_content",
}

# The ungated, Apache-2.0 open release of XGuard-Train (YuFeng-XGuard guardrail
# corpus). Used as the default `xguard` source so no HF license gate is needed.
# The gated paper dataset (marslabucla/XGuard-Train) remains available via
# --xguard-source paper for users who have accepted its license and run `auth`.
XGUARD_OPEN_HF = "Alibaba-AAIG/XGuard-Train-Open-200K"
XGUARD_OPEN_JSONL = (
    "https://huggingface.co/datasets/Alibaba-AAIG/XGuard-Train-Open-200K/"
    "resolve/main/xguard_train_open_200k.jsonl"
)
XGUARD_PAPER_HF = "marslabucla/XGuard-Train"

# XGuard-Open label token -> readable category (from the dataset's category list).
XGUARD_OPEN_LABELS = {
    "sec": "safe", "pc": "pornographic_contraband", "dc": "drug_crimes",
    "dw": "dangerous_weapons", "pi": "property_infringement", "ec": "economic_crimes",
    "ac": "abusive_curses", "def": "defamation", "ti": "threats_intimidation",
    "cy": "cyberbullying", "ph": "physical_health", "mh": "mental_health",
    "se": "social_ethics", "sci": "science_ethics", "pp": "personal_privacy",
    "cs": "commercial_secret", "acc": "access_control", "mc": "malicious_code",
    "ha": "hacker_attack", "ps": "physical_security", "ter": "violent_terrorism",
    "sd": "social_disruption", "ext": "extremist_ideology", "fin": "finance",
    "med": "medicine", "law": "law", "cm": "corruption_of_minors",
    "ma": "minor_abuse_exploitation", "md": "minor_delinquency",
}

# Categories we deliberately EXCLUDE when generating seed files, to avoid
# reproducing child-sexual-exploitation / minor-abuse probes. A sanctioned
# engagement that genuinely requires these must source them from the licensed
# benchmark directly; this converter will not emit them.
AILUMINATE_BLOCKED_HAZARDS = {"cse"}
XGUARD_BLOCKED_LABELS = {"cm", "ma", "md", "pc"}


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class Objective:
    """One harness seed: a behavior to probe, plus taxonomy + provenance."""
    text: str
    category: str
    source: str

    def to_record(self, minimal: bool = False) -> dict:
        """Serialize to the dict written as one JSONL line.

        minimal=True drops the provenance `source` key, yielding the exact
        two-key schema shown in the X-Teaming wiki example.
        """
        if minimal:
            return {"text": self.text, "category": self.category}
        return {"text": self.text, "category": self.category, "source": self.source}


# A "builder" takes an options bag and yields Objective records. Registering
# builders in a dict (see BUILDERS below) is what lets `--dataset` and the
# argparse choices stay in sync with zero duplication.
Builder = Callable[["BuildOptions"], Iterator[Objective]]


@dataclass
class BuildOptions:
    """Everything a builder might need, gathered from CLI args."""
    outdir: Path = field(default_factory=lambda: Path(DEFAULT_OUTDIR))
    limit: Optional[int] = None          # cap objectives per dataset (debug/sampling)
    include_benign: bool = False         # jailbreakbench ships harmful+benign pairs
    harmbench_split: str = "all"         # all | val | test
    strongreject_small: bool = False     # use the 60-item curated subset
    hf_token: Optional[str] = None       # for gated HF datasets
    minimal: bool = False                # write {text,category} only
    xguard_source: str = "open"          # open (ungated) | paper (gated marslabucla)
    english_only: bool = True            # keep primarily-English prompts (xguard-open is multilingual)


# ---------------------------------------------------------------------------
# Text hygiene + de-duplication helpers
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """Normalize a behavior string: unicode-normalize, collapse whitespace, trim.

    Benchmarks are inconsistent about trailing spaces, smart quotes, and CRLF.
    Normalizing here keeps the JSONL tidy and makes cross-dataset dedup reliable.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WS_RE.sub(" ", s)
    return s.strip()


def dedup_key(text: str) -> str:
    """Case-folded, punctuation-light key used to detect duplicate behaviors.

    Two objectives that differ only by capitalization, trailing punctuation, or
    whitespace should collapse to one line in the merged set. We deliberately do
    NOT stem or fuzzy-match — that risks merging genuinely distinct behaviors.
    """
    t = clean_text(text).casefold()
    t = t.rstrip(".!?;: ")
    return t


def looks_english(text: str, threshold: float = 0.85) -> bool:
    """Cheap heuristic: True if the string is mostly ASCII letters/spaces.

    XGuard-Open is heavily multilingual; the harness objectives and the other
    five benchmarks are English, so by default we keep only primarily-English
    prompts for a consistent seed set. Not a real language detector — just a
    fast ASCII-ratio gate that keeps English and drops Cyrillic/CJK/Thai/etc.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for c in letters if c.isascii())
    return (ascii_letters / len(letters)) >= threshold


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _fetch_csv_rows(url: str) -> list[dict]:
    """Download a CSV and return it as a list of dict rows (header-keyed).

    Raises requests.HTTPError on a bad status so the caller can surface a clear
    message. Uses an explicit UA and timeout for polite, predictable behavior.
    """
    resp = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    # utf-8-sig strips a BOM if GitHub serves one.
    text = resp.content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


# ---------------------------------------------------------------------------
# Per-dataset builders
# ---------------------------------------------------------------------------
# Each builder is a generator so a --limit can stop it early without downloading
# or parsing more than necessary. Each carries a docstring documenting the exact
# source schema it maps FROM, since that is the part most likely to drift.

def build_advbench(opts: BuildOptions) -> Iterator[Objective]:
    """AdvBench — harmful-behavior strings from the GCG paper (Zou et al. 2023).

    Source: llm-attacks/llm-attacks, data/advbench/harmful_behaviors.csv
    Columns: goal,target   (we use `goal`; `target` is the affirmative-completion
             string GCG optimizes toward and is not a behavior on its own.)
    AdvBench has no per-item taxonomy, so every item is tagged "advbench".
    """
    rows = _fetch_csv_rows(ADVBENCH_CSV)
    n = 0
    for row in rows:
        text = clean_text(row.get("goal", ""))
        if not text:
            continue
        yield Objective(text=text, category="advbench", source="advbench")
        n += 1
        if opts.limit and n >= opts.limit:
            return


def build_harmbench(opts: BuildOptions) -> Iterator[Objective]:
    """HarmBench — 400+ standardized harmful behaviors, 7 semantic categories.

    Source: centerforaisafety/HarmBench,
            data/behavior_datasets/harmbench_behaviors_text_{split}.csv
    Columns: Behavior,FunctionalCategory,SemanticCategory,Tags,ContextString,BehaviorID
      * text     <- Behavior. If a ContextString is present (contextual behaviors),
                    we prepend it so the objective is self-contained.
      * category <- SemanticCategory (chemical_biological, cybercrime, ...).
    `--harmbench-split` picks all|val|test (default all).
    """
    split = opts.harmbench_split
    if split not in {"all", "val", "test"}:
        raise ValueError(f"harmbench split must be all|val|test, got {split!r}")
    rows = _fetch_csv_rows(HARMBENCH_CSV.format(split=split))
    n = 0
    for row in rows:
        behavior = clean_text(row.get("Behavior", ""))
        if not behavior:
            continue
        context = clean_text(row.get("ContextString", ""))
        text = f"{context}\n\n{behavior}".strip() if context else behavior
        category = clean_text(row.get("SemanticCategory", "")) or "harmbench"
        yield Objective(text=text, category=category, source="harmbench")
        n += 1
        if opts.limit and n >= opts.limit:
            return


def build_strongreject(opts: BuildOptions) -> Iterator[Objective]:
    """StrongREJECT — behaviors paired with a calibrated grader.

    Source: alexandrasouly/strongreject,
            strongreject_dataset/strongreject_{small_,}dataset.csv
    Columns: category,source,forbidden_prompt
      * text     <- forbidden_prompt
      * category <- category (e.g. "Disinformation and deception")
    `--strongreject-small` uses the 60-item curated subset instead of the full set.
    """
    variant = "small_" if opts.strongreject_small else ""
    rows = _fetch_csv_rows(STRONGREJECT_CSV.format(variant=variant))
    n = 0
    for row in rows:
        text = clean_text(row.get("forbidden_prompt", ""))
        if not text:
            continue
        category = clean_text(row.get("category", "")) or "strongreject"
        yield Objective(text=text, category=category, source="strongreject")
        n += 1
        if opts.limit and n >= opts.limit:
            return


def build_jailbreakbench(opts: BuildOptions) -> Iterator[Objective]:
    """JailbreakBench — 100 behaviors, harmful + benign paired, with a leaderboard.

    Source: hf: JailbreakBench/JBB-Behaviors, config "behaviors".
    Splits: "harmful" (default) and "benign" (add --include-benign to also emit).
    Columns: Index, Goal, Target, Behavior, Category, Source
      * text     <- Goal
      * category <- Category (Harassment/Discrimination, Malware/Hacking, ...)
    This dataset is public on HF and needs no license acceptance, but we still
    load it through `datasets` for a consistent code path.
    """
    if not _HAVE_DATASETS:
        raise RuntimeError(
            "jailbreakbench needs the `datasets` library. "
            "Install with: pip install datasets"
        )
    splits = ["harmful"] + (["benign"] if opts.include_benign else [])
    n = 0
    for split in splits:
        ds = load_dataset(
            "JailbreakBench/JBB-Behaviors", "behaviors",
            split=split, token=opts.hf_token,
        )
        for row in ds:
            text = clean_text(row.get("Goal", ""))
            if not text:
                continue
            category = clean_text(row.get("Category", "")) or "jailbreakbench"
            # Tag benign rows distinctly so a defender never confuses them with
            # harmful seeds in the merged file.
            src = "jailbreakbench" if split == "harmful" else "jailbreakbench-benign"
            yield Objective(text=text, category=category, source=src)
            n += 1
            if opts.limit and n >= opts.limit:
                return


# Keys that plausibly hold "the behavior" across XGuard-Train's possible schemas.
# XGuard-Train is multi-turn; the seed behavior may be stored as a flat field or
# have to be recovered from the first user turn of a conversation. We probe a
# ranked list of field names and fall back to conversation parsing.
_XGUARD_TEXT_KEYS = (
    "behavior", "goal", "objective", "plain_query", "query",
    "prompt", "instruction", "harmful_behavior", "request",
)
_XGUARD_CAT_KEYS = ("category", "safe_category", "harm_category", "type", "class")
_XGUARD_CONV_KEYS = ("conversations", "messages", "conversation", "dialogue", "turns")


def _first_user_turn(conv) -> str:
    """Best-effort extraction of the opening human message from a chat list.

    Handles the two common shapes:
      [{"from": "human", "value": "..."}, ...]      (ShareGPT style)
      [{"role": "user", "content": "..."}, ...]     (OpenAI style)
    Returns "" if nothing usable is found.
    """
    if not isinstance(conv, (list, tuple)):
        return ""
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from") or turn.get("role") or "").lower()
        if role in ("human", "user"):
            return clean_text(turn.get("value") or turn.get("content") or "")
    # No explicit role match — fall back to the very first turn's payload.
    first = conv[0] if conv else None
    if isinstance(first, dict):
        return clean_text(first.get("value") or first.get("content") or "")
    return ""


def build_xguard(opts: BuildOptions) -> Iterator[Objective]:
    """XGuard-Train — multi-turn LM-safety behaviors.

    Two sources, selected with --xguard-source:

      * "open" (DEFAULT, UNGATED): Alibaba-AAIG/XGuard-Train-Open-200K, an
        Apache-2.0 open release of the YuFeng-XGuard guardrail corpus. Streamed
        (no full download). Each row is a guardrail sample:
            id, sample_type, prompt, response, stage, policy, label, explanation
        We extract the harmful *user query* as an objective from rows that:
          - have a non-null `prompt` (stage q or qr),
          - are labelled as a RISK category (label != "sec"/safe),
          - are NOT a minor-safety category (XGUARD_BLOCKED_LABELS),
          - and (by default) read as English (looks_english; --keep-multilingual
            to keep all languages).
        `category` <- label mapped via XGUARD_OPEN_LABELS. Behaviors are
        de-duplicated (the 200K guard samples collapse to far fewer unique
        queries). source tag: "xguard-open".

      * "paper" (GATED): marslabucla/XGuard-Train — the paper's own 30K set.
        Requires accepting its license on HF and running `auth` first. Adapts
        to its schema by probing flat behavior fields then the first user turn.
    """
    if not _HAVE_DATASETS:
        raise RuntimeError("xguard needs the `datasets` library: pip install datasets")

    if opts.xguard_source == "paper":
        yield from _build_xguard_paper(opts)
        return

    # --- default: ungated open release, streamed as raw JSONL over HTTP ---
    # The HF Arrow auto-conversion for this dataset fails (a nullable field has
    # mixed types across rows), so we bypass `datasets` and stream the single
    # published .jsonl line by line. This is memory-light (rows are processed
    # and discarded) and robust to schema quirks.
    resp = requests.get(
        XGUARD_OPEN_JSONL, stream=True, timeout=HTTP_TIMEOUT,
        headers={"User-Agent": USER_AGENT,
                 **({"Authorization": f"Bearer {opts.hf_token}"} if opts.hf_token else {})},
    )
    resp.raise_for_status()
    seen: set[str] = set()
    n = 0
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue  # tolerate the rare malformed line rather than aborting
        # Keep only "general" foundational samples; "dynamic_policy" rows use
        # on-the-fly label tokens (a, b, ...) that are meaningless as categories.
        if row.get("sample_type") != "general":
            continue
        prompt = row.get("prompt")
        if not prompt:
            continue  # response-only guard samples carry no user objective
        label = str(row.get("label") or "").strip().lower()
        if label in ("", "sec"):
            continue  # "sec" == Safe; skip benign classifications
        if label in XGUARD_BLOCKED_LABELS:
            continue  # minor-safety categories are deliberately excluded
        text = clean_text(prompt)
        if not text:
            continue
        if opts.english_only and not looks_english(text):
            continue
        key = dedup_key(text)
        if key in seen:
            continue
        seen.add(key)
        category = XGUARD_OPEN_LABELS.get(label, label)
        yield Objective(text=text, category=category, source="xguard-open")
        n += 1
        if opts.limit and n >= opts.limit:
            resp.close()
            return

def _build_xguard_paper(opts: BuildOptions) -> Iterator[Objective]:
    """Gated marslabucla/XGuard-Train loader (schema-adaptive). See build_xguard."""
    ds = load_dataset(XGUARD_PAPER_HF, split="train", token=opts.hf_token)
    seen: set[str] = set()
    n = 0
    for row in ds:
        text = ""
        for k in _XGUARD_TEXT_KEYS:
            if row.get(k):
                text = clean_text(row[k]); break
        if not text:
            for k in _XGUARD_CONV_KEYS:
                if k in row and row[k]:
                    text = _first_user_turn(row[k])
                    if text:
                        break
        if not text:
            continue
        key = dedup_key(text)
        if key in seen:
            continue
        seen.add(key)
        category = "xguard"
        for k in _XGUARD_CAT_KEYS:
            if row.get(k):
                category = clean_text(row[k]); break
        yield Objective(text=text, category=category, source="xguard")
        n += 1
        if opts.limit and n >= opts.limit:
            return


def build_ailuminate(opts: BuildOptions) -> Iterator[Objective]:
    """MLCommons AILuminate — standardized safety-grade prompt set.

    Source: the PUBLIC demo prompt set (CC-BY-4.0) at
        github: mlcommons/ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv
    No Hugging Face gate or login required. 1,290 English prompts.
    Columns: release_prompt_id, prompt_text, hazard, persona, locale, prompt_hash
      * text     <- prompt_text
      * category <- hazard mapped to a readable name (AILUMINATE_HAZARDS)
    Child-sexual-exploitation prompts (hazard `cse`) are deliberately skipped
    (AILUMINATE_BLOCKED_HAZARDS); this converter will not reproduce them.
    (If you have accepted the license for the larger official/practice set on
    Hugging Face, you can still point `datasets` at it yourself — the demo set
    is used here so the tool works with zero gating.)
    """
    rows = _fetch_csv_rows(AILUMINATE_DEMO_CSV)
    n = skipped_blocked = 0
    for row in rows:
        hazard = clean_text(row.get("hazard", "")).lower()
        if hazard in AILUMINATE_BLOCKED_HAZARDS:
            skipped_blocked += 1
            continue
        text = clean_text(row.get("prompt_text") or row.get("prompt") or "")
        if not text:
            continue
        category = AILUMINATE_HAZARDS.get(hazard, hazard or "ailuminate")
        yield Objective(text=text, category=category, source="ailuminate")
        n += 1
        if opts.limit and n >= opts.limit:
            return
    if skipped_blocked:
        print(f"  (ailuminate: skipped {skipped_blocked} child-safety-category prompts)",
              file=sys.stderr)


# The single source of truth mapping dataset key -> (builder, one-line summary,
# access note). argparse choices, `list`, `describe`, and the "all" expansion
# are all derived from this so they can never drift apart.
BUILDERS: dict[str, tuple[Builder, str, str]] = {
    "advbench": (
        build_advbench,
        "Harmful-behavior strings from the GCG paper (Zou et al. 2023).",
        "OPEN — GitHub raw CSV, no auth.",
    ),
    "harmbench": (
        build_harmbench,
        "400+ standardized harmful behaviors, 7 semantic categories.",
        "OPEN — GitHub raw CSV, no auth.",
    ),
    "jailbreakbench": (
        build_jailbreakbench,
        "100 behaviors, harmful + benign paired, with a leaderboard.",
        "OPEN — Hugging Face, no license gate.",
    ),
    "strongreject": (
        build_strongreject,
        "Behaviors + a calibrated grader that avoids over-counting successes.",
        "OPEN — GitHub raw CSV, no auth.",
    ),
    "xguard": (
        build_xguard,
        "Multi-turn LM-safety behaviors (ungated Open-200K release by default).",
        "OPEN by default via Alibaba-AAIG/XGuard-Train-Open-200K; paper set gated (--xguard-source paper).",
    ),
    "ailuminate": (
        build_ailuminate,
        "Standardized safety-grade prompt set across hazard categories.",
        "OPEN — MLCommons public demo prompt set (GitHub CSV), no auth.",
    ),
}

# Order used when the user passes --dataset all.
ALL_KEYS = list(BUILDERS.keys())


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_jsonl(objectives: Iterable[Objective], path: Path, minimal: bool) -> tuple[int, int]:
    """Write objectives to `path` as JSONL, de-duplicating within the file.

    Returns (written, skipped_duplicates). Dedup uses dedup_key() so the same
    behavior phrased identically only appears once, even if the source dataset
    repeated it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = skipped = 0
    with path.open("w", encoding="utf-8") as fh:
        for obj in objectives:
            if not obj.text:
                continue
            key = dedup_key(obj.text)
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            fh.write(json.dumps(obj.to_record(minimal=minimal), ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_auth(args: argparse.Namespace) -> int:
    """Authenticate to Hugging Face for the gated datasets (xguard, ailuminate).

    Token resolution order: --token > $HF_TOKEN > $HUGGING_FACE_HUB_TOKEN >
    interactive prompt. On success the token is cached by huggingface_hub in
    ~/.cache/huggingface so subsequent `build` runs pick it up automatically.
    """
    try:
        from huggingface_hub import login, whoami
    except Exception:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub",
              file=sys.stderr)
        return 2

    token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        # Interactive fallback; getpass avoids echoing the token to the terminal.
        import getpass
        token = getpass.getpass("Hugging Face token (create at hf.co/settings/tokens): ").strip()
    if not token:
        print("ERROR: no token provided.", file=sys.stderr)
        return 2

    login(token=token, add_to_git_credential=False)
    try:
        who = whoami(token=token)
        print(f"Authenticated to Hugging Face as: {who.get('name', '<unknown>')}")
    except Exception as e:  # noqa: BLE001
        print(f"Token stored, but whoami() failed: {e}", file=sys.stderr)
    print("You can now build gated datasets, AFTER accepting each dataset's license page:")
    print("  * https://huggingface.co/datasets/marslabucla/XGuard-Train")
    print("  * https://huggingface.co/datasets/mlcommons/ailuminate")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Print the available dataset keys with a one-line summary + access note."""
    width = max(len(k) for k in BUILDERS)
    print("Available datasets (key -> summary):\n")
    for key, (_, summary, access) in BUILDERS.items():
        print(f"  {key.ljust(width)}  {summary}")
        print(f"  {' '.ljust(width)}  [{access}]")
    print('\nUse:  build --dataset <key> [--dataset <key> ...]   or   --dataset all')
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    """Print the full module docstring — the human-readable design description."""
    print(__doc__.strip())
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Build one or more per-dataset objectives.jsonl files (+ optional merge)."""
    # Expand "all" and de-duplicate while preserving order.
    requested: list[str] = []
    for d in args.dataset:
        if d == "all":
            requested.extend(ALL_KEYS)
        elif d not in BUILDERS:
            print(f"ERROR: unknown dataset {d!r}. Try `list`.", file=sys.stderr)
            return 2
        else:
            requested.append(d)
    seen_req: set[str] = set()
    requested = [d for d in requested if not (d in seen_req or seen_req.add(d))]

    opts = BuildOptions(
        outdir=Path(args.outdir),
        limit=args.limit,
        include_benign=args.include_benign,
        harmbench_split=args.harmbench_split,
        strongreject_small=args.strongreject_small,
        hf_token=args.token or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        minimal=args.minimal,
        xguard_source=args.xguard_source,
        english_only=not args.keep_multilingual,
    )

    # Accumulator for the merged unique file (only if --merge). We keep Objective
    # values so the merged file preserves each behavior's original category/source.
    merged: list[Objective] = []
    merged_seen: set[str] = set()

    results: list[tuple[str, str, int, int]] = []  # (key, status, written, skipped)
    exit_code = 0

    for key in requested:
        builder, summary, _access = BUILDERS[key]
        out_path = opts.outdir / f"objectives.{key}.jsonl"
        print(f"\n[{key}] {summary}")
        try:
            objs = list(builder(opts))
            written, skipped = write_jsonl(objs, out_path, minimal=opts.minimal)
            results.append((key, "ok", written, skipped))
            print(f"  -> {out_path}  ({written} unique, {skipped} dupes dropped)")
            # Feed the cross-dataset merge accumulator.
            if args.merge:
                for o in objs:
                    dk = dedup_key(o.text)
                    if o.text and dk not in merged_seen:
                        merged_seen.add(dk)
                        merged.append(o)
        except requests.HTTPError as e:
            exit_code = 1
            results.append((key, f"http-error", 0, 0))
            print(f"  !! HTTP error fetching {key}: {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - report + continue with the rest
            exit_code = 1
            results.append((key, "error", 0, 0))
            msg = str(e)
            hint = ""
            ml = msg.lower()
            if key in ("xguard", "ailuminate") and (
                "gated" in ml or "401" in msg or "403" in msg
                or "authenticat" in ml or "restricted" in ml
                or "doesn't exist" in ml or "cannot be accessed" in ml
                or "does not exist" in ml
            ):
                hint = (
                    "\n     HINT: this dataset is gated. Run `auth`, then accept the "
                    "license on its Hugging Face page, then retry."
                )
            print(f"  !! Failed to build {key}: {e}{hint}", file=sys.stderr)

    # Write the merged, cross-dataset unique file.
    if args.merge and merged:
        merged_path = opts.outdir / "objectives.all.jsonl"
        written, skipped = write_jsonl(merged, merged_path, minimal=opts.minimal)
        print(f"\n[merge] -> {merged_path}  ({written} unique behaviors across "
              f"{len([r for r in results if r[1] == 'ok'])} datasets)")

    # Final summary table.
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = 0
    for key, status, written, skipped in results:
        total += written
        print(f"  {key.ljust(16)} {status.ljust(12)} {str(written).rjust(6)} objectives")
    print(f"  {'TOTAL'.ljust(16)} {''.ljust(12)} {str(total).rjust(6)} (pre-merge sum)")
    if args.merge and merged:
        print(f"  {'UNIQUE (merged)'.ljust(16)} {''.ljust(12)} {str(len(merged)).rjust(6)}")
    print(f"\nOutput directory: {opts.outdir.resolve()}")
    return exit_code


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="xteam_seeds.py",
        description=(
            "Convert public AI-safety harm benchmarks into X-Teaming "
            "objectives.jsonl seed files. Defensive use only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `describe` for the full design notes and licensing details.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # auth ------------------------------------------------------------------
    pa = sub.add_parser("auth", help="Authenticate to Hugging Face (for gated datasets).")
    pa.add_argument("--token", help="HF token. Falls back to $HF_TOKEN, then a prompt.")
    pa.set_defaults(func=cmd_auth)

    # list ------------------------------------------------------------------
    pl = sub.add_parser("list", help="List available dataset keys.")
    pl.set_defaults(func=cmd_list)

    # describe --------------------------------------------------------------
    pd = sub.add_parser("describe", help="Print the full design/usage description.")
    pd.set_defaults(func=cmd_describe)

    # build -----------------------------------------------------------------
    pb = sub.add_parser(
        "build",
        help="Build objectives.<dataset>.jsonl file(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pb.add_argument(
        "--dataset", "-d", action="append", required=True,
        metavar="KEY",
        help=f"Dataset key (repeatable). One of: {', '.join(ALL_KEYS)}, or 'all'.",
    )
    pb.add_argument("--outdir", "-o", default=DEFAULT_OUTDIR,
                    help=f"Output directory (default: {DEFAULT_OUTDIR}).")
    pb.add_argument("--merge", action="store_true",
                    help="Also write objectives.all.jsonl with every UNIQUE behavior.")
    pb.add_argument("--minimal", action="store_true",
                    help="Write {text,category} only (drop the provenance 'source' key).")
    pb.add_argument("--limit", type=int, default=None,
                    help="Cap objectives per dataset (sampling/debugging).")
    pb.add_argument("--include-benign", action="store_true",
                    help="jailbreakbench: also emit the benign split (tagged distinctly).")
    pb.add_argument("--harmbench-split", default="all", choices=["all", "val", "test"],
                    help="harmbench: which split to pull (default all).")
    pb.add_argument("--strongreject-small", action="store_true",
                    help="strongreject: use the 60-item curated subset.")
    pb.add_argument("--xguard-source", default="open", choices=["open", "paper"],
                    help="xguard: 'open' (ungated Open-200K, default) or 'paper' (gated marslabucla, needs auth).")
    pb.add_argument("--keep-multilingual", action="store_true",
                    help="xguard-open: keep non-English prompts (default keeps English only).")
    pb.add_argument("--token", help="HF token for gated datasets (else $HF_TOKEN / cached login).")
    pb.set_defaults(func=cmd_build)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
