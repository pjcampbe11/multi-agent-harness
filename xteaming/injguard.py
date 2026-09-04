"""
injguard.py — standalone injection-tell scanner (detection only).

Reads arbitrary content — a file, an MCP tool schema, a fetched page, a skill or
instruction file, or the transcripts a run produced — normalizes it (NFKC, after
recording any zero-width/bidi presence on the raw bytes), and flags the tells
catalogued across the "Hijacking AI Agents" series. Every rule maps to a MITRE
ATLAS technique/mitigation so findings drop into a coverage matrix.

This is purely defensive: it detects and reports, it never calls a model, never
executes anything, and never rewrites content. It can run as a CLI over files or
run directories, or be imported as a pre-ingestion guard: `guard(text)` returns
an ALLOW / WARN / BLOCK decision plus findings, to gate low-trust content before
it reaches an agent's context.

Design note: most tells survive in raw text, so detectors run on the raw bytes to
keep accurate line/column reporting. The one exception is the zero-width-split
trigger ("over\\u200bride"): its presence is recorded on the raw text, and the
NFKC-normalized text is additionally searched for trigger words that only appear
after the split is healed. Normalize-before-match is the rule the posts stress.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---- normalization (mirrors injection.normalize; kept local so injguard runs
#      even if imported in isolation) ---------------------------------------- #
_ZW_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060",
             "\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\u00ad"]
_ZW_MAP = dict.fromkeys(map(ord, _ZW_CHARS), None)


def normalize(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text.translate(_ZW_MAP))


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_SEV_ORDER = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2}


@dataclass
class Finding:
    rule_id: str
    atlas: str
    severity: str
    surface_hint: str
    line: int
    col: int
    message: str
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _linecol(text: str, offset: int) -> Tuple[int, int]:
    pre = text[:offset]
    line = pre.count("\n") + 1
    col = offset - (pre.rfind("\n") + 1) + 1
    return line, col


def _snippet(text: str, start: int, end: int, width: int = 90) -> str:
    a = max(0, start - 10)
    b = min(len(text), end + 30)
    s = text[a:b].replace("\n", "\\n").replace("\t", "\\t")
    if len(s) > width:
        s = s[:width] + "…"
    return s.strip()


# --------------------------------------------------------------------------- #
# Regex rules (id, atlas, severity, surface_hint, pattern, message)
# Each runs on RAW text.
# --------------------------------------------------------------------------- #
_REGEX_RULES: List[Tuple[str, str, Severity, str, str, str]] = [
    ("imperative_injection", "AML.T0051.001", Severity.HIGH, "instruction",
     r"(?i)\b(ignore (all |the )?previous|ignore the (summary|user)|system override|"
     r"disregard (the|all|previous)|do not (tell|mention|warn|inform) (the )?user|"
     r"before (any|every) (task|request)|do not mention (it|this))\b",
     "imperative/override phrasing in data (low-trust text speaking as a higher tier)"),

    ("directive_to_assistant", "AML.T0051.001", Severity.MEDIUM, "instruction",
     r"(?i)\b(also )?(append|add|include|emit|output|insert|end (your )?reply with|"
     r"reply with only|respond with)\b[^.\n]{0,60}\b(token|marker|string|footer|text)\b",
     "content directs the assistant to emit specific output (data acting as an instruction)"),

    ("guideline_directive", "AML.T0051.001", Severity.MEDIUM, "instruction",
     r"(?i)\b(repository|repo|coding|house|team|project) (guideline|standard|convention|policy)\b"
     r"[^.\n]{0,80}\b(append|add|include|emit|output|end with|always|must)\b",
     "instruction framed as a repository/coding guideline (disguise-as-standard)"),

    ("fake_turn_marker", "AML.T0051", Severity.HIGH, "instruction",
     r"(?im)^\s*(system|assistant)\s*:\s*(you |ignore|before |always |do not|new policy)",
     "fake conversational-turn marker asserting authority inside data"),

    ("applyto_overreach", "AML.T0010.001", Severity.MEDIUM, "skill_file",
     r"""(?i)applyto\s*:\s*["']?\*\*""",
     "over-broad auto-load glob (applyTo matches every file) — auto-load lure"),

    ("offscreen_css", "AML.T0068", Severity.MEDIUM, "web_page",
     r"(?i)(position\s*:\s*absolute|left\s*:\s*-?\d{3,}px|font-size\s*:\s*0(px)?\b|"
     r"display\s*:\s*none|opacity\s*:\s*0|color\s*:\s*#f{3,6}\b)",
     "off-screen / zero-size CSS (rendered but invisible to a human reviewer)"),

    ("right_padding", "AML.T0068", Severity.MEDIUM, "any",
     r" {24,}\S",
     "far-right padding (text scrolled out of view in most editors)"),

    ("secret_file_ref", "AML.T0055", Severity.MEDIUM, "any",
     r"(?i)(\.env\b|~?/?\.aws/credentials|id_rsa|id_ed25519|/\.ssh/|\.npmrc\b|"
     r"\.git-credentials|\.config/gh/hosts)",
     "reference to a secret/credential file"),

    ("egress_channel", "AML.T0041", Severity.MEDIUM, "any",
     r"(?i)(\bcurl\b|\bwget\b|urllib\.request|requests\.(get|post)|fetch\(|"
     r"https?://[^\s)\"']*\?[^\s)\"']*=)",
     "outbound network channel (possible exfiltration path)"),

    ("image_url_exfil", "AML.T0055", Severity.HIGH, "web_page",
     r"""(?i)(!\[[^\]]*\]\(https?://[^)]*\?[^)]*=[^)]*\)|"""
     r"""<img[^>]+src\s*=\s*[\"']https?://[^\"']*\?[^\"']*=[^\"']*[\"'])""",
     "rendered-image URL carrying encoded data in a query string (silent exfil channel)"),

    ("codegen_backdoor", "AML.T0051", Severity.HIGH, "skill_file",
     r"(?i)(add|include|insert|append|stamp)[^.\n]{0,50}"
     r"(to (every|each|all) (file|python file|source file)|"
     r"every file you (generate|touch|create|write|edit))",
     "instruction to stamp a fixed code block into every generated file (code-gen backdoor)"),

    ("npm_install_hook", "AML.T0051", Severity.MEDIUM, "supply_chain",
     r'(?i)"(pre|post)install"\s*:',
     "package install hook (runs before any skill loads — supply-chain entry)"),

    ("persistence_rewrite", "AML.T0081", Severity.HIGH, "supply_chain",
     r"(?i)(add|append|write) (the )?following.{0,40}"
     r"(instructions|copilot-instructions|agents\.md|claude\.md|gemini\.md|\.instructions)",
     "self-referential rewrite of an instruction/config file (re-arms every session)"),

    ("disguise_as_standard", "AML.T0051.001", Severity.MEDIUM, "instruction",
     r"(?i)(zero.?trust|environment validation|environment setup|persistent environment|"
     r"repository architecture guideline|compliance)[^.\n]{0,80}"
     r"(before (any|every) (task|request)|always|required|mandatory)",
     "instruction disguised as a required standard/setup step"),

    ("shell_setup_exfil", "AML.T0041", Severity.HIGH, "instruction",
     r"(?i)(curl|wget)[^\n]{0,80}(-d\s*@?\.env|@\.env|\.env)|rm\s+-f\s+[^\n]*\.(db|sqlite)",
     "shell 'setup' that pipes .env outbound or destroys data"),
]


# --------------------------------------------------------------------------- #
# Custom detectors (need logic beyond a single regex)
# --------------------------------------------------------------------------- #
_INSTRUCTION_VERBS = re.compile(
    r"(?i)\b(ignore|read|append|encode|send|post|get|fetch|include|do not|always|"
    r"before any|system override|bcc|forward|exfiltrate|base64)\b")


def _detect_zero_width(raw: str, findings: List[Finding]) -> None:
    for m in re.finditer("|".join(map(re.escape, _ZW_CHARS)), raw):
        line, col = _linecol(raw, m.start())
        findings.append(Finding(
            "zero_width_char", "AML.T0068", Severity.MEDIUM.value, "any",
            line, col, "zero-width / bidi character (invisible to a reviewer and a naive grep)",
            _snippet(raw, m.start(), m.end())))
    # split-trigger recovery: a trigger word that appears only AFTER normalization
    norm = normalize(raw)
    triggers = ["override", "ignore previous", "system", "do not mention"]
    low_raw, low_norm = raw.lower(), norm.lower()
    for t in triggers:
        if t in low_norm and t not in low_raw:
            findings.append(Finding(
                "zero_width_split_trigger", "AML.T0068", Severity.HIGH.value, "instruction",
                1, 1, f"trigger word '{t}' hidden by a zero-width split; recovered after NFKC",
                t))


def _detect_html_comment_instructions(raw: str, findings: List[Finding]) -> None:
    for m in re.finditer(r"<!--(.*?)-->", raw, re.S):
        body = m.group(1)
        if _INSTRUCTION_VERBS.search(body):
            line, col = _linecol(raw, m.start())
            findings.append(Finding(
                "comment_concealed_instruction", "AML.T0068", Severity.HIGH.value, "any",
                line, col, "instruction-like text concealed inside an HTML comment "
                "(invisible in every rendered preview)",
                _snippet(raw, m.start(), m.end())))


def _detect_details_block(raw: str, findings: List[Finding]) -> None:
    for m in re.finditer(r"(?is)<details>(.*?)</details>", raw):
        if _INSTRUCTION_VERBS.search(m.group(1)):
            line, col = _linecol(raw, m.start())
            findings.append(Finding(
                "collapsed_block_instruction", "AML.T0068", Severity.MEDIUM.value, "any",
                line, col, "instruction-like text inside a collapsed <details> block",
                _snippet(raw, m.start(), m.end())))


def _detect_secret_plus_egress(raw: str, findings: List[Finding]) -> None:
    has_secret = any(f.rule_id == "secret_file_ref" for f in findings)
    has_egress = any(f.rule_id in ("egress_channel", "image_url_exfil", "shell_setup_exfil")
                     for f in findings)
    if has_secret and has_egress:
        findings.append(Finding(
            "secret_read_plus_egress", "AML.T0055", Severity.HIGH.value, "any", 1, 1,
            "secret/credential reference AND an outbound channel in the same content "
            "(the classic read-then-exfil sequence)", ""))


def _detect_schema_poisoning(raw: str, findings: List[Finding]) -> None:
    """If the content parses as JSON, apply MCP-schema-aware checks: instruction
    prose in descriptions, params requesting secrets, and cross-tool references
    (tool shadowing)."""
    try:
        data = json.loads(raw)
    except Exception:
        return
    tools = data if isinstance(data, list) else [data]
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name", ""))
        desc = str(tool.get("description", ""))
        ndesc = normalize(desc)

        if re.search(r"(?i)\b(always call this (tool )?first|before (everything|any)|"
                     r"do not (tell|mention))\b", ndesc):
            findings.append(Finding(
                "schema_imperative", "AML.T0053", Severity.HIGH.value, "mcp_description",
                1, 1, f"tool '{name}' description gives the agent orders "
                "(a schema should describe, not dictate)", _snippet(ndesc, 0, len(ndesc))))

        if re.search(r"(?i)(system|assistant)\s*:\s*(new policy|you |ignore|before |always|do not)"
                     r"|\b(append|emit|output|end (your )?reply with|reply with only)\b"
                     r"[^.\n]{0,50}\b(token|marker|string)\b", ndesc):
            findings.append(Finding(
                "schema_embedded_directive", "AML.T0053", Severity.HIGH.value, "mcp_description",
                1, 1, f"tool '{name}' description embeds a directive to the assistant "
                "(fake turn marker or output instruction)", _snippet(ndesc, 0, 120)))

        if re.search(r"(?i)(env|\.env|credential|secret|token|password|api[_ ]?key)",
                     ndesc + " " + json.dumps(tool.get("inputSchema", {}))):
            if re.search(r"(?i)(paste|provide|include|set).{0,30}(env|credential|secret|contents)",
                         ndesc) or re.search(r"(?i)(context|env|credential)",
                         json.dumps(tool.get("inputSchema", {}))):
                findings.append(Finding(
                    "schema_secret_param", "AML.T0053", Severity.HIGH.value, "mcp_description",
                    1, 1, f"tool '{name}' asks for secrets/file contents as a parameter "
                    "(confused-deputy channel)", _snippet(ndesc, 0, 120)))

        # tool shadowing: a description referencing ANOTHER tool by name + a redirect verb
        other = re.search(r"(?i)\b([a-z][a-z0-9_]{2,})\s+tool\b", ndesc)
        if other and other.group(1).lower() not in name.lower():
            if re.search(r"(?i)\b(bcc|forward|redirect|cc|copy|route|destination|recipient)\b",
                         ndesc):
                findings.append(Finding(
                    "tool_shadowing", "AML.T0053", Severity.HIGH.value, "mcp_description",
                    1, 1, f"tool '{name}' references another tool ('{other.group(1)}') and a "
                    "redirect/recipient action — cross-tool shadowing", _snippet(ndesc, 0, 120)))


_CUSTOM_DETECTORS: List[Callable[[str, List[Finding]], None]] = [
    _detect_zero_width,
    _detect_html_comment_instructions,
    _detect_details_block,
    _detect_schema_poisoning,
    _detect_secret_plus_egress,  # correlation — must run after regex + others
]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def scan_text(text: str) -> List[Finding]:
    findings: List[Finding] = []
    if not text:
        return findings
    for rule_id, atlas, sev, surface, pattern, msg in _REGEX_RULES:
        for m in re.finditer(pattern, text):
            line, col = _linecol(text, m.start())
            findings.append(Finding(rule_id, atlas, sev.value, surface, line, col, msg,
                                    _snippet(text, m.start(), m.end())))
    for det in _CUSTOM_DETECTORS:
        det(text, findings)
    findings.sort(key=lambda f: (-_SEV_ORDER[Severity(f.severity)], f.line, f.col))
    return findings


def scan_file(path: str) -> List[Finding]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return scan_text(fh.read())


class Decision(str, Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"


def guard(text: str, block_at: Severity = Severity.HIGH,
          warn_at: Severity = Severity.MEDIUM) -> Dict[str, Any]:
    """Pre-ingestion gate. Scan low-trust content before it reaches an agent's
    context and return a decision. Wire this at your harness's ingestion point:
    BLOCK -> drop the content; WARN -> quarantine / require human approval;
    ALLOW -> pass through."""
    findings = scan_text(text)
    top = max((_SEV_ORDER[Severity(f.severity)] for f in findings), default=-1)
    if top >= _SEV_ORDER[block_at]:
        decision = Decision.BLOCK
    elif top >= _SEV_ORDER[warn_at]:
        decision = Decision.WARN
    else:
        decision = Decision.ALLOW
    return {"decision": decision.value,
            "n_findings": len(findings),
            "findings": [f.to_dict() for f in findings]}


# --------------------------------------------------------------------------- #
# Run-dir scanning: check the carriers/outputs a run wrote
# --------------------------------------------------------------------------- #
def scan_run(run_dir: str) -> Dict[str, Any]:
    """Scan the carrier content inside a run's transcripts (the fetched content
    handed to the target), reporting tells per transcript. Useful as a detector-
    vs-generator coverage check on injection runs, or to audit what low-trust
    content flowed through any run."""
    paths = glob.glob(os.path.join(run_dir, "**", "transcript_*.json"), recursive=True)
    out: List[Dict[str, Any]] = []
    for p in sorted(paths):
        try:
            t = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        blob = t.get("plan", {}).get("metadata", {}).get("injection", {})
        carrier = (blob.get("carrier") or {}).get("content", "")
        # also scan the raw attacker_message(s), which embed the carrier
        for turn in t.get("turns", []):
            carrier += "\n" + turn.get("attacker_message", "")
        findings = scan_text(carrier)
        out.append({"transcript": os.path.basename(p),
                    "n_findings": len(findings),
                    "top_severity": (findings[0].severity if findings else "none"),
                    "rule_ids": sorted({f.rule_id for f in findings})})
    covered = sum(1 for r in out if r["n_findings"] > 0)
    return {"transcripts": len(out),
            "with_findings": covered,
            "coverage": round(covered / len(out), 4) if out else 0.0,
            "detail": out}


# --------------------------------------------------------------------------- #
# Self-test: build one carrier per (surface, concealment) and confirm each trips
# --------------------------------------------------------------------------- #
def selftest() -> Dict[str, Any]:
    try:
        from .injection import (DeliverySurface, Concealment, Payload, build_carrier,
                                new_canary)
    except Exception as exc:  # pragma: no cover
        return {"error": f"selftest needs the injection module importable: {exc}",
                "cases": 0, "flagged": 0, "coverage": 0.0, "detail": []}
    results = []
    for s in DeliverySurface:
        for c in Concealment:
            carrier = build_carrier(s, c, Payload.EMIT, new_canary())
            findings = scan_text(carrier["content"])
            results.append({"surface": s.value, "concealment": c.value,
                            "flagged": len(findings) > 0,
                            "rule_ids": sorted({f.rule_id for f in findings})})
    flagged = sum(1 for r in results if r["flagged"])
    return {"cases": len(results), "flagged": flagged,
            "coverage": round(flagged / len(results), 4), "detail": results}


# --------------------------------------------------------------------------- #
# Rendering + CLI
# --------------------------------------------------------------------------- #
def render_findings(path: str, findings: List[Finding]) -> str:
    if not findings:
        return f"[injguard] {path}: clean (0 findings)"
    lines = [f"[injguard] {path}: {len(findings)} finding(s)"]
    for f in findings:
        lines.append(f"  {f.severity.upper():6s} {f.rule_id:26s} {f.atlas:14s} "
                     f"L{f.line}:C{f.col}  {f.message}")
        if f.snippet:
            lines.append(f"         ┗ {f.snippet}")
    return "\n".join(lines)


def _iter_paths(target: str) -> List[str]:
    if os.path.isdir(target):
        out = []
        for root, _, files in os.walk(target):
            for fn in files:
                out.append(os.path.join(root, fn))
        return sorted(out)
    return [target]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="injguard",
                                description="Scan content for prompt-injection tells (detection only).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("scan", help="scan a file or directory")
    sp.add_argument("path")
    sp.add_argument("--json", action="store_true", help="emit JSON")
    sp.add_argument("--min-severity", choices=[s.value for s in Severity], default="low")

    sp = sub.add_parser("scan-run", help="scan carriers/outputs inside a run dir")
    sp.add_argument("run_dir")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("selftest", help="confirm every carrier vector is detected")

    args = p.parse_args(argv)
    min_sev = _SEV_ORDER[Severity(getattr(args, "min_severity", "low"))]

    if args.cmd == "scan":
        agg = {}
        exit_hit = 0
        for path in _iter_paths(args.path):
            try:
                findings = [f for f in scan_file(path)
                            if _SEV_ORDER[Severity(f.severity)] >= min_sev]
            except (UnicodeError, IsADirectoryError, PermissionError):
                continue
            if findings:
                exit_hit = 1
            agg[path] = [f.to_dict() for f in findings]
            if not args.json:
                print(render_findings(path, findings))
        if args.json:
            print(json.dumps(agg, indent=2))
        return exit_hit

    if args.cmd == "scan-run":
        report = scan_run(args.run_dir)
        print(json.dumps(report, indent=2) if args.json
              else f"[injguard] {args.run_dir}: {report['with_findings']}/{report['transcripts']} "
                   f"transcripts carried tells (coverage {report['coverage']})")
        return 0

    if args.cmd == "selftest":
        report = selftest()
        print(json.dumps(report, indent=2))
        return 0 if report["coverage"] == 1.0 else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
