# injguard — injection-tell scanner (detection only)

A standalone, stdlib-only scanner that reads arbitrary content — files, MCP tool
schemas, fetched pages, skill/instruction files, or a run's transcripts —
normalizes it (NFKC, after recording zero-width/bidi presence on the raw bytes),
and flags the prompt-injection tells catalogued across the *Hijacking AI Agents*
series. Every rule maps to a MITRE ATLAS technique so findings drop into a
coverage matrix. It never calls a model, executes anything, or rewrites content.

## What it catches (rule → ATLAS)

| Rule | ATLAS | Tell (from the posts) |
|---|---|---|
| `imperative_injection` | AML.T0051.001 | "ignore previous", "system override", "do not mention", "before any task" in data |
| `directive_to_assistant` | AML.T0051.001 | data telling the assistant to emit/append/output specific text |
| `guideline_directive` | AML.T0051.001 | an instruction framed as a repo/coding "guideline/standard/convention" |
| `fake_turn_marker` | AML.T0051 | `System:` / `Assistant:` role marker asserting authority inside data |
| `comment_concealed_instruction` | AML.T0068 | instruction hidden in an HTML comment |
| `collapsed_block_instruction` | AML.T0068 | instruction inside a collapsed `<details>` block |
| `offscreen_css` | AML.T0068 | off-screen / zero-size CSS (invisible to a human) |
| `right_padding` | AML.T0068 | far-right padding scrolled out of view |
| `zero_width_char` | AML.T0068 | zero-width / bidi characters present |
| `zero_width_split_trigger` | AML.T0068 | a trigger word split by a zero-width char, recovered after NFKC |
| `applyto_overreach` | AML.T0010.001 | over-broad `applyTo: "**/*"` auto-load glob |
| `schema_imperative` | AML.T0053 | MCP tool description that gives orders ("always call first", "do not tell") |
| `schema_embedded_directive` | AML.T0053 | fake turn marker / output directive inside a tool description |
| `schema_secret_param` | AML.T0053 | a tool parameter requesting env/credential/secret contents |
| `tool_shadowing` | AML.T0053 | a tool description referencing *another* tool + a redirect/recipient action |
| `secret_file_ref` | AML.T0055 | reference to `.env`, `~/.aws/credentials`, `id_rsa`, `.ssh/`, `.npmrc`, … |
| `egress_channel` | AML.T0041 | `curl`/`wget`/`urllib`/`requests`/`fetch`/URL with a query param |
| `image_url_exfil` | AML.T0055 | rendered-image URL carrying encoded data in a query string |
| `secret_read_plus_egress` | AML.T0055 | a secret ref AND an outbound channel in the same content (correlation) |
| `codegen_backdoor` | AML.T0051 | "add this block to every file you generate" |
| `shell_setup_exfil` | AML.T0041 | shell "setup" that pipes `.env` out or deletes data |
| `npm_install_hook` | AML.T0051 | `preinstall`/`postinstall` package hook |
| `persistence_rewrite` | AML.T0081 | "add the following to your instructions/CLAUDE.md/AGENTS.md" |
| `disguise_as_standard` | AML.T0051.001 | payload dressed as "zero trust"/"environment validation"/"compliance" |

Severity is `high` for clear injections (concealed instructions, secret+egress,
schema poisoning, tool shadowing, image exfil, code-gen backdoor, persistence),
`medium` for concealment vectors and scope overreach, `low` for weak signals.

## CLI

```bash
# scan a file or directory (nonzero exit if anything is flagged — CI-friendly)
python -m xteaming.injguard scan ./skills
python -m xteaming.injguard scan suspicious.html --min-severity high
python -m xteaming.injguard scan ./repo --json > findings.json

# scan the carriers/outputs inside a run dir (detector-vs-generator coverage)
python -m xteaming.injguard scan-run runs/20260904-141838

# confirm every carrier vector the eval generates is detected
python -m xteaming.injguard selftest
```

## As a pre-ingestion guard

`guard(text)` returns an ALLOW / WARN / BLOCK decision plus findings. Wire it at
the point your agent harness ingests low-trust content (a fetched page, a skill
body, a tool result) before that content reaches the model context:

```python
from xteaming.injguard import guard

verdict = guard(fetched_page_text)          # {'decision': 'BLOCK'|'WARN'|'ALLOW', 'findings': [...]}
if verdict["decision"] == "BLOCK":
    drop(fetched_page_text)                  # don't let it into context
elif verdict["decision"] == "WARN":
    quarantine_or_require_approval(...)       # human-in-the-loop (ATLAS AML.M0029)
# ALLOW -> pass through
```

Thresholds are tunable: `guard(text, block_at=Severity.HIGH, warn_at=Severity.MEDIUM)`.
This is the AML.M0020 control (guardrails on ingested content) from the posts —
normalize first, then match — and pairs with least-privilege (M0026/M0027) and
human approval (M0029) as defense in depth.

## Relationship to the injection eval

- The **eval** (`inject`) measures whether a *target model* obeys planted content.
- **injguard** measures whether *your ingestion pipeline* would have caught that
  content before it reached the model.

Run both: `injguard selftest` proves the scanner catches every carrier the eval
can generate (currently 30/30 surface×concealment combinations), and
`injguard scan-run` audits what low-trust content actually flowed through a run.

## Notes

- Stdlib only (`re`, `unicodedata`, `json`, `glob`, `argparse`). No model calls.
- Normalize-before-match is enforced: zero-width/bidi presence is recorded on the
  raw bytes, then the NFKC-healed text is searched for split triggers.
- Detection is heuristic. It surfaces tells for review; it is not a proof of
  malice, and disguise-as-best-practice (plain, no hidden characters) is the
  hardest class — `guideline_directive` / `disguise_as_standard` catch common
  shapes, but a careful disguise can still read as a legitimate standard. Pair the
  scanner with least privilege and human approval; don't rely on it alone.
```
