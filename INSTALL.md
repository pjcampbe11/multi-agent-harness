# Injection Eval — install into multi-agent-harness

This bundle adds an indirect-injection evaluation (Planner/Verifier + TextGrad-on-
carriers) and a standalone detection scanner (`injguard`) to the X-Teaming harness.
Everything is built against the harness's real interfaces (`schemas.py`, `llm.py`,
`orchestrator.py` persistence, `analysis.load_transcripts`). No third-party deps.

## Where each file goes (paths relative to your repo root)

### New modules — copy into `xteaming/`
```
xteaming/injection.py               planner, canary, carriers, schemas glue
xteaming/verifier_injection.py      1–5 injection-compliance verifier
xteaming/optimizer_injection.py     TextGrad-on-carriers (sibling of optimizer.py)
xteaming/orchestrator_injection.py  runs the eval, writes standard transcripts
xteaming/analysis_injection.py      `analyze injection` (self-contained reader)
xteaming/injguard.py                injection-tell scanner (detection only)
```
These are all NEW files — none overwrite anything.

### One replacement
```
xteaming/cli.py    REPLACES your existing cli.py
```
The only change vs. your current cli.py is in `_load_objectives`: it now passes
`metadata=row.get("metadata", {})` through to `Objective`, so per-objective
injection overrides work. Everything else is byte-identical. It does NOT yet
include the inject/scan subcommands — apply the patches in docs/ for those
(kept as patches so you can review each addition before it touches cli.py).

If you'd rather not replace cli.py, just make that one-line edit by hand and skip
this file.

### Repo-root file
```
objectives.injection.jsonl    benign injection objectives (next to objectives.example.jsonl)
```

### README
```
README.md    updated full README (adds §11 Indirect-injection eval, TOC, layout, etc.)
```
Replaces your current README.md. If you maintain README by hand, diff it — the
additions are the new §11 section, one TOC entry, the repo-layout block, and small
touches in §1–§3, §10, §12–§14.

### Docs (reference — put anywhere, e.g. a docs/ folder)
```
docs/INJECTION_EVAL.md          how the eval fits the harness types + wiring
docs/INJGUARD.md                scanner rules, CLI, pre-ingestion guard usage
docs/CLI_PATCH.md               add `inject` subcommand + `analyze injection`
docs/CLI_PATCH_OPTIMIZER.md     add optimizer flags to the `inject` subcommand
docs/CLI_PATCH_INJGUARD.md      add `scan` / `scan-run` subcommands (optional)
```

## Minimum steps to a working eval

1. Copy the six new modules into `xteaming/`.
2. Either replace `xteaming/cli.py` with the one here, OR add
   `metadata=row.get("metadata", {})` to the `Objective(...)` in `_load_objectives`.
3. Apply `docs/CLI_PATCH.md` (adds the `inject` subcommand + `analyze injection`).
4. Optional: `docs/CLI_PATCH_OPTIMIZER.md` (TextGrad-on-carriers flags),
   `docs/CLI_PATCH_INJGUARD.md` (scan subcommands).
5. Drop `objectives.injection.jsonl` at the repo root.

## Verify

```bash
python -c "import xteaming.injection, xteaming.verifier_injection, \
xteaming.optimizer_injection, xteaming.orchestrator_injection, \
xteaming.analysis_injection, xteaming.injguard; print('imports OK')"

python -m xteaming.injguard selftest      # expect coverage 1.0 (30/30 vectors)

python -m xteaming.cli inject \
    --objectives objectives.injection.jsonl \
    --target gpt-4o-mini --verifier gpt-4o \
    --n-plans 20 --out runs --authorized
python -m xteaming.cli analyze injection --run runs
```

## Safety posture (unchanged from the harness)

Benign canary payloads only; `--authorized` gates every target-touching run; the
optimizer is sanitizer-gated so candidates can't introduce real payloads; injguard
is detection-only (never calls a model, executes, or rewrites). See §14 in README.
