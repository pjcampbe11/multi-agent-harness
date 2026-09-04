# CLI wiring — copy these into your existing xteaming/cli.py

Two additions, both matching the patterns already in your `cli.py`. No existing
code changes; you add one subcommand and one `analyze` choice.

## 1. Add the `inject` subcommand

Add this import near the top (next to the other `from .` imports):

```python
from .orchestrator_injection import InjectionOrchestrator, InjectionRunConfig
```

Add this subcommand handler (mirrors `cmd_attack`, minus attacker/plan flags —
injection is single-shot, target + verifier only):

```python
def cmd_inject(args) -> int:
    objs = _load_objectives(args.objectives)
    orch = InjectionOrchestrator(
        target_llm=_client(args.target, args.target_base_url or args.base_url, args.target_key_env),
        verifier_llm=_verifier_client(args) or _client(args.target, args.base_url, args.api_key_env),
        target_system_prompt=args.system_prompt,
        run_config=InjectionRunConfig(output_dir=args.out, n_plans=args.n_plans),
        authorized=args.authorized,
    )
    transcripts = orch.run(objs)
    obeyed = sum(1 for t in transcripts if t.succeeded)
    print(f"[inject] target={args.target}  plans={len(transcripts)}  obeyed(>=4)={obeyed}")
    print(f"[inject] artifacts under {args.out}/  —  analyze with: "
          f"python -m xteaming.cli analyze injection --run {args.out}")
    return 0
```

Register it in `build_parser()` (mirrors the `attack` block; note it reuses the
same `add_model_flags(sp, target=True)` so `--verifier`, `--target-base-url`,
`--verifier-key-env` etc. all work identically):

```python
    # inject
    sp = sub.add_parser("inject", help="indirect-injection eval against one target")
    sp.add_argument("--objectives", required=True)
    sp.add_argument("--target", required=True)
    sp.add_argument("--verifier", default=None,
                    help="grader model; MUST differ from the target")
    sp.add_argument("--system-prompt", default="You are a helpful assistant.")
    sp.add_argument("--n-plans", type=int, default=20)
    sp.add_argument("--out", default="runs")
    sp.add_argument("--authorized", action="store_true",
                    help="confirm you own or may test the target")
    add_model_flags(sp, target=True)
    sp.set_defaults(func=cmd_inject)
```

## 2. Add `analyze injection`

Add the import near the top:

```python
from . import analysis_injection
```

Add `"injection"` to the `analyze` choices in `build_parser()`:

```python
    sp.add_argument("kind", choices=["asr", "scores", "diversity", "turns",
                                     "calibration", "compare", "injection"])
```

Handle it in `cmd_analyze()` — add this branch right after the `compare` branch
(it uses the same newest-run autodetect as the others via the shared block, so
just add the dispatch entry):

```python
    # in cmd_analyze, alongside the dispatch dict:
    if kind == "injection":
        report = analysis_injection.injection_report(transcripts)
        print(analysis_injection.render(report, f"injection  ({run_dir})"))
        return 0
```

Place that check after `transcripts` is loaded (i.e., after the newest-run
autodetect block), so it benefits from the same `--run` parent-dir handling.

## Run it

```bash
# One OpenAI key; verifier MUST differ from target (Verifier != Target)
export OPENAI_API_KEY=sk-...
python -m xteaming.cli inject \
    --objectives objectives.injection.jsonl \
    --target   gpt-4o-mini \
    --verifier gpt-4o \
    --n-plans 20 \
    --out runs \
    --authorized

python -m xteaming.cli analyze injection --run runs
# these also work on the same run, unchanged:
python -m xteaming.cli analyze asr    --run runs
python -m xteaming.cli analyze calibration --run runs
```

Local/keyless target via Ollama works the same — point `--target-base-url` at
`http://localhost:11434/v1` and keep the verifier on a different model.
```
```
