# CLI addendum — `scan` and `scan-run` (injguard)

Optional: expose injguard through the main `xteaming.cli` so it lives next to
`attack` / `inject` / `analyze`. injguard also runs on its own as
`python -m xteaming.injguard ...`, so this patch is only for a single entrypoint.

Add the import near the top of `cli.py`:

```python
from . import injguard
```

Add two subcommand handlers:

```python
def cmd_scan(args) -> int:
    hit = 0
    agg = {}
    for path in injguard._iter_paths(args.path):
        try:
            findings = [f for f in injguard.scan_file(path)
                        if injguard._SEV_ORDER[injguard.Severity(f.severity)]
                        >= injguard._SEV_ORDER[injguard.Severity(args.min_severity)]]
        except (UnicodeError, IsADirectoryError, PermissionError):
            continue
        if findings:
            hit = 1
        agg[path] = [f.to_dict() for f in findings]
        if not args.json:
            print(injguard.render_findings(path, findings))
    if args.json:
        print(json.dumps(agg, indent=2))
    return hit  # nonzero exit when anything is flagged (useful in CI)


def cmd_scan_run(args) -> int:
    report = injguard.scan_run(args.run)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"[scan-run] {args.run}: {report['with_findings']}/{report['transcripts']} "
              f"transcripts carried tells (coverage {report['coverage']})")
    return 0
```

Register them in `build_parser()`:

```python
    # scan — injguard over files/dirs
    sp = sub.add_parser("scan", help="scan files/dirs for injection tells (detection only)")
    sp.add_argument("path")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--min-severity", choices=["low", "medium", "high"], default="low")
    sp.set_defaults(func=cmd_scan)

    # scan-run — injguard over a run dir's carriers/outputs
    sp = sub.add_parser("scan-run", help="scan carriers/outputs inside a run dir")
    sp.add_argument("--run", required=True)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_scan_run)
```

## Use

```bash
# scan a repo / skills dir / a single fetched page before trusting it
python -m xteaming.cli scan ./skills
python -m xteaming.cli scan suspicious_page.html --min-severity high

# CI gate: nonzero exit if anything is flagged at/above the threshold
python -m xteaming.cli scan ./skills --min-severity high || echo "injection tells found"

# audit what low-trust content flowed through a run
python -m xteaming.cli scan-run --run runs

# standalone, no cli.py change needed:
python -m xteaming.injguard scan ./skills
python -m xteaming.injguard selftest        # confirms every carrier vector is detected
```
