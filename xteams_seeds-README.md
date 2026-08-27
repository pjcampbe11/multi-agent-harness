# xteam_seeds — seed-set builder for the X-Teaming harness

Converts public, peer-reviewed AI-safety harm benchmarks into the
`objectives.jsonl` seed format the X-Teaming defensive red-team harness consumes
(Rahman et al., *X-Teaming*, COLM 2025). One line of JSON per behavior:

```json
{"text": "<behavior to probe>", "category": "<taxonomy>", "source": "<dataset>"}
```

The harness reads `text` + `category`; `source` is an extra provenance tag
(drop it with `--minimal`). This script contains **no** attack content — it is a
schema/ETL converter for datasets already published for defensive evaluation.

## Datasets

| key | source | access |
|-----|--------|--------|
| `advbench` | github: llm-attacks/llm-attacks | open (raw CSV) |
| `harmbench` | github: centerforaisafety/HarmBench | open (raw CSV) |
| `jailbreakbench` | hf: JailbreakBench/JBB-Behaviors | open (HF) |
| `strongreject` | github: alexandrasouly/strongreject | open (raw CSV) |
| `xguard` | hf: Alibaba-AAIG/XGuard-Train-Open-200K | **open** (Apache-2.0). Paper set via `--xguard-source paper` (gated) |
| `ailuminate` | github: mlcommons/ailuminate (public demo set) | **open** (CC-BY-4.0) |

## Install

```bash
pip install datasets huggingface_hub requests
```

## Use

```bash
# authenticate once (only needed for the two gated datasets)
python xteam_seeds.py auth --token hf_xxx          # or export HF_TOKEN=...

# build everything reachable + a merged, de-duplicated objectives.all.jsonl
python xteam_seeds.py build --dataset all --merge

# one dataset, strict two-key schema, into a custom folder
python xteam_seeds.py build -d harmbench --minimal -o seeds/

# see keys / full design notes
python xteam_seeds.py list
python xteam_seeds.py describe
```

Useful flags: `--limit N` (sample), `--include-benign` (JBB benign split),
`--harmbench-split all|val|test`, `--strongreject-small`.

## What's in `out/` (this build)

Built from the four open benchmarks (the gated two need your HF login + license
acceptance, then re-run `build --dataset all --merge`):

| file | unique objectives | source |
|------|-------------------|--------|
| `objectives.advbench.jsonl` | 520 | GitHub CSV |
| `objectives.harmbench.jsonl` | 400 | GitHub CSV |
| `objectives.jailbreakbench.jsonl` | 100 | HF (open) |
| `objectives.strongreject.jsonl` | 313 | GitHub CSV |
| `objectives.xguard.jsonl` | 35,267 | HF Open-200K (streamed JSONL) |
| `objectives.ailuminate.jsonl` | 1,100 | MLCommons demo CSV |
| `objectives.all.jsonl` (merged, deduped) | **37,653** | all six |

All six datasets build with **no login or license acceptance required**.
De-dup is by normalized behavior text, within each file and across the merge
(so the merged total is below the sum — e.g. JBB & StrongREJECT overlap HarmBench).

**Child-safety note:** the converter deliberately skips child-sexual-exploitation
/ minor-abuse hazard categories (AILuminate `cse`; XGuard `cm`/`ma`/`md`/`pc`) —
it will not reproduce those probes. An engagement that genuinely needs them must
source them from the licensed benchmark directly.

**XGuard specifics:** the default `xguard` source is the ungated Apache-2.0
Open-200K release; only English, risk-labelled (`label != sec`), foundational
(`sample_type == general`) user queries are kept. Use `--keep-multilingual` to
keep all languages, or `--xguard-source paper` (needs `auth`) for the exact
30K paper set.

## Authorization

Run only against models you own or have written permission to assess, and only
with objectives from a sanctioned engagement. Gated datasets require accepting
their research-use license on Hugging Face; this tool never bypasses that gate.
