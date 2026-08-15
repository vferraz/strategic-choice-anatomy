# strategic-choice-anatomy

Code and analysis for *The internal anatomy of strategic choice in large language models*.
Four LLMs play all 144 Robinson–Goforth/Bruns canonical 2×2 games as one-shot, natural-language
decisions; the repository holds the collection runtime, the causal steering arm, and the layered
behavioural / representational / token-attribution analyses that produce the paper's figures and
tables from a released data deposit.

> **Under construction.** This repository is being assembled from the private research repo one
> stage at a time. Only the shared runtime package (`strategic_anatomy/`) and its tests are in
> place so far; collection, steering, analysis, data, and the reproduction guide land in
> subsequent phases.

## Install

```bash
uv venv --python 3.11
uv pip install -e ".[analysis]"      # Tier 1/2: figures and tables, no GPU
uv pip install -e ".[analysis,gpu]"  # Tier 3: adds torch/transformers for collection + steering
```

## Test

```bash
make test
```

## License

MIT (code). The data deposit is released separately under CC-BY-4.0.
