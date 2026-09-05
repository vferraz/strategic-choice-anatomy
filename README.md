# strategic-choice-anatomy

Code, analysis and released data for **_The internal anatomy of strategic choice in large language
models_**.

Vinicius Ferraz¹˒²˒³ · Leon Houf¹ · Enrico Ferrea⁴
¹ Institute of Management & KD2 Lab, Karlsruhe Institute of Technology (KIT) ·
² Singularity AI Research, Singularity.Inc · ³ BrainSpot Labs · ⁴ salestech Data & AI

<!-- arXiv: TBD · Paper DOI: TBD · Data DOI: TBD -->

---

Four large language models play **all 144 Robinson–Goforth/Bruns canonical 2×2 games** as one-shot,
natural-language decisions. For every decision we record the choice *and* the model's internal state
at the point where it actually commits — the residual stream at the decision slot for dense models,
the harmony final-channel commit for the reasoning MoE. Three analysis layers then ask whether
choosing like a strategic agent means computing like one: **Layer A** measures the behaviour,
**Layer B** asks whether the game's incentive structure is encoded and *recruited*, **Layer C**
attributes the signal to individual prompt tokens, and a **causal steering** arm intervenes on the
representation to see what moves.

This repository holds the collection runtime, the steering arm, all four analysis layers, and every
table behind the paper's figures. The raw substrate (~19 GB of activations) is released separately
as a data deposit.

## Install

```bash
uv venv --python 3.11
uv pip install -e ".[analysis,dev]"       # Tier 1 + 2: figures and tables, no GPU
uv pip install -e ".[analysis,gpu,dev]"   # Tier 3: adds torch/transformers for collection
```

See [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md) for the GPU and GPT-OSS environments.

## Reproducibility tiers

| Tier | What you get | What you need |
|---|---|---|
| **1** | Rebuild figures and verify numbers from the committed tables | this clone, ~2 min |
| **2** | Regenerate every table from the released substrate | the deposit (~21 GB), CPU only |
| **3** | Re-collect the substrate itself | GPU (DGX-Spark-class), days |

### Tier 1 — clone only

```bash
make test           # 52 tests
make figures-tier1  # 6 of the 11 paper figures, from committed tables alone
make verify-data    # every committed table against data/MANIFEST.json
```

**6 of the 11 paper figures rebuild from a clean clone.** Three more need the deposit; the
remaining two additionally need raw human data we are not permitted to redistribute. The exact
per-figure requirements are in [`REPRODUCING.md`](REPRODUCING.md) — we would rather state this
plainly than claim a round number.

### Tier 2 — with the deposit

```bash
make download-data              # or: export SCA_DATA_ROOT=/path/to/deposit
make figures                    # all 11 (bar the two needing raw human data)
make tables                     # regenerate every table family
make verify                     # hash-compare regenerated vs committed
```

### Tier 3 — re-collect

Exact model ids, prompts, counterbalancing, seeds and capture positions are in
[`docs/METHODS.md`](docs/METHODS.md); the launchers are in `scripts/launchers/`. Read
[`docs/METHODS.md` §8](docs/METHODS.md) first — four documented gotchas will silently change your
numbers otherwise.

## Layout

| Path | Contents |
|---|---|
| `strategic_anatomy/` | Installable runtime: prompts, games, equilibrium engine, steering hooks, path config |
| `collection/` | Substrate collection (dense + GPT-OSS harmony + Layer-C capture) |
| `steering/` | Direction fitting and the causal steering runners |
| `features/` | Stage-F builders: game features, human reference tables |
| `analysis/` | `layer_a/` behaviour · `layer_b/` representation · `layer_c/` token attribution · `steering/` causal |
| `data/` | Git-tracked: game metadata, derived human aggregates, run manifests, **every table a figure reads**, and the 11 reference figure PDFs |
| `data_heavy/` | Where the deposit unpacks (gitignored) |
| `scripts/` | `launchers/` Tier-3 chains · `setup/` env provisioning · `dev/` gates · `download_data.py` |
| `docs/` | Methods, data schemas, GPT-OSS handling, the steering pre-registration, results map |
| `tests/` | `make test` |

Run `make help` for all targets.

## Documentation

| Document | For |
|---|---|
| [`REPRODUCING.md`](REPRODUCING.md) | Per-figure and per-table commands, and exactly what each needs |
| [`docs/METHODS.md`](docs/METHODS.md) | Models, prompt, counterbalancing, capture positions, steering, **the reproduction gotchas, locked seeds, limitations and the 12 hard constraints** |
| [`docs/DATA.md`](docs/DATA.md) | Released roots, per-artifact schemas, which GPT-OSS root to use |
| [`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md) | Claim → table → producing script |
| [`docs/GPTOSS_LAYER_B_HANDLING.md`](docs/GPTOSS_LAYER_B_HANDLING.md) | What may and may not be claimed about the MoE model |
| [`docs/AMENDMENT_uniform_site_geometry.md`](docs/AMENDMENT_uniform_site_geometry.md) | Why there are two GPT-OSS roots, and which numbers were superseded |
| [`docs/STEER_SAMPLE_RULE.md`](docs/STEER_SAMPLE_RULE.md) | The pre-registered steering sample and precision stopping rule |
| [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md) | The three environments and their pins |

## Data

- **Committed** (`data/`, ~40 MB): every table behind a figure, the 144-game metadata, derived
  per-game human aggregates, run manifests, and `MANIFEST.json` with a sha256 for each.
- **Deposited** (~21 GB): the substrate, GPT-OSS uniform-site recapture, Layer-C token scores,
  bridge residuals, and the steering outputs. CC-BY-4.0.
- **Not redistributed**: the two source studies' participant-level human exports. They belong to
  their authors. `data/human_refs/raw/README.md` says what to obtain and from where, and
  `data/MANIFEST.json` records their sha256 so you can verify your own copy is the file we used.

## Citation

If you use this code or data, please cite the paper and the software release — see
[`CITATION.cff`](CITATION.cff).

## License

Code: **MIT** ([`LICENSE`](LICENSE)). Data deposit: **CC-BY-4.0**. The human reference data remains
under the terms of its original publications.
