# Layer B Recruitment Final

One-figure Layer B rebuild for the paper story: **Recruitment, Not Representation**.

This folder is additive. It does not edit or read panel tables from `analysis/layer_b`
or `analysis/layerB_v2`, and it does not modify collected data or paper files.

## Data Contract

- Primary root: `output/oneshot_akata_main_dl/`
- Models: `qwen_instruct`, `qwen`, `llama31_instruct`, `gptoss`
- Behavioural target: realised generated or committed P1 action, aligned to
  `canonical_action_p1`
- Primary incentive axis: q=0.5 canonical-signed `Delta1c`, matching the Layer A
  lambda convention
- Empirical-q `Delta1c` is robustness only and is not the spine of the figure
- Metadata joins are from `datasets/processed/game_features.csv` and
  `datasets/processed/taxonomy/human_game_master_per_canonical.csv`

All cross-game quantities use the canonical axis. No plotted table aggregates raw
`move == 0`, raw `decoded_action == 0`, or label-space A/B choices.

## Outputs

- `_data/`: lightweight panel caches
- `tables/`: every plotted value, CI, audit, and status row
- `figures/fig_layer_b/recruitment.{png,pdf}`: compact paper-ready figure

## Figure Panels

1. Minimal representation inventory: decision/readout, q=0.5 incentive sign,
   cue-axis identity, and one stimulus control.
2. Decision crystallization across depth, plotted as absolute held-out AUC in
   architecture-specific facets rather than cross-architecture gain.
3. Decision-incentive geometry across depth, plotted as absolute angle in
   architecture-specific facets.
4. Final recruitment bridge: final-layer geometry predicts the within-model
   P(canonical) slope over 144 games.
5. Causal slot: reads A5 steering results if present; otherwise writes an explicit
   pending/source-missing panel and table.
6. Disposition dissociation: cue-shift angle to the decision axis vs behavioural
   cue aim, with point size encoding behavioural movement. LDA cue identity is
   retained in tables but omitted from the axis because it is near ceiling.

## Commands

```bash
.venv/bin/python -m analysis.layer_b/recruitment.src.build_all
```

The builder regenerates the tables, panel caches, and figure from the corrected
integrated root.

## GPT-OSS Note

GPT-OSS residual activations in the corrected root are treated as commit-aligned
state evidence. Because GPT-OSS is MoE, residual-only nulls are not interpreted as
absence of strategic computation. Router evidence is tracked separately in
`tables/router_status.csv` when available.

## Current Causal Status

The planned A5 steering run is expected at:

`analysis/block_b/tables/causal_oneshot/runs/a5full_20260620_200509/`

If that folder is unavailable or incomplete, `tables/causal_status.csv` and panel E
say pending/source missing. This is intentional: the figure reserves the causal
slot without fabricating a causal claim.
