# Layer B↔C bridge — re-capture runbook (GPU)

**Question.** At the payoff tokens, where the Layer C lens reads ≈0 (deferred integration, Result 6),
can a free-direction probe still decode the incentive? If yes → the incentive is *represented while
reading* but *recruited onto the choice axis only at commit* (Layer B's late crystallization and Layer
C's deferred integration are the same mechanism). If no → integration itself is deferred. See the
"Synthesis" section of `LAYER_C_FINDINGS.md`.

**Why a re-capture.** The Layer C capture stored only the scalar lens projection (`score_canonical`),
not the residual *vectors*. The probe needs the vectors. This runbook captures them at the
payoff / answer / verb positions only (baseline condition), which is small.

## Where it runs

GPU box (Spark), env `.venv` (same as the Qwen lens runs). **Llama needs `--chat`.** GPT-OSS excluded
(as everywhere in Layer C). Requires the same manifests the lens capture used — present in-repo:
`analysis/block_b/tables/causal_oneshot/manifests/{oneshot_config.json, game_universe_oneshot.csv}`,
`datasets/processed/game_features.csv`, and the local model weights.

## Steps

```bash
# 0. CPU sanity (no model) — confirms imports, selected positions, rows/game
python analysis/layer_c/capture_residuals_bridge.py --model qwen_instruct --dry-run

# 1. GPU capture (per model). ~one short forward per game×cb; baseline only.
python analysis/layer_c/capture_residuals_bridge.py --model qwen          --load_8bit
python analysis/layer_c/capture_residuals_bridge.py --model qwen_instruct --load_8bit
python analysis/layer_c/capture_residuals_bridge.py --model llama31_instruct --load_8bit --chat
#   -> output/oneshot_layerc_residuals/{model}/{game}/{resid.npy, meta.parquet, config.json, _DONE}

# 2. Probe vs lens (CPU — can run on the GPU box, or pull the residuals back and run anywhere)
python analysis/layer_c/probe_bridge.py --model qwen
python analysis/layer_c/probe_bridge.py --model qwen_instruct
python analysis/layer_c/probe_bridge.py --model llama31_instruct
#   -> analysis/layer_c/tables/bridge_probe_vs_lens.csv
```

## Reading the result

Per (region, layer): `probe_auc` (free-direction probe on residual PCs, game-grouped CV) vs `lens_auc`
(AUC of `score_canonical`, the output-direction projection), both decoding the incentive sign.

| pattern at `own_payoff` / `opponent_payoff` | conclusion |
|---|---|
| **probe_auc ≫ lens_auc (~0.5)** | incentive **represented but not projected** at the payoff token → recruited only at commit. **B and C unified.** |
| probe_auc ≈ lens_auc ≈ 0.5 | integration itself (not just projection) is deferred to commit. |

`answer_prefix` should show both high (the decision is on the output axis there); `control` (the `win`
verb) is the negative reference.

## Notes / size

- The capture script mirrors `analysis/block_c/generate_oneshot_layerc.py::run_game` exactly (same
  prompt, region map, layers, forward) and just keeps the residual vectors at the selected positions.
  It is **untested on GPU from this session** — run the `--dry-run` first.
- The CPU probe (`probe_bridge.py`) **is verified** here on a synthetic dataset
  (`--make-synthetic`): payoff probe_auc ≈ 0.98 vs lens_auc ≈ 0.50, answer_prefix both 1.0, control
  both ≈ 0.5 — i.e. it correctly separates "represented" from "projected".
- Footprint ≈ selected positions (~15) × 4 cb × 144 games × 2 layers × hidden(8192) × 2 bytes ≈
  **~0.2 GB per model** (float16). Trivial to pull back for the probe step.
- Layers, regions and the control token are top-of-file constants in `capture_residuals_bridge.py`.
