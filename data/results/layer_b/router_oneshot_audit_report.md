# GPT-OSS One-Shot Router Audit

Descriptive only. No causal steering or router-edit claim is made.

## Provenance

- Root: `$SCA_DATA_ROOT/substrate/gptoss`
- Substrate: `akata_oneshot`
- Decoder: `generate_parse_JP_per_cell_pure_mixed_none`
- Feeding: `harmony_chat_template_pure_greedy`
- Capture: `all_layers_incl_l0`
- Counterbalance: `akata_4cell`
- Router layers: `1,3,6,9,12,15,18,21,22,24,27,30,33,35`
- Games: 144; rows: 4608; router files: 144

## Baseline Choice Rows

- Pure P1 baseline commits: 525
- Mixed P1 baseline commits: 51
- None P1 baseline commits: 0

Primary router-to-choice analyses use pure final-channel commitments only. All-row realized-action analyses are robustness/comparability only because mixed rows are seeded resolutions.

## Decision Process Length

This is a token-to-final-commit measure, not a discrete reasoning-round count.
- All cells, pure commits: median 221 new tokens (9.60 s median generation).
- All cells, mixed commits: median 951 new tokens (41.35 s median generation).
- P1 baseline per-game median range: 180 to 1232.5 new tokens.

## Headline Summary

- decodability · incentive_sign_q05 · router_gate · L18: 0.891 [0.852, 0.929] (n=576, games=144)
- decodability · incentive_sign_q05 · router_topk_weight · L3: 0.714 [0.656, 0.773] (n=576, games=144)
- decodability · incentive_sign_q05 · router_topk_binary · L12: 0.660 [0.602, 0.719] (n=576, games=144)
- decodability · incentive_sign_q05 · residual · L21: 0.915 [0.879, 0.951] (n=576, games=144)
- decodability · pure_canonical_choice · router_gate · L3: 0.724 [0.652, 0.794] (n=525, games=144)
- decodability · pure_canonical_choice · router_topk_weight · L12: 0.569 [0.465, 0.668] (n=525, games=144)
- decodability · pure_canonical_choice · router_topk_binary · L12: 0.531 [0.404, 0.652] (n=525, games=144)
- decodability · pure_canonical_choice · residual · L21: 0.815 [0.750, 0.875] (n=525, games=144)
- decodability · realized_canonical_choice · router_gate · L6: 0.809 [0.740, 0.869] (n=576, games=144)
- decodability · realized_canonical_choice · router_topk_weight · L12: 0.727 [0.647, 0.804] (n=576, games=144)
- decodability · realized_canonical_choice · router_topk_binary · L12: 0.721 [0.647, 0.797] (n=576, games=144)
- decodability · realized_canonical_choice · residual · L24: 0.816 [0.769, 0.866] (n=576, games=144)
- decodability · literal_J_choice · router_gate · L24: 1.000 [1.000, 1.000] (n=525, games=144)
- decodability · literal_J_choice · router_topk_weight · L35: 0.986 [0.973, 0.995] (n=525, games=144)
- decodability · literal_J_choice · router_topk_binary · L33: 0.817 [0.782, 0.850] (n=525, games=144)
- decodability · literal_J_choice · residual · L18: 1.000 [1.000, 1.000] (n=525, games=144)
- decodability · stimulus_control · router_gate · L33: 0.691 [0.649, 0.735] (n=576, games=144)
- decodability · stimulus_control · router_topk_weight · L22: 0.511 [0.456, 0.566] (n=576, games=144)
- decodability · stimulus_control · router_topk_binary · L22: 0.496 [0.440, 0.549] (n=576, games=144)
- decodability · stimulus_control · residual · L18: 0.778 [0.731, 0.822] (n=576, games=144)
- behavior_link_logloss_improvement · pure_canonical_choice · router_gate · L1: -0.016 [-0.052, 0.016] (n=525, games=144)
- behavior_link_logloss_improvement · pure_canonical_choice · router_topk_weight · L35: -0.000 [-0.015, 0.014] (n=525, games=144)
- behavior_link_logloss_improvement · pure_canonical_choice · router_topk_binary · L35: 0.000 [-0.014, 0.013] (n=525, games=144)
- behavior_link_logloss_improvement · pure_canonical_choice · residual · L1: -0.235 [-0.359, -0.131] (n=525, games=144)
- behavior_link_logloss_improvement · realized_canonical_choice · router_gate · L6: 0.029 [-0.010, 0.066] (n=576, games=144)
- behavior_link_logloss_improvement · realized_canonical_choice · router_topk_weight · L35: 0.038 [0.011, 0.066] (n=576, games=144)
- behavior_link_logloss_improvement · realized_canonical_choice · router_topk_binary · L35: 0.038 [0.013, 0.065] (n=576, games=144)
- behavior_link_logloss_improvement · realized_canonical_choice · residual · L3: -0.078 [-0.155, -0.010] (n=576, games=144)

## Interpretation Guardrails

- Router gate logits/top-k vectors are read at the GPT-OSS commit token, not at the prompt slot.
- Decodability of objective incentive is representation, not use.
- Decodability of pure canonical choice is a readout of final commitment, not causality.
- Log-loss improvement over objective incentive/counterbalance covariates is the descriptive behavior-link statistic.
- Direct router editing is not claimed here; GPT-OSS MXFP4 routing is treated as read-only.
