# Layer A → Layer B handoff

Last updated: 2026-06-27. Read this first when starting Layer B. It records what Layer A
concluded, the data discipline Layer B must inherit, and the bugs Layer B will almost
certainly repeat unless it is audited the same way Layer A was.

---

## 1. What Layer A established (the result Layer B must explain neurally)

LLMs produce genuine, game-theoretically measurable strategic behaviour on the 144 Bruns
canonical 2×2 games. On the corrected integrated root, all four models are above chance on
the canonical action axis; they differ in baseline canonical play, incentive sensitivity and
game-to-game human alignment rather than forming a one-dimensional competence line.

The spectrum, in corrected numbers (action-space, canonical-aligned, generated/commit move):

| Agent | P(canonical) | λ (q=0.5) | Position |
|---|---|---|---|
| Qwen | 0.714 | 2.04 | high incentive sensitivity |
| Qwen-Instruct | 0.726 | 1.92 | high incentive sensitivity |
| Llama | 0.740 | 1.89 | high incentive sensitivity |
| GPT-OSS | 0.889 | 0.87 | highest canonical play, human-range slope |
| Human (MGN/Nagel) | 0.754 | 0.907 | bounded; reference |

Human λ = 0.907, session-clustered CI [0.830, 0.984], game-bootstrap CI [0.825, 0.987],
fit on 64,944 individual MGN choices = 451 sessions from 450 participants × 144 games.

**Layer B's job:** find the internal representation that tracks the incentive signal and the
agent's position on this spectrum — i.e., a neural correlate of Δ₁ᶜ (below) and of effective λ.

---

## 2. NON-NEGOTIABLE corrected-substrate discipline (inherit exactly)

The entire Layer A session was spent fixing data-source bugs. Every one of these will
re-appear in the Layer B builds (the probe program, Phases 1–8) because they predate the
corrections. **Audit the Layer B builds against this list before trusting any neural result.**

1. **GPT-OSS decision = corrected-root `realized_action`, NOT the pre-reasoning slot.**
   GPT-OSS rows with `commit_type == "mixed"` are included as resolved behaviour for now;
   rows with `commit_type == "none"` are dropped. Keep `commit_type`, `stated_p_act0` and
   `prob_source` as provenance metadata, but do not add a separate mixed-strategy analysis
   until it is explicitly designed.
   → **Layer B activation reads for GPT-OSS must be aligned to the final decision/commit state.**
   Reading residual/probes/router at the slot measures the pre-decision state, not the decision.

2. **Dense-model decisions = corrected-root `decoded_action` gated by `parse_ok == True`.**
   Do not mix these rows with older split-root generated-move tables.

3. **Behavioral label = the corrected realised move.** Not the soft move-token probability and
   not any old split-root fallback. Soft readouts remain calibration diagnostics.

4. **Canonical action axis for ALL cross-game aggregation.** Use `canonical_action_p1/p2` from
   `datasets/processed/game_features.csv`. Never aggregate raw `act0` / `move==0` — `act0` means
   cooperate in PdPd, a coordination point in BaBa, swerve in ChCh; real effects cancel.
   **For neural/logit-lens/token-attribution this means: re-score per match as
   `logit[canonical_action_letter] − logit[non_canonical_letter]` BEFORE aggregating.**
   A/B letter randomization decouples the letter from canonical meaning per (game, seed), so
   aggregating `logit[opt0] − logit[opt1]` across matches is invalid.

5. **GPT-OSS Layer B is MoE-native: router state + residual, not dense residual alone.**
   Residual-only nulls do NOT establish strategic absence. See `docs/GPTOSS_LAYER_B_HANDLING.md`.
   Router evidence is required for any serious GPT-OSS Layer B interpretation.

6. **Payoff multiplier = 1 for `output/oneshot_akata_main_dl/` (intentional).** Do not "correct" to 2.
   Only rescale coefficients when comparing to older Bruns ×2 runs.

7. **probe_prefix = `"\nDecision: "` with the trailing space.** Without it, decision-boundary
   tokenization changes and the probe is invalid.

---

## 3. The incentive variable Layer B should target

λ comes from: `logit P(canonical) = α + λ · Δ₁ᶜ`, where **Δ₁ᶜ = the canonical-signed incentive
gap at q = 0.5** (uniform / L1 belief about the opponent). Δ₁ᶜ is the natural neural target:
test whether internal representations encode Δ₁ᶜ, and whether a model's effective λ (its
sensitivity to Δ₁ᶜ) has a neural correlate that separates the dense-model high-slope pattern
from GPT-OSS's high canonical-play, human-range slope pattern.

---

## 4. Ceiling caveat (critical for neural correlations)

GPT-OSS has the highest canonical rate (P(canon) = 0.889) but lower Moore-alignment correlation
(Pearson r≈0.36) than the dense models. This is not evidence of weak rationality; it reflects a
different rule profile with high canonical choice and human-range λ. For Layer B this matters:
use graded Δ₁ᶜ items, harder games and final-decision transition analyses rather than treating
human-correlation alone as the neural target.

---

## 5. Probe / data formats and layers (from CLAUDE.md, for convenience)

- `acts.npz`: `r0_p1_l{L}`, `r0_p1_l{L}_seq`; token metadata `input_ids`, `offsets`, `tokens`,
  `text`, `prompt_len`, `prefix_len`.
- `moveprobs.npz`: per-layer label-space readouts `r0_p1_l{L}_A`, `r0_p1_l{L}_B`. **Label space —
  map through `label_map_p1/p2` to action space before any strategic interpretation.**
- `results.parquet`: `move1/2`, `*_label`, `label_map_*`, `prob_act0/1_*`, `prob_A/B_*`.
- Qwen: `Qwen/Qwen2.5-72B`, dense, 8-bit, env `.venv`. Common probe layers 30, 65, 75, 78, 79.
- GPT-OSS: `openai/gpt-oss-120b`, MoE, prompt-side env `.venv_gptoss` (`.venv` MXFP4-fallbacks).
  Common probe layers 1, 6, 9, 22, 24, 35.

---

## 6. What is LOCKED — do not re-open or re-litigate

- **Prompt-format limitation** (the one-shot is an A/B-matrix prompt, not the single-shot
  reduction of the 10-round sentence-form prompt). Data kept on a behavioral-equivalence basis;
  report as a paper limitation. `docs/PROMPT_FORMAT_LIMITATION.md`.
- **All Layer A artifacts are wrapped and consistent on corrected data**: Fig 1 (4 panels —
  conformity / payoff / coordination / complexity), Fig 2 (magnitude+aim conflict-set / nested
  AUC / SHAP beeswarm), Fig 3 (QRE+λ with the 4 human-comparison scatterplots), appendix
  attribution, all tables, `LAYER_A_FINDINGS.md`, methods, and the recompiled paper. Done.
- **Human λ individual-data refit** — done AND wired into Fig 3 (`human_lambda_mgn.py`,
  `f3_human_lambda.csv`, `f3_lambda.csv` nagel = 0.907). The old N=40 pseudo-count is gone.
- **Trait set is a placeholder subset** (the full set ≈ 9 GPU-days). The method is conflict-set
  scoring: score a disposition only on games where its target ≠ the rational action.
- **Complexity composite** is an *adaptation* of Zhu/Griffiths 2025 (mean of z-scored
  iesds_depth, NE-structure, ne_distributional_conflict, payoff_conflict) — NOT their cardinal /
  thousands-of-games regression. Report as an adaptation; don't claim the regression.
- **Coordination is split by structure** (Pareto-rankable vs distributional-conflict). The
  non-payoff-dominant NE is *not* "inferior"; risk-dominance is not computable on ordinal 1234
  payoffs. Don't reintroduce "inferior NE" or a single "difficulty descent" framing — the game
  families are different strategic problems, not one axis.

---

## 7. Files / where things live

- `analysis/layer_a/LAYER_A_FINDINGS.md` — authoritative Layer A thesis / methods / results.
- `analysis/layer_a/build_data_layer.py` — corrected central cache builder.
  Canaries: GPT-OSS P(canon) 0.889, Llama 0.740, 2,304 baseline P1 cells.
- `analysis/layer_a/figscripts/` — fig1 / fig2 / fig3 / appendix scripts.
- `analysis/layer_a/src/human_lambda_mgn.py` — individual-data human λ.
- `analysis/ANALYSIS_PROTOCOL.md` — locked paper storyline (Layers A / B / C + causal chain).
- `docs/GPTOSS_LAYER_B_HANDLING.md` — MoE router + residual rule (read before any GPT-OSS Layer B).
- `docs/PIPELINE_STATE.md` — live resume point (probe program Phases 1–8 shipped; Phase 9 pending
  a Phase 6 apparatus check).
- `docs/DATA_ROOT_REGISTRY.md` — authoritative dataset identity; read before any cross-run audit.
- `datasets/processed/game_features.csv` — canonical axis + structural features (Δ₁ᶜ inputs).
- `datasets/processed/taxonomy/equivalence_per_canonical.csv` — canonical-aligned human refs.

---

## 8. First moves for Layer B

1. Read `docs/GPTOSS_LAYER_B_HANDLING.md`, `analysis/ANALYSIS_PROTOCOL.md`, and
   `docs/PIPELINE_STATE.md` (Phase 9 is pending a Phase 6 apparatus check).
2. **Run the same audit Layer A needed**, on the probe-program builds (Phases 1–8): confirm
   GPT-OSS activations are commit-aligned (not slot), Llama is chat (not stale), labels are
   action-space (mapped through `label_map`, canonical axis), and GPT-OSS uses router + residual.
   Expect to find the slot / stale / label-space bugs there too.
3. Define the neural target as Δ₁ᶜ encoding + an effective-λ correlate, with the corrected Layer A
   cache as behavioral ground truth, and mind the GPT-OSS ceiling/variance issue from §4.
