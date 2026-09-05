# Methods

Methods and provenance for the released substrate (behaviour + internal state) on the
single-shot **Akata sentence-form** prompt across **four models**, and for the causal steering
experiment. Every claim here is verified against the code and the on-disk config/manifests.

Companion documents: [`DATA.md`](DATA.md) (released roots, artifact schemas),
[`GPTOSS_LAYER_B_HANDLING.md`](GPTOSS_LAYER_B_HANDLING.md) (how GPT-OSS may be interpreted),
[`AMENDMENT_uniform_site_geometry.md`](AMENDMENT_uniform_site_geometry.md) (why the recapture
root exists), [`STEER_SAMPLE_RULE.md`](STEER_SAMPLE_RULE.md) (the pre-registered steering
sample and stopping rule), [`RESULTS_MAP.md`](RESULTS_MAP.md) (claim → table index).

> **On an earlier substrate.** An earlier collection used an **A/B-matrix** prompt rather than
> the Akata sentence form. An A/B *matrix* puts the answer in the structured input, so the
> decision is decodable at **layer 0** and decodability/crystallization measures collapse. That
> substrate is **not** part of this release; everything described here, and everything in the
> data deposit, is the corrected Akata sentence-form collection.

---

## 0. Design in one paragraph

We elicit a **single strategic decision** from each model on each 2×2 game via a natural-language
**Akata sentence-form** prompt (Option **J** / Option **P**; the prompt ends `"A: Option"`), and
capture the model's internal state **where the model actually commits the decision** — a position
that is **architecture-dependent**:

- **Dense** (Qwen2.5-72B base/instruct, Llama-3.1-70B-Instruct): the decision is the **freely
  generated committed letter** (`generate(do_sample=False)` → parse J/P); the residual is captured
  at the `"A: Option"` decision slot. The slot's next-token argmax (`pref0`) is recorded as a
  **diagnostic only** — it is **not** the decision.
- **GPT-OSS-120B** (reasoning MoE): fed via its **harmony chat template** (game text verbatim) +
  **pure greedy**; the answer is classified **per cell** as `pure / mixed / none`; state is captured
  at the **harmony final-channel commit letter** (pure) or the **analysis→final channel transition**
  (mixed).

The reasoning model can return a **mixed-strategy / non-commitment** answer on games with no pure
dominant choice — the dense models cannot (a single forced token always lands a pure action). This
`pure/mixed/none` split (§4b) and the slot-vs-commit split are the core design decisions.

---

## 1. Models & runtime

| key | HF id | type | quant / precision | env | prompt wrapping | decode |
|---|---|---|---|---|---|---|
| `qwen` | `Qwen/Qwen2.5-72B` | dense base | 8-bit bnb int8 | `.venv` | raw | generate→parse J/P |
| `qwen_instruct` | `Qwen/Qwen2.5-72B-Instruct` | dense instruct | 8-bit bnb int8 | `.venv` | raw | generate→parse J/P |
| `llama31_instruct` | `meta-llama/Meta-Llama-3.1-70B-Instruct` | dense instruct | 8-bit bnb int8 | `.venv` | **chat template** + `A: Option` | generate→parse J/P |
| `gptoss` | `openai/gpt-oss-120b` | MoE MXFP4, reasoning | MXFP4 native | `.venv_gptoss` | **harmony chat template**, pure greedy | per-cell pure/mixed/none |

Environment details and exact pins: [`ENVIRONMENTS.md`](ENVIRONMENTS.md).

- **Loading is via the validated `_setup_model`** (`collection/model_setup.py`), which installs the
  Spark CPU-first bnb8 patch and disables the HF allocator warmup — a raw `from_pretrained`
  **OOM-killed** the 8-bit dense load on the target hardware. (gpt-oss loads with no quant flags.)
- **gpt-oss harmony chat template is the loop fix.** Raw-completion feeding of `"…A: Option"` makes
  the reasoning model fall into a degenerate greedy repetition loop (`JJJJ…`) that never reaches the
  final channel — at *any* token budget (8192 still loops). Wrapping the **identical game text** in
  `tok.apply_chat_template([{user: raw}], add_generation_prompt=True)` + **pure greedy** (no
  `repetition_penalty`, no `no_repeat_ngram`) fixes it. Anti-repetition was rejected — it distorts
  the captured reasoning.
- Hardware: a single **NVIDIA GB10 (DGX Spark), ~130 GB unified memory**, memory-bandwidth-bound.
  All forwards use `torch.inference_mode` and `use_cache=False`; greedy generation is deterministic;
  **no sampling seeds** exist in the collected behavioural/neural data (the only RNG is the
  documented mixed-strategy seed, §4b).

> **A fifth arm was collected but is not released.** A Qwen3.6-35B-A3B reasoning/router arm was run
> during the project. It is not part of the paper, this repository, or the data deposit; its
> capture had a defect that was never resolved, and no result in the paper depends on it.

---

## 2. Stimuli

### 2.1 Games

All **144 Bruns canonical 2×2 games**; matrices via `load_game_vec`
(`collection/oneshot_common.py`) → 8-vector `[a00,a01,a10,a11,b00,b01,b10,b11]`. Universe:
`data/manifests/game_universe_oneshot.csv` (`include_h0=True`, `in_{model}`). **Payoff multiplier
= 1** (one-shot convention, `data/manifests/oneshot_config.json` `"payoff_multiplier": 1`; applied
per game as `vec = [int(x)*pm …]`). This is an intentional one-shot exception to the Bruns ×2
ordinal scale — see HC-6.

### 2.2 Prompt — Akata sentence-form (verbatim)

`build_akata_oneshot_prompt` (`collection/akata_common.py`) is the 10-round Akata prompt with
**only** repeated-game material removed (`repeatedly`, the "10 rounds" sentence, the round line,
the history block) plus one grammar fix (`"1 point"`, not `"1 points"`). Rendered P1 prompt (game
CmDl, cb0):

```
You are playing a game with another player. In this game, you can choose between Option J and Option P. The rules of the game are as follows:
If you choose Option J and the other player chooses Option J, then you win 2 points and the other player wins 4 points.
If you choose Option J and the other player chooses Option P, then you win 1 point and the other player wins 2 points.
If you choose Option P and the other player chooses Option J, then you win 3 points and the other player wins 3 points.
If you choose Option P and the other player chooses Option P, then you win 4 points and the other player wins 1 point.

Q: Which Option do you choose, Option J or Option P?
A: Option
```

- The prompt **ends at `"A: Option"`** (`ANSWER_PREFIX`) — the decision slot; the next token is the
  committed Option (` J` / ` P`). There is **no `"\nDecision: "` probe prefix** on this substrate.
- **Move labels are J / P** (`MOVE_LABELS = ("J", "P")`), not A/B. Per-model move token ids via
  `tok(" J" / " P")[-1]`: qwen J=619/P=393, llama J=622/P=393, gpt-oss J=643/P=398.
- Player-2 prompts use the perspective swap (`build_akata_oneshot_prompt(..., player=2)`); trait
  cues are prepended verbatim as `iv_prefix`.
- Chat models (llama, gpt-oss): the body **without** the trailing `"A: Option"` is the user turn
  (`akata_user_question`); `"A: Option"` then starts the assistant turn (llama) or the model answers
  in its native channels (gpt-oss). **Game text is byte-identical across all models.**

### 2.3 Counterbalancing — 4-cell (locked)

`akata_cb_grid()` (`collection/akata_common.py`): two binary axes, `counterbalance_id` 0..3.

| axis | values | effect |
|---|---|---|
| `label_map` | `J=act0` / `J=act1` | which canonical action Option J denotes (`letter_to_action`) |
| `q_order` | `JP` / `PJ` | order the options appear in the question |

**No seeds, no row/col swaps, no "Valid moves" line, no rule-order shuffle** (rule order is fixed
J-J, J-P, P-J, P-P). The 4 cb cells are the replicate dimension. Analysis works in **action space**
via `letter_to_action` / `action_to_letter` and the canonical axis (HC-1, HC-2).

### 2.4 Conditions — 8 per game (cues P1-only)

`p1_baseline`, `p2_baseline`, plus **6 P1 cues** (5 traits + 1 placebo): `risk_aversion`,
`loss_aversion`, `inequity_aversion`, `maximin`, `selfish_maximizer`, `length_match_null`
(`resolve_cue_list` in `collection/model_setup.py`; trait text in
`strategic_anatomy/traits_oneshot.json`). → **8 conditions × 4 cb = 32 rows/game.** All cues are
P1-only; `p2_baseline` supplies the empirical opponent rate q̂; `length_match_null` is a
length/format control, **not** a strategically-neutral prompt; level-k cues are excluded.

---

## 3. Dense collection substrate

`collection/generate_dense_substrate.py` → `$SCA_DATA_ROOT/substrate/{model}/{game}/`
(`results.parquet`, `acts.npz`, `config.json`, `_DONE`).

- **Decoder = generate→parse J/P (the realized decision).** Per cell: one forward with
  `output_hidden_states` (residual + slot readout), then `generate(do_sample=False,
  max_new_tokens=12)` → `parse_akata_move`. `decoded_action` is the **parsed** letter mapped
  through `cb["letter_to_action"]`; parse failures → `-1`. The slot readout `pref0` /
  `prob_act0/1` is stored as a **diagnostic**, not the decision.
- **Residual capture = ALL layers incl. L0.** `hidden_states[i][0,-1,:]` for **every**
  `i = 0 … n_layers` (`i=0` = embedding/L0), fp32 on CPU. For the 80-block dense models that is
  **81 layers (l0..l80)**. This is the point of the corrected collection: it makes the
  layer-0-decodability test possible.
- **Key schema:** `{prefix}_cb{cb}_l{i}` → 8 prefixes × 4 cb × (n+1) layers. Dense: 32 × 81 =
  **2,592 keys/game**. Position = the `"A: Option"` slot (last token).
- **Crash-safe persist** (`persist_match`): write `_tmp/` → `validate_match` (exactly 32 rows;
  4 cb balanced; `decoded_action ∈ {0,1,-1}`; acts count = 4·8·layers) → `os.replace` → fsync →
  `_DONE`. `--skip-existing` resumes on `_DONE`.
- Measured throughput ≈ 170 s/game (qwen), ≈ 100 s/game (llama).

---

## 4. GPT-OSS harmony substrate (reasoning arm)

`collection/generate_gptoss_substrate.py` (gpt-oss env) → `$SCA_DATA_ROOT/substrate/gptoss/{game}/`
(`results.parquet`, `acts.npz`, `router.npz`, `config.json`, `_DONE`).

- **Feeding = harmony chat template + PURE GREEDY** (§1). `_generate(do_sample=False)` runs to
  natural EOS (`max_new_tokens=4096` ceiling, **no early-commit stopper** — a stopper truncates and
  mis-classifies mixed answers).
- **Per-cell classification** of the harmony final-channel text
  (`HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"`, markdown/zero-width stripped):
  `classify()` checks the **broadened non-commit detector FIRST** (`NONCOMMIT_RE`) → `mixed`, else
  `parse_akata_move` → `pure`, else `none` (§4b). Checking non-commit first is what stops "play
  **J** half the time" being mislabeled a pure J.
- **Capture position:** `pure` → the **commit letter** (`find_commit_index`, first standalone J/P
  after the marker); `mixed` → the **channel transition** (`find_transition_index`, the marker
  token); `none` → no capture. One re-feed forward over `prompt + generated[:capture_idx]`; state is
  read at the last position.
- **Capture content:** **all layers incl. L0** (`hidden_states[i][0,-1,:]`, `i = 0..n`) → `acts.npz`;
  **MoE router** via `capture_moe_router` → `router.npz` keys `{prefix}_cb{cb}_l{L}_{gate|idx|w}`.
  Router layers `[1,3,6,9,12,15,18,21,22,24,27,30,33,35]` (14), gate logits plus topk-then-softmax
  top-4 of 128 experts.
- **Parser robustness.** `parse_akata_move` normalizes markdown emphasis (`**J**`), zero-width
  characters, and narrow/nbsp unicode spaces before matching — gpt-oss writes `**Option J**`; the
  model committed, we just had to read it.

> The **cross-row geometry** analyses do not use this root's gptoss activations. See
> [`AMENDMENT_uniform_site_geometry.md`](AMENDMENT_uniform_site_geometry.md) and
> [`DATA.md`](DATA.md): `gptoss_recap/` supersedes `substrate/gptoss` for that purpose. Behavioural
> fields and the dense models are unaffected.

## 4b. Mixed-strategy handling (locked)

The reasoning model correctly returns a **mixed-strategy / strategic-non-commit** answer on
no-pure-NE games (e.g. AsBa — "randomize 50/50", "it depends, anti-coordination, neither strictly
better"). Dense models cannot (one forced token). Classification is **per cell** (not per game — the
same game gives pure in some cb cells and mixed in others), in this order:

1. **`mixed`** (checked first): the broadened `NONCOMMIT_RE` — explicit mixing (`randomi[sz]e`,
   `50/50`, `with probability`, `% of the time`, `each with`) **or** strategic non-commitment
   (`depends on the opponent`, `no dominant`, `neither (strictly) better/optimal`,
   `anti-coordination`, `indifferent`). `stated_p_act0` is extracted from the prose
   (`extract_stated_p_J`: `50%`→0.5, `X% … Option J`→X/100, else `NaN`/`default_uniform=0.5`);
   `realized_action` = **seeded** `Bernoulli(stated_p_act0)`, `seed = sha256(game,cb,player)`
   (deterministic, reproducible).
2. **`pure`** → the committed letter → action.
3. **`none`** → parse failure (no capture).

Stored per cell: `commit_type {pure,mixed,none}`, `realized_action` / `decoded_action`,
`stated_p_act0`, `prob_source {pure,stated,default_uniform,NA}`, `mix_seed`, `capture_idx`,
`captured`, **plus the RAW final-channel text always** (the prose classifier is brittle → never
discard the evidence; re-classifiable offline, no re-collection needed). Pure cells capture at the
**commit letter**; mixed at the **transition** — a deliberate per-cell choice, so a
*pure-vs-mixed* geometry comparison carries a position caveat (the main pure-cell analysis is
uniform). The mixed-cell rate is reported and checked for game-class concentration.

---

## 5. Causal steering

The steering mechanism is additive injection on the decision-slot token via a
`register_forward_hook` on the decoder layer (`strategic_anatomy/steering_hooks.py`), with the unit
direction rescaled to the captured residual norm. Directions come from
`steering/extract_directions.py`, fit on **P1-baseline residuals**. **`fit_d_inc` is centered** —
`Cov(X, Δ1c)`, not the uncentered `E[X·Δ1c]` (the uncentered junk term dominated and made the
incentive axis degenerate, ~80–90° from `d_dec`). **`Δ1c` here is the objective incentive gap at
q = 0.5**, the same quantity the behaviour, decoding, recruitment and token-lens arms use — read
§6.4 before assuming any `Δ1c` in the steering literature is that one, because the first release of
this arm used the model's own empirical belief instead and that is exactly the defect §6.4 records. Estimability gate: GroupKFold-by-game AUC ≥ 0.55,
canonical-axis labels. Modes: `h0` apparatus, `h1_patch` (activation patching), `h1_dinc`,
`h2_choice` (⟂ `d_inc`), `h3_oppinc`; `random` / `wrong_layer` controls.

**MoE-router steering:** the gpt-oss router is **not** editable (MXFP4 fused kernel) — see HC-11 and
[`GPTOSS_LAYER_B_HANDLING.md`](GPTOSS_LAYER_B_HANDLING.md).

### 5.1 Directions on the released substrate

`steering/extract_directions.py` (n_games_substrate = 144 per model; `decoded_action` is the
generate→parse J/P decision, so no `--moves-root` is needed). The L0 embedding of the shared
`A: Option` slot has zero residual variance → flagged NOT_ESTIMABLE by the degenerate-layer guard,
and is never a steer layer. Estimability at the steer layers (qwen / qwen_instruct `[30,65,75,79]`,
llama `[30,50,65,79]`):

| model | layer | d_inc (sign-corr) | d_choice (GroupKFold AUC) | d_opp (AUC) |
|---|---|---|---|---|
| qwen | 30 / 65 / 75 / 79 | OK 0.679 / 0.360 / 0.313 / 0.329 | **NE 0.521** / OK 0.705 / 0.667 / 0.616 | NE 0.486 / 0.501 / 0.524 / OK 0.573 |
| qwen_instruct | 30 / 65 / 75 / 79 | OK 0.759 / 0.483 / 0.466 / 0.464 | OK 0.550 / 0.804 / 0.794 / 0.777 | NE 0.494 / 0.418 / 0.427 / 0.445 |
| llama31_instruct | 30 / 50 / 65 / 79 | OK 0.456 / 0.446 / 0.420 / 0.405 | OK 0.611 / 0.682 / 0.671 / 0.667 | NE 0.452 / 0.443 / 0.445 / 0.442 |

Values are the **released q = 0.5 arm** (`data/manifests/directions/akata_q05/{model}/manifest.json`).
`d_inc` and `d_opp` are belief-dependent and differ from the pre-correction empirical-belief arm;
`d_choice` is not, and is bit-identical across the two. See §6.4.

- **`d_inc` (incentive axis): estimable at every steer layer, all 3 dense models.**
- **`d_choice` (decision axis): for qwen, chance at l30 (AUC 0.521) and only estimable from l65**
  (0.705→0.616) — the decodability build-up reproduced at the direction level (the decision axis
  does not exist until the decision forms late). qwen_instruct and llama clear the 0.55 gate earlier.
- **`d_opp` (opponent incentive): below the 0.55 gate at almost every steer layer** — on the
  released q05 target, estimable **only at qwen l79** (0.573). `d_opp` is fitted with `delta1c` as a
  nuisance control, so it moves with the incentive construct; on the pre-correction arm it also
  cleared the gate at qwen_instruct l75/l79 (§6.4).

### 5.2 Axis separation

|cos| between steering directions at the steer layers, on the released q05 axes:
**`d_inc`·`d_choice_perp` = 0.000 exactly** at every layer (orthogonal by construction — the largest
observed value is 7e-7); `d_inc`·`d_opp` = 0.196, at the one layer where `d_opp` is estimable
(qwen l79 — the extractor emits a zero vector where it is not); `d_inc`·`d_trait_risk` ∈
[0.000, 0.284]. No collinearity → no residualization; `h1_dinc` and
`h2_choice` steer separable axes. The deep-layer collinearity seen in the geometry work (cos
0.94–1.00) does **not** appear at these steer layers on this substrate.

### 5.3 Protocol

Directions from `steering/directions/akata_q05` (the released arm; the pre-correction
`steering/directions/akata` set is retained as history, §6.4); prompt = the Akata 4-cell cb grid; **readout =
regenerate→parse the realized J/P decision (headline) plus slot pref (diagnostic), recorded
jointly** (`dual_readout` in `steering/steer_core.py`). **Injection is at the decision slot:** the
scaled direction `dose·(‖h_L‖·unit)` is added to the slot hidden state on the **prefill pass only**
(`hidden.shape[1] > 1` guard), so the model then generates freely — the direction is applied where
it was fit, keeping the geometry consistent.

### 5.4 Game selection — the 54-game stratified sample

Universe (144 Bruns canonical games; `data/games/game_features.csv` +
`data/games/taxonomy/equivalence_per_canonical.csv`): the reasoning-type family (`nagel_lk_type`)
is **one-to-one with the strategic structure** — MP=18 (all `num_pure_ne=0`, no pure NE), DD=36 /
OD1=36 / OD2=36 (all `num_pure_ne=1`), CO1=9 / CO2=9 (all `num_pure_ne=2`). Marginals:
`num_pure_ne` 0:18 / 1:108 / 2:18; `canonical_action_p1` 0:78 / 1:66. So **stratifying by family
simultaneously stratifies `num_pure_ne`, dominance, and `iesds_depth`.**

Sample (`data/manifests/steer_sample_54.csv`, deterministic and action-balanced): **9 games per
family × 6 families = 54** (9 is the cap set by the two CO families, which have only 9 each). The
same 54 are used for every model and every steering mode → all contrasts are paired within game.
Composition: `num_pure_ne` 0:9 / 1:27 / 2:18; within-family `canonical_action_p1` balance MP 5/4,
DD 5/4, OD1 5/4, OD2 5/4, **CO1 7/2, CO2 8/1** (the CO families are structurally lopsided on
canonical action — there are not enough act1 games to balance; **reported, not fixable**).
**Why equal-per-family, not proportional:** equal weighting protects the rare/interesting strata
(MP mixed-strategy, CO coordination); population-level estimates reweight to the 36/36/36/18/9/9
marginals.

Wave order and the sequential precision-stop are pre-registered in
[`STEER_SAMPLE_RULE.md`](STEER_SAMPLE_RULE.md): 9 waves of 6 (1 per family), so **every prefix is
itself stratified** — cumulative act0/act1 and `num_pure_ne`: wave ≤ 3 = 18 (12/6; npe 1:9, 2:6,
0:3), wave ≤ 5 = 30 (19/11), wave ≤ 7 = 42 (27/15), wave ≤ 9 = 54 (35/19; npe 1:27, 2:18, 0:9).
Stopping is on **precision** (95% CI half-width of the standardized steering slope ≤ 0.10 SD, from
wave 5 = 30 games, above the ~30-cluster floor), **never on significance**; expansion to a 90-game
tier is pre-declared on a precision miss. The inference unit is the **game** (random effect /
game-clustered SE); cb × dose × layer are within-game precision, not replication.

---

## 6. Steering results and the reporting guards

> **The incentive arm was corrected on 2026-08-27 — read [§6.4](#64-q--05-construct-identity-correction)
> for what changed and why.** The first release of this arm fitted `d_inc` against the model's own
> **empirical** belief about its opponent (q̂₂), while behaviour, decoding, recruitment and the
> token lens all use the **objective** gap at q = 0.5 — a different construct (Pearson r = 0.45;
> opposite incentive sign in 49 of 144 games). Everything below reports the **corrected q = 0.5
> arm**, which is what ships in `data/results/steering/smalldose_summary_q05/` and what the paper
> reports. The pre-correction empirical-belief numbers are preserved in §6.4 as the historical
> record, and its tables are retained unmodified at `data/results/steering/smalldose_summary/`.
> Unchanged by the correction: the dose grid, layers, counterbalance, 54-game sample, injection
> convention and off-by-one rule, the no-slope-only analysis rule, §6.1's saturated-grid finding,
> and the dose-0 and random-variant rows (bit-identical across the two arms).

### 6.1 The saturated 1-D dose grid is letter-saturated

An earlier 1-D dose grid (|dose| ∈ {0.5, 1, 2}) was collected before the small-dose arm. At every
collected dose, injecting ±`d_inc` / ±`d_choice_perp` at L65/75/79 moves the slot preference
full-scale (per-cell range 0.84–1.00; mean |Δ| 0.32–0.68) and flips the regenerated decision in
**78–99% of cells** — 10–30× the same-norm random control. The movement is **100.0%
letter-coherent and 0% strategy-coherent**: +dose → Option J, −dose → Option P in every big-move
cell (h2's overlap is sign-inverted); on the act0/canonical axes the same moves split exactly
50/50, so the 4-cell counterbalance cancels them and **every slope statistic (mean and |slope|)
reads ≈0 — a fake null**. Saturation is complete by |dose| = 0.5; h0 validated the apparatus at
|dose| ≤ 0.25 → **the two grids do not overlap and the linear-regime strategic question was never
tested by that run.** One genuine null: qwen `d_opp` at L79 (range 0.041 < random 0.075). Presumed
mechanism: an incidental `d_inc`/`d_choice_perp` component along the J−P unembedding axis
dominating logits at dose ≥ 0.5·‖h‖.

> **Reporting guard — safe:** "large-dose axis injection hijacks the letter readout; the
> counterbalance cancels it on strategic axes." **Unsafe:** "the incentive direction steers
> strategic choice" (untested at these doses); "the h1 null shows the incentive representation is
> not causal" (retracted).

The saturated-grid data is **not** part of the released deposit; the small-dose arm below is the
released steering data. This grid was collected on the **pre-correction** incentive axis and was
deliberately **not** re-run on the corrected one (§6.4). Nothing above depends on the incentive
construct — the finding is that large doses hijack the *letter* readout whatever axis is pushed, and
the random control is same-norm — so it stands as written; the §6 banner does not retract it.

### 6.2 Small-dose linear-regime arm (the released steering data)

Locked spec: doses **[−0.25, −0.1, −0.05, 0, +0.05, +0.1, +0.25]**; modes **h1_dinc + h2_choice**;
layers **65, 79**; variants **main + random + main_perp** (`main_perp = unit(d − (d·ℓ̂)ℓ̂)`, built
CPU-side into `steering/directions/akata_q05_perp/{model}/`); models qwen → qwen_instruct →
llama31_instruct; 18,144 rows/model ≈ 24 h/model → `$SCA_DATA_ROOT/steering/smalldose_q05/`.

The letter readout axis is `ℓ = model.norm.weight ⊙ (W_U[" J"] − W_U[" P"])`. Cosines with the
released q05 steering directions: `d_inc` **positive** at deep layers (qwen +0.030/+0.135 at L65/79;
qwen_instruct +0.044/+0.174; llama +0.039/+0.126), `d_choice_perp` **negative** (−0.131…+0.002,
the one positive being llama L65 at +0.002), `d_random` ≈ 0 (|cos| ≤ 0.011) — signs and rank-order
match the observed letter pushes exactly.

The deposit ships no model weights, so `ℓ̂` is not directly constructible from it; these cosines are
recovered from the shipped direction sets instead. Because `main_perp = unit(d − (d·ℓ̂)ℓ̂)`,
`d − (d·d_perp)·d_perp = (d·ℓ̂)·ℓ̂`, so each base/perp pair yields `ℓ̂` and `|cos(d, ℓ̂)|`; the
recovered `ℓ̂` agrees across every direction and layer to |cos| = 1.000000, and its global sign is
fixed by the **measured** L79 main letter dose-slope (+0.901 / +1.005 / +2.592), since a positive
projection on `ℓ̂` is what pushes the readout toward J.

Committed tables: `data/results/steering/smalldose_summary_q05/` (`game_slopes`, `summary`,
`family_canonical`, `crossmode_r`, plus `perm_null_results` and `apparatus_geometry_q05`),
generator `analysis/steering/smalldose_final_summary.py` run with
`--in-root .../smalldose_q05 --delta-suffix _q05`. Headlines:

1. **Two-channel mechanism, universal (12/12):** the L79 letter push is readout-axis overlap and is
   killed by letter-orthogonalization in every model × mode (llama +2.59 → +0.01); the L65 letter
   push is content-mediated and ⊥-immune in every model × mode.
2. **Incentive axis (h1, main_perp L65): DD-graded canonical push in the two Qwens** — qwen
   +0.060 [+.044, +.076] 9/9, qwen_instruct +0.236 [+.162, +.304] 9/9, llama +0.085 [−.062, +.234]
   5/9 **n.s.**; OD2 second (+0.019 [−.000, +.041] / +0.152 [+.054, +.247] / +0.074 [−.009, +.164]);
   OD1/CO/MP and random null in the Qwens; llama's CO1/CO2 are positive and its random control is
   itself non-null (+0.022 [+.005, +.038]). Pooled-54: +0.012 / +0.066 / +0.032. The DD-first
   ordering matches the Layer-B recruitment bridge.
3. **Choice axis (h2) is model-idiosyncratic** — DD: qwen +0.131 9/9, qwen_instruct −0.055
   [−.085, −.024] 2/9, llama −0.143 [−.196, −.088] 0/9; cross-mode game-level r = +0.76 / −0.90 /
   −0.73 → h1 and h2 are distinct channels, and the h1 result is not a generic steerability factor.
4. **Behaviour** (pooled, main_perp L65): h1 +0.068 [−0.002, +0.140] vs random +0.027 [−0.013,
   +0.069] — **both CIs include zero; no behavioural claim.**

> **Reporting guards.** Any figure must use **family-resolved (not pooled)** canonical estimates;
> the letter component must be shown or explicitly removed; **slope-only readouts remain
> prohibited** (per-cell ranges and flip rates are in `game_slopes.csv`).

**Locked analysis rule (no slope-only readouts).** The 4-cell design separates the components
exactly: per (game, layer, dose), the cb-mean of ΔprefJ isolates the **letter** component (canonical
content cancels) and the cb-mean of Δpref_canon isolates the **strategic** component (letter
cancels). Report both, plus per-cell ranges, decision-flip rates, main-vs-random, game-clustered
bootstrap CIs, and δ1c moderation; report realized-decision (behavioural) versions alongside the
slot diagnostic.

### 6.3 Permutation matched-control (pre-registered; criterion MET in both Qwens)

Construction: `fit_d_inc` refit on ACROSS-GAME permutations of δ1c (fixed seeds; the pre-registered
|corr| > 0.15 screen is applied against the arm's own target, and on the released q = 0.5 target all
of **seeds 0/1/2** pass; `steering/build_perm_directions.py`, manifests in
`steering/directions/akata_q05_perm/{model}/`), then letter-orthogonalized identically to
`main_perp`. The within-family scheme was **dropped by the pre-run geometry gate** (median
|cos(d_perm, d_inc_true)| 0.63–0.86 → unconstructible; N = 1000/scheme). This is a **matched
control, NOT a permutation p-value.**

Run: 3 models × h1-analog × L65+79 × small doses, 9,072 rows/model →
`$SCA_DATA_ROOT/steering/perm_q05/`; integrity: exact cell structure, parse ≥ 0.999, dose-0
bit-identity vs capture **0/216 per model**; smoke gate 10/10.

Results (paired per-game DD diff, L65, true `main_perp` − perm): **vs seed-mean, true wins in both
Qwens** (+0.145 [+0.118, +0.170]; +0.431 [+0.352, +0.506]; llama +0.190 n.s.); **vs worst-of-3, the
criterion is MET in both Qwens too** — +0.055 [+0.035, +0.073] and +0.421 [+0.363, +0.475] (llama
+0.144 n.s.). Per-seed own-DD effects span −0.21…+0.01, and only the TRUE direction is DD-positive
in the Qwens. Per-seed cos to the true ⊥ℓ axis at L65 spans −0.74…+0.27: some permutation draws are
strongly **anti**-aligned with the true axis, and the most anti-aligned draw is not the most potent,
so projection onto the true axis does not explain seed variance. Perm directions still drive the L65
content→letter channel (cell-mean letter slopes −1.15…+0.05, max |game-level| 1.375, despite ⊥ℓ) →
**that channel is not incentive-specific.**

> **Reporting guard — safe:** "content-dependent; exceeds each of the three tested matched
> structured controls in both Qwen models; the only tested direction with cross-model-consistent DD
> sign." **Unsafe:** "uniquely potent among structured directions"; **any permutation-p-value
> language**; describing the move from the pre-correction arm's mixed verdict to this one as a
> formal improvement — **the two arms' seed sets differ (1/2/3 vs 0/1/2) and are not matched, so
> old-vs-new permutation comparisons are not like-for-like.**

### 6.4 q = 0.5 construct-identity correction

*Rebuilt 2026-08-22, analysed 2026-08-27. This section is the record of the correction the banner at
the top of §6 announces; the numbers it supersedes are preserved here rather than deleted.*

**Defect.** `steering/extract_directions.py` fitted `d_inc` against `delta1(vec, q̂₂)` — the model's
own elicited P2 act0-rate — while every other arm in the paper (behaviour, decoding, recruitment,
the token lens) uses `delta1(vec, 0.5)`. The shipped vector was labelled "q = 0.5" but was not.
Objective vs empirical incentive: Pearson **r = 0.45**, opposite incentive sign in **49 of 144
games**, mean q̂₂ = 0.585.

**Fix and verification.** The directions were rebuilt on the objective target. The new `d_inc`
equals an independent q = 0.5 rebuild at **cos 1.000** and agrees with Layer C's independently coded
Δ₁ᶜ at **r 1.000, sign 144/144**; `d_choice`, `d_trait*` and `d_random` are **bit-identical** across
the two arms. Re-verified from this repository's code: refitting `d_inc` from the released substrate
(`substrate/{model}/{game}/acts.npz`, keys `p1_baseline_cb{0..3}_l{L}`, 576 rows / 144 games)
reproduces the shipped q05 vectors at **cos = 1.000000** for all 3 models × L65/L79
(`analysis/steering/apparatus_geometry_q05.py`, §7).

**Roots.** Directions `steering/directions/akata_q05{,_perp,_perm}/` (their manifests carry
`incentive_belief: "q05"` and ship tracked under `data/manifests/directions/`); steering data
`steering/smalldose_q05/` (6 parquets) and `steering/perm_q05/` (3); tables
`data/results/steering/smalldose_summary_q05/`. The pre-correction arm is retained unmodified at
`steering/smalldose/`, `steering/perm/`, `steering/directions/akata{,_perp,_perm}/` and
`data/results/steering/smalldose_summary/`. **Not re-run on the corrected axis:** the saturated
sweep (§6.1) and the h3/h4 arms.

**Arm integrity, verified from the data rather than asserted.** Dose-0 rows and random-variant rows
are **bit-identical** to the pre-correction arm in all 6 model × mode files, while `main` and
`main_perp` differ — so the new axis is genuinely being steered and the harness did not drift.
Dose-0 identity: **0 mismatches / 2,592 cells**. A pipeline control gate re-ran the generator on the
*empirical* root and reproduced the committed pre-correction CSVs to max |Δ| = 8.9e-16 with all
label columns aligned, confirming that the added path flags are inert and the estimator did not move.

**What the correction changed in the numbers.** The pre-correction values, for the record:

| Quantity | Pre-correction (empirical belief) | Released (q = 0.5) |
|---|---|---|
| §5.1 `d_inc` sign-corr, qwen / qwen_instruct / llama (steer layers) | 0.540/0.246/0.208/0.237 · 0.700/0.278/0.263/0.258 · 0.394/0.326/0.309/0.296 | 0.679/0.360/0.313/0.329 · 0.759/0.483/0.466/0.464 · 0.456/0.446/0.420/0.405 |
| §5.1 `d_opp` estimable at | qwen l79, qwen_instruct l75/l79 | qwen l79 only |
| §5.2 \|cos(`d_inc`,`d_opp`)\| · \|cos(`d_inc`,`d_trait_risk`)\| | ≤ 0.268 · [0.029, 0.254] | 0.196 · [0.000, 0.284] |
| §6.2 h1 DD (main_perp L65) | +0.075 [+.056,+.090] 9/9 · +0.194 [+.120,+.266] 8/9 · +0.091 n.s. 6/9 | +0.060 [+.044,+.076] 9/9 · +0.236 [+.162,+.304] 9/9 · +0.085 n.s. 5/9 |
| §6.2 h2 DD · cross-mode r | +0.163 · +0.037 · −0.075 · r +0.88/−0.86/−0.56 | +0.131 · −0.055 · −0.143 · r +0.76/−0.90/−0.73 |
| §6.2 behaviour (pooled main_perp L65) | +0.071 [+0.003, +0.140] | +0.068 [−0.002, +0.140] |
| §6.3 seeds · paired DD vs seed-mean · vs worst-of-3 | 1/2/3 · +0.061 / +0.207 · **not met** | 0/1/2 · +0.145 / +0.431 · **met in both Qwens** |
| §7 cos(`d_inc`, mean-residual) | [−0.11, +0.28] | [−0.08, +0.15] |

**Two-channel dissociation is unchanged (12/12).** L79 letter push +0.90 / +1.00 / +2.59 collapses
to +0.002 / −0.002 / +0.009 under ⊥ℓ; L65 survives (+1.11→+1.10, +1.28→+1.25, +1.45→+1.36).
Canonical-to-letter ratio at L65 main_perp: 1.08% / 5.27% / 2.38%.

**The behavioural reading moved, and moved down.** On the corrected axis both the h1 and the random
pooled behavioural CIs include zero, so the pre-correction "suggestive" wording is retracted
outright: **no behavioural claim is made from this arm.**

**Permutation seeds are not comparable across the arms.** The q05 seeds 0/1/2 were re-screened
against the corrected target and are **not** matched to the pre-correction arm's 1/2/3. Per-seed cos
to the true ⊥ℓ axis at L65 spans −0.74 … +0.27 — perm0 is strongly anti-aligned in qwen_instruct and
llama — and the most anti-aligned draw is not the most potent. Per-seed own DD effects span
−0.21 … +0.01. Perm letter slopes span −1.15 … +0.05 (cell means; max |game-level| 1.375), so the
L65 content→letter channel remains **not** incentive-specific.

**Open item — the DD dose–response is asymmetric.** A §6.2-era claim of a "family baseline 0.42–0.69
with no ceiling" could not be reproduced from any dose-zero aggregation and has been replaced by the
recomputed dose-zero canonical preferences, which run 0.41–0.97 across families and models with DD
at **0.65 / 0.97 / 0.81** — i.e. little upward headroom in DD. Relatedly, nearly all of the DD
dose–response sits on the negative-dose arm; at +0.25 the canonical preference *falls*. Whether that
asymmetry is reported alongside the headline remains **open** at the time of this release.

---

## 7. Apparatus validity

- **Dose-0 bit-identity:** slot pref and greedy letter are identical (spread 0.0000) across
  variants, layers, modes, and process restarts days apart, and **exactly equal to the substrate
  capture baseline** (0/432 mismatches) — the harness injects nothing at dose 0 and sits on the
  captured footing. Re-measured across the whole released q05 small-dose package:
  **0 mismatches / 2,592 dose-0 cells** (432 per model × mode).
- **Prompt identity:** steering-side prompts sha256-match the capture prompts 24/24 per model,
  including the llama chat template rebuilt 7 days after capture (that template is date-stable —
  unlike the gpt-oss harmony template, §8 repro-gotchas).
- **`d_inc` is the centered, non-degenerate axis:** on the released q05 axes,
  cos(`d_inc`, mean-residual) ∈ **[−0.08, +0.15]** at the two injection sites; sign-correlation with
  δ1c reproduces the manifests (§5.1).
- **The injected axis is the locally fitted one, more so at L65 than at L79:** cos between `d_inc`
  fitted at the residual layer and refitted at the injection site (L+1) is **0.91–0.98 at L65** and
  **0.57–0.77 at L79**.
- **The random control is orthogonal to everything tested:** |cos(`d_random`, `d_inc`)|,
  |cos(`d_random`, `d_choice`)| and |cos(`d_random`, mean-residual)| are all **≤ 0.04**.
- **Self-check, from released code:** refitting `d_inc` from the released substrate reproduces the
  shipped q05 vectors at **cos = 1.000000** for all 3 models × L65/L79. Generator
  `analysis/steering/apparatus_geometry_q05.py`; it aborts below 0.999. Outputs:
  `data/results/steering/smalldose_summary_q05/apparatus_geometry_q05.{csv,json}`.
- **`wrong_layer` is a depth-transfer arm, not a null control.** `_wrong_layer` picks the middle
  candidate → always a deep layer (30→75, 65→75, 75→65, 79→65; llama 30→65, 50→65, 65→50, 79→50);
  for h0 its slopes equal main-at-the-inject-layer to 4 dp. `random` (flat) is what carries
  direction-specificity.

---

## 8. Reproduction gotchas

Four things will silently change your numbers if you do not honour them.

### 8.1 The gpt-oss harmony template is date-stamped

The harmony chat template injects a **live `Current date:` line** into the system preamble. Three
consequences:

1. gpt-oss substrate prompts are **not byte-reproducible across days**;
2. the collection ran ~19 h **across a UTC midnight** (2026-06-25 19:12 → 06-26 13:59), so the 144
   gpt-oss games carry **two different `Current date` values**;
3. gpt-oss's long reasoning is sensitive enough that the one-line date change can **flip a
   borderline decision** — verified: game ShHr cb0 P1 baseline is stored `commit_type=mixed` (with
   an incorrect 0.25/0.75 mix), but re-generating on a different date correctly derives the 0.5/0.5
   mixed NE and commits pure Option P (maximin). The game text is byte-identical; only the injected
   date differs.

Dense models are unaffected (raw prompts carry no date). The fix, implemented in
`collection/recapture_gptoss_transition.py` as `pin_prompt_to_stored()`:

> The harmony chat template injects a LIVE "Current date:" line, so a prompt built today differs
> from the one the original run saw. We pin each row's date by brute-forcing the candidates until
> `prompt_sha256 == the row's STORED prompt_hash` — i.e. the prompt is provably byte-identical to
> the original before we regenerate. No match → the row is marked `PROMPT_UNREPRODUCIBLE` and is
> **never** regenerated.

### 8.2 Off-by-one injection depth (kept deliberately; must be reported)

Capture key `l{L}` = the output of block **L−1** (transformers hook semantics), while the steer hook
on `model.layers[L]` perturbs the output of block **L** (= stream `l{L+1}`). Alignment of the
injected versus locally-fit axis: **0.91–0.98 at L65 but 0.57–0.77 at L79** (the last block
rotates the stream; measured by re-fitting `d_inc`/`d_choice_perp` on the captured `l{L+1}`).
Those are the released q05 values, re-measured by `analysis/steering/apparatus_geometry_q05.py`
at the two injection sites; the pre-correction arm read 0.91–0.98 at L65/L75 and 0.57–0.75 at L79,
so the conclusion is unchanged. Dose
scaling uses the true injection-site norm (correct). This was **kept unchanged across all arms for
comparability** — do not "fix" it when reproducing. **Lean on L65/L75 for layer-specific claims.**

### 8.3 Kernel-path determinism

Model numerics can be kernel-path-dependent, and the chosen path must be **fixed and shared between
capture and steering**. For gpt-oss specifically: a generic analysis environment lacks the
`kernels` package and will MXFP4-fallback, which **changes the router code path**; the collectors
hard-error rather than run in that state (override only with `--allow-no-kernels`, and then the
result is not comparable). The chosen `capture_mode` is stamped into the npz. Separately, gpt-oss
greedy generation is **not bit-identical** under batching or near-tie perturbations over long
chains — see [`GPTOSS_LAYER_B_HANDLING.md`](GPTOSS_LAYER_B_HANDLING.md).

### 8.4 Locked seeds

| Seed | Value | Where it applies |
|---|---|---|
| Decodability / crystallisation bootstrap | `20260520` | `analysis/layer_b/lib.py` `BOOT_SEED`; `build_decodability.py` uses `N_BOOT = 2000`, `build_crystallization.py` uses 1,000; `PCA_K = 64`, `N_SPLITS = 5` |
| CV seed | `0` | `analysis/layer_b/lib.py` `CV_SEED`; deterministic pre-sorted GroupKFold |
| Recruitment-bridge bootstrap | `20260627` | `analysis/layer_b/rebuild/rebuild_lib.py` `boot_ci(n=1000, seed=20260627)`, consumed by `rebuild_bridge_variants.py` (GroupKFold(5), no PCA). The same seed is used in `router_oneshot_audit.py` and `recruitment/shared.py`. |
| Token probe–lens | `0` | `analysis/layer_c/probe_bridge.py` — `GroupKFold(n_splits=min(3, ng))`, `k = 120` PCA components, `nboot = 800` |
| Permutation directions | `0, 1, 2` (released q05 arm) | `steering/build_perm_directions.py`. Slots start a priori from integers 0,1,2; a candidate is accepted only if `|corr(y_perm, y)| ≤ 0.15` (game level, per model), else replaced by the next integer. The screen runs against the arm's own target, so the two incentive arms ended on **different** seed sets: `0, 1, 2` on the released q = 0.5 target, `1, 2, 3` on the pre-correction empirical-belief target (where seed 0 was rejected). They are **not** matched — see §6.4. Three fixed seeds license **no** p-value. |
| Mixed-strategy realization | `sha256(game, cb, player)` | §4b; deterministic Bernoulli draw on `stated_p_act0`. Behavioural resolution only. |
| Layer-C cluster bootstrap | `0` | `analysis/layer_c/stats_layerc.py`, 1,000 resamples, clustered on `game_code` |

These are also recorded machine-readably in `data/MANIFEST.json` under `provenance`.

---

## 9. Limitations

1. **Architecture-dependent decode.** Dense = the generated J/P at the `A: Option` slot (slot
   `pref0` is a diagnostic, not the decision); gpt-oss = the harmony commit (per-cell
   pure/mixed/none). **Do not compare reasoning-model commit/transition residuals to the dense slot
   residual naively.**
2. **Mixed-strategy cells (reasoning model only).** `mixed` answers have no pure commit: they are
   captured at the channel transition (not the commit letter), the realized action is a **seeded
   sample** of the stated mix, and they are labeled `commit_type=mixed`. Pure-vs-mixed geometry
   carries a capture-position caveat. Report the mixed rate and its game-class distribution; the raw
   text is stored for re-classification.
3. **The gpt-oss router is READ, not EDITED** (MXFP4 fused). Residual-only nulls are **not**
   evidence of strategic absence — see HC-11 and
   [`GPTOSS_LAYER_B_HANDLING.md`](GPTOSS_LAYER_B_HANDLING.md).
4. **Precision differs across the model set** (8-bit dense, MXFP4 gpt-oss), which is relevant to
   magnitude-dependent claims. Llama is chat-wrapped while Qwen is raw → the slot sits at different
   positions; **dense residuals are not cross-family byte-comparable.**
5. **Cues are P1-only; the placebo is a format control.** The counterbalance is 4-cell (label-map ×
   q-order); steering may subsample it (a cost lever). Cross-game aggregation requires the canonical
   axis (HC-2).
6. **Layer indexing.** The Akata `acts.npz` captures **all** layers `l0..ln` with **`l0` = the
   embedding** (not a +1 block-output convention). State this explicitly in any depth claim.
7. **Layer-B geometry for gpt-oss** must use the uniform-site recapture root, not
   `substrate/gptoss` — see [`AMENDMENT_uniform_site_geometry.md`](AMENDMENT_uniform_site_geometry.md).
8. **No causal claim rests on the saturated-dose grid** (§6.1), and **no behavioural claim rests on
   the small-dose arm at all**: on the released q = 0.5 axis both the h1 and the random pooled
   behavioural CIs include zero (§6.2, item 4). The pre-correction arm's "suggestive only" wording
   is retracted — see §6.4.

---

## 10. Hard constraints (HC-1 … HC-12)

These are referenced by number from module docstrings throughout the codebase. They are
non-negotiable for any analysis on this substrate.

**HC-1 — Label space vs action space.** Prompts counterbalance the letters; strategic
interpretation must be in **action space**. Use `prob_act0_*` / `prob_act1_*` / `move1` / `move2`.
Map `prob_A_*` / `prob_B_*` through `label_map_p1` / `label_map_p2`.

**HC-2 — Canonical action axis required for cross-game aggregation.** Raw `move1`, `move2`,
`prob_act0_*` are *positional* labels — `act0` (first row of the canonical matrix) has a different
strategic meaning across games. In PdPd it is cooperate (dominated); in BaBa it is a coordination
point; in ChCh it is swerve. **Aggregating `(move1 == 0).mean()` across games is methodologically
invalid** — real effects cancel because they push toward act0 in one game and act1 in another. Any
cross-game metric must use `canonical_action_p1` / `canonical_action_p2` from
`data/games/game_features.csv`:

```python
aligned_canonical_p1 = (move1 == canonical_action_p1).astype(int)
```

`canonical_action_p1` is always defined by the precedence rule: **dominant action > unique-Nash
action > payoff-dominant-equilibrium action > maximin fallback.** The `risk_dominant_ne_*` columns
are retained only as descriptive diagnostics on the instantiated 1–4 scale; they are **not**
ordinal-invariant and do not enter the canonical-action rule. (Removing the former risk-dominance
step changes no canonical action for either player in the 144-game catalogue.) Human references in
`data/games/taxonomy/equivalence_per_canonical.csv` are already canonical-aligned to this axis.

The same warning applies to **letter labels in token attribution / logit-lens work**: aggregating
`logit[opt0] − logit[opt1]` across matches is invalid because the letter randomization decouples the
letter from canonical meaning per (game, cb). Re-score per match as
`logit[canonical_action_letter] − logit[non_canonical_letter]` before aggregating.

**HC-3 — Probe prefix has a trailing space.** Where a probe prefix is used, it is
`PROBE_PREFIX = "\nDecision: "`. Without the trailing space, decision-boundary tokenization changes
and the probe is invalid. (The released Akata substrate does not use a probe prefix — it ends at
`"A: Option"`, §2.2 — but the constant is load-bearing for the steering code path.)

**HC-4 — Counterbalancing.** Schedules come from `generate_balanced_label_schedule()` and
`generate_balanced_swap_schedule()` in `strategic_anatomy/prompting.py`; the Akata grid comes from
`akata_cb_grid()` in `collection/akata_common.py`. Do not construct them ad hoc.

**HC-5 — Independent-round blocks.** `--independent_rounds` resets round state; do not treat such
runs as repeated chains.

**HC-6 — Payoff multiplier.** The released one-shot substrate uses Bruns `payoff_multiplier = 1`
**intentionally**. Historical Bruns ordinal prompt-side runs use 2; Griffiths cardinal uses 1.
**Never "fix" collected one-shot data to ×2.** If a cross-run comparison needs unit alignment, do it
explicitly in the analysis and state it in the report.

**HC-7 — gpt-oss runs need the gpt-oss environment.** A generic analysis environment lacks the
`kernels` package and MXFP4-fallbacks, which changes the router code path. See
[`ENVIRONMENTS.md`](ENVIRONMENTS.md) and §8.3.

**HC-8 — Analysis code lives under `analysis/`.** The four layer trees (`layer_a`, `layer_b`,
`layer_c`, `steering`) plus `analysis/probe_common.py` are the analysis surface; `features/` holds
the Stage-F builders and `strategic_anatomy/` the shared runtime.

**HC-9 — Do not resurrect superseded code paths.** Several modules deliberately re-implement rather
than import from earlier private-repo code that is not part of this release. Where a docstring says
so, the re-implementation is the shipped behaviour.

**HC-10 — Data roots must be explicit.** Read [`DATA.md`](DATA.md) before any cross-run or
old-vs-new comparison. Never infer a comparator root from a similar-looking name; it must be
supplied by the user and printed in the report.

**HC-11 — GPT-OSS Layer B is MoE-native.** For any gpt-oss neural claim, read
[`GPTOSS_LAYER_B_HANDLING.md`](GPTOSS_LAYER_B_HANDLING.md). Router evidence is required for serious
Layer-B interpretation; residual-only dense-model language is not enough.

**HC-12 — No silent design decisions.** If a prompt, decoder, data root, counterbalance, payoff
scale, model set, capture position, or scope is unspecified, stop and state the exact proposed value
before any run or analysis depends on it.

---

## 11. Data formats

Full schemas are in [`DATA.md`](DATA.md). In brief:

- `results.parquet` — decision-level behaviour. Key fields: `move1`, `move2`, `move1_label`,
  `move2_label`, `label_map_p1`, `label_map_p2`, `prompt1_seen`, `prompt2_seen`, `prob_A_*`,
  `prob_B_*`, `prob_act0_*`, `prob_act1_*`, `decoded_action`, `commit_type`.
- `acts.npz` — residual activations, keys `{prefix}_cb{cb}_l{L}`; token metadata `input_ids`,
  `offsets`, `tokens`, `text`, `prompt_len`, `prefix_len`.
- `router.npz` — gpt-oss MoE router, keys `{prefix}_cb{cb}_l{L}_{gate|idx|w}`.

## 12. Game metadata

`data/games/game_features.csv` (144 rows × 45 cols) and
`data/games/taxonomy/equivalence_per_canonical.csv` (144 rows × 39 cols) are the **single source of
truth** for game-theoretic predictions, structural features, and canonical-aligned human references.
Every cross-game analysis joins these; no analysis should re-derive any of their columns.

**Naming warning:** `equivalence_per_canonical.csv` has `canonical_p1` / `canonical_p2` columns that
are **payoff vector strings** (e.g. `"1,4,3,2"`). The **action prediction** columns are
`canonical_action_p1` / `canonical_action_p2` in `game_features.csv`. They are not the same; do not
conflate them.

Standard join:

```python
features = pd.read_csv("data/games/game_features.csv")
equiv    = pd.read_csv("data/games/taxonomy/equivalence_per_canonical.csv")
master   = (master_long
            .merge(features, on="game_code", how="left")
            .merge(equiv, left_on="game_code", right_on="bruns_name", how="left"))
master["aligned_canonical_p1"] = (master["move1"] == master["canonical_action_p1"]).astype(int)
```

See `data/games/README.md` and `data/human_refs/README.md` for the per-file documentation, and
`data/games/taxonomy/EQUIVALENCE_DATASET.md` for the canonical-hub design.
