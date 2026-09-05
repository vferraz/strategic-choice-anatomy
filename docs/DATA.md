# Data

Two kinds of data ship with this project.

| | Where | Size | Tracked in git |
|---|---|---|---|
| **Committed tables** — every number behind a paper figure | `data/` in this repository | ~41 MB | yes |
| **Released deposit** — the raw substrate, activations, and steering outputs | downloaded into `data_heavy/` | ~19 GB | no |

Everything resolves through `strategic_anatomy/config.py`. Nothing in the codebase hardcodes a
data path.

```bash
# fetch the deposit (all components, ~19 GB)
python scripts/download_data.py

# or a single component
python scripts/download_data.py --component layerc

# or point at a copy you already have
export SCA_DATA_ROOT=/mnt/big/sca_data
```

---

## 1. Released roots

`$SCA_DATA_ROOT` defaults to `<repo>/data_heavy`. Each root below has an accessor in
`strategic_anatomy/config.py`; use the accessor, never the literal.

| Deposit path | Accessor | Size | Contents |
|---|---|---|---|
| `substrate/{model}/{game}/` | `substrate_root()` | ~18 GB | The main collection: 4 models × 144 games. `results.parquet`, `acts.npz`, `config.json`, `_DONE`; gptoss additionally `router.npz`. |
| `gptoss_recap/{game}/` | `gptoss_recap_root()` | 138 MB | GPT-OSS uniform-site recapture, 144 games. Adds `genids.npz` (full token streams). **Supersedes `substrate/gptoss` for cross-row geometry** — see §4. |
| `layerc/{model}/{game}/` | `layerc_root()` | 34 MB | Layer-C per-token logit-lens scores, dense models only. `tokens.parquet`, `config.json`, `_DONE`. |
| `layerc_bridge_residuals/{model}/{game}/` | `layerc_bridge_root()` | 1.1 GB | The Layer-B↔C bridge: per-token residuals. `resid.npy`, `meta.parquet`, `config.json`, `_DONE`. |
| `steering/smalldose_q05/*.parquet` | `steering_root()` | 172 KB | **The released steering arm** (objective q = 0.5 incentive target): 6 parquets, 54,432 rows. |
| `steering/perm_q05/*.parquet` | `steering_root()` | 84 KB | Permutation matched-control runs for the released arm: 3 parquets, seeds 0/1/2. |
| `steering/directions/akata_q05{,_perp,_perm}/{model}/` | `steering_root()` | 59 / 12 MB / 564 KB | Fitted steering directions for the released arm: `directions.npz` + `manifest.json` (`incentive_belief: "q05"`). |
| `steering/smalldose/*.parquet` | `steering_root()` | 168 KB | *Historical:* the pre-correction empirical-belief arm (`METHODS.md` §6.4). |
| `steering/perm/*.parquet` | `steering_root()` | small | *Historical:* permutation runs for the pre-correction arm, seeds 1/2/3. |
| `steering/directions/{akata,akata_perp,akata_perm}/{model}/` | `steering_root()` | small | *Historical:* the pre-correction direction sets. |

There is also a regenerable working cache at `layer_b_cache/` (`layer_b_cache_root()`, ~4.5 GB).
It is **not** part of the deposit — `analysis/layer_b/build_residual_cache.py` rebuilds it from
`substrate/` in about 90 seconds.

Models are `qwen` (Qwen2.5-72B base), `qwen_instruct` (Qwen2.5-72B-Instruct), `llama31_instruct`
(Llama-3.1-70B-Instruct), `gptoss` (gpt-oss-120b). Layer C and the bridge cover the three dense
models only.

---

## 2. Artifact schemas

Verified against the released files, not from a spec.

### `substrate/{model}/{game}/results.parquet`

One row per (player, condition, counterbalance cell): **32 rows/game** = 8 conditions × 4 cb.

Dense models (20 columns):

```
model  game_code  player  condition  counterbalance_id  label_map  q_order
label_map_p1  label_map_p2  prompt_hash
move_letter  parse_ok  decoded_action  decoded_label  gen_text
pref_J  pref_P  prob_act0  prob_act1  pref0
```

`decoded_action` is **the decision** (the generated letter mapped through the cell's
`letter_to_action`); `-1` marks a parse failure. `pref0` / `prob_act*` are the slot readout and are
a **diagnostic only** — see `METHODS.md` §3.

GPT-OSS (23 columns) replaces the slot-decode fields with the harmony commit fields:

```
model  game_code  player  condition  counterbalance_id  label_map  q_order
label_map_p1  label_map_p2  prompt_hash
commit_type  move_letter  decoded_action  realized_action  stated_p_act0  prob_source
capture_idx  captured  final_marker  n_new_tokens  slot_pref_J  final_text  gen_s
```

`commit_type ∈ {pure, mixed, none}`; on `mixed` cells `realized_action` is the **seeded** Bernoulli
resolution of `stated_p_act0` (`METHODS.md` §4b). `final_text` retains the raw final-channel text so
the prose classifier can be re-run offline.

### `substrate/{model}/{game}/acts.npz`

Keys `{prefix}_cb{cb}_l{L}`, one float32 vector per key, at the capture position.

| | prefixes | cb | layers | keys/game | dim |
|---|---|---|---|---|---|
| dense | 8 | 4 | 81 (`l0`..`l80`) | 2,592 | 8192 |
| gptoss | 8 | 4 | 37 (`l0`..`l36`) | 1,184 (captured cells only) | 2880 |

**`l0` is the embedding**, not a block output — see `METHODS.md` §9 item 6. Prefixes are
`p1_baseline`, `p2_baseline`, and `p1_cue_{risk_aversion,loss_aversion,inequity_aversion,maximin,selfish_maximizer,length_match_null}`.

### `substrate/gptoss/{game}/router.npz`

Keys `{prefix}_cb{cb}_l{L}_{gate|idx|w}` over 14 router layers
`[1,3,6,9,12,15,18,21,22,24,27,30,33,35]`:

| suffix | shape | dtype | meaning |
|---|---|---|---|
| `_gate` | (128,) | float32 | gate logits, **pre-softmax**, all 128 experts |
| `_idx` | (4,) | int32 | top-k selected expert indices |
| `_w` | (4,) | float32 | top-k routing weights |

`idx` and `w` are **derived** from the gate logits under the topk-then-softmax convention
(`GptOssTopKRouter`), which is robust to the router module's return signature across the MXFP4
kernel path and the native path. The `capture_mode` actually used is stamped into the npz.

### `gptoss_recap/{game}/`

Same shape as the gptoss substrate plus `genids.npz` (full generated token-id streams,
`{prefix}_cb{cb}_genids`, int32) so no future GPU pass is needed. `results.parquet` carries 31
columns, adding the recapture-specific fields:

```
capture_site  pin_date  transition_idx  letter_idx  stated_p_J  p_act0  p_canonical
use_in_neural_target  repro_ok  repro_mismatch  val_cos  val_site  val_ok  final_text_full
```

- `capture_site` — the uniform analysis→final transition for every row (that is the point of the
  recapture; see §4).
- `pin_date` — which `Current date:` value reproduced the stored `prompt_hash`. See
  `METHODS.md` §8.1: rows whose prompt could not be pinned were **never regenerated**.
- `use_in_neural_target` — the stated-policy neural target flag. **35 harness-imputed
  `default_uniform` rows are flagged out and never imputed** → n = 541 of 576.
- `val_cos` / `val_ok` — stream reproduction check; all 576 rows reproduced exactly (`val_cos = 1.0`).

### `layerc/{model}/{game}/tokens.parquet`

Per-token logit-lens scores, lens layers L79 + L40:

```
model  game_code  cb_id  condition  layer  token_index  token_str  char_start  char_end
region  is_canonical_row  score_canonical  score_trait_target  canonical_action_letter  prompt_hash
```

`score_canonical` is already re-scored per match onto the canonical action letter — do **not**
aggregate raw `logit[opt0] − logit[opt1]` across games (`METHODS.md` HC-2).

### `layerc_bridge_residuals/{model}/{game}/`

`resid.npy` — float16, shape `(n_tokens_selected, hidden)` e.g. `(152, 8192)` — paired row-for-row
with `meta.parquet`:

```
model  game_code  cb_id  layer  token_index  token_str  region  is_canonical_row  is_final
digit  score_canonical  canonical_action  delta1c  incentive_sign
```

### `steering/smalldose_q05/*.parquet` (and the historical `steering/smalldose/`)

One parquet per (model, mode); 9,072 rows each. Identical schema in both arms; the dose-0 and
`random` rows are bit-identical between them, `main` and `main_perp` are not (`METHODS.md` §6.4):

```
model  mode  game_code  wave  counterbalance_id  steer_layer  variant  inject_layer  dose
status  slot_pref_J  realized_letter  realized_action  canonical_action_p1  realized_canonical
```

`variant ∈ {main, random, main_perp}`; `dose ∈ {−0.25, −0.1, −0.05, 0, +0.05, +0.1, +0.25}`;
`wave` is the pre-registered stratified wave (`STEER_SAMPLE_RULE.md`). Note `inject_layer` versus
`steer_layer` — the documented off-by-one, `METHODS.md` §8.2.

### `steering/directions/{set}/{model}/`

`set ∈ {akata_q05, akata_q05_perp, akata_q05_perm}` for the released arm and
`{akata, akata_perp, akata_perm}` for the historical one.

`directions.npz` with keys `d_inc_l{L}`, `d_choice_l{L}`, `d_choice_perp_l{L}`,
`d_trait_l{L}_{trait}`, `d_opp_l{L}`, `d_random_l{L}`, plus `manifest.json` recording the fit
statistics, estimability verdicts, removed cosines, the per-vector sha256, `incentive_belief`, and a
git commit. A direction the estimability gate rejected is written as a zero vector, so check
`per_direction[...]["status"]` before using one.

The `manifest.json` files for the q05 sets also ship **tracked**, under
`data/manifests/directions/akata_q05{,_perp,_perm}/{model}/`, so the construct identity and the
per-vector checksums travel with the repository rather than only with the deposit. **Caveat on
their `git_commit` field:** it records the commit the GPU machine had checked out, not the code that
ran — the `--belief q05` extractor invocation was working-tree state there and is not in any commit.
The load-bearing evidence is instead the refit self-check in
`analysis/steering/apparatus_geometry_q05.py`, which reproduces every shipped q05 `d_inc` from the
released substrate at cos = 1.000000.

---

## 3. Committed tables (`data/`)

| Path | Contents |
|---|---|
| `data/games/` | `game_features.csv` (144 × 45) — the canonical action axis and structural covariates. `taxonomy/` — equivalence tables and human-reference crosswalks. See `data/games/README.md`. |
| `data/human_refs/` | Derived **per-game** human aggregates only, plus the LLM trait-effect tables. `unified_pairs.parquet` (7,466 × 92). See `data/human_refs/README.md`. |
| `data/manifests/` | Run manifests: `oneshot_config.json`, `game_universe_oneshot.csv`, `steer_sample_54.csv`, the h1/h3 game pair lists. |
| `data/results/` | Every table a figure reads, by layer: `layer_a/`, `layer_b/` (+ `fusion/`, `rebuild/`, `recruitment/`), `layer_c/`, `steering/`. Plus `figures_reference/` — the 11 final paper PDFs, for diffing your rebuild. |
| `data/MANIFEST.json` | sha256 + size for every file above, the builder provenance chain, and the deposit component list. |

### Human data is not redistributed

Only **derived per-game aggregates** ship. The two source studies' participant-level exports belong
to their authors; `data/human_refs/raw/README.md` says what to obtain, from where, and — via
`data/MANIFEST.json` `human_raw_sources` — the sha256 to check your copy against, so the human λ fit
is verifiable without us redistributing anything. Two paper figures need those raw files; see
`REPRODUCING.md`.

---

## 4. Which gpt-oss root to use

This is the one place where picking the wrong root silently produces a wrong number.

- **Behavioural fields, and anything about the dense models:** `substrate/`. Unaffected.
- **Cross-row gpt-oss geometry:** `gptoss_recap/`, **not** `substrate/gptoss`.

The original gpt-oss capture anchored the read-out position to the model's own output (pure → the
commitment letter; mixed → the analysis→final transition). Cross-row geometry on that substrate is
**unidentifiable**: the position covaries with the model's strategy type. The recapture uses a
**uniform** capture site for every row. Full reasoning, the superseded numbers, and what still
stands: [`AMENDMENT_uniform_site_geometry.md`](AMENDMENT_uniform_site_geometry.md).

The recap was verified array-by-array at release time: **46,080 arrays, 0 corrupt, the L0 anchor
exactly invariant**, and all 576 P1-baseline streams reproduced bit-identically (`val_cos = 1.000`).

---

## 5. Payoff scale (do not "fix" this)

The released one-shot substrate uses Bruns **`payoff_multiplier = 1`, intentionally**. Historical
Bruns ordinal prompt-side runs use 2; Griffiths cardinal payoffs use 1.

> **Never "fix" collected one-shot data to ×2.** If a cross-run coefficient comparison requires unit
> alignment, do it explicitly in the analysis and state it in the report.

This is HC-6 in `METHODS.md`.

---

## 6. Comparator discipline

Any report comparing data roots must print, and must not infer a root from a similar-looking name:

```text
PRIMARY_ROOT_USED                 = <path>
COMPARATOR_ROOT_USED              = <path, semicolon-separated, or NONE>
COMPARATOR_ROOT_WAS_USER_SUPPLIED = True/False
AUTODETECT_USED                   = True/False
```

A comparator root must be **supplied by the user**, never auto-detected. Analyses that require one
abort if it is missing. This is HC-10.

Earlier substrates from this project — the A/B-matrix one-shot collection and the DESIGN_V2
round-based collection — are **not** released and are not in the deposit. A handful of shipped
scripts still target them (their module docstrings say so, and they exit with an explanatory
message rather than running); they are retained because their estimator functions are used
elsewhere.

---

## 7. Integrity checks

```bash
# every committed file against the manifest
python scripts/download_data.py --verify-committed

# a single deposit component after download
python scripts/download_data.py --component substrate --verify-only
```

Per-game, the deposit carries `_DONE` sentinels and a `sha256` field inside each `config.json`;
`_DONE` counts should be 144/144 per model for `substrate/`, `gptoss_recap/`, `layerc/` and
`layerc_bridge_residuals/`.
