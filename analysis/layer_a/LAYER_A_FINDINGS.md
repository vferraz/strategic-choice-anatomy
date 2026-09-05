# Layer A — Thesis, Findings & Interpretations (AUTHORITATIVE)

Single source of truth for Layer A behavioural findings and how to interpret them.
Supersedes ad-hoc numbers in older notes. Last updated 2026-06-27.

**Status legend:** ✓ = verified on corrected data · ⚠ = caveat/limited scope ·
✗ = stale/wrong, do not cite.

**Corrected-data rule (applies to every number below).** Behaviour = the corrected
integrated-root realised action from `output/oneshot_akata_main_dl/{model}/{game}/results.parquet`.
Dense models use `decoded_action` gated by `parse_ok`; GPT-OSS uses `realized_action`
gated by `commit_type != "none"`. GPT-OSS `mixed` rows are included as their resolved
0/1 action, with `stated_p_act0` / `prob_source` retained as provenance metadata.
The soft move-token probability (`pref0`) is a shaded calibration readout only, never
the behavioural headline. The corrected caches are rebuilt by
`analysis/layer_a/build_data_layer.py`, which delegates to `src/shared_data.py`
and asserts corrected-root provenance (current canaries: GPT-OSS P(canonical)=0.889;
Llama P(canonical)=0.740; 4 usable baseline cells per game).

---

## 1. Main thesis

**Large language models produce genuine, game-theoretically measurable strategic
behaviour: their choices follow conventional solution concepts, are lawfully governed
by game complexity as human play is, and occupy distinct points in a bounded-rationality
space where mean canonical play, incentive sensitivity and soft calibration separate.**

This is the foundation of the paper. It establishes that there is real strategic
structure to explain — not noise, not a surface heuristic — that the human sample
corroborates the structure is genuine, and that models occupy a structured range of
behaviour. That range is the variable the neural layers (B/C) exist to explain, and the
precision/intercept decomposition is formalised by the quantal-response model (Fig 3).

We do **not** claim models reproduce or match human decisions. We take raw model output,
measure it with conventional game theory and behavioural economics, and explain it. The
human sample is corroboration (same complexity law, same incentive-response form), not a
target the models must hit.

---

## Methods — how every quantity is computed (paper-ready definitions)

All quantities are on the **canonical action axis** (precedence rule: dominant action >
unique NE > payoff-dominant NE > risk-dominant NE > maximin). The per-game behavioural
unit is **P(canonical)** = the realised rate of playing the canonical action across the
4 corrected-root counterbalance cells (realised decision; flipped from positional act0 by
`canonical_action_p1`). Source: `scripts/compute_game_features.py` (structure),
`analysis/layer_a/build_regime_rationality.py` (behaviour).

**M.1 Equilibrium conformity / distance to equilibrium (Fig 1a).** For a game with a
unique pure NE at (n₁, n₂) ∈ {0,1}² (n = the NE action's act0-probability) and realised
joint play (p₁, p₂) = (P1 act0 rate, P2 act0 rate), conformity is the **chance-normalised
inverse Euclidean distance** R = 1 − d/d_chance, with d = ‖(p₁,p₂) − (n₁,n₂)‖₂ and
d_chance = ‖(0.5,0.5) − (n₁,n₂)‖₂. So **R = 1 on the equilibrium, 0 at random play, < 0
worse than random.** Computed per game on dominance-solvable games only (unique pure NE),
averaged within family; realised (generated) is the headline, soft `pref0` the shaded
overlay. (On unique-pure-NE games d_chance is constant within the regime, so
raw/scaled/chance normalisations are monotone equivalent — chance is chosen for readability.)

**M.2 Payoff efficiency (Fig 1b).** Expected total payoff under independent play at the
realised marginals, E[π₁+π₂] = Σ_cell P(cell)·(u₁+u₂), P(cell) = product of the players'
marginal rates; **min–max normalised** to the achievable total range:
efficiency = (E[π₁+π₂] − min cell-sum)/(max cell-sum − min cell-sum), 0 = worst cell,
1 = best cell. Equilibrium and random references use the same map at the NE profile and at
(0.5,0.5). In MP the two references coincide (descriptive only).

**M.3 Coordination, by equilibrium structure (Fig 1c).** A cell (i,j) is a **pure NE** if
neither player gains by unilateral deviation (u₁[i,j] ≥ u₁[¬i,j] and u₂[i,j] ≥ u₂[i,¬j]).
For two-pure-NE coordination games:
- **Pareto-rankable** = one NE Pareto-dominates the other (both payoffs ≥, one >).
- **Distributional-conflict** = the two NE are Pareto-incomparable and P1's preferred NE ≠
  P2's preferred NE (Battle-of-Sexes type).
Complementary; split the 18 CO games **9 / 9**, cross-cutting CO1/CO2 (rankable = 5 CO1 +
4 CO2; conflict = 4 CO1 + 5 CO2). Per agent, the cb-matched joint profile (P1, P2 actions
in the same counterbalance cell) is scored: on **rankable** → Pareto-best NE / other NE /
miscoordinate (unconditional, sums to 1); on **conflict** → coordinate / miscoordinate
only (the P1-/P2-favoured split is symmetric and uninformative). Humans use the
marginal-product joint distribution. **The non-payoff-dominant NE is "the other NE," never
"inferior"; risk-dominance is not computable on ordinal {1,2,3,4} payoffs** (defined for
only 9/18 CO games), so it is not certified risk-dominant.

**M.4 Quantal sensitivity λ (Fig 3a).** logit P(canonical) = α + λ·Δ₁ᶜ, with Δ₁ᶜ the
**canonical-signed level-1 incentive gap at a uniform belief q = 0.5**:
Δ₁ = q·u₁(0,0)+(1−q)·u₁(0,1) − q·u₁(1,0)−(1−q)·u₁(1,1), signed positive toward the
canonical action. Fit by binomial MLE on realised choices: LLM rows use corrected-root
counterbalance cells, and the human row uses the Moore et al. session-level choices matched
by perspective (451 complete sessions from 450 participants × 144 games). Headline CIs are
bootstrap-by-game; the human row also records a session-clustered interval. λ =
quantal-response precision (random at
λ→0; sharp best-response/Nash at λ→∞). λ is a **slope**, not the whole curve:
the intercept α captures baseline canonical-choice propensity, so two agents can share
similar λ while having visibly different curve heights. Soft `pref0` λ is a calibration overlay.

**M.5 Game complexity (Fig 1d).** A **structural composite**: mean of z-scored components
(z-scored on the 144-game Bruns distribution) — `iesds_depth` (rounds of iterated
strict-dominance elimination, 0–2); an NE-structure term (1 if 0 or 2 pure NE, 0 if a
unique pure NE); `ne_distributional_conflict` (2-NE games only); and `payoff_conflict`
(1 − Pearson r between the players' payoff vectors; opposed interests → high).
`payoff_variance` is dropped (constant 1.25 across {1,2,3,4} permutations), so the
composite averages 3 components for non-coordination games, 4 for coordination games.
**Relation to Zhu, Peterson, Enke & Griffiths (2025):** an *adaptation of their logic*
(game complexity predicts deviation from optimal play), **not** a replication. They
estimate complexity by regression/ML over thousands of procedurally generated **cardinal**
games (many instances per class); our design is the **complete set of 144 unique ordinal
2×2 games**, which cannot support an instances-per-class regression, so we use a
theory-grounded structural composite on the ordinal matrices. Cite Zhu et al. for the
motivating logic, not the method.

**M.6 Complexity response (Fig 1d).** Per agent, association of per-game complexity with
per-game P(canonical): Spearman ρ (primary, scale-free) + OLS slope; quintile-binned means
for display; humans overlaid. Report all-games and dominance-only (cleanest where
"canonical" is unambiguous). The partial check controlling for Δ₁ᶜ is reported in
`f3_complexity_partial.csv`: it remains negative for dense LLMs, is near-zero for
GPT-OSS because of ceiling performance, and is positive/small for the human sample.
Thus the raw complexity-response panel is shared by humans and models, but the
beyond-incentive complexity claim is model-specific.

**M.7 Statistics.** Uncertainty = nonparametric **bootstrap by game** (2000 resamples,
seed 20260520), 95% percentile CIs, everywhere (λ, conformity, efficiency, coordination
shares). Complexity associations over n = 144 games (all |ρ| ≥ 0.27 are significant at
this n; report exact p). Equilibrium-vs-bounded model comparison (NE rejection) by
leave-games-out predictive log-loss (supplement).

**M.8 Trait cues — targets and conflict-set scoring (Fig 2a).** Each cue's *target action*
is the action its instruction prescribes, via an ordinally-valid heuristic on the canonical
8-vector: **security family (risk, loss, maximin)** → the maximin action (argmaxᵢ of the
worst-case payoff minⱼ p1[i,j]); **selfish** → expected-payoff max under a uniform opponent
(argmaxᵢ meanⱼ p1[i,j]); **inequity** → the most-equal action (argminᵢ meanⱼ |p1−p2|).
Ties drop the game. Risk/loss/maximin share the maximin target and are **not separately
identifiable** on ordinal 2×2 games (each action holds two of {1,2,3,4}, so all risk/loss
criteria coincide) — report as one *security* family, but keep the cues as separate points
(the dispositional cues and the explicit maximin rule can behave differently). **Aim is
scored only on each cue's CONFLICT set** — the games where its target ≠ the rational
action — the only set where the disposition's effect is identified apart from rationality:
security on the **48 mean-preserving-spread** games, inequity on its **~20** conflict games,
selfish on its **96** games (a manipulation check, since its target *is* the rational
action). Net = placebo-corrected shift toward target; aim = net ÷ magnitude (down-weight
near-floor magnitudes). On the conflict sets the dispositional cues aim toward target for
the capable models; the earlier all-games "anti-aimed" result was an identifiability
artifact (96/144 security games where safe = rational).

**M.9 Bounded-rationality map (Fig 3b).** Per agent, a quantal level-k (QLk) model is fit
to realised choices by binomial MLE: P(canonical) = π₀·0.5 + Σ_{k=1,2,3} π_k·σ(λ·gap_Lk),
with gap_Lk the canonical-signed incentive gap at the level-k belief (L1 = opponent
uniform; L2 = opponent plays its L1 action; L3 = its L2 action) and π a 4-simplex over
levels 0–3 with a shared precision λ. The map plots **reasoning depth** = Σ_k k·π_k (x) and
**decision sharpness** = mean_g max(p_g, 1−p_g) (y) — the fitted **mean modal-action
probability**, bounded [0.5,1]. Sharpness, **not λ, is the y-axis**: λ is unidentified once
choices are near-deterministic (GPT-OSS pins the optimiser ceiling at 50), whereas the
modal-action probability stays identified there. NB max(p,1−p) is modal-action determinism,
**not** literally P(best response) — the two coincide only for a single pure level, not a
finite-λ mixture. Grey ◇ landmarks are the λ→∞ limits of the pure levels — Random/L0 at 0.5;
det-Lk = decisiveness 0.5+0.5·(strict-response rate), giving det-L1 = det-L3 = 0.833 and
det-L2 = 0.944 (a tie-counting property of the 144-game set, **not** a competence order) — and
are reference points, **not** an achievable ceiling. The deep-deterministic corner is an
**idealised limit, NOT Nash** (finite level-k does not converge to Nash on the cyclic/mixed
games). CIs are bootstrap-by-game refits. Canonical build:
`analysis/layer_a/figscripts/fig3_qre_to_levelk.py` (panel a and panel b are built
together; the script asserts the corrected-root cache: 144 games, n=4 cells, integral
counts); numbers in `tables/f3_fingerprint.csv`.

---

## 2. Findings

### 2.1 Conformity in dominance-solvable games (Fig 1a)
Where a unique pure equilibrium exists, all four models are above chance, and conformity
falls as the equilibrium requires deeper reasoning (DD → OD1 → OD2). GPT-OSS remains the
highest mean canonical player, but the corrected root no longer supports the old
low-Llama reading.
- Per-model baseline P(canonical): **GPT-OSS 0.889**, **Llama 0.740**, Qwen-Instruct
  0.726, Qwen 0.714.
- Dominance-solvable conformity \(R\): GPT-OSS is strongest (DD 1.00, OD1 0.84, OD2
  0.72); the dense models all show the DD → OD depth drop.
- The depth ordering (DD>OD1>OD2 for bounded agents) is the expected backdrop — a
  property of the games, shared by humans and theory — not an LLM-specific result.

### 2.2 Game complexity governs play — the key "same structure" result (Fig 1d)
Per-game complexity (composite structural score) lawfully predicts play for **every
agent, humans included** — same direction, comparable magnitude. Spearman ρ of
complexity vs canonical-action rate ✓:

| agent | all games | dominance-only |
|---|---|---|
| Qwen2.5 | −0.67 | −0.70 |
| Qwen2.5-Instruct | −0.53 | −0.59 |
| Llama-3.1-Instruct | −0.51 | −0.56 |
| Moore et al. (human) | −0.43 | −0.56 |
| GPT-OSS | −0.27 | −0.24 |

Interpretation: models respond to the same structural variable that governs human play
(the NHB-relevant point). Agent-specific slopes rule out a mechanical artifact (if
complexity merely defined the canonical action, all agents would share one ρ). GPT-OSS's
shallow slope is a positive signal — a near-rational agent is complexity-*resistant* (it
solves hard games too). Even Llama has a clear negative slope, so it is weak-but-
structure-sensitive, not random.

Partial check: controlling for the canonical incentive gap Δ₁ᶜ, complexity remains
negative for the dense LLMs (partial r: Qwen −0.50, Qwen-I −0.24, Llama −0.26), is
near-zero for GPT-OSS (−0.03; ceiling), and is positive/small for the human sample
(+0.17). Therefore the raw complexity law is shared by humans and models, but
``complexity beyond incentives'' should be claimed for the LLMs only.

Caveat: the composite `complexity_score` does **not** cleanly rank the game families
(use it as a continuous per-game regressor, not to order DD<OD<CO<MP).

### 2.3 Coordination and equilibrium selection (Fig 1c)
Split by equilibrium *structure*, not CO1/CO2 (the structural cut crosses the families).
- **Pareto-rankable games (9: 5 CO1, 4 CO2)** — one NE Pareto-dominates. Unconditional
  outcome split (coordinate on Pareto-best / on the other / miscoordinate) ✓:
  GPT-OSS 42/14/44 · Llama 28/0/72 · Qwen2.5-I 22/0/78 · Qwen2.5 17/0/83 ·
  Moore (human) 37/15/47 · Griffiths (human) 56/6/38.
  Finding: on the four-cell corrected root, model coordination is weaker than the old
  split-root estimate; GPT-OSS is the only model that appreciably reaches the other NE.
- **Distributional-conflict games (9: 4 CO1, 5 CO2)** — no payoff-dominant NE (each
  equilibrium favours a different player). **Confounded — descriptive only.** A
  label-locked agent trivially lands on the diagonal NE (Llama spuriously high), a
  payoff-decisive agent clashes with itself (GPT-OSS spuriously low), and the
  P1-/P2-favoured split is symmetric (uninformative). Do not read as competence.
- **Terminology (hard rule):** never call the non-payoff-dominant NE "inferior" — it is
  "the other NE." Risk-dominance is **not** computable on ordinal {1,2,3,4} payoffs
  (defined for only 9/18 CO games), so the other NE is not certified risk-dominant.

### 2.4 Payoff efficiency across regimes (Fig 1b)
Realised earnings track the equilibrium where a game is solvable and collapse toward
chance in mixed games. Capable agents capture near-equilibrium payoffs in dominance
games; the gap to random vanishes in MP (where equilibrium and random payoffs coincide).
Regime-agnostic performance measure; consistent with 2.1–2.3.

### 2.5 The rationality continuum (per-model summary)
Action-space quantal sensitivity λ (q=0.5 / L1-uniform basis; generated choices):

| agent | λ | status |
|---|---|---|
| Qwen2.5 | 2.04 [1.77, 2.40] | ✓ |
| Qwen2.5-Instruct | 1.92 [1.63, 2.32] | ✓ |
| Llama-3.1-Instruct | 1.89 [1.60, 2.30] | ✓ |
| Moore et al. (human) | 0.91 [0.82, 0.99] | ✓ 451 sessions / 450 participants |
| GPT-OSS | 0.87 [0.59, 1.28] | ✓ final-channel; mixed resolved |

Reading: λ is a slope, not the whole competence ranking. The corrected root separates
**precision** from **baseline canonical propensity**: dense models have steep incentive
slopes, while GPT-OSS has the highest average P(canonical) and a high intercept but a
human-range λ. GPT-OSS remains strong in dominance-solvable games but is **not** globally
Nash and does not resolve coordination cleanly.

### 2.6 Human corroboration (and the ceiling caveat)
Per-game P(canonical) vs the human sample (Nagel), corrected data ✓:
Qwen 0.63 · Qwen-Instruct 0.62 · Llama 0.54 · GPT-OSS 0.36.
**Do not headline this correlation.** It is level-confounded: a high-baseline agent
has less game-to-game variance to covary with humans, and the metric rewards sharing
humans' *mistakes*. GPT-OSS's lower r therefore does not by itself imply weak strategic
structure. Use complexity-governance
(2.2) and the shared incentive-response form (QRE/λ, Fig 3) as the human-corroboration
evidence instead. The per-game scatter belongs in a supplement with this caveat.

### 2.7 Bounded-rationality map — two reasoning regimes (Fig 3b) ✓
In the level-k plane (reasoning depth × decision determinism), frontier LLMs do **not** lie
on a single rationality scale — they split into two regimes:

| agent | reasoning depth [95% CI] | decision determinism [95% CI] |
|---|---|---|
| GPT-OSS | **2.07 [1.73, 2.31]** | 0.855 [0.822, 0.892] |
| Qwen2.5 | 1.07 [0.84, 1.25] | 0.777 [0.738, 0.814] |
| Qwen2.5-Instruct | 1.22 [0.91, 1.47] | 0.775 [0.739, 0.814] |
| Llama-3.1-Instruct | 1.00 [0.83, 1.24] | 0.779 [0.740, 0.819] |
| Moore et al. (human) | 1.03 [0.95, 1.15] | 0.745 [0.715, 0.778] |

- **Dense LLMs + human = one inseparable cluster.** The three dense LLMs and Moore et al.
  share a common depth interval [0.95, 1.15] and determinism interval [0.74, 0.78] — shallow
  (~Level-1), noisy, best-responding to a naive opponent. Their CIs mutually overlap; the
  figure draws them as one envelope and asserts **no fine ordering** among them.
- **GPT-OSS is the lone second regime.** Deeper (a Level-2/3 mixture; π₀≈0.071) — its depth
  CI lower bound 1.73 clears the cluster's max depth_hi 1.47 — and more deterministic.

**Why this earns the main text** (it is not a restatement of 2.5):
1. It **decomposes "rationality" into two separable axes** — reasoning depth and execution
   determinism — that a single λ conflates, and so resolves the GPT-OSS paradox of 2.1/2.5:
   its high canonical play with a flat λ comes from greater **depth**, not lower noise.
2. It is the **methodologically clean human corroboration**: the human sits **inside** the
   dense-LLM cluster (same depth, similar determinism, overlapping CIs) — a shared-bounded-
   regime result, explicitly **not** the level-confounded rate-matching of 2.6. This is the
   form of human corroboration the thesis (§1, §4) endorses.
3. It **names the explanandum for Layers B/C**: reasoning depth is the latent axis the neural
   work must account for (deep-deterministic GPT-OSS vs shallow-noisy human-like cluster).

**Guardrails (baked into the figure caption).** The separation is **depth-led** — GPT-OSS's
determinism edge over the cluster is **near zero with overlapping bootstrap CIs**, so do **not** claim
GPT-OSS is "sharper". "Decision sharpness" = mean **modal-action probability**, not
P(best response). The deep-deterministic corner is an **idealised limit, not Nash**. The
discriminating content is exactly two facts — "GPT-OSS separates on depth" and "the human
sits with the dense models" — so the panel must **not** be read as a richly resolved
fingerprint. Source: `tables/f3_fingerprint.csv`.

---

## 3. The unifying quantity — hand-off to Fig 3
λ (quantal-response precision; logit P(canonical) = α + λ·Δ₁ᶜ) is the spine that carries
Layer A forward. Fig 1 establishes the phenomenon; **Fig 3 formalises it** with the
incentive-response curve beside the complexity-response curve. λ is the quantity carried
into the activations (Layers B/C). Fig 3b adds a second, orthogonal axis — **reasoning
depth** (the level-k mixture; §2.7) — and the behavioural heterogeneity along it (a deep,
deterministic GPT-OSS vs a shallow, noisy human-like cluster) is itself an explanandum the
neural layers target.
Belief basis: headline λ uses q=0.5, the model-independent stimulus basis that makes
cross-agent comparison apples-to-apples.

---

## 4. Interpretation stance (claim / do not claim)
- CLAIM: model play is genuinely strategic, theory-measurable, structure-governed, and
  separates along precision, intercept and complexity sensitivity; the human sample
  corroborates that the structure is real.
- DO NOT CLAIM: that models reproduce/match human choice rates ("human-comparable" as a
  level claim). The shared object is the *response to structure/incentives*, not the level.
- The DD→OD→CO→MP difficulty gradient is the expected backdrop (holds for everyone), not
  an LLM finding; it is evidence the models track the same structure.
- "Competence depends on the game" is true for all agents and is not the thesis.
- GPT-OSS has the highest mean canonical play but a human-range incentive slope; its
  lower human-correlation is a level/variance artifact, not a standalone disconfirmation.

---

## 5. Methodological guardrails (non-negotiable — these were real bugs)
1. Behaviour = corrected integrated-root realised action, never slot argmax.
2. GPT-OSS readout uses `realized_action`; `mixed` rows are included as resolved actions,
   `none` rows are dropped.
3. Llama from the chat-wrapped substrate, never the stale raw capture.
4. Rebuild `build_data_layer` caches before locking numbers; assert corrected root =
   `output/oneshot_akata_main_dl`, 576 configs, payoff_multiplier=1 and 4 cells/game.
5. Equilibrium-proximity only on dominance-solvable games (unique pure NE). MP is not
   identifiable from a single deterministic choice (mixed NE; all actions payoff-equal at
   the equilibrium; the ≈0.5 counterbalance marginal is an artifact, not mixing) →
   descriptive only. Coordination measured by structure (selection on Pareto-rankable;
   conflict descriptive).
6. Never "inferior NE"; risk-dominance not computable on ordinal {1,2,3,4}.
7. Canonical action axis for all cross-game aggregation; bootstrap-by-game CIs.
8. Human λ on individual MGN decisions (451 sessions from 450 participants × 144 games), with session-clustered uncertainty recorded.

---

## 6. Open items / pending
- No separate mixed-strategy analysis is included yet; GPT-OSS mixed rows are only used
  as resolved choices in the current Layer-A behavioural rates.

---

## 7. Figure mapping (final architecture — see `SPEC_layerA_figures_final.md`)
**Fig 1 (foundations), four panels:**
(a) Unique-equilibrium conformity (dominance-solvable; realised solid + soft shaded; depth-ordered).
(b) Payoff efficiency across the six regimes (equilibrium / random reference lines; MP descriptive).
(c) Coordination by equilibrium structure (Pareto-rankable 3-way; distributional-conflict greyed/descriptive; CO1/CO2 counts in headers).
(d) Game complexity governs play (per-agent complexity→P(canonical) slope + human overlay; GPT-OSS most resistant).

**Fig 2 (traits; GPT-OSS = full corrected-root sample):**
(a) Magnitude vs target alignment, each cue scored on its CONFLICT set, coloured by family — dispositional cues (risk/loss/inequity) aim toward target for the capable models (the all-games "anti-aimed" result was an identifiability artifact). Keep cues as separate points (risk/loss vs the maximin rule can diverge).
(b) TreeSHAP feature attribution for canonical conformity, using the de-collinearized corrected feature set; positive SHAP values increase predicted canonical play.
(c) Nested held-out AUC: model → +structure (large jump) → +trait (modest additional
gain) → +interaction. Dense-primary AUC goes 0.49 → 0.68 → 0.72 → 0.75; GPT-OSS full
goes 0.46 → 0.79 → 0.83 → 0.91. The variance-partition bar is retired from the main text.
The old incentive-gating panel is REMOVED. Data source is the corrected integrated root
`output/oneshot_akata_main_dl` for all four models.
Appendix = 3-panel attribution remainder (`figS_attribution.png`): signed-incentive SHAP dependence, nested variance/AUC progression, GLMM forest. The wrong-sign SHAP bug is fixed (ρ≈+0.945).

**Fig 3 (behavioural model):**
(a) Incentive response — QRE curve, P(canonical) vs Δ₁ᶜ, λ per agent, human overlay (the λ spine).
(b) Bounded-rationality map — reasoning depth × **decision determinism** (bounded mean
modal-action probability; **replaces** the unidentified log-λ fingerprint). Two regimes:
dense LLMs + human cluster (not separable) vs a deeper, more deterministic GPT-OSS;
level-k archetype landmarks; bootstrap CIs; **depth-led** separation (determinism edge
marginal/overlapping); **no Nash** claim. Built in
`figscripts/fig3_qre_to_levelk.py`; see §2.7 / M.9.
Model-human alignment scatterplots, model selection (NE rejection), MGN rule-classification, L1(α) → supplement.

**Appendix:** remaining trait-attribution checks (3-panel: signed-incentive SHAP dependence,
nested variance/AUC progression, GLMM forest); mixed-game descriptives (human aggregate
frequency; LLM soft-policy entropy); model-selection + model-human alignment.

## 8. Provenance
Corrected behaviour: `output/oneshot_akata_main_dl/{model}/{game}/results.parquet`.
Structure/complexity:
`datasets/processed/game_features.csv`. Equilibria: `analysis/_shared/eq_engine.py`.
Human samples: Moore, Germano & Nagel (2026) [`datasets/nagel/`]; Zhu, Peterson, Enke &
Griffiths (2025). Build: `analysis/layer_a/build_regime_rationality.py`,
`figscripts/fig1_rationality.py`. Spec: `analysis/SPEC_layerA_fig1_rationality_corrected.md`.
