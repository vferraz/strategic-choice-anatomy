# Storyline, the "associative fusion" finding, and neuroscience — memo

Author: analysis pass, 2026-07-02. Scope requested: (1) run the incentive×decision
orthogonality idea against the data, (2) scan for unmined findings, (3) pin down the
storyline and the neuroscience relations. Not a paper edit — a decision document.

Data used: `output/oneshot_akata_main_dl` via the cached decision-slot residuals in
`analysis/layer_b/_data/baseline/` (P1 baseline, 576 rows = 144 games × 4 cb, all
layers). New code/tables/figures live in `analysis/fusion_associative/`.

---

## 0. One-paragraph verdict

The project is in strong shape and your instinct about orthogonality is right and
productive. The paper already contains the winning thesis — *strategic structure is
represented broadly but recruited into choice selectively* — and `STORYLINE_STRATEGY.md`
already lays out the correct re-sequencing. What was missing is exactly what you intuited:
the incentive and decision axes are **not** orthogonal, and the interesting fact is **where
along depth they stop being orthogonal**. They start near-orthogonal (separate streams) and
**fuse in the late layers** — a co-representation→binding→commitment profile that is the
clean LLM analogue of an *associative area*. This is new relative to the paper (which
reports only the final-layer angle), it is robust as a depth phenomenon, and it gives the
neuroscience framing something concrete to cash in. Two cautions, both important and both
below: the *base-vs-instruct* version of the effect is belief-basis dependent, and the
correct statistical null is not 90°.

---

## 1. The storyline — validated, with three refinements

Keep the spine in `STORYLINE_STRATEGY.md`: **one estimand (λ), three windows (behaviour /
representation / token-localisation), two anchors (human data, neuroscience)**, thesis =
*represented broadly, recruited selectively*. It is the right paper. Three updates the draft
should absorb:

**1a. "Base-vs-instruct" is now *graded recruitment*, not a clean binary.** The corrected
Layer C (`layer_c/LAYER_C_FINDINGS.md`, 2026-06-28) supersedes the numbers in the
storyline doc. λ in vocabulary space at the final pre-choice token: Qwen-Instruct **1.43**
(r=0.74), **Llama 0.79** (r=0.66, clearly recruits), Qwen-**base** **0.17** (r=0.13, weak)
— while behavioural λ is ≈1.9–2.0 for all three. So the headline is not "instruct recruits,
base doesn't"; it is *"behaviour hides a graded internal recruitment axis; the pretrained
base model is the weak case."* That is still a controlled, striking result (same weights ±
alignment; behaviourally indistinguishable; internally ordered), just stated correctly.

**1b. There is an unresolved Llama contradiction the paper must not paper over.** Layer B's
recruitment bridge classes Llama as a non-recruiter (partial slope ≈ 0); Layer C (corrected,
final-token) says Llama clearly recruits (λ_lens 0.79). Layer C's own notes suspect the
Layer B Llama readout has the chat-scaffolding dilution that was just fixed in C. **My
geometry analysis casts an independent third vote and it sides with Layer C: Llama fuses
strongly (see §2).** Recommendation: re-run the Layer B bridge for Llama at the final
pre-choice token before freezing any "Llama doesn't recruit" sentence. Right now that claim
is unsettled and three methods do not agree.

**1c. Recruitment is deferred in *two* coordinates, not one.** Layer C Result 6 ("deferred
integration": payoffs are read but not projected onto the choice axis until the commit
token — a *position* axis) and my depth result (alignment forms only in late layers — a
*depth* axis) are the same phenomenon on orthogonal coordinates. Unify them: *the model
stores the incentive as it reads, and projects it onto the decision in one late operation,
localised to late layers and to the answer position.* This is a cleaner, more memorable
statement than either result alone.

---

## 2. New finding: the "associative" fusion transition (your orthogonality idea, tested)

**Question, made precise.** Define, per layer L, the same two axes the paper uses at the
final layer (`analysis/layer_b/build_geometry.py`): a **decision axis** (difference of
mean residuals for canonical vs non-canonical realised choices) and an **incentive axis**
(residual covariance with the continuous canonical-signed level-1 gap Δ₁ᶜ). Their angle
measures whether the choice direction *is* the incentive direction. Compute it at **every**
layer, not just the last.

**Result (robust).** In every model the two axes are near-orthogonal and statistically
indistinguishable from chance through the early and middle layers, then the angle **drops
sharply in the late layers** to the ~27–38° the paper reports. This is a genuine transition,
not a gradual drift — Qwen-Instruct shows a near-cliff at ~0.7 relative depth
(`figures/fig_fusion_depth.png`, top-left). Interpretation: the incentive and the choice are
carried in *separate* subspaces while the model reads the game, and are *bound into a common
subspace* only late — the co-representation→binding→commitment profile of an associative
region.

**The correct null matters (a real fix for the paper).** The draft compares the angle to a
90° "orthogonal null." That is the angle of two *independent* random directions, and it is
too liberal here: the decision and the incentive are correlated *targets* (r≈0.56 across
games), so even with no joint neural encoding the two estimated axes are pulled together. A
permutation null that shuffles (decision, incentive) jointly across rows — preserving that
target-correlation but destroying any residual↔target link — has median ≈55° (uniform
belief), not 90°. (Sanity check: shuffling them *independently* recovers ≈88°, confirming the
machinery.) Fusion should be claimed where the observed angle drops **below this permutation
band**, which happens only in the late layers. The paper's "far below 90°" is technically
true but rests on the wrong null; switching to the permutation null makes the claim
defensible *and* turns the flat final-layer number into the depth story.

**Per-model summary** (`tables/fusion_summary.csv`; "sig late" = layers in the deep 40%
whose angle is below the permutation null, p<0.05):

| model | final angle | fusion onset (uniform) | sig late (uniform) | sig late (empirical) |
|---|---|---|---|---|
| Qwen2.5-Instruct | 31.0° | ~0.71 depth | 20/33 | 25/33 |
| Qwen2.5 (base) | 26.7° | none | 0/33 | 21/33 |
| Llama-3.1-Instruct | 30.0° | ~0.60 depth | 33/33 | 33/33 |
| GPT-OSS-120B | 37.5° | (early, see below) | 5/15 | 15/15 |

Two reads that survive: (i) **the depth transition itself is robust** across belief bases and
across all models; (ii) **the geometry orders recruitment the same way Layer C does** —
Instruct and Llama fuse strongly under the objective incentive; the pretrained base is the
weak case; this is why the geometry adjudicates the Llama tension toward "Llama recruits."

**GPT-OSS is architecturally different and consistent with being read at the commit token.**
Its angle is already low at shallow depth and stays low — expected, because GPT-OSS is read
*after* its reasoning trace, so by the commit token the incentive is already integrated at
every layer rather than assembled late. This is a nice cross-architecture contrast, not a
contradiction.

**The honest caveat (do not skip).** The clean "base does *not* fuse" result holds under the
*objective/uniform* incentive (matching Layer C's canonical λ_lens, where base = 0.17). Under
the model's *own empirical* opponent belief, the base model's late angle *does* dip below its
null (21/33 late layers). So the base model aligns its choice with the incentive computed
under its own belief but not with the objective incentive — suggestive, but belief-basis
dependent. Therefore: present the **depth transition** as the robust finding and let the
**base-vs-instruct** contrast *corroborate* (not carry) the recruitment claim that λ_lens and
the bridge carry. Both belief bases are shown side by side in `fig_fusion_depth.png`.

## 2b. Companion: "represented early, recruited late"

Same machinery yields a second, complementary depth signature
(`figures/fig_represent_vs_recruit_depth.png`). The *decodability* of the realised choice
(Cohen's d of the decision-axis projection) is already large in the **early** layers
(Qwen-Instruct d≈1.9 at ~0.15 depth) — the model "knows its answer direction" early — but the
*alignment of that direction with the incentive* only appears **late**. That is the
represent-vs-recruit thesis rendered as a within-network time-course: the choice is
represented before it is grounded in the incentive. This is the depth-axis twin of Layer C's
position-axis "deferred integration," and together they are a strong, novel mechanistic
statement.

---

## 2c. GPT-OSS: the MoE-native test, and why there is no depth transition

The dense fusion story (early-separate → late-fuse) does not transfer to GPT-OSS, and it is
worth being explicit about why rather than waving at it. GPT-OSS is read at the *post-reasoning
commit token* and is a mixture-of-experts, so two hypotheses could explain its flat profile:
(i) its strategic computation is in the *router*, not the residual stream this analysis uses;
or (ii) by the commit token the incentive is already bound at every depth, because the binding
happened earlier — across the reasoning trace (token/time), not across layers.

I tested (i) directly by running the identical geometry on the per-layer **router gate-logit
vectors** (128-dim; the substrate behind paper panel d), head-to-head with the residual stream
(`oss_router_fusion.py`, `figures/fig_oss_router_fusion.png`, `tables/oss_router_fusion.csv`).
Result: the router behaves **like the residual, not differently** — no early→late transition in
either; the incentive and choice axes are already aligned (angle ~10–25°) from the shallowest
layers. Under the model's *own empirical* opponent belief the angle is below the null at nearly
every layer in both substrates (residual 36/36, router 12/14); under the *objective* incentive
both are weaker and scattered (residual 10/36, router 3/14). So the router is **not** the missing
piece — hypothesis (i) is rejected. What is left is (ii) plus a second, interesting fact:
GPT-OSS's choice aligns far more cleanly with the incentive computed under *its own* belief than
with the objective one — the fingerprint of a model that actually mentalises the opponent
(consistent with its deeper level-k behaviour and the Coricelli-Nagel bridge below).

Takeaway for the paper: do **not** put GPT-OSS on the dense depth-transition claim. State that
for a reasoning model read at commit, incentive and choice are already fused across depth, in
residual *and* router, most strongly against its own belief; the binding for such a model unfolds
across the reasoning trace, not across layers — which is exactly the sequence-position ("deferred
integration") axis Layer C flags. Confirming *where* along the reasoning trace it binds needs the
per-token `_seq` residuals, which the commit-token capture did not store (a re-capture, same one
Layer C's discriminating test already requires).

## 3. Neuroscience relations — how to cash them in (your associative-areas idea)

Frame all of these as **shared computational problem + shared workflow (define task → record
→ decode → perturb), now with full observability** — not homology. That is both defensible
and the actual novelty. Four bridges, each tied to a specific result:

**Bridge A — Associative integration / feature binding (your idea, and the best new hook).**
In cortex, separate streams (e.g. what/where, vision/movement) are carried independently and
integrated in associative areas; binding is a late, convergent operation. Your data now show
the LLM analogue directly: incentive and decision occupy near-orthogonal subspaces early and
**bind into a shared subspace in the late layers** (§2). This gives the paper a principled
reason to read across depth ("we watched the choice bind to the incentive") and a crisp,
citable metaphor — the late layers act as an *associative/binding stage* for value and action.
This is the single most novel neuroscience contribution available to you and it is your idea.

**Bridge B — Co-representation → commitment.** Cisek's affordance competition and Gold &
Shadlen's accumulation-to-bound describe competing options held in parallel then resolved to
one. Layer B depth-crystallisation + my angle collapse + Layer C's late projection are the
LLM version: two candidate moves co-represented, a decision axis sharpening across depth,
commitment at the answer slot / final channel. Cite where you introduce the depth panels.

**Bridge C — Depth of strategic reasoning has a substrate (Coricelli & Nagel 2009; Nagel
2018).** Your tightest, most authentic link (Nagel is a co-author; the human data is that
lineage). Two results are the analogue: (i) the depth×determinism bounded-rationality map
places humans with the shallow dense models and GPT-OSS alone at depth — a cross-species
placement on the same reasoning-depth axis mPFC encodes; (ii) Layer C Result 2 — **opponent
payoffs move the decision only in Qwen-Instruct** (−0.65, CI excludes 0; base and Llama n.s.)
— is the token-level theory-of-mind / level-2 signature, the mPFC-linked "I think that you
think" operation, localised to tokens. Note this mentalising signature is *narrower* than
recruitment (Instruct only), which is worth saying honestly.

**Bridge D — Representation vs read-out (the recruitment gate itself).** Systems neuroscience
separates *what a population encodes* from *what a downstream area reads out* — the reason
choice-probability and causal-necessity tests exist alongside decoding. Recruitment is the
LLM instantiation of that read-out gate; that instruction-tuning *installs* it (graded,
§1a) is a learned gating of a pre-existing representation. This is the principled home for
your "decodability ≠ use" guardrail — stated with this citation it sounds principled, not
defensive.

One caution: do not call the residual/router sites homologous to mPFC. The binding-stage
language (Bridge A) is the safe, strong version.

---

## 4. Other unmined leads (ranked by value / effort)

1. **Cue axis fuses at the same associative layer? (tests the abstract's "same gate" claim
   directly.)** The abstract promises dispositions are "represented orthogonally, recruited
   through the same gate." You can *show* it: build the disposition axis per layer (cued −
   baseline residual) and test whether its angle to the decision axis collapses at the *same*
   depth as the incentive axis. If yes, "the same gate" stops being an assertion and becomes a
   figure. Effort: medium (needs per-layer cue residuals; the cue cache currently stores only
   the deepest layer, so a re-extract from `acts.npz` is required). Highest scientific upside.

2. **Re-run the Layer B recruitment bridge for Llama at the final pre-choice token.** Cheap,
   and it resolves the three-way Llama disagreement (§1b) that a referee *will* find. Until
   done, soften every "Llama doesn't recruit" sentence.

3. **Upgrade Fig 2c's null to the permutation null.** Small change, removes a real referee
   attack surface, and it is what licenses the depth-transition story. (§2)

4. **Own-vs-opponent recruitment as a depth trajectory.** Layer C gives the opponent-payoff
   effect only at L79. Computing it across depth would show *when* mentalising enters — a
   direct Coricelli-Nagel depth analogue. Effort: medium.

5. **"Build-then-collapse" for the base model, quantified.** The depth curves hint the base
   model represents (decision decodable early) but never binds (angle stays in-null under the
   objective incentive). If you want the base-as-weak-case story, this is the figure that
   makes it, stated with the belief-basis caveat.

---

## 5. Risk register (what a referee pushes on — most already in STORYLINE_STRATEGY §8)

- **Causal leg is incomplete / dense causal expectation is null.** Do not hang the thesis on
  steering. Load-bearing evidence = convergence of behavioural λ, the partial recruitment
  bridge, the λ_lens dissociation, **and now the depth-fusion geometry** — four non-causal
  windows + one controlled model comparison. Report steering as bounded confirmation +
  limitation.
- **Base-vs-instruct geometry is belief-basis dependent** (§2 caveat). Corroborates, does not
  carry.
- **n=4 models** for cross-model geometry — keep angle/onset↔λ relations descriptive; the
  inferential unit is games (144), which is fine within-model.
- **Angle null** — fixed by the permutation null (§2); use it.
- **"Canonical action" circularity** — the precedence rule is fixed and theory-derived, human
  data scored on the same axis, partial bridge controls for the objective gap. One sentence
  neutralises it.
- **In-sample axes in 8192-dim over-fit** — the permutation null is computed the same in-sample
  way, so it absorbs the over-fit; that is why the null is wide. State this.

---

## 6. Concrete next actions

1. Absorb §1a–1c into the draft (graded recruitment; Llama caveat; deferred-integration
   unification of depth + position).
2. Add the depth-fusion transition (§2) as a panel in the Layer B figure or a short new
   figure; switch the angle null to the permutation null.
3. Run lead #1 (cue-axis fusion at the same layer) — it directly earns the abstract's "same
   gate" clause.
4. Do lead #2 (Llama bridge at final token) before freezing Layer B text.
5. Cash in Bridge A (associative binding stage) at the depth panels and in the Discussion;
   Bridge C (Coricelli-Nagel) at the own-vs-opponent result.

Artifacts: `analysis/fusion_associative/` — `build_fusion_figures.py` (one self-contained,
repo-relative script that regenerates everything: `python analysis/fusion_associative/build_fusion_figures.py`);
`tables/fusion_depth_table.csv` (uniform), `tables/fusion_depth_empirical.csv`,
`tables/fusion_summary.csv`; `figures/fig_fusion_depth.{png,pdf}`,
`figures/fig_represent_vs_recruit_depth.png`. Depends on the residual cache in
`analysis/layer_b/_data/baseline/` (built by `analysis/layer_b/build_residual_cache.py`).
