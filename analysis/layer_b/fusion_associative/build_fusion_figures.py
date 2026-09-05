#!/usr/bin/env python3
"""Depth-resolved decision-incentive geometry: the "associative fusion" transition.

Self-contained, repo-relative regeneration of every table and figure in
`analysis/fusion_associative/`. Replicates the paper's Layer-B geometry recipe
(analysis/layer_b/lib.py + build_geometry.py) at EVERY captured residual layer
instead of only the deepest, and adds a permutation null so "non-orthogonal" is tested
against the correct finite-sample, target-correlation-preserving baseline rather than the
theoretical 90 deg.

Axes (raw residual space, per layer L, P1-baseline decision-slot residuals):
  d_dec = unit(mean X[aligned==1] - mean X[aligned==0])            # decision axis
  d_inc = unit(columnwise centered OLS slope of X on Delta1c)      # incentive axis
  angle = arccos(clip(<d_dec, d_inc>, -1, 1))                      # matches lib.angle_deg

Delta1c = canonical-signed level-1 incentive gap, computed under two beliefs:
  * uniform q=0.5  -> objective / model-independent stimulus incentive  (PRIMARY)
  * empirical q    -> model's own P2-baseline opponent act0 rate        (ROBUSTNESS)

Permutation null: permute (aligned, Delta1c) JOINTLY across games, keeping all four
counterbalanced presentation forms of a game together (destroys the residual<->target
link while preserving both the decision<->incentive target correlation and within-game
dependence), recompute both axes as two matmuls, take the angle. p = one-sided fraction
of null <= observed.

Inputs (must exist; built by analysis/layer_b/build_residual_cache.py):
  analysis/layer_b/_data/baseline/{meta_<model>.parquet, X_<model>_l<L>.npy}
  data/games/taxonomy/equivalence_per_canonical.csv   (canonical_8vec)
  data/games/game_features.csv                        (canonical_action_p1)
Outputs (written next to this script):
  tables/fusion_depth_table.csv       (uniform belief, all layers)
  tables/fusion_depth_empirical.csv   (empirical belief, all layers)
  tables/fusion_summary.csv           (per-model onset / significance summary)
  tables/gptoss_capture_site_sensitivity.csv (all vs pure-row vs complete-pure-game geometry)
  figures/fig_fusion_depth.{png,pdf}          (2 rows x 4 models: angle vs depth + null)
  figures/fig_represent_vs_recruit_depth.png  (decodable-early vs aligned-late, Qwen pair)
  figures/fig_gptoss_capture_site_sensitivity.{png,pdf}

Run from anywhere:  python analysis/fusion_associative/build_fusion_figures.py
"""
from __future__ import annotations
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from strategic_anatomy.config import data_root, game_features_csv, results_root, substrate_root, taxonomy_dir
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- repo-relative paths (this file: analysis/fusion_associative/build_fusion_figures.py) ----
HERE = Path(__file__).resolve().parent
# phase-4: git does not track empty directories, so figures/ and tables/ do not exist in a
# fresh clone and savefig() would raise FileNotFoundError. Create them up front.
(HERE / "figures").mkdir(parents=True, exist_ok=True)
(HERE / "tables").mkdir(parents=True, exist_ok=True)
ROOT = HERE.parents[2]
CACHE = data_root() / "layer_b_cache" / "baseline"
FEAT = game_features_csv()
EQUIV = taxonomy_dir() / "equivalence_per_canonical.csv"
TAB = results_root() / "layer_b" / "fusion"
FIG = HERE / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MODELS = ["qwen_instruct", "qwen", "llama31_instruct", "gptoss"]
GPT = "gptoss"
DENSE = [m for m in MODELS if m != GPT]
#: GPT-OSS uniform-site recapture cache, built by recap_cache.py in this directory.
RECAP_CACHE = data_root() / "layer_b_cache" / "recap_baseline"
#: The amended primary GPT-OSS scope (docs/METHODS.md, uniform-site amendment 2026-07-13).
SCOPE_UNIFORM = "uniform_transition_policy"
RECRUIT_TAB = results_root() / "layer_b" / "recruitment"
NICE = {"qwen_instruct": "Qwen2.5-Instruct", "qwen": "Qwen2.5 (base)",
        "llama31_instruct": "Llama-3.1-Instruct", "gptoss": "GPT-OSS-120B"}
COL = {"qwen_instruct": "#1b7837", "qwen": "#a6611a",
       "llama31_instruct": "#2166ac", "gptoss": "#762a83"}
N_PERM = 200
SEED = 20260702


# ---- geometry (math identical to analysis/layer_b/lib.py) -------------------------
def _unit(v):
    v = np.asarray(v, np.float64); n = float(np.linalg.norm(v))
    return v / n if n > 0 else np.zeros_like(v)


def _norm_cols(M):
    n = np.linalg.norm(M, axis=0); n[n == 0] = 1.0
    return M / n


def decision_axis(X, a):
    y = np.asarray(a, float); k = np.isfinite(y); Xk, yk = X[k], y[k].astype(int)
    return _unit(Xk[yk == 1].mean(0) - Xk[yk == 0].mean(0)) if len(np.unique(yk)) >= 2 else np.zeros(X.shape[1])


def incentive_axis(X, d):
    d = np.asarray(d, float); k = np.isfinite(d)
    if k.sum() < 5 or np.nanstd(d[k]) == 0:
        return np.zeros(X.shape[1])
    Xc = X[k] - X[k].mean(0); xc = d[k] - d[k].mean()
    return _unit((Xc.T @ xc) / float(xc @ xc))


def angle_deg(a, b):
    a, b = _unit(a), _unit(b)
    return np.nan if (not a.any() or not b.any()) else float(np.degrees(np.arccos(np.clip(float(a @ b), -1, 1))))


# ---- uniform-site recap estimator (LOCKED; ported verbatim from the 2026-07-13 generator)
# `oss_migration/recovered_20260713/recap_variants_freeze.py` in the private repo is the
# source of truth for every line below. Only path literals and write destinations were
# rewritten; the estimator is byte-verbatim, because the committed p-values are quantised in
# 1/201 steps and therefore encode the exact permutation stream.
#
# Why these two helpers exist alongside _unit/angle_deg instead of reusing them: at the
# uniform capture site every row shares one token, so the layer-0 residual is *exactly*
# invariant across rows and the centered matrix is identically zero. The release angle_deg
# returns NaN for a zero axis, which makes run() drop the layer -- correct for the dense
# models, whose layer-0 slot token is likewise shared (that is why their tables start at L1).
# The recovered generator instead returns arccos(0) = 90 deg, emits the row, AND consumes a
# full 200-permutation block from the generator. Dropping it would shift every subsequent
# layer's p-value, so the recap path must keep the recovered convention.
def _unit_recap(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def angle_deg_recap(a, b):
    a, b = _unit_recap(a), _unit_recap(b)
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1, 1))))


def policy_axis(X, p):
    """Unit centered OLS slope of X on the graded stated-policy target P(canonical).

    The same estimator as :func:`incentive_axis` -- on this substrate the decision target is
    continuous (pure -> 0/1, stated-mixed -> the stated probability), so the class-mean
    :func:`decision_axis` of the answer-slot analysis does not apply.
    """
    return incentive_axis(X, p)


def null_angles_policy(X, p, d, groups, forms, rng, n_perm=N_PERM):
    """Joint game-block permutation null with the slope form on BOTH axes.

    Twin of :func:`null_angles`; differs only in that the policy axis is a centered slope on
    a graded target rather than a class-mean contrast, so both axes use the same matmul.
    """
    Xc = X - X.mean(0)
    perms = _game_block_permutations(groups, forms, rng, n_perm=n_perm)
    P = (p[perms] - p[perms].mean(1, keepdims=True)).T
    D = (d[perms] - d[perms].mean(1, keepdims=True)).T
    cos = np.clip(np.einsum("dp,dp->p", _norm_cols(Xc.T @ P), _norm_cols(Xc.T @ D)), -1, 1)
    return np.degrees(np.arccos(cos))


def _game_block_permutations(groups, forms, rng, n_perm=N_PERM):
    """Map each target game's rows to one source game, matching presentation forms.

    The returned integer array can index a row-aligned target vector directly.  Every
    permutation therefore moves a game's retained presentation block together rather
    than treating repeated presentations as independent rows. Games are permuted only
    within identical form-availability patterns; the primary analysis has four forms
    for every game, while the 525-row pure-only sensitivity has variable patterns.
    """
    groups, forms = np.asarray(groups), np.asarray(forms)
    if groups.ndim != 1 or forms.ndim != 1 or len(groups) != len(forms):
        raise ValueError("groups and forms must be same-length one-dimensional arrays")

    game_ids = pd.unique(groups)
    blocks_by_pattern, covered = {}, []
    for game_id in game_ids:
        idx = np.flatnonzero(groups == game_id)
        order = np.argsort(forms[idx], kind="stable")
        idx = idx[order]
        game_forms = tuple(forms[idx].tolist())
        if len(set(game_forms)) != len(game_forms):
            raise ValueError(f"duplicate counterbalance forms within game {game_id}")
        blocks_by_pattern.setdefault(game_forms, []).append(idx)
        covered.extend(idx.tolist())

    if len(covered) != len(groups) or len(np.unique(covered)) != len(groups):
        raise ValueError("game blocks must partition every analysis row exactly once")

    perms = np.empty((n_perm, len(groups)), dtype=int)
    for b in range(n_perm):
        for pattern_blocks in blocks_by_pattern.values():
            blocks = np.stack(pattern_blocks)
            target_rows = blocks.ravel()
            source_blocks = blocks[rng.permutation(len(blocks))]
            perms[b, target_rows] = source_blocks.ravel()
    return perms


def null_angles(X, a, d, groups, forms, rng, n_perm=N_PERM):
    """Vectorised joint game-block permutation null for both geometry axes."""
    n = len(a); n1 = int((a == 1).sum()); n0 = int((a == 0).sum()); Xc = X - X.mean(0)
    perms = _game_block_permutations(groups, forms, rng, n_perm=n_perm)
    al = a[perms]; W = np.where(al == 1, 1.0 / n1, -1.0 / n0).T
    dv = d[perms]; D = (dv - dv.mean(1, keepdims=True)).T
    cos = np.clip(np.einsum("dp,dp->p", _norm_cols(X.T @ W), _norm_cols(Xc.T @ D)), -1, 1)
    return np.degrees(np.arccos(cos))


# ---- per-game canonical-signed incentive gap ------------------------------------------
_V = (pd.read_csv(EQUIV).set_index("bruns_name")["canonical_8vec"]
      .apply(lambda s: [float(x) for x in str(s).split(",")]).to_dict())
_CA = pd.read_csv(FEAT).set_index("game_code")["canonical_action_p1"].to_dict()


def d1c_map(qmap=None):
    """canonical-signed Delta1c per game. qmap None -> uniform q=0.5 (all games); else a
    per-game empirical belief dict -> games absent from qmap get NaN (dropped), matching the
    exclusion in lib.empirical_delta1c."""
    out = {}
    for g, v in _V.items():
        if qmap is None:
            q = 0.5
        elif g in qmap:
            q = float(qmap[g])
        else:
            out[g] = float("nan"); continue
        d1 = q * v[0] + (1 - q) * v[1] - q * v[2] - (1 - q) * v[3]   # EU1(act0)-EU1(act1)
        out[g] = d1 if int(_CA[g]) == 0 else -d1                      # sign toward canonical
    return out


def qhat_p2(model):
    """Empirical P2-baseline act0 rate per game, replicating lib.empirical_delta1c's
    behavioural source: dense = parse_ok-gated decoded_action; GPT-OSS = realized_action with
    commit_type='none' dropped."""
    cols = ["game_code", "player", "condition", "decoded_action"]
    cols += ["realized_action", "commit_type"] if model == "gptoss" else ["parse_ok"]
    files = glob.glob(str(substrate_root() / model / "*" / "results.parquet"))
    R = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    b = R[(R.player == 2) & (R.condition == "baseline")].copy()
    if model == "gptoss":
        b = b[b["commit_type"] != "none"]
        b["a"] = pd.to_numeric(b["realized_action"], errors="coerce")
    else:
        b = b[b["parse_ok"] == True]  # noqa: E712  (parse_ok may be object/bool)
        b["a"] = pd.to_numeric(b["decoded_action"], errors="coerce")
    b = b[b["a"].isin([0, 1])]
    return b.groupby("game_code")["a"].apply(lambda s: float((s == 0).mean())).to_dict()


def _attach_gptoss_commit_type(meta):
    """Join capture-site provenance to the 576 cached GPT-OSS baseline rows."""
    # Same defect class as lib.attach_baseline_commit_type: a cache built by the current
    # build_residual_cache.py already carries commit_type, and the merge below would then
    # collide into commit_type_x/_y. Drop the cache-borne copy first.
    meta = meta.drop(columns=["commit_type"], errors="ignore")
    files = glob.glob(str(substrate_root() / "gptoss" / "*" / "results.parquet"))
    parts = []
    for f in files:
        d = pd.read_parquet(
            f, columns=["game_code", "player", "condition", "counterbalance_id", "commit_type"]
        )
        parts.append(d[(d.player == 1) & (d.condition == "baseline")][
            ["game_code", "counterbalance_id", "commit_type"]
        ])
    commit = pd.concat(parts, ignore_index=True)
    out = meta.merge(
        commit,
        on=["game_code", "counterbalance_id"],
        how="left",
        validate="one_to_one",
    )
    if len(out) != len(meta) or out.commit_type.isna().any():
        raise AssertionError("GPT-OSS commit-type join did not cover the residual cache")
    return out


def _scope_mask(meta, model, scope):
    if model != "gptoss":
        return np.ones(len(meta), dtype=bool), "parsed_generated_choice"
    if scope == "heterogeneous_all":
        return np.ones(len(meta), dtype=bool), scope
    pure = meta.commit_type.eq("pure")
    if scope == "pure_commitment":
        return pure.to_numpy(), scope
    if scope == "pure_complete_games":
        complete = pure.groupby(meta.game_code).transform("all")
        return complete.to_numpy(), scope
    if scope == SCOPE_UNIFORM:
        # Uniform-site recapture population: the 541 rows carrying a usable stated policy
        # (525 pure + 16 stated-mixed). The 35 harness-imputed default_uniform rows are
        # flagged out and never imputed. Only reachable from the recap cache, whose meta
        # carries use_in_neural_target.
        return meta["use_in_neural_target"].eq(1).to_numpy(), scope
    raise ValueError(f"unknown GPT-OSS geometry scope: {scope}")


def _load(model):
    meta = pd.read_parquet(CACHE / f"meta_{model}.parquet").reset_index(drop=True)
    if model == "gptoss":
        meta = _attach_gptoss_commit_type(meta)
    aligned = (pd.to_numeric(meta["decoded_action"], errors="coerce")
               == pd.to_numeric(meta["canonical_action_p1"], errors="coerce")).astype(float).to_numpy()
    layers = sorted(int(re.search(r"_l(\d+)\.npy", f).group(1))
                    for f in glob.glob(str(CACHE / f"X_{model}_l*.npy")))
    return meta, aligned, layers


def _load_recap():
    """Recap cache meta (locked row order) + the sorted residual layer list."""
    meta_path = RECAP_CACHE / f"meta_{GPT}.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"GPT-OSS uniform-site recap cache not found at {RECAP_CACHE}.\n"
            f"Build it first:\n"
            f"    uv run python analysis/layer_b/fusion_associative/recap_cache.py"
        )
    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    layers = sorted(int(re.search(r"_l(\d+)\.npy", f).group(1))
                    for f in glob.glob(str(RECAP_CACHE / f"X_{GPT}_l*.npy")))
    return meta, layers


def run_recap(belief, *, scope=SCOPE_UNIFORM):
    """GPT-OSS decision-incentive geometry on the uniform-site recapture.

    The amended primary GPT-OSS scope. Same geometry as :func:`run`, with two differences
    forced by the substrate: the read-out site is uniform across rows (so the decision target
    is the model's graded STATED policy rather than a realized binary choice), and the
    population is the 541 rows with ``use_in_neural_target == 1``.

    LOCKED, verbatim port -- see the helper comment above. In particular layer 0 is computed,
    emitted, and consumes a permutation block; the caller drops it from the depth tables and
    keeps it in the capture-site sensitivity table.
    """
    rng = np.random.default_rng(SEED)
    meta, layers = _load_recap()
    dmap = d1c_map(None if belief == "uniform" else qhat_p2(GPT))
    dvec = meta["game_code"].map(dmap).astype(float).to_numpy()
    pvec = meta["p_canonical"].to_numpy(float)
    population, row_scope = _scope_mask(meta, GPT, scope)
    mask = np.isfinite(pvec) & np.isfinite(dvec) & population
    p, dv = pvec[mask], dvec[mask]
    groups = meta.loc[mask, "game_code"].to_numpy()
    forms = meta.loc[mask, "counterbalance_id"].to_numpy()
    binm = np.isin(p, [0.0, 1.0])   # pure rows only: the Cohen d is a two-class contrast
    Lmax = max(layers)
    rows = []
    for L in layers:
        X = np.load(RECAP_CACHE / f"X_{GPT}_l{L}.npy").astype(np.float64)[mask]
        if X.std() == 0:
            continue
        d_pol, d_inc = policy_axis(X, p), incentive_axis(X, dv)
        ang = angle_deg_recap(d_pol, d_inc)
        if not np.isfinite(ang):
            continue
        proj = (X - X.mean(0)) @ d_pol
        gain_r = float(np.corrcoef(dv, proj)[0, 1]) if np.std(proj) > 0 else np.nan
        g1, g0 = proj[binm & (p == 1)], proj[binm & (p == 0)]
        pooled = np.sqrt(0.5 * (g1.var() + g0.var())) if len(g1) and len(g0) else np.nan
        dec_d = float((g1.mean() - g0.mean()) / pooled) if pooled and pooled > 0 else np.nan
        nu = null_angles_policy(X, p, dv, groups, forms, rng)
        lo, med, hi = np.nanpercentile(nu, [2.5, 50, 97.5])
        pv = float((np.sum(nu <= ang) + 1) / (N_PERM + 1))
        rows.append(dict(model=GPT, layer=L, depth_frac=L / Lmax, angle_deg=ang,
                         null_lo=lo, null_med=med, null_hi=hi, perm_p=pv, sig=pv < 0.05,
                         gain_r=gain_r, dec_cohen_d=dec_d, belief=belief,
                         row_scope=row_scope, n_obs=int(mask.sum()),
                         n_games=int(pd.unique(groups).size)))
    out = pd.DataFrame(rows)
    print(f"[{belief:9s}] {GPT} (recap/{row_scope}): {len(out)} layers, "
          f"final angle={out.angle_deg.iloc[-1]:.1f}")
    return out


def run(belief, *, models=MODELS, gptoss_scope="pure_complete_games"):
    """belief in {'uniform', 'empirical'}; returns per-layer dataframe with angle + null + diagnostics."""
    rows = []
    for m in models:
        rng = np.random.default_rng(SEED)
        meta, aligned, layers = _load(m)
        dmap = d1c_map(None if belief == "uniform" else qhat_p2(m))
        dvec = meta["game_code"].map(dmap).astype(float).to_numpy()
        population, row_scope = _scope_mask(meta, m, gptoss_scope)
        mask = np.isfinite(aligned) & np.isfinite(dvec) & population
        al, dv = aligned[mask], dvec[mask]
        groups = meta.loc[mask, "game_code"].to_numpy()
        forms = meta.loc[mask, "counterbalance_id"].to_numpy()
        Lmax = max(layers)
        for L in layers:
            X = np.load(CACHE / f"X_{m}_l{L}.npy").astype(np.float64)[mask]
            if X.std() == 0:
                continue
            d_dec, d_inc = decision_axis(X, al), incentive_axis(X, dv)
            ang = angle_deg(d_dec, d_inc)
            if not np.isfinite(ang):
                continue
            proj = (X - X.mean(0)) @ d_dec
            gain_r = float(np.corrcoef(dv, proj)[0, 1]) if np.std(proj) > 0 else np.nan
            g1, g0 = proj[al == 1], proj[al == 0]
            pooled = np.sqrt(0.5 * (g1.var() + g0.var())) if len(g1) and len(g0) else np.nan
            dec_d = float((g1.mean() - g0.mean()) / pooled) if pooled and pooled > 0 else np.nan
            nu = null_angles(X, al, dv, groups, forms, rng)
            lo, med, hi = np.nanpercentile(nu, [2.5, 50, 97.5])
            p = float((np.sum(nu <= ang) + 1) / (N_PERM + 1))
            rows.append(dict(model=m, layer=L, depth_frac=L / Lmax, angle_deg=ang,
                             null_lo=lo, null_med=med, null_hi=hi, perm_p=p, sig=p < 0.05,
                             gain_r=gain_r, dec_cohen_d=dec_d, belief=belief,
                             row_scope=row_scope, n_obs=int(mask.sum()),
                             n_games=int(pd.unique(groups).size)))
        print(f"[{belief:9s}] {m}: {rows[-1]['layer']} layers, final angle={rows[-1]['angle_deg']:.1f}")
    return pd.DataFrame(rows)


def onset(df, m, run_len=3):
    s = df[(df.model == m) & (df.layer > 0)].sort_values("layer").reset_index(drop=True)
    sig = (s.perm_p < 0.05).to_numpy()
    for i in range(len(sig) - run_len + 1):
        if sig[i] and sig[i:i + run_len].all():
            return int(s.loc[i, "layer"]), round(float(s.loc[i, "depth_frac"]), 2)
    return None, None


def summarise(U, E):
    rows = []
    for m in MODELS:
        su, se = U[(U.model == m) & (U.layer > 0)], E[(E.model == m) & (E.layer > 0)]
        lu, le = su[su.depth_frac >= 0.6], se[se.depth_frac >= 0.6]
        rows.append(dict(model=NICE[m],
                         final_angle=round(su.sort_values("layer").iloc[-1].angle_deg, 1),
                         onset_depth_uniform=onset(U, m)[1], onset_depth_empirical=onset(E, m)[1],
                         sig_late_uniform=f"{int(lu.sig.sum())}/{len(lu)}",
                         sig_late_empirical=f"{int(le.sig.sum())}/{len(le)}",
                         minp_uniform=round(su.perm_p.min(), 3), minp_empirical=round(se.perm_p.min(), 3)))
    return pd.DataFrame(rows)


def fig_depth(U, E):
    data = {"Uniform belief (objective incentive, q=0.5)": U,
            "Empirical belief (model's own opponent belief)": E}
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=True)
    for r, (tag, df) in enumerate(data.items()):
        for c, m in enumerate(MODELS):
            ax = axes[r, c]
            s = df[(df.model == m) & (df.layer > 0)].sort_values("depth_frac")
            x = s.depth_frac.to_numpy()
            ax.fill_between(x, s.null_lo, s.null_hi, color="0.86", label="permutation null 95%")
            ax.plot(x, s.null_med, color="0.55", lw=1, ls="--", label="null median")
            ax.plot(x, s.angle_deg, color=COL[m], lw=2.3, label="observed")
            sig = s[s.perm_p < 0.05]
            ax.scatter(sig.depth_frac, sig.angle_deg, color=COL[m], s=20, zorder=5,
                       edgecolor="k", lw=0.4, label="fused (p<0.05)")
            ax.axhline(90, color="k", lw=0.7, ls=":", alpha=0.6)
            ax.set_ylim(0, 135); ax.set_xlim(0, 1)
            if r == 0:
                title = NICE[m]
                if m == "gptoss":
                    title += "\n(108 all-pure games)"
                ax.set_title(title, fontsize=12)
            if r == 1:
                ax.set_xlabel("relative depth")
            if c == 0:
                ax.set_ylabel(f"{tag}\n\ndecision-incentive angle (deg)", fontsize=9)
    axes[0, 3].legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle("Decision and incentive axes are near-orthogonal early and FUSE in late layers "
                 "(the 'associative' transition). Dots: angle below permutation null.", fontsize=13, y=0.98)
    fig.text(0.5, 0.005, "90 deg dotted = classic orthogonal null (independent random vectors); "
             "grey band = finite-sample permutation null. Fusion = observed below the band.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(FIG / "fig_fusion_depth.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / "fig_fusion_depth.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_represent_recruit(U):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for j, m in enumerate(["qwen_instruct", "qwen"]):
        s = U[(U.model == m) & (U.layer > 0)].sort_values("depth_frac")
        a = ax[j]
        a.plot(s.depth_frac, s.dec_cohen_d, color="#d95f02", lw=2, label="decision decodability (Cohen d)")
        a2 = a.twinx()
        a2.plot(s.depth_frac, 90 - s.angle_deg, color=COL[m], lw=2, label="incentive alignment (90-angle)")
        a.set_title(NICE[m]); a.set_xlabel("relative depth")
        if j == 0:
            a.set_ylabel("decision decodability (Cohen d)", color="#d95f02")
        a2.set_ylabel("incentive alignment (90-angle, deg)", color=COL[m])
        a.set_ylim(0, 2); a2.set_ylim(0, 70)
    fig.suptitle("Represented early, recruited late: the choice is decodable in early layers, "
                 "but aligns with incentive only in late layers", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_represent_vs_recruit_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_gptoss_capture_sensitivity(S):
    """Show how heterogeneous capture sites alter GPT-OSS residual geometry."""
    labels = {
        "heterogeneous_all": "all rows (pure + mixed)",
        "pure_commitment": "pure rows (pattern-stratified null)",
        "pure_complete_games": "108 all-pure games",
    }
    colors = {
        "heterogeneous_all": "#b35806",
        "pure_commitment": "#8073ac",
        "pure_complete_games": "#542788",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for scope in labels:
        s = S[(S.row_scope == scope) & (S.layer > 0)].sort_values("depth_frac")
        ax.plot(s.depth_frac, s.angle_deg, lw=2, color=colors[scope], label=labels[scope])
        sig = s[s.perm_p < .05]
        ax.scatter(sig.depth_frac, sig.angle_deg, s=18, color=colors[scope],
                   edgecolor="white", linewidth=.3, zorder=4)
    ax.axhline(90, color="0.35", lw=.8, ls=":")
    ax.set(xlim=(0, 1), ylim=(0, 135), xlabel="relative depth",
           ylabel="decision-incentive angle (deg)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("GPT-OSS geometry is sensitive to heterogeneous commitment sites", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_gptoss_capture_site_sensitivity.png", dpi=300)
    fig.savefig(FIG / "fig_gptoss_capture_site_sensitivity.pdf")
    plt.close(fig)


#: recruitment_geometry_depth.csv is this table under other column names.
RECRUIT_RENAME = {"angle_deg": "angle_decision_incentive_deg", "null_med": "null_median_deg",
                  "null_lo": "null_lo_deg", "null_hi": "null_hi_deg", "sig": "below_null"}
RECRUIT_COLS = ["model", "layer", "angle_decision_incentive_deg", "null_median_deg",
                "null_lo_deg", "null_hi_deg", "perm_p", "below_null", "row_scope", "n_obs",
                "n_games", "n_perm", "independent_shuffle_median_deg"]


def write_recruitment_geometry(U):
    """Emit recruitment_geometry_depth.{csv,parquet} as a rename of the uniform depth table.

    The recruitment panel's depth-resolved geometry is not an independent estimate: it is
    this module's uniform-belief table under other column names, for every model. The
    recruitment tree's own geometry path sweeps a `selected_layers` subset and computes no
    permutation null, so it cannot produce the committed table -- which is why
    `recruitment/build_all.py` no longer writes this file and reads it instead.

    `independent_shuffle_median_deg` is the one column with no producer anywhere in the
    release: three scalars, one per dense model at its deepest layer, from a
    row-independent shuffle diagnostic whose generator was not carried into the release.
    They are carried forward from the existing file when present and left NaN otherwise;
    they are NOT regenerated here, and nothing downstream reads them.
    """
    G = U.rename(columns=RECRUIT_RENAME).copy()
    G["n_perm"] = N_PERM
    G["independent_shuffle_median_deg"] = np.nan
    # Committed row order: dense models alphabetically, then the GPT-OSS block last.
    out = pd.concat([G[G.model != GPT].sort_values(["model", "layer"]),
                     G[G.model == GPT].sort_values("layer")],
                    ignore_index=True)[RECRUIT_COLS]
    prior = RECRUIT_TAB / "recruitment_geometry_depth.csv"
    if prior.exists():
        P = pd.read_csv(prior)
        if "independent_shuffle_median_deg" in P.columns:
            keyed = P.set_index(["model", "layer"])["independent_shuffle_median_deg"]
            out["independent_shuffle_median_deg"] = keyed.reindex(
                pd.MultiIndex.from_arrays([out.model, out.layer])).to_numpy()
    RECRUIT_TAB.mkdir(parents=True, exist_ok=True)
    (RECRUIT_TAB / "_data").mkdir(parents=True, exist_ok=True)
    out.to_csv(prior, index=False)
    # figures.py:_read_panel prefers the parquet, so both must be written or the recruitment
    # panel silently renders a stale generation.
    out.to_parquet(RECRUIT_TAB / "_data" / "recruitment_geometry_depth.parquet", index=False)
    print(f"wrote recruitment_geometry_depth.{{csv,parquet}} ({len(out)} rows) -> {RECRUIT_TAB}")


def main():
    # GPT-OSS runs on the uniform-site recapture (the amended primary scope); the dense
    # models keep the answer-slot substrate. Layer 0 is dropped from the two depth tables
    # and kept in the capture-site sensitivity table -- the freeze-then-L0-removal sequence
    # of 2026-07-13, reproduced here in one pass.
    RU, RE = run_recap("uniform"), run_recap("empirical")
    U = pd.concat([run("uniform", models=DENSE), RU[RU.layer > 0]], ignore_index=True)
    U.to_csv(TAB / "fusion_depth_table.csv", index=False)
    E = pd.concat([run("empirical", models=DENSE), RE[RE.layer > 0]], ignore_index=True)
    E.to_csv(TAB / "fusion_depth_empirical.csv", index=False)
    GS = pd.concat(
        [run("uniform", models=[GPT], gptoss_scope=scope)
         for scope in ("heterogeneous_all", "pure_commitment", "pure_complete_games")] + [RU],
        ignore_index=True,
    )
    GS.to_csv(TAB / "gptoss_capture_site_sensitivity.csv", index=False)
    S = summarise(U, E); S.to_csv(TAB / "fusion_summary.csv", index=False)
    print("\n" + S.to_string(index=False))
    write_recruitment_geometry(U)
    fig_depth(U, E); fig_represent_recruit(U); fig_gptoss_capture_sensitivity(GS)
    print(f"\nwrote tables -> {TAB}\nwrote figures -> {FIG}")


if __name__ == "__main__":
    main()
