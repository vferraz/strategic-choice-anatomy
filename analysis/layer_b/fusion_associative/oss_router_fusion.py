#!/usr/bin/env python3
"""MoE-native fusion for GPT-OSS: decision-incentive geometry in ROUTER gate-logit space.

The residual-stream fusion analysis is a dense-model story read at the answer slot. GPT-OSS
is (i) read at the post-reasoning commit token and (ii) a mixture-of-experts, so much of its
strategic computation is carried by the ROUTER, not the residual stream (cf. paper panel d:
incentive sign decodes from gate logits at AUC 0.89). This script runs the *same* geometry
(decision axis vs incentive axis, permutation null) on the per-layer router gate-logit vectors
and compares it head-to-head with the residual stream, to see whether GPT-OSS shows the
fusion transition in the substrate where its computation actually lives.

  python analysis/fusion_associative/oss_router_fusion.py
"""
from __future__ import annotations
import sys
import glob
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
from strategic_anatomy.config import gptoss_recap_root, results_root, substrate_root
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
# phase-4: git does not track empty directories, so figures/ and tables/ do not exist in a
# fresh clone and savefig() would raise FileNotFoundError. Create them up front.
(HERE / "figures").mkdir(parents=True, exist_ok=True)
(HERE / "tables").mkdir(parents=True, exist_ok=True)
ROOT = HERE.parents[2]
from strategic_anatomy import paper_style as PS  # noqa: E402
PS.apply()
from analysis.layer_b.fusion_associative.build_fusion_figures import (  # noqa: E402  (reuse identical math)
    decision_axis, incentive_axis, angle_deg, null_angles, d1c_map, qhat_p2, CACHE,
    _attach_gptoss_commit_type, _scope_mask, _load_recap, angle_deg_recap, policy_axis,
    null_angles_policy, SCOPE_UNIFORM)

GPT = "gptoss"
#: Primary router path reads the uniform-site recapture -- the same substrate the residual
#: geometry moved to on 2026-07-13. Its router.npz keys are identical in form to the main
#: substrate's (p1_baseline_cb{cb}_l{L}_gate, 14 layers), so the regex below is unchanged.
GATE_DIR = gptoss_recap_root()
#: The superseded answer-slot substrate, still read by the sensitivity writer in main().
SENSITIVITY_GATE_DIR = substrate_root() / GPT
N_PERM = 200
SEED = 20260702


def load_router_gate(gate_dir=None):
    """Return {(game_code, cb, layer): gate_vec} for P1 baseline commit cells + sorted layers."""
    gate_dir = GATE_DIR if gate_dir is None else gate_dir
    store, layers = {}, set()
    pat = re.compile(r"^p1_baseline_cb(\d+)_l(\d+)_gate$")
    for f in sorted(glob.glob(str(gate_dir / "*" / "router.npz"))):
        game = Path(f).parent.name
        with np.load(f) as z:
            for k in z.files:
                mobj = pat.match(k)
                if mobj:
                    cb, L = int(mobj.group(1)), int(mobj.group(2))
                    store[(game, cb, L)] = z[k].astype(np.float64)
                    layers.add(L)
    return store, sorted(layers)


def build_matrix(store, meta, L):
    """Rows aligned to meta order; return X (n x 128), row-mask of present cells."""
    rows, present = [], []
    for _, r in meta.iterrows():
        key = (r["game_code"], int(r["counterbalance_id"]), L)
        v = store.get(key)
        present.append(v is not None)
        rows.append(v if v is not None else None)
    present = np.array(present)
    X = np.stack([rows[i] for i in np.where(present)[0]]) if present.any() else np.empty((0, 128))
    return X, present


def run(belief, *, scope="pure_complete_games", gate_dir=None):
    """Answer-slot substrate router geometry -- the SUPERSEDED capture site.

    Retained unchanged as the sensitivity writer (oss_router_fusion_sensitivity.csv). The
    primary table now comes from :func:`run_recap`.
    """
    gate_dir = SENSITIVITY_GATE_DIR if gate_dir is None else gate_dir
    rng = np.random.default_rng(SEED)
    meta = pd.read_parquet(CACHE / f"meta_{GPT}.parquet").reset_index(drop=True)
    meta = _attach_gptoss_commit_type(meta)
    aligned_all = (pd.to_numeric(meta["decoded_action"], errors="coerce")
                   == pd.to_numeric(meta["canonical_action_p1"], errors="coerce")).astype(float).to_numpy()
    dmap = d1c_map(None if belief == "uniform" else qhat_p2(GPT))
    dvec_all = meta["game_code"].map(dmap).astype(float).to_numpy()
    groups_all = meta["game_code"].to_numpy()
    forms_all = meta["counterbalance_id"].to_numpy()
    population, row_scope = _scope_mask(meta, GPT, scope)
    store, layers = load_router_gate(gate_dir)
    Lmax = max(layers)
    out = []
    for L in layers:
        X, present = build_matrix(store, meta, L)
        al, dv = aligned_all[present], dvec_all[present]
        groups, forms = groups_all[present], forms_all[present]
        keep = np.isfinite(al) & np.isfinite(dv) & population[present]
        X, al, dv = X[keep], al[keep], dv[keep]
        groups, forms = groups[keep], forms[keep]
        if X.shape[0] < 20 or X.std() == 0:
            continue
        ang = angle_deg(decision_axis(X, al), incentive_axis(X, dv))
        nu = null_angles(X, al, dv, groups, forms, rng, n_perm=N_PERM)
        lo, med, hi = np.nanpercentile(nu, [2.5, 50, 97.5])
        p = float((np.sum(nu <= ang) + 1) / (N_PERM + 1))
        out.append(dict(substrate="router_gate", belief=belief, layer=L, depth_frac=L / Lmax,
                        angle_deg=ang, null_lo=lo, null_med=med, null_hi=hi, perm_p=p,
                        sig=p < 0.05, n=int(X.shape[0]), n_games=int(pd.unique(groups).size),
                        row_scope=row_scope))
    return pd.DataFrame(out)


def run_recap(belief, *, scope=SCOPE_UNIFORM):
    """Router-gate geometry on the uniform-site recapture -- the primary table.

    The MoE twin of build_fusion_figures.run_recap: same population (541 in-target rows),
    same graded stated-policy target, same locked permutation null, read in gate-logit space
    instead of the residual stream. LOCKED, verbatim port of variants C/D of the 2026-07-13
    generator; the router layer list carries no layer 0, so unlike the residual path the
    generator is consumed over the 14 captured layers only.
    """
    rng = np.random.default_rng(SEED)
    meta, _ = _load_recap()
    dmap = d1c_map(None if belief == "uniform" else qhat_p2(GPT))
    dvec_all = meta["game_code"].map(dmap).astype(float).to_numpy()
    pvec_all = meta["p_canonical"].to_numpy(float)
    groups_all = meta["game_code"].to_numpy()
    forms_all = meta["counterbalance_id"].to_numpy()
    population, row_scope = _scope_mask(meta, GPT, scope)
    store, layers = load_router_gate(GATE_DIR)
    Lmax = max(layers)
    out = []
    for L in layers:
        X, present = build_matrix(store, meta, L)
        p, dv = pvec_all[present], dvec_all[present]
        groups, forms = groups_all[present], forms_all[present]
        keep = np.isfinite(p) & np.isfinite(dv) & population[present]
        X, p, dv = X[keep], p[keep], dv[keep]
        groups, forms = groups[keep], forms[keep]
        if X.shape[0] < 20 or X.std() == 0:
            continue
        ang = angle_deg_recap(policy_axis(X, p), incentive_axis(X, dv))
        nu = null_angles_policy(X, p, dv, groups, forms, rng, n_perm=N_PERM)
        lo, med, hi = np.nanpercentile(nu, [2.5, 50, 97.5])
        pv = float((np.sum(nu <= ang) + 1) / (N_PERM + 1))
        out.append(dict(substrate="router_gate", belief=belief, layer=L, depth_frac=L / Lmax,
                        angle_deg=ang, null_lo=lo, null_med=med, null_hi=hi, perm_p=pv,
                        sig=pv < 0.05, n=int(X.shape[0]), n_games=int(pd.unique(groups).size),
                        row_scope=row_scope))
    return pd.DataFrame(out)


def main():
    RU, RE = run_recap("uniform"), run_recap("empirical")
    R = pd.concat([RU, RE], ignore_index=True)
    R.to_csv(results_root() / "layer_b" / "fusion" / "oss_router_fusion.csv", index=False)
    RS = pd.concat(
        [run("uniform", scope=scope)
         for scope in ("heterogeneous_all", "pure_commitment", "pure_complete_games")],
        ignore_index=True,
    )
    RS.to_csv(results_root() / "layer_b" / "fusion" / "oss_router_fusion_sensitivity.csv", index=False)

    # residual gptoss from the main table (uniform + empirical)
    resU = pd.read_csv(results_root() / "layer_b" / "fusion" / "fusion_depth_table.csv")
    resU = resU[(resU.model == GPT) & (resU.layer > 0)].assign(substrate="residual", belief="uniform")
    resE = pd.read_csv(results_root() / "layer_b" / "fusion" / "fusion_depth_empirical.csv")
    resE = resE[(resE.model == GPT) & (resE.layer > 0)].assign(substrate="residual", belief="empirical")

    def summ(df, name):
        s = df[df.layer > 0]
        return (f"  {name:16s} sig {int(s.sig.sum()):2d}/{len(s):2d} | "
                f"min angle {s.angle_deg.min():4.1f} | min p {s.perm_p.min():.3f}")
    print("GPT-OSS decision-incentive fusion, by substrate (108 all-pure games):")
    print("[uniform]");  print(summ(resU, "residual")); print(summ(RU, "router gate"))
    print("[empirical]"); print(summ(resE, "residual")); print(summ(RE, "router gate"))

    # ---- figure: 2x2 substrate (rows) x belief (cols), GPT-OSS (NHB house style) ----
    grid = [[("residual · objective", resU, "a"),
             ("residual · own belief", resE, "b")],
            [("router gate · objective", RU, "c"),
             ("router gate · own belief", RE, "d")]]
    fig, axes = plt.subplots(2, 2, figsize=(PS.NHB_TEXTWIDTH_IN * 0.72, 4.3), sharey=True, sharex=True)
    for r in range(2):
        for c in range(2):
            ax = axes[r, c]
            name, df, letter = grid[r][c]
            s = df[df.layer > 0].sort_values("depth_frac")
            x = s.depth_frac.to_numpy()
            ax.fill_between(x, s.null_lo, s.null_hi, color="0.86", lw=0)
            ax.plot(x, s.null_med, color=PS.GREY, lw=0.8, ls=(0, (4, 2)))
            ax.plot(x, s.angle_deg, color="#9467bd", lw=1.6)
            sg = s[s.perm_p < 0.05]
            ax.scatter(sg.depth_frac, sg.angle_deg, color="#9467bd", s=11, zorder=5,
                       edgecolor="white", linewidths=0.3)
            ax.axhline(90, color=PS.INK, lw=0.6, ls=":", alpha=0.55)
            ax.set_xlim(0, 1); ax.set_ylim(0, 135); ax.set_xticks([0, 0.5, 1.0])
            ax.tick_params(labelsize=PS.NHB_FS_TICK)
            if r == 1:
                ax.set_xlabel("relative depth", fontsize=PS.NHB_FS_AXIS)
            PS.nhb_panel_title(ax, letter, name, title_x=0.075, fontsize=PS.NHB_FS_MINI_TITLE)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    for r in range(2):
        axes[r, 0].set_ylabel("decision–incentive\nangle (°)", fontsize=PS.NHB_FS_AXIS)
    axes[0, 0].text(0.03, 92, "orthogonal", fontsize=PS.NHB_FS_FOOT, color=PS.FAINT, va="top")
    axes[0, 0].text(0.03, 4, "fused", fontsize=PS.NHB_FS_FOOT, color=PS.FAINT, va="bottom")
    handles = [
        Line2D([0], [0], color="#9467bd", lw=1.6, label="observed angle"),
        Patch(facecolor="0.86", label="permutation null (95%)"),
        Line2D([0], [0], color=PS.GREY, lw=0.8, ls=(0, (4, 2)), label="null median"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9467bd",
               markeredgecolor="white", markersize=4, label="angle < null (p < 0.05)"),
    ]
    fig.legend(handles=handles, fontsize=PS.NHB_FS_LEGEND, loc="lower center", ncol=4,
               frameon=False, handlelength=1.7, columnspacing=1.4, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.925, bottom=0.15, wspace=0.12, hspace=0.28)
    fig.savefig(HERE / "figures" / "fig_oss_router_fusion.pdf", dpi=300)
    fig.savefig(HERE / "figures" / "fig_oss_router_fusion.png", dpi=600)
    print("\nwrote figures/fig_oss_router_fusion.{pdf,png} and primary/sensitivity router tables")


if __name__ == "__main__":
    main()
