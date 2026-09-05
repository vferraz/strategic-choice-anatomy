#!/usr/bin/env python3
"""B3 — dispositions are clearly REPRESENTED but weakly RECRUITED ("heard, not obeyed").

Deepest layer, P1, conditions = baseline + 5 disposition cues + placebo (length_match_null).
Cue-shift vector for a row = Z[cue] - Z[matched baseline (same game, cb)], with Z z-scored to
the baseline mean/std (mirrors fig6_disposition_state_space / fig6b_lda_dispositions).

B3a  Held-out LDA: PCA->Fisher LDA on the 5 disposition labels, trained on a GAME split,
     projected on HELD-OUT games; placebo lands central. Per-disposition held-out recall +
     overall accuracy + placebo max-class share (low => central/null). Expect 70B+ incl. Llama
     near-perfect -> the sharpest dissociation (perfect representation, ~zero behavioural aim).
B3b  Disposition axis _|_ decision axis: angle between the common cue-shift direction and the
     decision axis (diff-of-means canonical vs not on baseline). ~90deg => "what to do" and
     "what kind of player" on independent tracks.
B3c  Representation-vs-recruitment dissociation: per (model, trait), LDA recall (representation,
     x) vs signed behavioural aim (recruitment, y) = how much the cue shifts the choice toward the
     trait's preferred action (trait_targets a_<trait>). Llama: high representation, ~zero aim.

  .venv/bin/python analysis/layer_b/build_dispositions.py
Outputs: tables/b3_lda.csv, tables/b3_orthogonality.csv, tables/b3_dissociation.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_b import lib  # noqa: E402

PCA_K_LDA = 30
SPLIT_SEED = 0


def _cue_shifts(model: str):
    """Return (Z, meta) for the cue cache with z-scoring to baseline, plus a per-row matched
    baseline shift S (NaN rows where no matched baseline). meta carries game_code, condition, trait."""
    cm, cx = lib.load_cues(model)
    cx = np.asarray(cx, dtype=np.float64)
    cm = cm.reset_index(drop=True)
    base = (cm["condition"] == "baseline").to_numpy()
    mu, sd = cx[base].mean(0), cx[base].std(0) + 1e-6
    Z = (cx - mu) / sd
    bidx = {(g, cb): i for g, cb, i in
            zip(cm.loc[base, "game_code"], cm.loc[base, "counterbalance_id"], np.where(base)[0])}
    cm["trait"] = cm["condition"].str.replace("^cue_", "", regex=True)
    return Z, cm, base, bidx


def build_lda_and_ortho():
    lda_rows, ortho_rows, shift_store = [], [], {}
    rng = np.random.default_rng(SPLIT_SEED)
    for m in lib.MODELS:
        Z, cm, base, bidx = _cue_shifts(m)
        # shifts for the 5 traits + placebo
        rows = []
        for i, (cond, trait, g, cb) in enumerate(zip(cm.condition, cm.trait, cm.game_code, cm.counterbalance_id)):
            if cond == "baseline":
                continue
            j = bidx.get((g, cb))
            if j is None:
                continue
            rows.append((Z[i] - Z[j], trait, g))
        SH = np.array([r[0] for r in rows])
        cue = np.array([r[1] for r in rows])
        game = np.array([r[2] for r in rows])
        shift_store[m] = (SH, cue, game)

        tr_mask = np.isin(cue, lib.TRAITS)                       # train classes = 5 traits (not placebo)
        ug = rng.permutation(np.unique(game))
        train_g = set(ug[:len(ug) // 2])
        is_tr = np.array([g in train_g for g in game])

        mu, comp = lib.pca_fit(SH[tr_mask & is_tr], PCA_K_LDA)
        P = (SH - mu) @ comp.T
        ylab = {t: k for k, t in enumerate(lib.TRAITS)}
        W = lib.lda_fit(P[tr_mask & is_tr], np.array([ylab[c] for c in cue[tr_mask & is_tr]]))
        E = P @ W
        cent = {k: E[tr_mask & is_tr][np.array([ylab[c] for c in cue[tr_mask & is_tr]]) == k].mean(0)
                for k in range(len(lib.TRAITS))}

        def _assign(pts):
            return np.array([min(cent, key=lambda k: float(np.sum((p - cent[k]) ** 2))) for p in pts])

        te = ~is_tr
        # per-trait held-out recall
        accs = {}
        for t in lib.TRAITS:
            sel = te & (cue == t)
            if sel.sum() == 0:
                accs[t] = np.nan; continue
            yhat = _assign(E[sel])
            accs[t] = float(np.mean(yhat == ylab[t]))
            lda_rows.append(dict(model=m, disposition=t, held_out_acc=accs[t], n_heldout=int(sel.sum())))
        # overall held-out accuracy on the 5 traits
        sel = te & tr_mask
        overall = float(np.mean(_assign(E[sel]) == np.array([ylab[c] for c in cue[sel]]))) if sel.sum() else np.nan
        lda_rows.append(dict(model=m, disposition="OVERALL", held_out_acc=overall, n_heldout=int(sel.sum())))
        # placebo centrality (proper control): distance of the placebo centroid from the trait
        # cloud center, RELATIVE to the mean trait-centroid distance. <1 => placebo lands central
        # (null), i.e. the discriminant is not just firing on any prompt perturbation.
        psel = te & (cue == lib.PLACEBO)
        center = E[sel].mean(0) if sel.sum() else E[te].mean(0)
        trait_d = float(np.mean([np.linalg.norm(E[te & (cue == t)].mean(0) - center)
                                 for t in lib.TRAITS if (te & (cue == t)).sum()]))
        plac_d = float(np.linalg.norm(E[psel].mean(0) - center)) if psel.sum() else np.nan
        centrality = plac_d / (trait_d + 1e-12) if np.isfinite(plac_d) else np.nan

        # B3b: angle between common cue-shift direction and the decision axis (on baseline Z)
        common = SH[tr_mask].mean(0)
        yb = (cm.loc[base, "decoded_action"].to_numpy() == cm.loc[base, "canonical_action_p1"].to_numpy()).astype(int)
        d_dec = lib.decision_axis_diffmeans(Z[base], yb)
        ang = lib.angle_deg(common, d_dec)
        ortho_rows.append(dict(model=m, angle_disposition_decision_deg=ang,
                               overall_lda_acc=overall, placebo_centrality=centrality,
                               n_traits=len(lib.TRAITS)))
        print(f"[{lib.SHORT[m]}] B3a LDA held-out overall={overall:.2f} placebo_centrality={centrality:.2f} | "
              f"B3b angle(disp,decision)={ang:.1f}deg")

    pd.DataFrame(lda_rows).to_csv(lib.TAB_DIR / "b3_lda.csv", index=False)
    pd.DataFrame(ortho_rows).to_csv(lib.TAB_DIR / "b3_orthogonality.csv", index=False)
    return shift_store


def build_dissociation():
    """B3c: per (model, trait) LDA recall (representation) vs signed behavioural aim (recruitment)."""
    tt = lib.trait_targets().set_index("game_code")           # a_<trait>, d_<trait> per game
    lda = pd.read_csv(lib.TAB_DIR / "b3_lda.csv")
    rows = []
    for m in lib.MODELS:
        cm, _ = lib.load_cues(m)
        cm = cm.reset_index(drop=True)
        for t in lib.TRAITS:
            a_col = f"a_{t}"
            base = cm[cm["condition"] == "baseline"].copy()
            cued = cm[cm["condition"] == f"cue_{t}"].copy()
            for d in (base, cued):
                d["a_pref"] = d["game_code"].map(tt[a_col])
                d["chose_pref"] = (d["decoded_action"] == d["a_pref"]).astype(float)
                d.loc[d["a_pref"].isna(), "chose_pref"] = np.nan
            # match by (game, cb): aim = E[chose_pref | cue] - E[chose_pref | baseline]
            bm = base.groupby(["game_code", "counterbalance_id"])["chose_pref"].mean()
            cmn = cued.groupby(["game_code", "counterbalance_id"])["chose_pref"].mean()
            join = pd.concat([bm.rename("b"), cmn.rename("c")], axis=1).dropna()
            aim = float((join["c"] - join["b"]).mean()) if len(join) else np.nan
            rec = lda[(lda.model == m) & (lda.disposition == t)]["held_out_acc"]
            rows.append(dict(model=m, trait=t, lda_acc=float(rec.iloc[0]) if len(rec) else np.nan,
                             behavioral_aim=aim, n=int(len(join))))
    out = pd.DataFrame(rows)
    out.to_csv(lib.TAB_DIR / "b3_dissociation.csv", index=False)
    for m in lib.MODELS:
        sub = out[out.model == m]
        print(f"[{lib.SHORT[m]}] B3c mean LDA acc={sub.lda_acc.mean():.2f}  mean |aim|={sub.behavioral_aim.abs().mean():.3f}")
    print(f"wrote b3_lda.csv, b3_orthogonality.csv, b3_dissociation.csv")


if __name__ == "__main__":
    build_lda_and_ortho()
    build_dissociation()
