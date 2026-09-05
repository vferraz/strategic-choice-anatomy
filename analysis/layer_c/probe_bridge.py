"""
Layer B<->C bridge analysis (CPU): probe (free direction) vs lens (output direction).

Reads the residual re-capture (output/oneshot_layerc_residuals/{model}/{game}/{resid.npy, meta.parquet}).
For each (region, layer): a game-grouped CV logistic probe decodes the incentive sign from the residual
VECTORS (any linear direction) and is compared to the LENS decoder (AUC of the scalar score_canonical,
the output-direction projection). The discriminating contrast lives at the payoff tokens:

  probe_auc >> lens_auc (~0.5)  =>  incentive is REPRESENTED while reading but not yet projected
                                    onto the choice axis  =>  recruited only at commit (B and C unified).
  probe_auc ~ lens_auc ~ 0.5    =>  integration itself, not just projection, is deferred to commit.

Writes tables/bridge_probe_vs_lens.csv.

Run after the GPU capture:   python analysis/layer_c/probe_bridge.py --model qwen_instruct
Self-test (no GPU data):     python analysis/layer_c/probe_bridge.py --make-synthetic /tmp/bridge_syn
                             python analysis/layer_c/probe_bridge.py --root /tmp/bridge_syn --model synth
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from strategic_anatomy.config import layerc_bridge_root, results_root

ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = layerc_bridge_root()   # matches capture_residuals_bridge.py
TAB = results_root() / "layer_c"


def load_model(root, model):
    metas, Xs, off = [], [], 0
    for f in sorted(glob.glob(str(Path(root) / model / "*" / "meta.parquet"))):
        d = Path(f).parent
        m = pd.read_parquet(f)
        X = np.load(d / "resid.npy")
        assert len(m) == len(X), f"len mismatch {f}"
        m = m.copy(); m["row"] = np.arange(off, off + len(m)); off += len(m)
        metas.append(m); Xs.append(X)          # keep float16 in memory; cast per-fold subset (lean)
    if not metas:
        raise SystemExit(f"no data under {Path(root)/model} (run capture_residuals_bridge.py first)")
    return pd.concat(metas, ignore_index=True), np.vstack(Xs)


def probe_auc(X, y, groups, k=120, seed=0):
    """Out-of-fold AUC of a game-grouped logistic probe on PCA comps. Leak-free (fit per fold).
    k=120 PCs so the probe is a fairer 'any linear direction' test (must be able to recover the single
    lens direction; PCA=50 compressed away the low-variance decision axis)."""
    X = np.asarray(X, dtype=np.float32)          # cast the (small) region subset only
    y = np.asarray(y); ng = len(np.unique(groups))
    if len(np.unique(y)) < 2 or ng < 3:
        return np.nan, None
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=min(3, ng))
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        kk = int(min(k, X.shape[1], len(tr) - 1))
        sc = StandardScaler().fit(X[tr])
        pca = PCA(n_components=kk, random_state=seed).fit(sc.transform(X[tr]))
        Ztr, Zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, y[tr])
        oof[te] = lr.predict_proba(Zte)[:, 1]
    ok = ~np.isnan(oof)
    auc = float(roc_auc_score(y[ok], oof[ok])) if len(np.unique(y[ok])) == 2 else np.nan
    return auc, oof


def lens_auc(score, y):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, np.asarray(score)))


def auc_ci(y, s, groups, nboot=800, seed=0):
    """95% game-cluster bootstrap CI on the AUC (resample games; no refitting)."""
    y = np.asarray(y); s = np.asarray(s); groups = np.asarray(groups)
    m = ~np.isnan(s) & ~np.isnan(y.astype(float))
    y, s, groups = y[m], s[m], groups[m]
    if len(np.unique(y)) < 2:
        return (np.nan, np.nan)
    idx = [np.where(groups == g)[0] for g in np.unique(groups)]
    ng = len(idx); rng = np.random.default_rng(seed); est = []
    for _ in range(nboot):
        rows = np.concatenate([idx[i] for i in rng.integers(0, ng, ng)])
        if len(np.unique(y[rows])) == 2:
            est.append(roc_auc_score(y[rows], s[rows]))
    return tuple(np.nanpercentile(est, [2.5, 97.5])) if est else (np.nan, np.nan)


def analyse(meta, X, model):
    rows = []
    meta = meta[meta["incentive_sign"] != 0].copy()
    meta["y"] = (meta["incentive_sign"] > 0).astype(int)
    regions = ["own_payoff", "opponent_payoff"]
    if "is_final" in meta.columns:
        regions.append("final_answer")        # the exact final pre-choice token (primary readout)
    for region in regions:
        if region == "control":
            sub_region = meta[meta["token_str"].astype(str).str.strip().isin(["win", "wins"])]
        elif region == "final_answer":
            sub_region = meta[meta["is_final"] == True]
        else:
            sub_region = meta[meta["region"] == region]
        for L in sorted(meta["layer"].unique()):
            sub = sub_region[sub_region["layer"] == L]
            if len(sub) < 30:
                continue
            Xs = X[sub["row"].values]
            pa, oof = probe_auc(Xs, sub["y"].values, sub["game_code"].values)
            la = lens_auc(sub["score_canonical"].values, sub["y"].values)
            g = sub["game_code"].values
            plo, phi = auc_ci(sub["y"].values, oof, g) if oof is not None else (np.nan, np.nan)
            llo, lhi = auc_ci(sub["y"].values, sub["score_canonical"].values, g)
            rnd = lambda v: round(v, 3) if v == v else np.nan
            rows.append(dict(model=model, region=region, layer=int(L), n=len(sub),
                             n_games=int(sub["game_code"].nunique()),
                             probe_auc=rnd(pa), probe_lo=rnd(plo), probe_hi=rnd(phi),
                             lens_auc=rnd(la), lens_lo=rnd(llo), lens_hi=rnd(lhi),
                             probe_minus_lens=rnd(pa - la) if (pa == pa and la == la) else np.nan))
    return pd.DataFrame(rows)


def make_synthetic(outdir, n_games=40, D=64, seed=0):
    """Tiny fake dataset in the on-disk layout. Payoff tokens: incentive REPRESENTED (probe-decodable)
    but NOT on the lens axis. Answer tokens: on BOTH. Control: neither. Validates the pipeline."""
    rng = np.random.default_rng(seed)
    base = Path(outdir) / "synth"
    d_rep, d_out = 1, 0
    for gi in range(n_games):
        sign = rng.choice([-1, 1]); d1c = sign * rng.uniform(0.3, 2.0)
        rows, Xs = [], []
        for cb in range(4):
            for L in (79, 40):
                def emit(region, tok, on_rep, on_out, lens_signal, is_can=False, digit=-1, is_final=False):
                    x = rng.normal(0, 1, D)
                    if on_rep: x[d_rep] += sign * 3.0
                    if on_out: x[d_out] += sign * 3.0
                    Xs.append(x.astype(np.float16))
                    rows.append(dict(model="synth", game_code=f"G{gi:03d}", cb_id=cb, layer=L,
                                     token_index=len(rows), token_str=tok, region=region,
                                     is_canonical_row=is_can, is_final=is_final, digit=digit,
                                     score_canonical=float(lens_signal + rng.normal(0, 1)),
                                     canonical_action=0, delta1c=d1c, incentive_sign=int(sign)))
                for _ in range(4): emit("own_payoff", "3", True, False, 0.0, digit=3)      # represented only
                for _ in range(4): emit("opponent_payoff", "2", True, False, 0.0, digit=2)
                for k in range(3): emit("answer_prefix", " Option", True, True, sign * 2.5, is_final=(k == 2))
                emit("control", "win", False, False, 0.0)
        gdir = base / f"G{gi:03d}"; gdir.mkdir(parents=True, exist_ok=True)
        np.save(gdir / "resid.npy", np.vstack(Xs).astype(np.float16))
        pd.DataFrame(rows).to_parquet(gdir / "meta.parquet", index=False)
        (gdir / "config.json").write_text(json.dumps({"game_code": f"G{gi:03d}", "n_rows": len(rows)}))
        (gdir / "_DONE").write_text("synthetic")
    print(f"wrote synthetic dataset -> {base}  ({n_games} games)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen_instruct")
    p.add_argument("--root", default=str(OUT_DEFAULT))
    p.add_argument("--make-synthetic", default="")
    args = p.parse_args()
    if args.make_synthetic:
        make_synthetic(args.make_synthetic); return
    meta, X = load_model(args.root, args.model)
    res = analyse(meta, X, args.model)
    TAB.mkdir(parents=True, exist_ok=True)
    out = TAB / "bridge_probe_vs_lens.csv"
    if out.exists():
        prev = pd.read_csv(out); res = pd.concat([prev[prev["model"] != args.model], res], ignore_index=True)
    res.to_csv(out, index=False)
    print(res.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
