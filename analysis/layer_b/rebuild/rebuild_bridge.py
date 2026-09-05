"""Rebuild within_model_bridge.csv (Layer-B panel e — the load-bearing table).

Estimator (reconstructing the tex, Results 'Recruitment is selective'):
  Per model, per candidate layer:
    d_inc = centred covariance direction between P1-baseline residuals and
            continuous canonical-signed Delta1c, fit on train games only
            (GroupKFold-by-game, 5 folds); held-out rows projected (OOF).
    Projections z-scored, averaged to game level -> x_g.
    y_g = P(canonical): mean realized aligned-canonical over valid cb rows.
    RAW slope: OLS y_g ~ x_g  (units: delta P(canonical) per SD of projection).
    PARTIAL:   residualize x_g and y_g each on Delta1c_g, slope of residuals.
    95% CIs: bootstrap games (1000).
  Candidate layers (pre-registered probe layers + final):
    dense: 30, 65, 75, 78, 79, 80   gptoss: 1, 6, 9, 22, 24, 35, 36
  Acceptance targets (paper_NHB_v3.tex): raw 0.179/0.055/0.135/0.090;
  partial +0.047/-0.012/+0.006/+0.072 (qwen_instruct/qwen/llama/gptoss order
  rearranged per model below). Mismatch -> report, do not edit the paper.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from analysis.layer_b.rebuild.rebuild_lib import (ROOT, MODELS, games, meta, load_game_rows,
                         realized_aligned, boot_ci)

CAND = {"qwen": [30, 65, 75, 78, 79, 80],
        "qwen_instruct": [30, 65, 75, 78, 79, 80],
        "llama31_instruct": [30, 65, 75, 78, 79, 80],
        "gptoss": [1, 6, 9, 22, 24, 35, 36]}

TEX = {"qwen_instruct": (0.179, 0.047), "qwen": (0.055, -0.012),
       "llama31_instruct": (0.135, 0.006), "gptoss": (0.090, 0.072)}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)


def slope(y, x):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def residualize(v, on):
    on = np.asarray(on, float)
    A = np.c_[np.ones_like(on), on]
    beta, *_ = np.linalg.lstsq(A, np.asarray(v, float), rcond=None)
    return np.asarray(v, float) - A @ beta


def run_model(model):
    M = meta()
    gl = games(model)
    layers = CAND[model]
    X_all = {L: [] for L in layers}
    y_rows, ok_rows, g_rows, d_rows = [], [], [], []
    for gi, g in enumerate(gl):
        X, df = load_game_rows(model, g, layers)
        y, ok = realized_aligned(df, model, M.loc[g, "canonical_action_p1"])
        for L in layers:
            X_all[L].append(X[L].astype(np.float32))
        y_rows.append(y); ok_rows.append(ok)
        g_rows.append(np.full(len(df), gi)); d_rows.append(
            np.full(len(df), M.loc[g, "delta1c"], dtype=float))
        if gi % 36 == 0:
            print(f"[{model}] loaded {gi+1}/{len(gl)}", flush=True)
    y = np.concatenate(y_rows); ok = np.concatenate(ok_rows)
    grp = np.concatenate(g_rows); dlt = np.concatenate(d_rows)
    res = []
    for L in layers:
        X = np.concatenate(X_all[L]).astype(np.float64)
        proj = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X, groups=grp):
            Xt = X[tr] - X[tr].mean(0)
            d = Xt.T @ (dlt[tr] - dlt[tr].mean())
            n = np.linalg.norm(d)
            if n > 0:
                d /= n
            proj[te] = (X[te] - X[tr].mean(0)) @ d
        proj = (proj - np.nanmean(proj)) / np.nanstd(proj)
        gids = np.unique(grp)
        xg = np.array([proj[(grp == i)].mean() for i in gids])
        yg = np.array([y[(grp == i) & ok].mean() if ((grp == i) & ok).any()
                       else np.nan for i in gids])
        dg = np.array([dlt[grp == i][0] for i in gids])
        keep = np.isfinite(yg)
        xg, yg, dg = xg[keep], yg[keep], dg[keep]
        raw = slope(yg, xg)
        par = slope(residualize(yg, dg), residualize(xg, dg))
        def st_raw(take): return slope(yg[take], xg[take])
        def st_par(take): return slope(residualize(yg[take], dg[take]),
                                       residualize(xg[take], dg[take]))
        rlo, rhi = boot_ci(xg, st_raw); plo, phi = boot_ci(xg, st_par)
        res.append(dict(model=model, layer=L, raw_slope=raw, raw_lo=rlo,
                        raw_hi=rhi, partial_slope=par, partial_lo=plo,
                        partial_hi=phi, n_games=int(keep.sum())))
        t_raw, t_par = TEX[model]
        print(f"[{model}] L{L}: raw {raw:+.3f} [{rlo:+.3f},{rhi:+.3f}] "
              f"(tex {t_raw:+.3f}) | partial {par:+.3f} [{plo:+.3f},{phi:+.3f}] "
              f"(tex {t_par:+.3f})", flush=True)
    pd.DataFrame(res).to_csv(os.path.join(OUT, f"bridge_{model}.csv"), index=False)


if __name__ == "__main__":
    for m in (sys.argv[1:] or MODELS):
        run_model(m)
    with open(os.path.join(OUT, "provenance.json"), "w") as f:
        json.dump({
            "analysis": "analysis/layer_b/rebuild (evidence-freeze regeneration)",
            "data_root": ROOT,
            "models": MODELS,
            "n_boot": 1000,
            "primary_behavior_target":
                "realized P1 baseline action aligned to canonical_action_p1",
            "primary_incentive_axis": "canonical-signed Delta1c at q=0.5",
            "seed": 20260627,
            "note": "parameters match the original run's provenance.json "
                    "(recovered 2026-07-09); candidate-layer reconstruction "
                    "per the locked spec",
        }, f, indent=2)
    print("DONE", flush=True)
