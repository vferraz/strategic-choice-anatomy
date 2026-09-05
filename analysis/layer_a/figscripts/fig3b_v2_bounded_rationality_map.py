#!/usr/bin/env python3
"""Fig 3b (v2) — bounded-rationality map: reasoning depth x decision sharpness.

Status: FINAL (2026-06-27). Default n_boot=200 with honest COLD-start bootstrap fits
(warm-starting was rejected — it collapses boundary CIs; see ``bootstrap``). Raise
``FIG3B_BOOT`` offline for tighter CIs. The robust, defensible claim is DEPTH-led:
GPT-OSS is deeper (depth CIs disjoint from the cluster); its decision-sharpness edge over
the dense-LLM+human cluster is marginal (within bootstrap noise). Do not sell GPT-OSS as
"sharper".

STANDALONE. Does NOT touch ``fig3_qre_to_levelk.py`` or the QRE panel (a); it is an
additive alternative for panel (b) only.

Why this exists
---------------
The original panel (b) plots the QLk precision ``lambda`` on a log axis. ``lambda`` is
only weakly identified at the top: once choices are near-deterministic the likelihood
goes flat, so the optimiser pins GPT-OSS at its ceiling (50) and the log axis implies a
precision the data cannot support, squashing every other agent into the floor.

This panel keeps the SAME fitted QLk models and the SAME depth x-axis, but replaces the
y-axis with **decision sharpness** = the fitted model's mean MODAL-ACTION probability,
``mean_g max(p_g, 1-p_g)`` in [0.5, 1]. (This is decision determinism, NOT literally
"P(best response)": max(p,1-p) is the more-likely action of the fitted mixture, which
coincides with a best response only for a single pure level, not a finite-lambda mixture.)
That functional stays identified exactly where ``lambda`` does not (the predicted choice
probabilities are pinned even when lambda is censored), so GPT-OSS lands honestly near the
idealized deep, deterministic limit instead of at "50".

Enrichments
-----------
* Level-k archetype LANDMARKS: Random/L0 (depth 0, sharpness 0.5) and deterministic
  Level-1/2/3 (pure pi_k=1, lambda->inf). These are *pure-level reference points*, NOT an
  achievable ceiling (a deep mixture can exceed the line). det-L2 peaks because L2
  point-beliefs leave the fewest L-indifferent games.
* 95% bootstrap-by-game CIs (error bars). They show the honest result: GPT-OSS's depth CI
  clears the pack, while the three dense LLMs + the human (Moore et al.) overlap so
  completely they are one statistical group (drawn as a shaded envelope).

Honesty notes (carried into the caption)
----------------------------------------
* y is decision sharpness / determinism (a composite of precision, L0 tremble and depth),
  not "precision lambda". It is not strictly monotone in lambda for mixtures whose levels
  disagree.
* Risk-dominant NE is not used anywhere (undefined on ordinal {1,2,3,4} payoffs).

Inputs (current Layer-A caches / source data)
---------------------------------------------
  data/games/taxonomy/human_game_master_per_canonical.csv   canonical axis + gaps
  analysis/layer_a/_data/layerA_game_level.parquet             LLM realised per-game rate/n
  datasets/nagel/normalized_data_20241612.csv                      human individual choices

Outputs
-------
  analysis/layer_a/figures/fig3b_v2_bounded_rationality_map.{png,pdf}
  analysis/layer_a/tables/f3b_v2_fingerprint_sharpness.csv

Run
---
  python analysis/layer_a/figscripts/fig3b_v2_bounded_rationality_map.py
  FIG3B_BOOT=500 python .../fig3b_v2_bounded_rationality_map.py   # heavier CIs
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from strategic_anatomy.config import human_refs_root, results_root, taxonomy_dir

ROOT = Path(__file__).resolve().parents[3]
MASTER = taxonomy_dir() / "human_game_master_per_canonical.csv"
GAME_LEVEL = results_root() / "layer_a" / "_data" / "layerA_game_level.parquet"
NORM = human_refs_root() / "raw" / "nagel" / "normalized_data_20241612.csv"
FIG_DIR = ROOT / "analysis" / "layer_a" / "figures"
TAB_DIR = results_root() / "layer_a"

SEED = 20260520
N_BOOT = int(os.environ.get("FIG3B_BOOT", "200"))  # cold-start CIs; raise offline for more

LLMS = ["qwen", "qwen_instruct", "llama31_instruct", "gptoss"]
DISPLAY = {"qwen": "Qwen2.5", "qwen_instruct": "Qwen2.5-I",
           "llama31_instruct": "Llama", "gptoss": "GPT-OSS", "nagel": "Moore et al."}
COLOR = {"qwen": "#1f77b4", "qwen_instruct": "#17becf", "llama31_instruct": "#2ca02c",
         "gptoss": "#9467bd", "nagel": "#c83b31"}
CLUSTER = ["qwen", "qwen_instruct", "llama31_instruct", "nagel"]  # dense LLMs + human


# ---------------------------------------------------------------- gaps (canonical axis)
def _vec(s):
    return [float(x) for x in str(s).split(",")]


def _delta1(v, q):
    u = np.array(v[:4]).reshape(2, 2)
    return (q * u[0, 0] + (1 - q) * u[0, 1]) - (q * u[1, 0] + (1 - q) * u[1, 1])


def _q_from_action(a):
    if pd.isna(a):
        return 0.5
    a = int(a)
    return 1.0 if a == 0 else (0.0 if a == 1 else 0.5)


def build_gaps(master: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Per game: canonical-signed level-k incentive gaps (L1=q.5, L2=P2 plays L1 act, L3=L2)."""
    out = {}
    for _, r in master.iterrows():
        v = _vec(r["canonical_8vec"])
        sgn = 1 if int(r["canonical_action_p1"]) == 0 else -1
        out[r["game_code"]] = (
            sgn * _delta1(v, 0.5),
            sgn * _delta1(v, _q_from_action(r.get("l1_action_p2"))),
            sgn * _delta1(v, _q_from_action(r.get("l2_action_p2"))),
        )
    return out


# ----------------------------------------------------------------- QLk binomial MLE
def _simplex4(theta):
    z = np.concatenate([[0.0], theta])
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _pcanon(pi, lam, G):
    return (pi[0] * 0.5 + pi[1] * expit(lam * G[:, 0])
            + pi[2] * expit(lam * G[:, 1]) + pi[3] * expit(lam * G[:, 2]))


def _nll(par, s, n, G):
    pi = _simplex4(par[:3])
    p = np.clip(_pcanon(pi, par[3], G), 1e-6, 1 - 1e-6)
    return -np.sum(s * np.log(p) + (n - s) * np.log(1 - p))


def fit_qlk(s, n, G, multi=False, x0=None):
    """Return (depth, sharpness, pi, lam, raw_params). depth=sum_k k*pi_k; sharpness=mean max(p,1-p).

    ``x0`` warm-starts a single fit (used by the bootstrap for speed); otherwise a small
    multi-restart is used for the point estimate.
    """
    inits = [list(x0)] if x0 is not None else (
        [[0, 0, 0, 1.0]] + ([[0, 0, 0, 0.3], [0.5, 0.5, 0.5, 2.0]] if multi else []))
    best = None
    for init in inits:
        r = minimize(_nll, init, args=(s, n, G), method="L-BFGS-B",
                     bounds=[(-8, 8)] * 3 + [(0.0, 50.0)])
        if best is None or r.fun < best.fun:
            best = r
    pi = _simplex4(best.x[:3])
    lam = float(best.x[3])
    p = _pcanon(pi, lam, G)
    depth = float(sum(k * pi[k] for k in range(4)))
    sharp = float(np.mean(np.maximum(p, 1 - p)))
    return depth, sharp, pi, lam, np.asarray(best.x)


def bootstrap(s, n, G, n_boot=N_BOOT, seed=SEED):
    """95% CI on (depth, sharpness) by resampling games, with an honest COLD fit per
    resample (single fixed init).

    Warm-starting every resample from the full-sample MLE was tried and REJECTED: it
    collapses boundary CIs (a near-pure-Level-1 agent's depth CI degenerates to a point),
    understating uncertainty. Cold starts cost more compute but give a faithful CI.
    """
    rng = np.random.default_rng(seed)
    ds, ss = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(s), len(s))
        try:
            d, sh, _, _, _ = fit_qlk(s[i], n[i], G[i])
            ds.append(d)
            ss.append(sh)
        except Exception:
            continue
    return (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)),
            float(np.percentile(ss, 2.5)), float(np.percentile(ss, 97.5)))


# ----------------------------------------------------------------- per-agent data
def llm_arrays(gl: pd.DataFrame, gaps, model):
    d = gl[gl["model"] == model].dropna(subset=["p_canon", "n_ok_p1_baseline"]).copy()
    d = d.sort_values("game_code").reset_index(drop=True)
    # corrected-root invariants — fail loudly so a stale/wrong cache cannot silently
    # change the figure (matches the asserts in shared_data.audit_clean).
    assert len(d) == 144, f"{model}: expected 144 games in layerA_game_level cache, got {len(d)}"
    assert d["game_code"].nunique() == 144, (
        f"{model}: expected 144 unique games, got {d['game_code'].nunique()}")
    n = d["n_ok_p1_baseline"].to_numpy(float)
    assert np.all(n == 4), (
        f"{model}: corrected root expects n_ok_p1_baseline == 4 cells/game; got "
        f"{sorted(set(n.tolist()))} — verify the substrate/cache before trusting this figure")
    s_float = d["p_canon"].to_numpy() * n
    assert np.allclose(s_float, np.rint(s_float), atol=1e-6), (
        f"{model}: p_canon*n is not integral — realised rate is not a clean k/4, cache may be corrupt")
    g = d["game_code"].to_numpy()
    s = np.rint(s_float)
    return s, n, np.array([gaps[x] for x in g])


def human_arrays(master: pd.DataFrame, nd: pd.DataFrame, gaps):
    s, n, G = [], [], []
    for _, r in master.iterrows():
        col = f"P{int(r['nagel_p_id'])}"
        if col not in nd.columns:
            continue
        nn = int(nd[col].notna().sum())
        fr = float(r["nagel_frac_choose_act0_canonical"])           # positional act0 rate
        rate = fr if int(r["canonical_action_p1"]) == 0 else 1 - fr  # -> canonical-action rate
        s.append(round(rate * nn))
        n.append(nn)
        G.append(gaps[r["game_code"]])
    return np.array(s, float), np.array(n, float), np.array(G)


def landmark_ceiling(G1):
    """Sharpness of a deterministic single-level reasoner (lambda->inf) on gap vector G1."""
    p = expit(1e6 * np.asarray(G1))
    return float(np.mean(np.maximum(p, 1 - p)))


# ----------------------------------------------------------------- figure
def render(rows: pd.DataFrame, lm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    R = rows.set_index("agent")
    fig, ax = plt.subplots(figsize=(5.35, 3.45))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
    })

    ax.axhline(0.5, color="#bbb", ls=":", lw=0.8, zorder=1)
    ax.text(3.12, 0.505, "chance", fontsize=7.2, color="#999", ha="right", va="bottom")

    # archetype landmarks (pure-level reference points; NOT a ceiling).
    # DISCRETE points only — deliberately NO connecting line: their vertical order
    # (L1<L2>L3) is a tie-counting artifact of this game set, not a trajectory or a
    # competence ranking, so a line would mislead.
    ld = [0, 1, 2, 3]
    ls = [0.5, lm["L1"], lm["L2"], lm["L3"]]
    ax.scatter(ld, ls, marker="D", s=28, color="#8a8a8a", zorder=2)
    landmark_labels = [
        (0, 0.5, "Random\n(L0)", (12, 9), "left"),
        (1, lm["L1"], "deterministic\nL1", (-7, 8), "right"),
        (2, lm["L2"], "deterministic\nL2", (0, 9), "center"),
        (3, lm["L3"], "deterministic\nL3", (8, 8), "left"),
    ]
    for d, s, lab, offset, ha in landmark_labels:
        ax.annotate(lab, (d, s), textcoords="offset points", xytext=offset,
                    ha=ha, va="bottom", fontsize=6.9, color="#777")
    ax.annotate("idealized deep,\ndeterministic limit", (2.56, 0.975),
                fontsize=7.3, color="#555", style="italic", ha="center", va="top")
    ax.annotate("grey diamonds: pure level-k references", (3.12, 0.468),
                fontsize=6.2, color="#9a9a9a", ha="right")

    # cluster envelope = dense LLMs + human (their CIs mutually overlap)
    dlo = R.loc[CLUSTER, "depth_lo"].min(); dhi = R.loc[CLUSTER, "depth_hi"].max()
    slo = R.loc[CLUSTER, "sharp_lo"].min(); shi = R.loc[CLUSTER, "sharp_hi"].max()
    ax.add_patch(Ellipse(((dlo + dhi) / 2, (slo + shi) / 2), dhi - dlo, shi - slo,
                         facecolor="#888", alpha=0.09, edgecolor="#aaa", ls="--",
                         lw=0.8, zorder=1))
    ax.annotate("dense LLMs + human\n95% CIs overlap",
                ((dlo + dhi) / 2, slo - 0.030), ha="center", va="top",
                fontsize=6.8, color="#777")

    # agents with 95% CI error bars
    lab_off = {
        "qwen": (8, 13),
        "qwen_instruct": (10, -18),
        "llama31_instruct": (-15, 15),
        "nagel": (-22, -1),
        "gptoss": (8, 17),
    }
    label_box = dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.18)
    for ag, r in R.iterrows():
        c = COLOR[ag]
        ax.errorbar(r["depth"], r["sharp"],
                    xerr=[[r["depth"] - r["depth_lo"]], [r["depth_hi"] - r["depth"]]],
                    yerr=[[r["sharp"] - r["sharp_lo"]], [r["sharp_hi"] - r["sharp"]]],
                    fmt="D" if ag == "nagel" else "o",
                    ms=6.2 if ag == "gptoss" else 5.5, color=c, ecolor=c,
                    elinewidth=0.95, capsize=2.0, mec="white", mew=0.55, zorder=5)
        dx, dy = lab_off[ag]
        ax.annotate(DISPLAY[ag], (r["depth"], r["sharp"]), textcoords="offset points",
                    xytext=(dx, dy), ha="left" if dx >= 0 else "right",
                    va="center", fontsize=8.2, fontweight="bold" if ag == "gptoss" else "normal",
                    color=c, bbox=label_box,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.42, alpha=0.55,
                                    shrinkA=2, shrinkB=4) if ag != "gptoss" else None)
    g = R.loc["gptoss"]
    ax.annotate(f"L2/L3 mix\nL0 weight ≈ {g['pi0'] * 100:.0f}%",
                (g["depth"], g["sharp"]), textcoords="offset points", xytext=(11, -19),
                ha="left", va="top", fontsize=6.9, color="#6b4fa0",
                bbox=label_box,
                arrowprops=dict(arrowstyle="-", color="#6b4fa0", lw=0.45, alpha=0.55,
                                shrinkA=2, shrinkB=4))

    ax.set_xlim(-0.10, 3.18)
    ax.set_ylim(0.45, 0.995)
    ax.set_xlabel("reasoning depth (quantal level-k mixture)", fontsize=8.8)
    ax.set_ylabel("decision sharpness\nmean modal-action probability", fontsize=8.8)
    ax.set_title("Bounded-rationality map", fontsize=10.2, loc="left", pad=6,
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.8, length=3.0)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.45, alpha=0.75, zorder=0)
    # robust read, kept in the source/provenance; the visual panel stays clean.
    dep_disjoint = R.loc["gptoss", "depth_lo"] > R.loc[CLUSTER, "depth_hi"].max()
    sh_margin = R.loc["gptoss", "sharp_lo"] - R.loc[CLUSTER, "sharp_hi"].max()
    print(f"[fig3b_v2] robust read: depth_disjoint={dep_disjoint} "
          f"sharpness_margin={sh_margin:+.3f}")
    fig.tight_layout(rect=(0, 0.00, 1, 1))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig3b_v2_bounded_rationality_map.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3b_v2] wrote {FIG_DIR / 'fig3b_v2_bounded_rationality_map.png'}")


def main() -> int:
    master = pd.read_csv(MASTER)
    gaps = build_gaps(master)
    gl = pd.read_parquet(GAME_LEVEL)
    nd = pd.read_csv(NORM)

    # landmark ceilings from the gap vectors over all 144 games
    G = np.array([gaps[g] for g in master["game_code"]])
    lm = {"L1": landmark_ceiling(G[:, 0]), "L2": landmark_ceiling(G[:, 1]),
          "L3": landmark_ceiling(G[:, 2])}
    print(f"[fig3b_v2] landmark ceilings  det-L1={lm['L1']:.3f}  det-L2={lm['L2']:.3f}  det-L3={lm['L3']:.3f}")

    rows = []
    for ag in LLMS + ["nagel"]:
        if ag == "nagel":
            s, n, Ga = human_arrays(master, nd, gaps)
        else:
            s, n, Ga = llm_arrays(gl, gaps, ag)
        depth, sharp, pi, lam, _ = fit_qlk(s, n, Ga, multi=True)
        dlo, dhi, slo, shi = bootstrap(s, n, Ga)
        rows.append({"agent": ag, "display": DISPLAY[ag], "depth": depth, "sharp": sharp,
                     "depth_lo": dlo, "depth_hi": dhi, "sharp_lo": slo, "sharp_hi": shi,
                     "pi0": float(pi[0]), "pi1": float(pi[1]), "pi2": float(pi[2]),
                     "pi3": float(pi[3]), "lambda_qlk": lam,
                     "lambda_censored": lam >= 49.0, "n_games": int(len(s))})
        print(f"  {DISPLAY[ag]:12s} depth={depth:.2f}[{dlo:.2f},{dhi:.2f}]  "
              f"sharp={sharp:.3f}[{slo:.3f},{shi:.3f}]  pi0={pi[0]:.3f}  "
              f"lam={lam:.1f}{'(censored)' if lam >= 49 else ''}")
    rows = pd.DataFrame(rows)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(TAB_DIR / "f3b_v2_fingerprint_sharpness.csv", index=False)
    print(f"[fig3b_v2] wrote {TAB_DIR / 'f3b_v2_fingerprint_sharpness.csv'}")
    render(rows, lm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
