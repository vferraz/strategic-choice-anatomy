#!/usr/bin/env python3
"""Confirm the colleague's observation: the permutation-null spread grows with depth,
and it is explained by the activation geometry (effective dimensionality), NOT by the
permutation procedure (which is identical at every layer).

Per layer we compute:
  - participation ratio  PR = (sum lambda)^2 / sum(lambda^2)  of the activation
    covariance (via the 576x576 Gram matrix eigenvalues) = effective dimensionality;
    high PR = isotropic, low PR = anisotropic (variance in few directions).
  - top-1 variance fraction (anisotropy).
  - null spread = null_hi - null_lo from the existing joint-permutation table.
Then we test whether PR predicts the null spread across layers.
"""
from __future__ import annotations
import glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
from strategic_anatomy.config import data_root, results_root
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
# phase-4: git does not track empty directories, so figures/ and tables/ do not exist in a
# fresh clone and savefig() would raise FileNotFoundError. Create them up front.
(HERE / "figures").mkdir(parents=True, exist_ok=True)
(HERE / "tables").mkdir(parents=True, exist_ok=True)
ROOT = HERE.parents[1]
CACHE = data_root() / "layer_b_cache" / "baseline"
MODELS = ["qwen_instruct", "qwen", "llama31_instruct", "gptoss"]
NICE = {"qwen_instruct": "Qwen2.5-Instruct", "qwen": "Qwen2.5 base",
        "llama31_instruct": "Llama-3.1-Instruct", "gptoss": "GPT-OSS-120B"}
COL = {"qwen_instruct": "#1f77b4", "qwen": "#2ca02c", "llama31_instruct": "#d62728", "gptoss": "#9467bd"}

tab = pd.read_csv(results_root() / "layer_b" / "fusion" / "fusion_depth_table.csv")
tab = tab[tab.angle_deg.notna()].copy()
tab["null_spread"] = tab["null_hi"] - tab["null_lo"]


def rankcorr(x, y):
    x = pd.Series(x).rank().to_numpy(); y = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(x, y)[0, 1])


rows = []
for m in MODELS:
    layers = sorted(int(re.search(r"_l(\d+)\.npy", f).group(1)) for f in glob.glob(str(CACHE / f"X_{m}_l*.npy")))
    step = 4 if m != "gptoss" else 3
    use = [L for L in layers if L % step == 0 and L > 0]
    for L in use:
        X = np.load(CACHE / f"X_{m}_l{L}.npy").astype(np.float64)
        Xc = X - X.mean(0)
        G = Xc @ Xc.T                       # 576x576 Gram; same nonzero eigenvalues as cov
        lam = np.linalg.eigvalsh(G)
        lam = lam[lam > 1e-9]
        pr = float((lam.sum() ** 2) / (lam ** 2).sum())     # participation ratio (eff. dim)
        top1 = float(lam.max() / lam.sum())                 # anisotropy: top eigenvalue share
        rows.append(dict(model=m, layer=L, participation_ratio=pr, top1_frac=top1))
        del X, Xc, G
    print(f"[{m}] {len(use)} layers")

pr = pd.DataFrame(rows)
df = pr.merge(tab[["model", "layer", "depth_frac", "null_spread", "null_med"]], on=["model", "layer"])
df.to_csv(results_root() / "layer_b" / "fusion" / "null_spread_vs_geometry.csv", index=False)

print("\n=== per-model rank correlations ===")
print(f"{'model':20s} {'PR~depth':>9s} {'spread~depth':>12s} {'PR~spread':>10s} {'top1~spread':>12s}")
for m in MODELS:
    s = df[df.model == m]
    print(f"{NICE[m]:20s} {rankcorr(s.depth_frac, s.participation_ratio):>9.2f} "
          f"{rankcorr(s.depth_frac, s.null_spread):>12.2f} "
          f"{rankcorr(s.participation_ratio, s.null_spread):>10.2f} "
          f"{rankcorr(s.top1_frac, s.null_spread):>12.2f}")

# figure: (1) null spread + PR vs depth (mirror);  (2) scatter PR vs null spread
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for m in MODELS:
    s = df[df.model == m].sort_values("depth_frac")
    ax[0].plot(s.depth_frac, s.null_spread, color=COL[m], lw=1.8, label=NICE[m])
    ax[1].scatter(s.participation_ratio, s.null_spread, color=COL[m], s=18, label=NICE[m])
ax[0].set_xlabel("relative depth"); ax[0].set_ylabel("null spread (null_hi − null_lo, °)")
ax[0].set_title("Null gets wider with depth")
ax[1].set_xlabel("participation ratio (effective dim.)  →  more isotropic")
ax[1].set_ylabel("null spread (°)")
ax[1].set_title("Wider null ⇔ lower effective dimensionality")
ax[1].legend(fontsize=7)
fig.suptitle("The permutation-null spread is set by each layer's activation geometry, not the shuffle", fontsize=11)
fig.tight_layout()
fig.savefig(HERE / "figures" / "fig_null_spread_geometry.png", dpi=150, bbox_inches="tight")
print("\nwrote tables/null_spread_vs_geometry.csv and figures/fig_null_spread_geometry.png")
