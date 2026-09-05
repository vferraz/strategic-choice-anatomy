#!/usr/bin/env python3
"""Read-only check: reproduce the complexity->P(canonical) Spearman rho and test which
complexity variable (if any) yields the draft's -0.67/-0.53/-0.51 vs the table's -0.41.

Does NOT write outside analysis/fusion_associative/. Prints to stdout only.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from strategic_anatomy.config import data_root, game_features_csv


def spearman_rho(x, y):
    x = pd.Series(np.asarray(x, float)).rank().to_numpy()
    y = pd.Series(np.asarray(y, float)).rank().to_numpy()
    return float(np.corrcoef(x, y)[0, 1])

ROOT = Path(__file__).resolve().parents[2]
CACHE = data_root() / "layer_b_cache" / "baseline"
gf = pd.read_csv(game_features_csv())

MODELS = {"qwen": "Qwen2.5 base", "qwen_instruct": "Qwen2.5-Instruct",
          "llama31_instruct": "Llama", "gptoss": "GPT-OSS"}

# per-game realised P(canonical) from the cached P1-baseline decision slot
pcanon = {}
for m in MODELS:
    meta = pd.read_parquet(CACHE / f"meta_{m}.parquet")
    a = (pd.to_numeric(meta["decoded_action"], errors="coerce")
         == pd.to_numeric(meta["canonical_action_p1"], errors="coerce")).astype(float)
    meta = meta.assign(al=a)
    pcanon[m] = meta.groupby("game_code")["al"].mean()

cand = ["complexity_score", "complexity_score_n_components", "iesds_depth",
        "num_pure_ne", "ne_distributional_conflict", "payoff_conflict", "payoff_variance"]
# derived indicator used inside the composite
gf["eq_ambiguity"] = gf["num_pure_ne"].isin([0, 2]).astype(int)
cand.append("eq_ambiguity")

print("Spearman rho( complexity_variable , realised P(canonical) ) over 144 games\n")
hdr = f"{'variable':30s}" + "".join(f"{MODELS[m]:>18s}" for m in MODELS)
print(hdr); print("-" * len(hdr))
for v in cand:
    if v not in gf.columns:
        continue
    row = f"{v:30s}"
    cv = gf.set_index("game_code")[v]
    for m in MODELS:
        s = pd.concat([cv, pcanon[m]], axis=1, join="inner").dropna()
        rho = spearman_rho(s.iloc[:, 0], s.iloc[:, 1])
        row += f"{rho:>18.3f}"
    print(row)

print("\nReference: table f1_complexity_response.csv = "
      "qwen -0.407, qwen_instruct -0.396, llama -0.425, gptoss -0.175")
print("Reference: draft paper_NHB_v2.tex L344-347 = "
      "qwen -0.67, qwen_instruct -0.53, llama -0.51, gptoss -0.27")
