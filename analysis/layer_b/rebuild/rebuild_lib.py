"""Shared loaders/estimators for the Layer-B table regeneration (evidence freeze).

Locked spec (2026-07-09, see docs/METHODS.md):
- Root: $SCA_DATA_ROOT/substrate (corrected Akata substrate, config-verified).
- Decision label: realized generated move — dense decoded_action gated by parse_ok;
  GPT-OSS realized_action excluding commit_type == "none".
- Canonical axis: data/games/game_features.csv (canonical_action_p1).
- Incentive: committed analysis/layer_c/tables/incentive_delta1c.csv.
- P1-baseline rows only for strategic probes (4 cb cells / game).
- Probes: logistic regression, GroupKFold by game (5 folds), out-of-fold predictions.
- CIs: bootstrap by game (1000 resamples, 95%).
- GPT-OSS read at its commit capture; dense at the answer slot. No _seq keys.
- No code or values sourced from _archive/_legacy.
"""
import os
import numpy as np
import pandas as pd
from strategic_anatomy.config import game_features_csv, results_root, substrate_root

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ROOT = str(substrate_root())
FEATURES = str(game_features_csv())
DELTA1C = os.path.join(str(results_root() / "layer_c"),
                       "incentive_delta1c.csv")

MODELS = ["qwen", "qwen_instruct", "llama31_instruct", "gptoss"]
N_LAYERS = {"qwen": 81, "qwen_instruct": 81, "llama31_instruct": 81, "gptoss": 37}


def games(model):
    d = os.path.join(ROOT, model)
    return sorted(g for g in os.listdir(d)
                  if os.path.isdir(os.path.join(d, g)) and not g.startswith("_"))


def meta():
    feat = pd.read_csv(FEATURES)[["game_code", "canonical_action_p1"]]
    dlt = pd.read_csv(DELTA1C)
    m = feat.merge(dlt, on="game_code", how="inner")
    assert len(m) == 144, f"meta join expected 144 games, got {len(m)}"
    return m.set_index("game_code")


def load_game_rows(model, game, layers, condition="baseline", player=1):
    """Return (X[list per layer], rows_df) for the 4 cb cells of one condition."""
    gd = os.path.join(ROOT, model, game)
    df = pd.read_parquet(os.path.join(gd, "results.parquet"))
    df = df[(df.player == player) & (df.condition == condition)].sort_values(
        "counterbalance_id")
    z = np.load(os.path.join(gd, "acts.npz"))
    cond = condition if condition == "baseline" else condition
    key = lambda cb, L: f"p{player}_{cond}_cb{cb}_l{L}"
    X = {L: np.stack([z[key(cb, L)] for cb in df.counterbalance_id]) for L in layers}
    return X, df


def realized_aligned(df, model, canon_p1):
    """Per-row aligned-canonical label (1/0) and validity mask, realized moves only."""
    if model == "gptoss":
        ok = df.commit_type.astype(str) != "none"
        act = df.realized_action
    else:
        ok = df.parse_ok.astype(bool)
        act = df.decoded_action
    y = (act == canon_p1).astype(float)
    return y.to_numpy(), ok.to_numpy()


def boot_ci(games_arr, stat, n=1000, seed=20260627):
    # seed 20260627 = the original run's seed, recovered from the surviving
    # _archive/layerB_recruitment_final/tables/provenance.json (read-only
    # verification exception, locked 2026-07-09).
    rng = np.random.default_rng(seed)
    idx = np.arange(len(games_arr))
    vals = []
    for _ in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        v = stat(take)
        if v is not None and np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi
