#!/usr/bin/env python3
"""Corrected Layer-A substrate loader.

The final Layer-A behavioural substrate is the integrated per-game tree:

    $SCA_DATA_ROOT/substrate/{model}/{game}/results.parquet

It replaces the older split roots (``oneshot_moves``, ``oneshot_commit`` and
``oneshot_main/substrate``) for the paper rebuild. Dense models expose parsed
generated decisions in ``decoded_action`` with ``parse_ok``. GPT-OSS exposes the
resolved final-channel decision in ``realized_action`` and retains mixed-strategy
metadata alongside it. Mixed GPT-OSS cells are already resolved to a deterministic
0/1 action and are included as behaviour; only ``commit_type == "none"`` is
dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from strategic_anatomy.config import substrate_root

ROOT = Path(__file__).resolve().parents[3]
CORRECTED_ROOT = substrate_root()

MODELS = ("qwen", "qwen_instruct", "llama31_instruct", "gptoss")
DENSE_MODELS = ("qwen", "qwen_instruct", "llama31_instruct")


def model_dirs(model: str) -> list[Path]:
    root = CORRECTED_ROOT / model
    if not root.exists():
        raise FileNotFoundError(f"corrected substrate model root missing: {root}")
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name != "_tmp"]
    return [d for d in dirs if (d / "results.parquet").exists()]


def result_files(model: str) -> list[Path]:
    files = [d / "results.parquet" for d in model_dirs(model)]
    if len(files) != 144:
        raise AssertionError(f"{model}: expected 144 corrected results files, found {len(files)}")
    return files


def config_files(model: str) -> list[Path]:
    files = [d / "config.json" for d in model_dirs(model) if (d / "config.json").exists()]
    if len(files) != 144:
        raise AssertionError(f"{model}: expected 144 corrected config files, found {len(files)}")
    return files


def read_model_results(model: str, columns: list[str] | None = None) -> pd.DataFrame:
    frames = [pd.read_parquet(path, columns=columns) for path in result_files(model)]
    return pd.concat(frames, ignore_index=True)


def _j_maps_to_act0(label_map_value) -> bool:
    text = str(label_map_value)
    if "J=act0" in text:
        return True
    if "J=act1" in text:
        return False
    raise ValueError(f"cannot infer J/action map from {label_map_value!r}")


def _gptoss_pref0(df: pd.DataFrame) -> pd.Series:
    """Convert GPT-OSS P(J) to action-space P(act0), role by role."""
    p_j = pd.to_numeric(df["slot_pref_J"], errors="coerce").astype(float)
    map_values = df["label_map_p1"].where(df["player"].eq(1), df["label_map_p2"])
    fallback = df["label_map"] if "label_map" in df else pd.Series(index=df.index, dtype=object)
    map_values = map_values.fillna(fallback)
    j_is_act0 = map_values.map(_j_maps_to_act0)
    return pd.Series(np.where(j_is_act0, p_j, 1.0 - p_j), index=df.index, dtype=float)


def normalized_results(model: str) -> pd.DataFrame:
    """Return integrated-root rows with common decision/soft columns.

    Columns include ``cb``, ``action`` (0/1 candidate), ``ok`` (usable decision),
    and ``pref0`` (action-space P(act0) readout). GPT-OSS mixed-strategy metadata
    is retained when present.
    """
    d = read_model_results(model).rename(columns={"counterbalance_id": "cb"})
    d["model"] = model
    if model == "gptoss":
        d["action"] = pd.to_numeric(d["realized_action"], errors="coerce")
        d["ok"] = d["action"].isin([0, 1]) & ~d["commit_type"].eq("none")
        d["pref0"] = _gptoss_pref0(d)
        d["source"] = "$SCA_DATA_ROOT/substrate gptoss realized_action (mixed resolved; none dropped)"
    else:
        d["action"] = pd.to_numeric(d["decoded_action"], errors="coerce")
        d["ok"] = d["parse_ok"].astype(bool) & d["action"].isin([0, 1])
        d["pref0"] = pd.to_numeric(d["pref0"], errors="coerce")
        d["source"] = "$SCA_DATA_ROOT/substrate dense decoded_action (parse_ok-gated)"
        for col in ("commit_type", "stated_p_act0", "prob_source"):
            if col not in d:
                d[col] = np.nan
    d["action"] = d["action"].where(d["action"].isin([0, 1]), np.nan)
    return d


def decisions(model: str) -> pd.DataFrame:
    d = normalized_results(model)
    keep = [
        "model", "game_code", "player", "condition", "cb", "action", "ok",
        "commit_type", "stated_p_act0", "prob_source", "source",
    ]
    out = d[[c for c in keep if c in d.columns]].copy()
    out.attrs["source"] = str(CORRECTED_ROOT)
    return out


def soft(model: str) -> pd.DataFrame:
    d = normalized_results(model)
    return d[["model", "game_code", "player", "condition", "cb", "pref0"]].copy()


def p1_decisions(model: str, *, conditions: set[str] | None = None) -> pd.DataFrame:
    d = decisions(model)
    p1 = d[d["player"].eq(1) & d["ok"]].copy()
    if conditions is not None:
        p1 = p1[p1["condition"].isin(conditions)].copy()
    p1["decoded_action"] = p1["action"].astype(int)
    return p1


def assert_complete_pm1() -> pd.DataFrame:
    """Validate corrected-root completeness and return per-model config summaries."""
    rows: list[dict] = []
    for model in MODELS:
        for cfg_path in config_files(model):
            cfg = json.loads(cfg_path.read_text())
            rows.append({
                "model": model,
                "game_code": cfg.get("game_code", cfg_path.parent.name),
                "payoff_multiplier": cfg.get("payoff_multiplier"),
                "n_rows": cfg.get("n_rows"),
                "substrate": cfg.get("substrate", ""),
                "decoder": cfg.get("decoder", ""),
                "chat": cfg.get("chat", np.nan),
            })
    out = pd.DataFrame(rows)
    pms = set(pd.to_numeric(out["payoff_multiplier"], errors="coerce").dropna().astype(int))
    if pms != {1}:
        raise AssertionError(f"corrected substrate payoff_multiplier != {{1}}: {sorted(pms)}")
    counts = out.groupby("model")["game_code"].nunique()
    missing = set(MODELS) - set(counts.index)
    if missing or not counts.eq(144).all():
        raise AssertionError(f"corrected substrate incomplete: {counts.to_dict()}, missing={missing}")
    return out


def gptoss_commit_summary() -> pd.DataFrame:
    d = normalized_results("gptoss")
    return (d.groupby(["player", "condition", "commit_type"], dropna=False)
              .size()
              .rename("n")
              .reset_index())
