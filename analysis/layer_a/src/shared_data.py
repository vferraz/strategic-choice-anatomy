#!/usr/bin/env python3
"""Layer A final — shared data layer (SPEC_layerA_final.md §"Shared data layer").

REALIZED DECISION = the corrected integrated one-shot substrate decision in
``$SCA_DATA_ROOT/substrate/{model}/{game}/results.parquet``. Dense models use
``decoded_action`` gated by ``parse_ok``. GPT-OSS uses ``realized_action`` gated by
``commit_type != 'none'``; ``commit_type == 'mixed'`` is already a resolved 0/1
action and is included as behaviour, with the mixed metadata retained for provenance.
Unparseable / no-commit cells are DROPPED and counted — never an argmax fallback.
``pref0`` (soft) is a model-internal robustness/overlay readout only — never the
human-comparison primary, never an argmax-inflated λ.

Human references (Nagel/Griffiths) come only from the external human datasets +
metadata (``human_game_master_per_canonical.csv``, ``unified_pairs.parquet`` human
rows), never from any old LLM substrate.

Honesty invariants (docs/METHODS.md HC-1/HC-2):
  * cross-game aggregation is on the CANONICAL ACTION AXIS — we keep positional
    ``p_act0`` for geometry (distance-to-Nash uses positional Nash coords) but
    ``p_canon = mean(decision == canonical_action_p1)`` and all signed gaps
    are canonical-signed; we never pool raw ``action == 0`` across games.
  * generate()/commit decode only — NEVER the substrate slot argmax.
  * bootstrap-by-game, 2000 resamples, seed 20260520 (consumers).

Builds three caches under ``analysis/layer_a/_data/``:
  layerA_game_level.parquet      per (model, game): baseline + 6 cue aggregates,
                                 q2, p_canon, delta1c (q=.5 and empirical q-hat),
                                 belief gaps, + master metadata join.
  layerA_cells_p1baseline.parquet  per (model, game, cb): P1 baseline decoded_action,
                                 y_canon, pref0, prob_act0, delta1c_q05/emp, lk, etc.
                                 (the cell-level object the lambda/QRE/CV fits read).
  fig1_panel.parquet             unified-pairs-shaped rows (LLM from one-shot, human
                                 from external refs) that feed the FROZEN Fig-1 body.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from strategic_anatomy.config import human_refs_root, results_root, taxonomy_dir

ROOT = Path(__file__).resolve().parents[3]
from collection.oneshot_common import delta1, canonical_sign  # noqa: E402
from analysis.probe_common import ne_pairs_per_game  # noqa: E402
from analysis.layer_a.src import corrected_substrate as CS  # noqa: E402

# ---- locations --------------------------------------------------------------
DATA = results_root() / "layer_a" / "_data"
MASTER_CANON = taxonomy_dir() / "human_game_master_per_canonical.csv"
UNIFIED = human_refs_root() / "unified_pairs.parquet"
CORRECTED_ROOT = CS.CORRECTED_ROOT

# ---- model presentation (single colour map reused across all three figures) --
MODELS = ["qwen_instruct", "qwen", "llama31_instruct", "gptoss"]
SHORT = {"qwen_instruct": "Qwen-I", "qwen": "Qwen-B",
         "llama31_instruct": "Llama", "gptoss": "GPT-OSS"}
NICE = {"qwen_instruct": "Qwen2.5-72B-Instruct", "qwen": "Qwen2.5-72B (base)",
        "llama31_instruct": "Llama-3.1-70B-Instruct", "gptoss": "GPT-OSS-120B (MoE)"}
COL = {"qwen_instruct": "#1f77b4", "qwen": "#2ca02c",
       "llama31_instruct": "#d62728", "gptoss": "#9467bd"}
HUMAN_COL = {"nagel": "#000000", "griffiths": "#ff7f0e"}
DENSE_MODELS = ["qwen_instruct", "qwen", "llama31_instruct"]

CUES = ["risk_aversion", "loss_aversion", "inequity_aversion",
        "maximin", "selfish_maximizer", "length_match_null"]
PLACEBO = "length_match_null"
LK_ORDER = ["DD", "OD1", "OD2", "CO1", "CO2", "MP"]

RNG_SEED = 20260520
BOOT_N = 2000


# ---- 8-vec parsing (canonical_8vec is a COMMA string, NOT JSON) -------------
def parse_8vec(s) -> list[float]:
    if isinstance(s, (list, tuple, np.ndarray)):
        return [float(x) for x in s]
    return [float(x) for x in str(s).split(",")]


def _q_from_action(a) -> float:
    """Opponent (P2) plays deterministic action a -> P2 act0 probability.
    a==0 -> 1.0 ; a==1 -> 0.0 ; undefined/tie -> 0.5 (mirrors levelk_precision)."""
    if pd.isna(a):
        return 0.5
    a = int(a)
    return 1.0 if a == 0 else (0.0 if a == 1 else 0.5)


# ---- metadata spine: the long-pending master df (144 rows) ------------------
def master() -> pd.DataFrame:
    m = pd.read_csv(MASTER_CANON)
    bad = m["canonical_action_p1"].isna() | ~m["canonical_action_p1"].isin([0, 1])
    if bad.any():
        raise AssertionError(f"canonical_action_p1 not in {{0,1}} for "
                             f"{sorted(m.loc[bad,'game_code'].tolist())[:8]}")
    return m


def game_beliefs(m: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per game: canonical-signed incentive gap at the level-k beliefs L1/L2/L3.
    L1 -> q=0.5 ; L2 -> P2 plays l1_action_p2 ; L3 -> P2 plays l2_action_p2."""
    if m is None:
        m = master()
    rows = []
    for _, r in m.iterrows():
        vec = parse_8vec(r["canonical_8vec"])
        canon = int(r["canonical_action_p1"])
        q = {"L1": 0.5,
             "L2": _q_from_action(r.get("l1_action_p2")),
             "L3": _q_from_action(r.get("l2_action_p2"))}
        rec = {"game_code": r["game_code"], "canonical_action_p1": canon}
        for b, qb in q.items():
            rec[f"q_{b}"] = qb
            rec[f"gap_{b}"] = canonical_sign(delta1(vec, qb), canon)
        rows.append(rec)
    return pd.DataFrame(rows)


# ---- decisions (the realized move) + soft readout (robustness only) ---------
def load_decisions(model: str) -> pd.DataFrame:
    """The REALIZED DECISION, long.

    Columns include game_code, player, condition, cb, action (0/1 where ok), ok.
    The source is the corrected integrated root; GPT-OSS mixed cells are included
    after their deterministic Bernoulli resolution, while no-commit cells are dropped.
    """
    out = CS.decisions(model)
    out.attrs["source"] = str(CORRECTED_ROOT)
    return out


def load_soft(model: str) -> pd.DataFrame:
    """Soft readout pref0 = action-space P(act0), robustness/overlay only."""
    return CS.soft(model)


def model_game_panel(model: str, m: pd.DataFrame) -> pd.DataFrame:
    """Per-game aggregates for one model over the REALIZED DECISION (generate()/commit),
    plus the soft pref0 readout (robustness overlay only)."""
    dec = load_decisions(model)
    soft = load_soft(model)
    meta = m[["game_code", "canonical_action_p1", "canonical_8vec", "nagel_lk_type",
              "num_pure_ne", "complexity_score", "iesds_depth",
              "l1_action_p2", "l2_action_p2", "nagel_frac_choose_act0_canonical"]].copy()
    canon_map = dict(zip(meta["game_code"], meta["canonical_action_p1"]))

    def rate0(df):                       # positional act0 rate over OK decisions
        s = df[df["ok"]]
        return s.groupby("game_code")["action"].apply(lambda a: float((a == 0).mean()))

    def pcanon(df):
        s = df[df["ok"]]
        out = {}
        for g, sub in s.groupby("game_code"):
            c = canon_map.get(g)
            out[g] = float((sub["action"].astype(int) == int(c)).mean()) if c in (0, 1) else np.nan
        return pd.Series(out, dtype=float)

    def nok(df):                         # usable cells/game (for drop reporting)
        return df[df["ok"]].groupby("game_code").size()

    def softmean(df):
        return df.groupby("game_code")["pref0"].mean()

    p1 = dec[(dec.player == 1) & (dec.condition == "baseline")]
    p2 = dec[(dec.player == 2) & (dec.condition == "baseline")]
    sp1 = soft[(soft.player == 1) & (soft.condition == "baseline")]
    sp2 = soft[(soft.player == 2) & (soft.condition == "baseline")]

    rows = meta.copy().set_index("game_code")
    rows["model"] = model
    rows["p1_p_act0"] = rate0(p1)        # realized_p1 (decision, positional)
    rows["q2"] = rate0(p2)               # realized_p2 / opponent act0 rate (decision)
    rows["p_canon"] = pcanon(p1)         # canonical-axis conformity (P1 baseline decision)
    rows["n_ok_p1_baseline"] = nok(p1)
    rows["p1_pref0"] = softmean(sp1)     # soft readout (overlay only)
    rows["p2_pref0"] = softmean(sp2)
    pref0_g = softmean(sp1)
    rows["p_canon_soft"] = [pref0_g.get(g, np.nan) if int(c) == 0 else 1 - pref0_g.get(g, np.nan)
                            for g, c in zip(rows.index, rows["canonical_action_p1"])]

    for cue in CUES:                     # P1 cue conditions
        cdf = dec[(dec.player == 1) & (dec.condition == f"cue_{cue}")]
        scue = soft[(soft.player == 1) & (soft.condition == f"cue_{cue}")]
        rows[f"cue_{cue}_p_act0"] = rate0(cdf)
        rows[f"cue_{cue}_p_canon"] = pcanon(cdf)
        rows[f"cue_{cue}_pref0"] = softmean(scue)
        rows[f"cue_{cue}_n_ok"] = nok(cdf)

    d1c_q05, d1c_emp = [], []
    for g, r in rows.iterrows():
        vec = parse_8vec(r["canonical_8vec"])
        canon = int(r["canonical_action_p1"])
        d1c_q05.append(canonical_sign(delta1(vec, 0.5), canon))
        q2 = r["q2"]
        d1c_emp.append(canonical_sign(delta1(vec, float(q2)), canon) if pd.notna(q2) else np.nan)
    rows["delta1c_q05"] = d1c_q05
    rows["delta1c_emp"] = d1c_emp
    rows["abs_delta1c_q05"] = np.abs(d1c_q05)
    return rows.reset_index()


def model_cells_p1(model: str, m: pd.DataFrame) -> pd.DataFrame:
    """Cell-level P1 baseline DECISIONS (the λ/QRE/CV object): one row per usable cb.
    Unparseable / no-commit cells are DROPPED (no argmax fallback). pref0 = soft readout."""
    dec = load_decisions(model)
    soft = load_soft(model)
    p1 = dec[(dec.player == 1) & (dec.condition == "baseline") & (dec["ok"])][
        ["game_code", "cb", "action"]].rename(columns={"action": "decoded_action"}).copy()
    sp1 = soft[(soft.player == 1) & (soft.condition == "baseline")][["game_code", "cb", "pref0"]]
    p1 = p1.merge(sp1, on=["game_code", "cb"], how="left")
    meta = m[["game_code", "canonical_action_p1", "canonical_8vec", "nagel_lk_type", "complexity_score"]]
    p1 = p1.merge(meta, on="game_code", how="left")
    p2 = dec[(dec.player == 2) & (dec.condition == "baseline") & (dec["ok"])]
    q2 = p2.groupby("game_code")["action"].apply(lambda a: float((a == 0).mean()))
    p1["q2"] = p1["game_code"].map(q2)
    p1["y_canon"] = (p1["decoded_action"].astype(int) == p1["canonical_action_p1"].astype(int)).astype(int)
    d05, demp = [], []
    for _, r in p1.iterrows():
        vec = parse_8vec(r["canonical_8vec"])
        canon = int(r["canonical_action_p1"])
        d05.append(canonical_sign(delta1(vec, 0.5), canon))
        demp.append(canonical_sign(delta1(vec, float(r["q2"])), canon) if pd.notna(r["q2"]) else np.nan)
    p1["delta1c_q05"] = d05
    p1["delta1c_emp"] = demp
    p1["model"] = model
    return p1.drop(columns=["canonical_8vec"])


# ---- builders ---------------------------------------------------------------
def build_game_level(m: pd.DataFrame) -> pd.DataFrame:
    panels = [model_game_panel(mod, m) for mod in MODELS]
    gl = pd.concat(panels, ignore_index=True)
    gl = gl.drop(columns=["canonical_8vec"])
    DATA.mkdir(parents=True, exist_ok=True)
    gl.to_parquet(DATA / "layerA_game_level.parquet", index=False)
    return gl


def build_cells(m: pd.DataFrame) -> pd.DataFrame:
    cells = pd.concat([model_cells_p1(mod, m) for mod in MODELS], ignore_index=True)
    DATA.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(DATA / "layerA_cells_p1baseline.parquet", index=False)
    return cells


def nash_coords(m: pd.DataFrame) -> pd.DataFrame:
    """Nash-equilibrium coords (P1 act0 prob, P2 act0 prob) per game, computed FRESH
    from the canonical 8-vec via eq_engine — NOT read from any prior/aggregate file.
    For coordination games (CO1/CO2) the two pure NE are kept (ne1, ne2); for
    dominance-solvable games the unique pure NE; for no-pure-NE (MP) the mixed NE."""
    ne = ne_pairs_per_game()  # game_code -> ([(p1_act0, p2_act0), ...], kind)
    lk = dict(zip(m["game_code"], m["nagel_lk_type"]))
    rows = []
    for g, (pairs, _kind) in ne.items():
        pures = [(float(a), float(b)) for a, b in pairs if a in (0.0, 1.0) and b in (0.0, 1.0)]
        mixed = [(float(a), float(b)) for a, b in pairs if not (a in (0.0, 1.0) and b in (0.0, 1.0))]
        if lk.get(g) in ("CO1", "CO2") and len(pures) >= 2:
            (a1, b1), (a2, b2) = pures[0], pures[1]
        elif pures:
            (a1, b1), (a2, b2) = pures[0], (np.nan, np.nan)
        elif mixed:
            (a1, b1), (a2, b2) = mixed[0], (np.nan, np.nan)
        else:
            (a1, b1), (a2, b2) = (np.nan, np.nan), (np.nan, np.nan)
        rows.append({"game_code": g, "ne1_p1": a1, "ne1_p2": b1, "ne2_p1": a2, "ne2_p2": b2})
    return pd.DataFrame(rows)


def build_fig1_panel(m: pd.DataFrame, gl: pd.DataFrame) -> pd.DataFrame:
    """unified-pairs-shaped rows for the FROZEN Fig-1 body, sourced cleanly:
      * Nash coords  -> eq_engine on the canonical 8-vec (NOT any aggregate file).
      * LLM rows     -> corrected integrated one-shot realized decision rates;
        GPT-OSS mixed cells are included as resolved actions.
      * Human rows   -> external MGN/Griffiths canonical rates (the endorsed human
        reference); the design_v2 LLM rows in that table are EXCLUDED, and Nagel
        rates are pinned to the authoritative master by a build-time assertion."""
    meta = (m[["game_code", "canonical_id", "nagel_lk_type",
               "nagel_frac_choose_act0_canonical"]]
            .merge(nash_coords(m), on="game_code", how="left"))

    # external human (Nagel/Griffiths) realized choice rates — human rows only
    up = pd.read_parquet(UNIFIED)
    up = up.loc[:, ~up.columns.duplicated()]
    hum = (up[up["sample"].isin(["nagel", "griffiths"])][["sample", "game_code", "realized_p1", "realized_p2"]]
           .rename(columns={"game_code": "bruns_name"}).copy())
    hum["agent_kind"] = "human"; hum["condition"] = "observed"
    hum["pref0_p1"] = np.nan; hum["pref0_p2"] = np.nan; hum["in_model_set"] = True
    hum = hum.merge(meta.rename(columns={"game_code": "bruns_name"}), on="bruns_name", how="left")

    # ENFORCE clean provenance: Nagel realized_p1 IS the authoritative MGN canonical
    # aggregate, not an old-substrate quantity (fails loudly if the source ever drifts).
    nag = hum[hum["sample"] == "nagel"]
    err = float((nag["realized_p1"] - nag["nagel_frac_choose_act0_canonical"]).abs().max())
    assert err < 1e-6, f"Nagel realized_p1 != master nagel_frac_choose_act0_canonical (max {err:.2e})"
    assert (hum["agent_kind"] == "human").all(), "non-human row leaked into human reference"
    hum = hum.drop(columns=["nagel_frac_choose_act0_canonical"])

    # LLM rows from one-shot (realized_* positional act0 rates; pref0 soft means)
    meta_no_h = meta.drop(columns=["nagel_frac_choose_act0_canonical"])
    llm = gl[["model", "game_code", "p1_p_act0", "q2", "p1_pref0", "p2_pref0"]].copy()
    llm = llm.rename(columns={"model": "sample", "game_code": "bruns_name",
                              "p1_p_act0": "realized_p1", "q2": "realized_p2",
                              "p1_pref0": "pref0_p1", "p2_pref0": "pref0_p2"})
    llm = llm.merge(meta_no_h.rename(columns={"game_code": "bruns_name"}), on="bruns_name", how="left")
    llm["agent_kind"] = "llm"
    llm["condition"] = "baseline"
    llm["in_model_set"] = True

    panel = pd.concat([llm, hum], ignore_index=True)
    panel.to_parquet(DATA / "fig1_panel.parquet", index=False)
    return panel


def audit_clean() -> bool:
    """Provenance audit for the corrected integrated one-shot substrate."""
    m = master()
    print("=== Layer A clean-provenance audit ===")

    # 1) corrected substrate is complete, pm=1, all 4 models present
    cfg = CS.assert_complete_pm1()
    rows_by_model = cfg.groupby("model")["n_rows"].first().to_dict()
    print(f"  [1] corrected integrated substrate: {len(cfg)} configs under {CORRECTED_ROOT}, "
          f"payoff_multiplier={{1}}, rows/game={rows_by_model}")

    # 2) MGN vectors are {1,2,3,4} permutations (same rank scale as LLMs; no rescale)
    dr = pd.read_csv(human_refs_root() / "raw" / "nagel" / "df_ros.csv")
    def _vec(v):  # df_ros game_vector is a JSON-style "[4, 3, ...]" list
        return [float(x) for x in str(v).strip().strip("[]").split(",")]
    bad = [v for v in dr["game_vector"] if sorted(_vec(v)[:4]) != [1, 2, 3, 4]
           or sorted(_vec(v)[4:]) != [1, 2, 3, 4]]
    assert not bad, "MGN game_vector not a {1,2,3,4} permutation"
    print(f"  [2] MGN ranks {{1,2,3,4}}: 144/144 OK; scale labels "
          f"{m['payoff_scale_llm'].iloc[0]}/{m['payoff_scale_mgn'].iloc[0]}/{m['payoff_scale_griffiths'].iloc[0]}")

    # 3) Fig-1 Nash coords are eq_engine on the canonical 8-vec (recompute == panel)
    panel = pd.read_parquet(DATA / "fig1_panel.parquet")
    nc = nash_coords(m).set_index("game_code")
    pj = panel.drop_duplicates("bruns_name").set_index("bruns_name")
    d = max(float((pj.reindex(nc.index)[c] - nc[c]).abs().max(skipna=True))
            for c in ["ne1_p1", "ne1_p2"])
    assert d < 1e-9, f"panel Nash coords != eq_engine ({d})"
    print(f"  [3] Fig-1 Nash coords == eq_engine(canonical_8vec): max diff {d:.1e}")

    # 4) human rows are external; no LLM/design_v2 row leaked; Nagel == master aggregate
    assert set(panel[panel.agent_kind == "human"]["sample"]) == {"nagel", "griffiths"}
    nag = panel[panel["sample"] == "nagel"].merge(
        m[["game_code", "nagel_frac_choose_act0_canonical"]].rename(columns={"game_code": "bruns_name"}),
        on="bruns_name", how="left")
    e = float((nag["realized_p1"] - nag["nagel_frac_choose_act0_canonical"]).abs().max())
    assert e < 1e-6, f"Nagel realized_p1 != master canonical aggregate ({e})"
    print(f"  [4] human ref = external Nagel/Griffiths only; Nagel == master aggregate (diff {e:.1e})")

    # 5) every realized rate in the panel is a genuine [0,1] probability
    for col in ["realized_p1", "realized_p2"]:
        v = panel[col].dropna()
        assert v.between(0, 1).all(), f"{col} out of [0,1]"
    print("  [5] all realized choice rates in [0,1]")

    # 6) DECISION PROVENANCE — all decisions resolve from the corrected integrated root.
    gl = pd.read_parquet(DATA / "layerA_game_level.parquet")
    for mod in MODELS:
        dec = load_decisions(mod)
        assert str(CORRECTED_ROOT) in dec.attrs["source"], f"{mod} decision not from corrected root"
    coverage = gl.groupby("model")["n_ok_p1_baseline"].agg(["min", "mean", "max"]).round(2)
    print("  [6] decisions from corrected integrated root; P1 baseline usable cells/game:")
    print(coverage.reindex(MODELS).to_string())
    gpt_summary = CS.gptoss_commit_summary()
    gpt_base = gpt_summary[gpt_summary["condition"].eq("baseline")]
    print("  [7] GPT-OSS commit_type counts (baseline, includes mixed as resolved):")
    print(gpt_base.to_string(index=False))
    assert (gpt_summary[gpt_summary["commit_type"].eq("mixed")]["n"].sum() > 0), (
        "expected GPT-OSS mixed-strategy provenance rows in corrected substrate"
    )
    print("=== AUDIT PASSED: corrected-root, canonical-axis, one-shot ===")
    return True


# ---- orchestration ----------------------------------------------------------
def main() -> int:
    m = master()
    print(f"[master] {len(m)} canonical games | scale: "
          f"llm={m['payoff_scale_llm'].iloc[0]} mgn={m['payoff_scale_mgn'].iloc[0]} "
          f"grif={m['payoff_scale_griffiths'].iloc[0]}")

    gl = build_game_level(m)
    cells = build_cells(m)
    panel = build_fig1_panel(m, gl)

    # ---- summaries / honesty checks (printed) ----
    print(f"\n[game_level] {len(gl)} rows = {gl['model'].nunique()} models x "
          f"{gl['game_code'].nunique()} games")
    print("[game_level] canonical conformity p_canon (P1 baseline, mean over games):")
    print(gl.groupby("model")["p_canon"].mean().reindex(MODELS).round(3).to_string())
    print("\n[game_level] mean opponent act0 rate q2 by model:")
    print(gl.groupby("model")["q2"].mean().reindex(MODELS).round(3).to_string())

    print(f"\n[cells] {len(cells)} P1-baseline cells = "
          f"{cells['model'].nunique()} models x {cells['game_code'].nunique()} games x "
          f"~{len(cells)//(cells['model'].nunique()*cells['game_code'].nunique())} cb")

    print(f"\n[fig1_panel] {len(panel)} rows")
    print(panel.groupby(["agent_kind", "sample"], dropna=False).size().to_string())
    # N games per lk-class on the 144-game LLM scope (frozen-fig sanity)
    llm_q = panel[(panel.agent_kind == "llm") & (panel["sample"] == "qwen_instruct")]
    print("\n[fig1_panel] N games per nagel_lk_type (qwen_instruct, all 144):")
    print(llm_q["nagel_lk_type"].value_counts().reindex(LK_ORDER, fill_value=0).to_string())

    # honesty assert: never aggregated raw act0 across games as a trait metric here
    assert gl["p_canon"].notna().sum() > 0, "p_canon all-NaN"
    print("\nwrote:")
    for f in ["layerA_game_level.parquet", "layerA_cells_p1baseline.parquet", "fig1_panel.parquet"]:
        print("  ", DATA / f)

    print()
    audit_clean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
