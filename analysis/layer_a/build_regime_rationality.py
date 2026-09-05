#!/usr/bin/env python3
"""Corrected Layer-A Figure 1 tables: regime-matched rationality measures.

Reads corrected integrated-root decisions directly from
``$SCA_DATA_ROOT/substrate/{model}/{game}/results.parquet``. Dense models use
``decoded_action`` gated by ``parse_ok``. GPT-OSS uses ``realized_action`` gated by
``commit_type != 'none'``; mixed rows are already resolved to 0/1 and are included
as behaviour.

The stale one-shot unified cache is not used for LLM behaviour. Human rows are
reused from data/human_refs/unified_pairs.parquet. Moore et al. rows
are aggregated per canonical game. Zhu et al. payoff and coordination measures
are instead computed on each original cardinal matrix (``griffiths_game_id``)
before aggregation; a cardinal matrix is never selected from a collapsed
canonical-game group.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.probe_common import game_meta, parse_8vec  # noqa: E402
from strategic_anatomy import eq_engine as EQ  # noqa: E402
from analysis.layer_a.src import corrected_substrate as CS  # noqa: E402
from strategic_anatomy.config import human_refs_root, repo_root, results_root

ROOT = repo_root()


TAB_DIR = results_root() / "layer_a"
UNIFIED = human_refs_root() / "unified_pairs.parquet"

SEED = 20260520
N_BOOT = 2000
LLMS = ("qwen", "qwen_instruct", "llama31_instruct", "gptoss")
HUMANS = ("nagel", "griffiths")
AGENTS = LLMS + HUMANS
DISPLAY = {
    "qwen": "Qwen2.5-72B",
    "qwen_instruct": "Qwen2.5-72B-Instruct",
    "llama31_instruct": "Llama-3.1-70B-Instruct",
    "gptoss": "GPT-OSS-120B",
    "nagel": "Moore et al. (human)",
    "griffiths": "Zhu et al. (human)",
}
FAMILIES = ("DD", "OD1", "OD2", "CO1", "CO2", "MP")
DOM_FAMILIES = ("DD", "OD1", "OD2")
CO_FAMILIES = ("CO1", "CO2")


def _act0_prob(action: int) -> float:
    return 1.0 if int(action) == 0 else 0.0


def _clip01(p: float) -> float:
    return float(min(1.0, max(0.0, p)))


def _boolish(value) -> bool | float:
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except TypeError:
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return np.nan


def parse_matrix_8vec(s) -> list[float]:
    """Parse either bare ``1,4,...`` strings or bracketed human-cache vectors."""
    if isinstance(s, (list, tuple, np.ndarray)):
        return [float(x) for x in s]
    text = str(s).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return parse_8vec(text)


def cell_label(cell: tuple[int, int] | None) -> str:
    return "" if cell is None else f"{int(cell[0])},{int(cell[1])}"


def parse_cell_set(text: str) -> set[tuple[int, int]]:
    cleaned = str(text).replace("(", "").replace(")", "")
    return {tuple(map(int, part.split(","))) for part in cleaned.split(";") if part}


def parse_cell_label(text: str) -> tuple[int, int]:
    return tuple(map(int, str(text).split(",")))  # type: ignore[return-value]


def entropy_binary(p: float) -> float:
    p = _clip01(float(p))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def expected_payoffs(p1_act0: float, p2_act0: float, vec8) -> tuple[float, float]:
    U1, U2 = EQ.matrices_from_8vec(vec8)
    p = float(p1_act0)
    q = float(p2_act0)
    probs = np.array([[p * q, p * (1 - q)], [(1 - p) * q, (1 - p) * (1 - q)]])
    return float((probs * U1).sum()), float((probs * U2).sum())


def game_reference_rows() -> pd.DataFrame:
    meta = game_meta().copy()
    rows = []
    for r in meta.itertuples(index=False):
        vec = parse_matrix_8vec(r.canonical_8vec)
        U1, U2 = EQ.matrices_from_8vec(vec)
        pure = EQ.pure_nash_equilibria(U1, U2)
        ne_pairs, ne_kind = EQ.enumerate_nash(U1, U2)
        cell_sums = U1 + U2
        rec = {
            "game_code": r.game_code,
            "family": r.nagel_lk_type,
            "canonical_action_p1": int(r.canonical_action_p1),
            "canonical_action_p2": int(r.canonical_action_p2),
            "vec8": ",".join(str(int(x)) for x in vec),
            "payoff_kind": "ordinal_rank",
            "num_pure_ne_recomputed": len(pure),
            "ne_kind_recomputed": ne_kind,
            "pure_ne_cells": ";".join(f"{i},{j}" for i, j in pure),
            "worst_total": float(cell_sums.min()),
            "best_total": float(cell_sums.max()),
            "random_total": sum(expected_payoffs(0.5, 0.5, vec)),
            "mp_mixed_p1": np.nan,
            "mp_mixed_p2": np.nan,
        }
        if len(pure) == 1:
            i, j = pure[0]
            rec.update({
                "unique_ne_p1": _act0_prob(i),
                "unique_ne_p2": _act0_prob(j),
                "ne_total": float(U1[i, j] + U2[i, j]),
            })
        else:
            rec.update({"unique_ne_p1": np.nan, "unique_ne_p2": np.nan})
            pd_ne = EQ.payoff_dominant_ne(pure, U1, U2)
            if pd_ne is not None:
                i, j = pd_ne
                rec["ne_total"] = float(U1[i, j] + U2[i, j])
            elif pure:
                rec["ne_total"] = float(np.mean([U1[i, j] + U2[i, j] for i, j in pure]))
            elif ne_pairs:
                p, q = ne_pairs[0]
                rec["mp_mixed_p1"] = float(p)
                rec["mp_mixed_p2"] = float(q)
                rec["ne_total"] = sum(expected_payoffs(p, q, vec))
            else:
                rec["ne_total"] = np.nan
        if len(pure) == 2:
            pd_ne = EQ.payoff_dominant_ne(pure, U1, U2)
            rd_ne = EQ.risk_dominant_ne(pure, U1, U2)
            rec["payoff_ne_cell"] = cell_label(pd_ne)
            rec["risk_ne_cell"] = cell_label(rd_ne)
            feature_rankable = _boolish(getattr(r, "ne_pareto_rankable", np.nan))
            computed_rankable = pd_ne is not None
            if not pd.isna(feature_rankable) and bool(feature_rankable) != computed_rankable:
                raise AssertionError(f"{r.game_code}: feature rankability disagrees with recomputed NE")
            if computed_rankable:
                feature_p1 = getattr(r, "payoff_dominant_ne_p1", np.nan)
                feature_p2 = getattr(r, "payoff_dominant_ne_p2", np.nan)
                if not pd.isna(feature_p1) and not pd.isna(feature_p2):
                    feature_pd = (int(feature_p1), int(feature_p2))
                    if feature_pd != pd_ne:
                        raise AssertionError(f"{r.game_code}: payoff-dominant NE mismatch")
                other = next(cell for cell in pure if cell != pd_ne)
                rec["co_structure"] = "rankable"
                rec["co_structure_label"] = "Pareto-rankable"
                rec["pdom_cell"] = cell_label(pd_ne)
                rec["other_ne_cell"] = cell_label(other)
            else:
                rec["co_structure"] = "conflict"
                rec["co_structure_label"] = "Distributional conflict"
                rec["pdom_cell"] = ""
                rec["other_ne_cell"] = ""
            rec["ne_pareto_rankable"] = bool(computed_rankable)
            rec["ne_distributional_conflict"] = bool(not computed_rankable)
        else:
            rec["payoff_ne_cell"] = ""
            rec["risk_ne_cell"] = ""
            rec["co_structure"] = ""
            rec["co_structure_label"] = ""
            rec["pdom_cell"] = ""
            rec["other_ne_cell"] = ""
            rec["ne_pareto_rankable"] = np.nan
            rec["ne_distributional_conflict"] = np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    counts = out.groupby("family")["num_pure_ne_recomputed"].agg(["size", "unique"]).to_dict("index")
    if not (out[out.family.isin(DOM_FAMILIES)]["num_pure_ne_recomputed"].eq(1).all()):
        raise AssertionError("dominance-solvable families are not all unique-pure-NE")
    if not (out[out.family.isin(CO_FAMILIES)]["num_pure_ne_recomputed"].eq(2).all()):
        raise AssertionError("coordination families are not all two-pure-NE")
    if not (out[out.family.eq("MP")]["num_pure_ne_recomputed"].eq(0).all()):
        raise AssertionError("MP family is not all zero-pure-NE")
    co = out[out["family"].isin(CO_FAMILIES)]
    co_counts = co.groupby(["co_structure", "family"]).size()
    expected = {("rankable", "CO1"): 5, ("rankable", "CO2"): 4,
                ("conflict", "CO1"): 4, ("conflict", "CO2"): 5}
    for key, value in expected.items():
        if int(co_counts.get(key, 0)) != value:
            raise AssertionError(f"unexpected CO structure count {key}: {co_counts.get(key, 0)}")
    print("  [assert] recomputed pure-NE regimes:", counts)
    return out


def _load_llm_generated_cells() -> pd.DataFrame:
    CS.assert_complete_pm1()
    frames = []
    for model in LLMS:
        d = CS.decisions(model)
        b = d[(d["condition"].eq("baseline")) & (d["ok"])].copy()
        b = b.rename(columns={"cb": "cell_id"})
        b["agent"] = model
        frames.append(b[["agent", "game_code", "cell_id", "player", "action"]])
    out = pd.concat(frames, ignore_index=True)
    out["action"] = out["action"].astype(int)
    return out


def _load_llm_soft_rates() -> pd.DataFrame:
    frames = []
    for model in LLMS:
        d = CS.soft(model)
        b = d[d["condition"].eq("baseline")].copy()
        b["agent"] = model
        frames.append(b[["agent", "game_code", "player", "pref0"]])
    s = pd.concat(frames, ignore_index=True)
    return (s.groupby(["agent", "game_code", "player"], as_index=False)
             .agg(pref0=("pref0", "mean"), n_soft=("pref0", "size")))


def llm_game_rates() -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = _load_llm_generated_cells()
    hard = (cells.groupby(["agent", "game_code", "player"], as_index=False)
            .agg(realized_act0=("action", lambda x: float((x == 0).mean())),
                 n_cells=("action", "size")))
    h1 = hard[hard.player.eq(1)].rename(columns={"realized_act0": "realized_p1", "n_cells": "n_p1"})
    h2 = hard[hard.player.eq(2)].rename(columns={"realized_act0": "realized_p2", "n_cells": "n_p2"})
    game = h1[["agent", "game_code", "realized_p1", "n_p1"]].merge(
        h2[["agent", "game_code", "realized_p2", "n_p2"]], on=["agent", "game_code"], how="inner")

    soft = _load_llm_soft_rates()
    s1 = soft[soft.player.eq(1)].rename(columns={"pref0": "pref0_p1", "n_soft": "n_soft_p1"})
    s2 = soft[soft.player.eq(2)].rename(columns={"pref0": "pref0_p2", "n_soft": "n_soft_p2"})
    game = game.merge(s1[["agent", "game_code", "pref0_p1", "n_soft_p1"]],
                      on=["agent", "game_code"], how="left")
    game = game.merge(s2[["agent", "game_code", "pref0_p2", "n_soft_p2"]],
                      on=["agent", "game_code"], how="left")
    game["agent_kind"] = "llm"
    game["display"] = game["agent"].map(DISPLAY)
    return game, cells


def human_game_rates() -> pd.DataFrame:
    up = pd.read_parquet(UNIFIED)
    up = up.loc[:, ~up.columns.duplicated()].copy()
    h = up[up["agent_kind"].eq("human")].copy()
    # This canonical-game panel is used for action-axis measures only. Zhu et al.
    # cardinal matrices are deliberately absent: payoff and coordination use
    # ``griffiths_original_rows`` below.
    keep = ["sample", "game_code", "realized_p1", "realized_p2", "payoff_kind"]
    h = h[keep].dropna(subset=["realized_p1", "realized_p2"])
    g = (h.groupby(["sample", "game_code"], as_index=False)
         .agg(realized_p1=("realized_p1", "mean"),
              realized_p2=("realized_p2", "mean"),
              payoff_kind=("payoff_kind", "first"),
              n_human_rows=("realized_p1", "size")))
    g = g.rename(columns={"sample": "agent"})
    g["agent_kind"] = "human"
    g["display"] = g["agent"].map(DISPLAY)
    g["pref0_p1"] = np.nan
    g["pref0_p2"] = np.nan
    g["n_p1"] = g["n_human_rows"]
    g["n_p2"] = g["n_human_rows"]
    return g.drop(columns=["n_human_rows"])


def griffiths_original_rows(refs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return Zhu et al. marginals with their own cardinal matrix intact.

    Each original game has two role-oriented rows. The payoff and independent
    coordination functionals are invariant to that joint role/action
    transformation, so downstream code verifies invariance and collapses the
    pair to one ``griffiths_game_id`` before family-level averaging.
    """
    up = pd.read_parquet(UNIFIED)
    up = up.loc[:, ~up.columns.duplicated()].copy()
    keep = [
        "game_code", "griffiths_game_id", "griffiths_unique_id",
        "realized_p1", "realized_p2", "matrix_8vec",
    ]
    g = up.loc[up["sample"].eq("griffiths"), keep].dropna(
        subset=["griffiths_game_id", "realized_p1", "realized_p2", "matrix_8vec"]
    ).copy()
    if refs is None:
        refs = game_reference_rows()
    ref_cols = [
        "game_code", "family", "pure_ne_cells", "co_structure",
        "co_structure_label", "pdom_cell", "other_ne_cell",
    ]
    g = g.merge(refs[ref_cols], on="game_code", how="left", validate="many_to_one")
    g["agent"] = "griffiths"
    g["display"] = DISPLAY["griffiths"]
    g["payoff_kind"] = "cardinal"
    g["vec8"] = g["matrix_8vec"].map(
        lambda value: ",".join(str(float(x)) for x in parse_matrix_8vec(value))
    )

    pair_sizes = g.groupby("griffiths_game_id").size()
    if not pair_sizes.eq(2).all():
        bad = pair_sizes[~pair_sizes.eq(2)].head().to_dict()
        raise AssertionError(f"Zhu original games must have two oriented rows: {bad}")
    if (g.groupby("griffiths_game_id")["family"].nunique() != 1).any():
        raise AssertionError("Zhu oriented rows disagree on game family")
    if (g.groupby("griffiths_game_id")["co_structure"].nunique(dropna=False) != 1).any():
        raise AssertionError("Zhu oriented rows disagree on coordination structure")
    print(
        "  [assert] Zhu cardinal source: "
        f"{g['griffiths_game_id'].nunique()} original games, {len(g)} oriented rows"
    )
    return g


def build_game_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    refs = game_reference_rows()
    llm, cells = llm_game_rates()
    human = human_game_rates()
    panel = pd.concat([llm, human], ignore_index=True)
    panel = panel.merge(refs, on="game_code", how="left", suffixes=("", "_ref"))
    if "payoff_kind_ref" in panel:
        panel["payoff_kind"] = panel["payoff_kind"].fillna(panel["payoff_kind_ref"])
    panel["display"] = panel["agent"].map(DISPLAY)
    return panel, cells


def proximity_to_unique_ne(p1: float, p2: float, ne1: float, ne2: float) -> float:
    d = math.sqrt((p1 - ne1) ** 2 + (p2 - ne2) ** 2)
    d_chance = math.sqrt((0.5 - ne1) ** 2 + (0.5 - ne2) ** 2)
    return float(1.0 - d / d_chance)


def _boot_mean(vals: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(SEED)
    means = vals[rng.integers(0, vals.size, size=(N_BOOT, vals.size))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def dominance_conformity(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel[panel["family"].isin(DOM_FAMILIES)].copy()
    d["R_realized_game"] = [
        proximity_to_unique_ne(p1, p2, n1, n2)
        for p1, p2, n1, n2 in zip(d.realized_p1, d.realized_p2, d.unique_ne_p1, d.unique_ne_p2)
    ]
    d["R_soft_game"] = [
        proximity_to_unique_ne(p1, p2, n1, n2) if np.isfinite(p1) and np.isfinite(p2) else np.nan
        for p1, p2, n1, n2 in zip(d.pref0_p1, d.pref0_p2, d.unique_ne_p1, d.unique_ne_p2)
    ]
    rows = []
    for (agent, fam), sub in d.groupby(["agent", "family"], sort=False):
        m, lo, hi = _boot_mean(sub["R_realized_game"].to_numpy())
        sm, slo, shi = _boot_mean(sub["R_soft_game"].to_numpy())
        rows.append({
            "agent": agent,
            "display": DISPLAY[agent],
            "family": fam,
            "n_games": int(sub["game_code"].nunique()),
            "R_realized": m,
            "R_realized_ci_lo": lo,
            "R_realized_ci_hi": hi,
            "R_soft": sm,
            "R_soft_ci_lo": slo,
            "R_soft_ci_hi": shi,
        })
    return pd.DataFrame(rows)


def _payoff_game_records(source: pd.DataFrame) -> pd.DataFrame:
    """Evaluate payoff functionals row-wise using each row's own matrix."""
    rows = []
    for r in source.itertuples(index=False):
        vec = parse_matrix_8vec(r.vec8)
        e1, e2 = expected_payoffs(r.realized_p1, r.realized_p2, vec)
        rec = r._asdict()
        rec["realized_total_game"] = e1 + e2
        U1, U2 = EQ.matrices_from_8vec(vec)
        rec["worst_total_game"] = float((U1 + U2).min())
        rec["best_total_game"] = float((U1 + U2).max())
        rec["random_total_game"] = sum(expected_payoffs(0.5, 0.5, vec))
        pure = EQ.pure_nash_equilibria(U1, U2)
        if len(pure) == 1:
            i, j = pure[0]
            rec["ne_total_game"] = float(U1[i, j] + U2[i, j])
        elif pure:
            pd_ne = EQ.payoff_dominant_ne(pure, U1, U2)
            if pd_ne is not None:
                i, j = pd_ne
                rec["ne_total_game"] = float(U1[i, j] + U2[i, j])
            else:
                rec["ne_total_game"] = float(np.mean([U1[i, j] + U2[i, j] for i, j in pure]))
        else:
            ne_pairs, _ = EQ.enumerate_nash(U1, U2)
            rec["ne_total_game"] = sum(expected_payoffs(*ne_pairs[0], vec)) if ne_pairs else np.nan
        rows.append(rec)
    g = pd.DataFrame(rows)
    if g.empty:
        return g
    denom = (g["best_total_game"] - g["worst_total_game"]).replace(0, np.nan)
    g["realized_efficiency_game"] = (g["realized_total_game"] - g["worst_total_game"]) / denom
    g["eq_efficiency_game"] = (g["ne_total_game"] - g["worst_total_game"]) / denom
    g["random_efficiency_game"] = (g["random_total_game"] - g["worst_total_game"]) / denom
    return g


def _payoff_summary_record(
    agent: str,
    family: str,
    sub: pd.DataFrame,
    *,
    n_games: int,
    n_game_codes: int,
    n_source_rows: int,
    aggregation_unit: str,
) -> dict:
    return {
        "agent": agent,
        "display": DISPLAY[agent],
        "family": family,
        "payoff_kind": sub["payoff_kind"].iloc[0],
        "aggregation_unit": aggregation_unit,
        "n_games": int(n_games),
        "n_game_codes": int(n_game_codes),
        "n_source_rows": int(n_source_rows),
        "realized_total": float(sub["realized_total_game"].mean()),
        "eq_total": float(sub["ne_total_game"].mean()),
        "random_total": float(sub["random_total_game"].mean()),
        "worst_total": float(sub["worst_total_game"].mean()),
        "best_total": float(sub["best_total_game"].mean()),
        "realized_efficiency": float(sub["realized_efficiency_game"].mean()),
        "eq_efficiency": float(sub["eq_efficiency_game"].mean()),
        "random_efficiency": float(sub["random_efficiency_game"].mean()),
    }


def payoff_efficiency(
    panel: pd.DataFrame,
    griffiths_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expected payoff efficiency under independently elicited role marginals.

    LLM and Moore rows are one record per canonical game. Zhu rows are evaluated
    against the original cardinal matrix, verified across the two oriented
    records, and averaged once per original ``griffiths_game_id``.
    """
    if griffiths_rows is None:
        griffiths_rows = griffiths_original_rows()

    # The collapsed Zhu canonical panel intentionally has no cardinal matrix and
    # must never enter this calculation.
    g = _payoff_game_records(panel[~panel["agent"].eq("griffiths")].copy())
    out = []
    for (agent, fam), sub in g.groupby(["agent", "family"], sort=False):
        out.append(_payoff_summary_record(
            agent,
            fam,
            sub,
            n_games=sub["game_code"].nunique(),
            n_game_codes=sub["game_code"].nunique(),
            n_source_rows=len(sub),
            aggregation_unit="canonical_game_code",
        ))

    zhu = _payoff_game_records(griffiths_rows)
    metric_cols = [
        "realized_total_game", "ne_total_game", "random_total_game",
        "worst_total_game", "best_total_game", "realized_efficiency_game",
        "eq_efficiency_game", "random_efficiency_game",
    ]
    for column in metric_cols:
        span = zhu.groupby("griffiths_game_id")[column].agg(
            lambda values: float(values.max() - values.min())
        )
        if (span > 1e-9).any():
            raise AssertionError(f"Zhu oriented rows disagree on {column}")

    one_per_original = (
        zhu.groupby("griffiths_game_id", as_index=False)
        .agg(
            family=("family", "first"),
            payoff_kind=("payoff_kind", "first"),
            **{column: (column, "mean") for column in metric_cols},
        )
    )
    for family, sub in one_per_original.groupby("family", sort=False):
        source_sub = zhu[zhu["family"].eq(family)]
        out.append(_payoff_summary_record(
            "griffiths",
            family,
            sub,
            n_games=len(sub),
            n_game_codes=source_sub["game_code"].nunique(),
            n_source_rows=len(source_sub),
            aggregation_unit="griffiths_game_id",
        ))
    return pd.DataFrame(out)


def coordination_metrics(
    panel: pd.DataFrame,
    cells: pd.DataFrame,
    griffiths_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expected coordination implied by separately elicited role policies."""
    if griffiths_rows is None:
        griffiths_rows = griffiths_original_rows()
    refs = game_reference_rows()
    co_refs = refs[refs["family"].isin(CO_FAMILIES)][[
        "game_code", "family", "pure_ne_cells", "co_structure",
        "co_structure_label", "pdom_cell", "other_ne_cell",
    ]].copy()
    count_table = (co_refs.groupby(["co_structure", "family"]).size()
                   .unstack(fill_value=0).reindex(["rankable", "conflict"]))

    def base_counts(structure: str) -> dict[str, int]:
        row = count_table.loc[structure]
        return {
            "co1_count": int(row.get("CO1", 0)),
            "co2_count": int(row.get("CO2", 0)),
        }

    def row_record(
        agent: str,
        structure: str,
        method: str,
        n_games: int,
        n_cells,
        pdom: float,
        other: float,
        coord: float,
        miscoord: float,
        *,
        n_game_codes: int | None = None,
        n_source_rows: int | None = None,
        aggregation_unit: str,
        family_counts: dict[str, int] | None = None,
    ) -> dict:
        counts = base_counts(structure) if family_counts is None else family_counts
        return {
            "agent": agent,
            "display": DISPLAY[agent],
            "structure": structure,
            "structure_label": "Pareto-rankable" if structure == "rankable" else "Distributional conflict",
            "method": method,
            "aggregation_unit": aggregation_unit,
            "n_games": int(n_games),
            "n_game_codes": int(n_games if n_game_codes is None else n_game_codes),
            "n_source_rows": int(n_games if n_source_rows is None else n_source_rows),
            "n_cells": n_cells,
            "co1_count": counts["co1_count"],
            "co2_count": counts["co2_count"],
            "pdom_share": pdom,
            "other_share": other,
            "coord_share": coord,
            "miscoord_share": miscoord,
        }

    rows = []
    for agent in LLMS:
        acells = cells[(cells["agent"].eq(agent))].copy()
        p1 = acells[acells.player.eq(1)].rename(columns={"action": "action_p1"})
        p2 = acells[acells.player.eq(2)].rename(columns={"action": "action_p2"})
        pair = p1[["game_code", "cell_id", "action_p1"]].merge(
            p2[["game_code", "cell_id", "action_p2"]], on=["game_code", "cell_id"], how="inner")
        pair = pair.merge(co_refs, on="game_code", how="inner")
        for structure, sub in pair.groupby("co_structure"):
            pdom = []
            other = []
            coord = []
            miscoord = []
            for r in sub.itertuples(index=False):
                pure = parse_cell_set(r.pure_ne_cells)
                prof = (int(r.action_p1), int(r.action_p2))
                if structure == "rankable":
                    pdom_cell = parse_cell_label(r.pdom_cell)
                    other_cell = parse_cell_label(r.other_ne_cell)
                    pdom.append(float(prof == pdom_cell))
                    other.append(float(prof == other_cell))
                    miscoord.append(float(prof not in pure))
                else:
                    coord.append(float(prof in pure))
                    miscoord.append(float(prof not in pure))
            if structure == "rankable":
                pdom_share = float(np.mean(pdom))
                other_share = float(np.mean(other))
                coord_share = pdom_share + other_share
            else:
                pdom_share = np.nan
                other_share = np.nan
                coord_share = float(np.mean(coord))
            rows.append(row_record(
                agent, structure, "matched_cells", sub["game_code"].nunique(), int(len(sub)),
                pdom_share, other_share, coord_share, float(np.mean(miscoord)),
                n_game_codes=sub["game_code"].nunique(),
                n_source_rows=len(sub),
                aggregation_unit="matched_counterbalance_cell",
            ))
    def independent_row_metrics(row) -> dict[str, float]:
        pure = parse_cell_set(row.pure_ne_cells)
        probs = {
            (0, 0): row.realized_p1 * row.realized_p2,
            (0, 1): row.realized_p1 * (1 - row.realized_p2),
            (1, 0): (1 - row.realized_p1) * row.realized_p2,
            (1, 1): (1 - row.realized_p1) * (1 - row.realized_p2),
        }
        if row.co_structure == "rankable":
            pdom = probs[parse_cell_label(row.pdom_cell)]
            other = probs[parse_cell_label(row.other_ne_cell)]
            coord = pdom + other
            return {
                "pdom": float(pdom), "other": float(other),
                "coord": float(coord), "miscoord": float(1.0 - coord),
            }
        coord = sum(probs[cell] for cell in pure)
        return {
            "pdom": np.nan, "other": np.nan,
            "coord": float(coord), "miscoord": float(1.0 - coord),
        }

    # Moore aggregate: expected joint profile under independently elicited role
    # frequencies, one record per canonical game.
    hp = panel[(panel["agent"].eq("nagel")) & (panel["family"].isin(CO_FAMILIES))].copy()
    for (agent, structure), sub in hp.groupby(["agent", "co_structure"]):
        metrics = pd.DataFrame(independent_row_metrics(r) for r in sub.itertuples(index=False))
        if structure == "rankable":
            pdom_share = float(metrics["pdom"].mean())
            other_share = float(metrics["other"].mean())
        else:
            pdom_share = np.nan
            other_share = np.nan
        rows.append(row_record(
            agent, structure, "aggregate_independent", sub["game_code"].nunique(), np.nan,
            pdom_share, other_share, float(metrics["coord"].mean()),
            float(metrics["miscoord"].mean()),
            n_game_codes=sub["game_code"].nunique(),
            n_source_rows=len(sub),
            aggregation_unit="canonical_game_code",
        ))

    # Zhu aggregate: evaluate both oriented records with their original-game
    # marginals, verify invariance, then count each original matrix once.
    zhu_source = griffiths_rows[griffiths_rows["family"].isin(CO_FAMILIES)].copy()
    zhu_metrics = []
    for r in zhu_source.itertuples(index=False):
        rec = {
            "griffiths_game_id": r.griffiths_game_id,
            "game_code": r.game_code,
            "family": r.family,
            "structure": r.co_structure,
        }
        rec.update(independent_row_metrics(r))
        zhu_metrics.append(rec)
    zhu_metrics = pd.DataFrame(zhu_metrics)
    for column in ["pdom", "other", "coord", "miscoord"]:
        span = zhu_metrics.groupby("griffiths_game_id")[column].agg(
            lambda values: float(values.max() - values.min())
        )
        finite = span[np.isfinite(span)]
        if (finite > 1e-9).any():
            raise AssertionError(f"Zhu oriented rows disagree on {column}")
    one_per_original = (
        zhu_metrics.groupby("griffiths_game_id", as_index=False)
        .agg(
            family=("family", "first"),
            structure=("structure", "first"),
            pdom=("pdom", "mean"),
            other=("other", "mean"),
            coord=("coord", "mean"),
            miscoord=("miscoord", "mean"),
        )
    )
    for structure, sub in one_per_original.groupby("structure", sort=False):
        source_sub = zhu_metrics[zhu_metrics["structure"].eq(structure)]
        family_counts = {
            "co1_count": int(sub["family"].eq("CO1").sum()),
            "co2_count": int(sub["family"].eq("CO2").sum()),
        }
        rows.append(row_record(
            "griffiths", structure, "original_game_independent", len(sub), np.nan,
            float(sub["pdom"].mean()) if structure == "rankable" else np.nan,
            float(sub["other"].mean()) if structure == "rankable" else np.nan,
            float(sub["coord"].mean()), float(sub["miscoord"].mean()),
            n_game_codes=source_sub["game_code"].nunique(),
            n_source_rows=len(source_sub),
            aggregation_unit="griffiths_game_id",
            family_counts=family_counts,
        ))
    return pd.DataFrame(rows)


def mp_descriptive(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mp = panel[panel["family"].eq("MP")].copy()
    for r in mp.itertuples(index=False):
        vec = parse_matrix_8vec(r.vec8)
        U1, U2 = EQ.matrices_from_8vec(vec)
        ne_pairs, _ = EQ.enumerate_nash(U1, U2)
        p_mix, q_mix = ne_pairs[0] if ne_pairs else (np.nan, np.nan)
        rec = {
            "agent": r.agent,
            "display": DISPLAY[r.agent],
            "game_code": r.game_code,
            "payoff_kind": r.payoff_kind,
            "realized_p1": r.realized_p1,
            "realized_p2": r.realized_p2,
            "mixed_ne_p1": p_mix,
            "mixed_ne_p2": q_mix,
            "abs_freq_gap_p1": abs(r.realized_p1 - p_mix) if np.isfinite(p_mix) else np.nan,
            "abs_freq_gap_p2": abs(r.realized_p2 - q_mix) if np.isfinite(q_mix) else np.nan,
            "pref_entropy_p1": entropy_binary(r.pref0_p1) if np.isfinite(r.pref0_p1) else np.nan,
            "pref_entropy_p2": entropy_binary(r.pref0_p2) if np.isfinite(r.pref0_p2) else np.nan,
        }
        rows.append(rec)
    out = pd.DataFrame(rows)
    # Add per-agent summaries comparing MP entropy to dominance games.
    dom = panel[panel["family"].isin(DOM_FAMILIES)].copy()
    summaries = []
    for agent in AGENTS:
        a_mp = out[out.agent.eq(agent)]
        a_dom = dom[dom.agent.eq(agent)]
        summaries.append({
            "agent": agent,
            "display": DISPLAY[agent],
            "game_code": "__SUMMARY__",
            "payoff_kind": a_mp["payoff_kind"].iloc[0] if len(a_mp) else np.nan,
            "realized_p1": np.nan,
            "realized_p2": np.nan,
            "mixed_ne_p1": np.nan,
            "mixed_ne_p2": np.nan,
            "abs_freq_gap_p1": float(a_mp["abs_freq_gap_p1"].mean()) if len(a_mp) else np.nan,
            "abs_freq_gap_p2": float(a_mp["abs_freq_gap_p2"].mean()) if len(a_mp) else np.nan,
            "pref_entropy_p1": float(a_mp["pref_entropy_p1"].mean()) if len(a_mp) else np.nan,
            "pref_entropy_p2": float(a_mp["pref_entropy_p2"].mean()) if len(a_mp) else np.nan,
            "dominance_pref_entropy_p1": float(a_dom["pref0_p1"].map(lambda x: entropy_binary(x) if np.isfinite(x) else np.nan).mean()),
            "dominance_pref_entropy_p2": float(a_dom["pref0_p2"].map(lambda x: entropy_binary(x) if np.isfinite(x) else np.nan).mean()),
        })
    return pd.concat([out, pd.DataFrame(summaries)], ignore_index=True)


def build_all() -> dict[str, pd.DataFrame]:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    panel, cells = build_game_panel()
    zhu_original = griffiths_original_rows()
    dominance = dominance_conformity(panel)
    payoff = payoff_efficiency(panel, zhu_original)
    coord = coordination_metrics(panel, cells, zhu_original)
    mp = mp_descriptive(panel)

    dominance.to_csv(TAB_DIR / "f1_dominance_conformity.csv", index=False)
    payoff.to_csv(TAB_DIR / "f1_payoff_efficiency.csv", index=False)
    coord.to_csv(TAB_DIR / "f1_coordination.csv", index=False)
    mp.to_csv(TAB_DIR / "f1_mp_descriptive.csv", index=False)

    print("  [source] universe = full 144 one-shot games for LLM/MGN; Griffiths has observed subset")
    print("  [source] realized choices = $SCA_DATA_ROOT/substrate; soft = pref0 overlay only")
    print("  [source] Zhu payoff/coordination = 832 original cardinal matrices; paired orientations verified")
    print("\n--- f1_dominance_conformity ---")
    print(dominance[["agent", "family", "n_games", "R_realized", "R_soft"]].round(3).to_string(index=False))
    return {"panel": panel, "dominance": dominance, "payoff": payoff, "coordination": coord, "mp": mp}


def main() -> int:
    build_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
