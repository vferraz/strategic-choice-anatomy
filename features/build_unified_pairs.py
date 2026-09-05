"""Build the unified pair-level equivalence dataset.

The output grain is one row per sample/condition/game pair. Realized choices
and theoretical predictions are expressed as P(act0) for both players.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategic_anatomy.eq_engine import action_to_act0_prob, compute_matrix_metrics
from strategic_anatomy.config import games_root, human_refs_root, taxonomy_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

from strategic_anatomy.games import bruns_games  # noqa: E402


DEFAULT_OUT = human_refs_root() / "unified_pairs.parquet"
RUN_DIR_RE = re.compile(r"^(?P<game>.+)_s(?P<seed>\d+)_(?P<condition>.+)$")
LLM_COLUMNS = [
    "game_code",
    "round",
    "move1",
    "move2",
    "pref_opt0_p1_round",
    "pref_opt0_p2_round",
    "cell_name",
    "trait_p1",
    "trait_p2",
]


def _parse_csv_vec(value: Any) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    if pd.isna(value):
        raise ValueError("cannot parse NaN payoff vector")
    return [float(part.strip()) for part in str(value).split(",")]


def _json_vec(vec: list[float] | list[int]) -> str:
    vals: list[int | float] = []
    for value in vec:
        f = float(value)
        vals.append(int(f) if f.is_integer() else f)
    return json.dumps(vals, separators=(",", ":"))


def _boolish(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_run_dir(path: Path) -> dict[str, Any]:
    match = RUN_DIR_RE.match(path.name)
    if not match:
        raise ValueError(f"cannot parse DESIGN_V2 run directory: {path}")
    return {
        "dir_game": match.group("game"),
        "seed": int(match.group("seed")),
        "dir_condition": match.group("condition"),
    }


def _matrix_for_bruns(game_code: str) -> list[float]:
    if game_code not in bruns_games:
        raise KeyError(f"unknown Bruns game code: {game_code}")
    return [float(v) for v in bruns_games[game_code][0]]


def _feature_table(root: Path) -> pd.DataFrame:
    features = pd.read_csv(games_root() / "game_features.csv")
    features = features.drop_duplicates("game_code")
    return features


def _equivalence_table(root: Path) -> pd.DataFrame:
    eq = pd.read_csv(taxonomy_dir() / "equivalence_per_canonical.csv")
    eq = eq.copy()
    eq["canonical_id"] = eq["canonical_id"].astype(int)
    return eq


def _theory_columns(matrix_8vec: list[float]) -> dict[str, Any]:
    return compute_matrix_metrics(matrix_8vec)


def _base_row(
    *,
    sample: str,
    agent_kind: str,
    condition: str,
    trait_p1: str,
    trait_p2: str,
    game_code: str,
    bruns_name: str,
    canonical_id: int,
    nagel_lk_type: str,
    griffiths_game_id: float = math.nan,
    griffiths_unique_id: float = math.nan,
    griffiths_role: str | float = math.nan,
    n_pooled: int = 1,
    matrix_8vec: list[float] | None = None,
    payoff_kind: str = "ordinal",
    realized_p1: float = math.nan,
    realized_p2: float = math.nan,
    pref0_p1: float = math.nan,
    pref0_p2: float = math.nan,
) -> dict[str, Any]:
    if matrix_8vec is None:
        matrix_8vec = _matrix_for_bruns(bruns_name)
    row: dict[str, Any] = {
        "sample": sample,
        "agent_kind": agent_kind,
        "condition": condition,
        "trait_p1": trait_p1,
        "trait_p2": trait_p2,
        "game_code": game_code,
        "bruns_name": bruns_name,
        "canonical_id": int(canonical_id),
        "nagel_lk_type": nagel_lk_type,
        "griffiths_game_id": griffiths_game_id,
        "griffiths_unique_id": griffiths_unique_id,
        "griffiths_role": griffiths_role,
        "n_pooled": int(n_pooled),
        "matrix_8vec": _json_vec(matrix_8vec),
        "payoff_kind": payoff_kind,
        "realized_p1": float(realized_p1),
        "realized_p2": float(realized_p2),
        "pref0_p1": float(pref0_p1) if pd.notna(pref0_p1) else math.nan,
        "pref0_p2": float(pref0_p2) if pd.notna(pref0_p2) else math.nan,
    }
    row.update(_theory_columns(matrix_8vec))
    return row


def _load_llm_rows(root: Path, eq: pd.DataFrame) -> tuple[pd.DataFrame, set[int], dict[str, int]]:
    design_root = root / "output/design_v2_main"
    if not design_root.exists():
        # phase-4: this already failed loudly, which is the right behaviour — the message
        # now says why. The LLM half of unified_pairs.parquet was built from the DESIGN_V2
        # round-based substrate, which plan §1/§5 exclude from the release and which is not
        # in the data deposit. The built table itself ships at data/human_refs/.
        raise FileNotFoundError(
            f"missing DESIGN_V2 root: {design_root}\n"
            "This Stage-F builder reads the DESIGN_V2 round-based substrate, which is NOT "
            "part of this release or the data deposit (see docs/DATA.md). The table it "
            "produces ships pre-built at data/human_refs/unified_pairs.parquet and is what "
            "every downstream tier reads."
        )

    eq_by_game = eq.set_index("bruns_name")
    records: list[dict[str, Any]] = []
    invalid_counts = {"move1_invalid": 0, "move2_invalid": 0}

    model_dirs = [
        path
        for path in sorted(design_root.iterdir())
        if path.is_dir() and list(path.glob("*/results.parquet"))
    ]
    for model_dir in model_dirs:
        sample = model_dir.name
        for parquet_path in sorted(model_dir.glob("*/results.parquet")):
            parsed = _parse_run_dir(parquet_path.parent)
            df = pd.read_parquet(parquet_path, columns=LLM_COLUMNS)
            df = df[df["round"] == 1].copy()
            if df.empty:
                continue
            invalid_counts["move1_invalid"] += int((df["move1"] == -1).sum())
            invalid_counts["move2_invalid"] += int((df["move2"] == -1).sum())
            first = df.iloc[0]
            records.append(
                {
                    "sample": sample,
                    "game_code": str(first["game_code"]),
                    "condition": str(first["cell_name"]),
                    "trait_p1": str(first["trait_p1"]),
                    "trait_p2": str(first["trait_p2"]),
                    "seed": int(parsed["seed"]),
                    "realized_p1": float((df["move1"] == 0).mean()),
                    "realized_p2": float((df["move2"] == 0).mean()),
                    "pref0_p1": float(df["pref_opt0_p1_round"].mean()),
                    "pref0_p2": float(df["pref_opt0_p2_round"].mean()),
                }
            )

    if not records:
        raise ValueError(f"found no LLM rows under {design_root}")

    raw = pd.DataFrame(records)
    grouped_rows: list[dict[str, Any]] = []
    for keys, grp in raw.groupby(["sample", "game_code", "condition"], sort=True):
        sample, game_code, condition = keys
        if game_code not in eq_by_game.index:
            raise KeyError(f"LLM game has no equivalence row: {game_code}")
        eq_row = eq_by_game.loc[game_code]
        for col in ("trait_p1", "trait_p2"):
            values = sorted(grp[col].dropna().unique())
            if len(values) != 1:
                raise ValueError(f"{keys} has multiple {col} values: {values}")
        grouped_rows.append(
            _base_row(
                sample=str(sample),
                agent_kind="llm",
                condition=str(condition),
                trait_p1=str(grp["trait_p1"].iloc[0]),
                trait_p2=str(grp["trait_p2"].iloc[0]),
                game_code=str(game_code),
                bruns_name=str(game_code),
                canonical_id=int(eq_row["canonical_id"]),
                nagel_lk_type=str(eq_row["nagel_lk_type"]),
                n_pooled=int(grp["seed"].nunique()),
                matrix_8vec=_matrix_for_bruns(str(game_code)),
                payoff_kind="ordinal",
                realized_p1=float(grp["realized_p1"].mean()),
                realized_p2=float(grp["realized_p2"].mean()),
                pref0_p1=float(grp["pref0_p1"].mean()),
                pref0_p2=float(grp["pref0_p2"].mean()),
            )
        )

    out = pd.DataFrame(grouped_rows)
    model_set = set(out["canonical_id"].astype(int).unique())
    return out, model_set, invalid_counts


def _load_nagel_rows(root: Path, eq: pd.DataFrame) -> pd.DataFrame:
    perspective = pd.read_csv(human_refs_root() / "raw" / "nagel" / "perspective_info.csv")
    paired_from_raw = perspective.set_index("p_id")["p_paired"].to_dict()
    rate_by_pid = eq.set_index("nagel_p_id")["nagel_frac_choose_act0_canonical"].to_dict()

    rows: list[dict[str, Any]] = []
    for _, eq_row in eq.sort_values("canonical_id").iterrows():
        p_id = int(eq_row["nagel_p_id"])
        paired = int(eq_row["nagel_p_paired"])
        if int(paired_from_raw[p_id]) != paired:
            raise AssertionError(f"Nagel pairing mismatch for p_id={p_id}")
        if paired not in rate_by_pid:
            raise KeyError(f"missing paired Nagel perspective p_id={paired}")
        game_code = str(eq_row["bruns_name"])
        rows.append(
            _base_row(
                sample="nagel",
                agent_kind="human",
                condition="observed",
                trait_p1="observed",
                trait_p2="observed",
                game_code=game_code,
                bruns_name=game_code,
                canonical_id=int(eq_row["canonical_id"]),
                nagel_lk_type=str(eq_row["nagel_lk_type"]),
                n_pooled=1,
                matrix_8vec=_matrix_for_bruns(game_code),
                payoff_kind="ordinal",
                realized_p1=float(eq_row["nagel_frac_choose_act0_canonical"]),
                # partner perspective's act0 axis is the complement of this game's P2 axis (see TICKET_fix_p2_orientation)
                realized_p2=1.0 - float(rate_by_pid[paired]),
            )
        )
    return pd.DataFrame(rows)


def _oriented_vec(row: pd.Series, p1_col: str, p2_col: str) -> list[float]:
    p1 = np.array(_parse_csv_vec(row[p1_col]), dtype=float).reshape(2, 2)
    p2 = np.array(_parse_csv_vec(row[p2_col]), dtype=float).reshape(2, 2)
    if _boolish(row["swap_sr"]):
        p1 = p1[::-1, :]
        p2 = p2[::-1, :]
    if _boolish(row["swap_sc"]):
        p1 = p1[:, ::-1]
        p2 = p2[:, ::-1]
    return [float(v) for v in np.concatenate([p1.reshape(-1), p2.reshape(-1)])]


def _load_griffiths_rows(root: Path, eq: pd.DataFrame) -> pd.DataFrame:
    grif = pd.read_csv(taxonomy_dir() / "griffiths_to_canonical.csv")
    keep_game_ids = (
        grif.groupby("game_id")["has_ties"].any().loc[lambda s: ~s].index.tolist()
    )
    grif = grif[grif["game_id"].isin(keep_game_ids)].dropna(subset=["canonical_id"]).copy()
    grif["canonical_id"] = grif["canonical_id"].astype(int)
    eq_by_cid = eq.set_index("canonical_id")
    canonical_vec_by_cid = {
        int(row["canonical_id"]): [float(v) for v in _parse_csv_vec(row["canonical_8vec"])]
        for _, row in eq.iterrows()
    }

    rows: list[dict[str, Any]] = []
    for game_id, grp in grif.groupby("game_id", sort=True):
        if len(grp) != 2:
            raise AssertionError(f"Griffiths game_id={game_id} has {len(grp)} rows")
        grp = grp.sort_values("unique_id").reset_index(drop=True)
        for idx in range(2):
            row = grp.iloc[idx]
            partner = grp.iloc[1 - idx]
            cid = int(row["canonical_id"])
            eq_row = eq_by_cid.loc[cid]
            ordinal_vec = _oriented_vec(row, "ordinal_p1", "ordinal_p2")
            expected = canonical_vec_by_cid[cid]
            if [int(v) for v in ordinal_vec] != [int(v) for v in expected]:
                raise AssertionError(
                    f"Griffiths ordinal swap failed for game_id={game_id}, canonical_id={cid}"
                )
            matrix = _oriented_vec(row, "cardinal_p1", "cardinal_p2")
            bruns_name = str(eq_row["bruns_name"])
            rows.append(
                _base_row(
                    sample="griffiths",
                    agent_kind="human",
                    condition="observed",
                    trait_p1="observed",
                    trait_p2="observed",
                    game_code=bruns_name,
                    bruns_name=bruns_name,
                    canonical_id=cid,
                    nagel_lk_type=str(eq_row["nagel_lk_type"]),
                    griffiths_game_id=float(game_id),
                    griffiths_unique_id=float(row["unique_id"]),
                    griffiths_role=str(row["role"]),
                    n_pooled=1,
                    matrix_8vec=matrix,
                    payoff_kind="cardinal",
                    realized_p1=float(row["up_choice_canonical"]),
                    # partner perspective's act0 axis is the complement of this game's P2 axis (see TICKET_fix_p2_orientation)
                    realized_p2=1.0 - float(partner["up_choice_canonical"]),
                )
            )
    return pd.DataFrame(rows)


def _is_missing(value: Any) -> bool:
    return pd.isna(value)


def _raw_action_to_target(value: Any) -> float:
    if _is_missing(value):
        return math.nan
    return action_to_act0_prob(int(value))


def _substitute_dominance(row: pd.Series, level_col: str, dom_col: str) -> float:
    action = row[level_col]
    if pd.notna(action) and int(action) != -1:
        return _raw_action_to_target(action)
    dom = row[dom_col]
    if pd.notna(dom):
        return _raw_action_to_target(dom)
    return _raw_action_to_target(action)


def _unique_mixed_target(row: pd.Series, side: str) -> float:
    for idx in range(1, 4):
        p1 = row[f"ne{idx}_p1"]
        p2 = row[f"ne{idx}_p2"]
        if pd.notna(p1) and pd.notna(p2) and 0.0 < p1 < 1.0 and 0.0 < p2 < 1.0:
            return float(row[f"ne{idx}_{side}"])
    return math.nan


def _nearest_pure_ne_target(row: pd.Series, side: str) -> float:
    candidates: list[tuple[float, float, float]] = []
    for idx in range(1, 4):
        p1 = row[f"ne{idx}_p1"]
        p2 = row[f"ne{idx}_p2"]
        if pd.isna(p1) or pd.isna(p2):
            continue
        if p1 in (0.0, 1.0) and p2 in (0.0, 1.0):
            dist = math.sqrt(
                (float(row["realized_p1"]) - float(p1)) ** 2
                + (float(row["realized_p2"]) - float(p2)) ** 2
            )
            candidates.append((dist, float(p1), float(p2)))
    if not candidates:
        return math.nan
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][1 if side == "p1" else 2]


def _assign_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    target_p1: list[float] = []
    target_p2: list[float] = []
    for _, row in df.iterrows():
        lk = row["nagel_lk_type"]
        if lk == "DD":
            target_p1.append(float(row["dom_p1"]))
            target_p2.append(float(row["dom_p2"]))
        elif lk == "OD1":
            target_p1.append(_substitute_dominance(row, "_l1_action_p1", "_dom_action_p1"))
            target_p2.append(_substitute_dominance(row, "_l1_action_p2", "_dom_action_p2"))
        elif lk == "OD2":
            target_p1.append(_substitute_dominance(row, "_l2_action_p1", "_dom_action_p1"))
            target_p2.append(_substitute_dominance(row, "_l2_action_p2", "_dom_action_p2"))
        elif lk == "MP":
            target_p1.append(_unique_mixed_target(row, "p1"))
            target_p2.append(_unique_mixed_target(row, "p2"))
        elif lk in {"CO1", "CO2"}:
            target_p1.append(_nearest_pure_ne_target(row, "p1"))
            target_p2.append(_nearest_pure_ne_target(row, "p2"))
        else:
            target_p1.append(math.nan)
            target_p2.append(math.nan)
    df["target_p1"] = target_p1
    df["target_p2"] = target_p2
    return df


def _finalize(
    df: pd.DataFrame, features: pd.DataFrame, model_set: set[int]
) -> pd.DataFrame:
    df = _assign_targets(df)
    df["canonical_id"] = df["canonical_id"].astype(int)
    df["in_model_set"] = df["canonical_id"].isin(model_set)

    private_cols = [col for col in df.columns if col.startswith("_")]
    df = df.drop(columns=private_cols)
    df = df.merge(features, on="game_code", how="left", validate="many_to_one")

    if df["canonical_id"].isna().any():
        raise AssertionError("canonical_id has NaN after assembly")
    if df["nagel_lk_type"].isna().any():
        raise AssertionError("nagel_lk_type has NaN after assembly")
    for col in ("realized_p1", "realized_p2"):
        bad = df[col].notna() & ~df[col].between(0.0, 1.0)
        if bad.any():
            raise AssertionError(f"{col} has values outside [0, 1]")
    return df


def build_dataset(data_root: str | Path = ".") -> tuple[pd.DataFrame, dict[str, int]]:
    """Build and return the unified dataset without writing it."""
    root = Path(data_root).resolve()
    eq = _equivalence_table(root)
    features = _feature_table(root)
    llm, model_set, invalid_counts = _load_llm_rows(root, eq)
    nagel = _load_nagel_rows(root, eq)
    grif = _load_griffiths_rows(root, eq)
    all_rows = pd.concat([llm, nagel, grif], ignore_index=True, sort=False)
    final = _finalize(all_rows, features, model_set)
    return final, invalid_counts


def write_dataset(df: pd.DataFrame, out: str | Path) -> tuple[Path, Path]:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    head_path = out_path.with_name(f"{out_path.stem}.head.csv")
    df.head(100).to_csv(head_path, index=False)
    return out_path, head_path


def build_unified_pairs(data_root: str | Path = ".", out: str | Path = DEFAULT_OUT) -> pd.DataFrame:
    df, invalid_counts = build_dataset(data_root)
    out_path, head_path = write_dataset(df, out)
    if invalid_counts["move1_invalid"] or invalid_counts["move2_invalid"]:
        print(
            "warning: invalid round-1 LLM moves treated as not-act0: "
            f"move1={invalid_counts['move1_invalid']}, move2={invalid_counts['move2_invalid']}"
        )
    print(f"wrote {out_path} ({len(df):,} rows)")
    print(f"wrote {head_path}")
    counts = (
        df.groupby(["sample", "nagel_lk_type"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(counts.to_string())
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=".", help="repository/data root")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output parquet path")
    args = parser.parse_args(argv)
    build_unified_pairs(args.data_root, args.out)


if __name__ == "__main__":
    main()
