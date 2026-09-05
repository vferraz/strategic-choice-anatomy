#!/usr/bin/env python3
"""Causal steering pilot for Experiment 1 vectors.

This runner loads Qwen once, plays baseline prompts, and applies P1-only
activation steering at selected transformer layers. It uses vector candidates
prepared by analysis/prepare_steering_vectors.py.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import hashlib
import json
import pathlib
import random
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from strategic_anatomy.games import bruns_games
from strategic_anatomy.runtime import (
    Agent,
    MOVE_LABELS,
    Trait,
    _build_quant_config,
    _disable_hf_allocator_warmup,
    _free_gpu_cache,
    _install_spark_cpu_first_bnb8_patch,
    _load_model,
    _load_tokenizer,
    build_prompt,
    generate_balanced_label_schedule,
    generate_balanced_swap_schedule,
    generate_label_map,
    load_traits,
)


# NOTE (phase-4): these defaulted to a private-repo run directory under `_legacy/`
# (`_legacy/validation_logs/exp1_20260411_STEERING_PREP/…`) which is never copied into the
# release — so public `--help` advertised a path that cannot exist and leaked an internal
# run id. They are now empty: the flags are required, and `resolve_vector_files()` raises an
# actionable error naming them. Repointing at a shipped vector set instead would silently
# change which steering vectors a bare run uses, so it is deliberately not done.
DEFAULT_VECTOR_FILE = ""
DEFAULT_METADATA_FILE = ""


def resolve_vector_files(vector_file: str, metadata_file: str) -> tuple[str, str]:
    """Validate the steering-vector CLI pair, failing with an actionable message.

    Released direction sets live under ``$SCA_DATA_ROOT/steering/directions/{akata,
    akata_perp,akata_perm}/{model}/`` (see ``docs/DATA.md``); pass the ``.npz`` and its
    ``manifest.json``/metadata CSV explicitly.
    """
    missing = [n for n, v in (("--vector_file", vector_file),
                              ("--metadata_file", metadata_file)) if not v]
    if missing:
        raise SystemExit(
            f"{' and '.join(missing)} required — there is no default. The historical default "
            "named a private-repo `_legacy/` run directory that is not part of this release.\n"
            "Released direction sets: $SCA_DATA_ROOT/steering/directions/{akata,akata_perp,"
            "akata_perm}/{model}/ (see docs/DATA.md)."
        )
    return vector_file, metadata_file

CORE_GAMES = ["PdPd", "ChCh", "ShSh", "BaBa"]
DEFAULT_TARGET_TRAITS = [
    "risk_aversion_heavy",
    "inequity_aversion_light",
    "selfish_maximizer",
]
DEFAULT_SEEDS = [100, 200, 300]

# Offline-ranked candidates from STEERING_VECTOR_PREP_REPORT.md.
BEST_CANDIDATES = {
    "risk_aversion_heavy": ("action_aligned_residual", 79),
    "risk_aversion_light": ("action_aligned_residual", 72),
    "inequity_aversion_light": ("action_aligned_residual", 78),
    "inequity_aversion_heavy": ("action_aligned_residual", 78),
    "maximin": ("action_aligned_residual", 72),
    "selfish_maximizer": ("action_aligned_raw", 79),
}


@dataclass
class SteeringCondition:
    game: str
    seed: int
    rounds: int
    block: str
    target_trait: str = "none"
    treatment: str = "baseline"
    vector_type: str = ""
    source_layer: int = -1
    inject_layer: int = -1
    dose: float = 0.0
    random_control: bool = False
    wrong_layer_control: bool = False
    history: bool = True
    payoff_multiplier: int = 2
    balanced_labels: bool = True
    balanced_display: bool = True


def parse_csv(value: str, cast=str) -> list:
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)


def stable_seed(*parts: object) -> int:
    h = hashlib.blake2b("|".join(map(str, parts)).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % (2**32)


def transformer_layers(model) -> List[torch.nn.Module]:
    """Return decoder blocks for common HF causal LM layouts."""
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("base_model", "model", "layers"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return list(obj)
    raise ValueError("Could not locate transformer layers for activation steering.")


def make_random_like(vec: np.ndarray, *key_parts: object) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(*key_parts))
    rnd = rng.normal(size=vec.shape).astype(np.float32)
    rnd_norm = float(np.linalg.norm(rnd))
    vec_norm = float(np.linalg.norm(vec))
    if rnd_norm == 0.0 or not np.isfinite(rnd_norm):
        raise ValueError("Generated invalid random control vector.")
    return (rnd * (vec_norm / rnd_norm)).astype(np.float32)


class SteeringAgent(Agent):
    """Agent that can add a vector to the final-token residual stream."""

    def __init__(
        self,
        *args,
        steering_vector: np.ndarray | None = None,
        steering_layer: int | None = None,
        steering_dose: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.steering_vector = steering_vector
        self.steering_layer = steering_layer
        self.steering_dose = float(steering_dose)

    @contextlib.contextmanager
    def steering_hook(self):
        if self.steering_vector is None or self.steering_layer is None or self.steering_dose == 0.0:
            yield
            return

        layers = transformer_layers(self.model)
        if not (0 <= int(self.steering_layer) < len(layers)):
            raise ValueError(f"Steering layer {self.steering_layer} out of range 0..{len(layers)-1}")

        vec_np = np.asarray(self.steering_vector, dtype=np.float32)
        dose = float(self.steering_dose)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[-1] != vec_np.shape[0]:
                raise ValueError(
                    f"Steering dim mismatch: hidden {hidden.shape[-1]} vs vector {vec_np.shape[0]}"
                )
            vec = torch.as_tensor(vec_np, device=hidden.device, dtype=hidden.dtype)
            hidden2 = hidden.clone()
            hidden2[:, -1, :] = hidden2[:, -1, :] + dose * vec
            if isinstance(output, tuple):
                return (hidden2,) + output[1:]
            return hidden2

        handle = layers[int(self.steering_layer)].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def choose(self, prompt: str, max_history: int = 0):
        with self.steering_hook():
            return super().choose(prompt, max_history=max_history)


def vector_key(metadata: pd.DataFrame, vector_type: str, game: str, target_trait: str, layer: int) -> str:
    hit = metadata.loc[
        (metadata["vector_type"] == vector_type)
        & (metadata["game"] == game)
        & (metadata["trait_p1"].fillna("") == target_trait)
        & (metadata["layer"] == int(layer))
    ]
    if hit.empty:
        raise KeyError(
            f"No vector found for type={vector_type}, game={game}, trait={target_trait}, layer={layer}"
        )
    return str(hit.iloc[0]["vector_key"])


def build_treatment_condition(
    game: str,
    seed: int,
    rounds: int,
    target_trait: str,
    treatment: str,
    block: str,
    wrong_layer: int = 30,
) -> SteeringCondition:
    best_type, best_layer = BEST_CANDIDATES[target_trait]

    if treatment == "best":
        return SteeringCondition(game, seed, rounds, block, target_trait, treatment, best_type, best_layer, best_layer, 1.0)
    if treatment == "best_x2":
        return SteeringCondition(game, seed, rounds, block, target_trait, treatment, best_type, best_layer, best_layer, 2.0)
    if treatment == "best_neg":
        return SteeringCondition(game, seed, rounds, block, target_trait, treatment, best_type, best_layer, best_layer, -1.0)
    if treatment == "raw":
        return SteeringCondition(game, seed, rounds, block, target_trait, treatment, "raw", best_layer, best_layer, 1.0)
    if treatment == "common":
        return SteeringCondition(game, seed, rounds, block, target_trait, treatment, "common", best_layer, best_layer, 1.0)
    if treatment == "residual":
        return SteeringCondition(game, seed, rounds, block, target_trait, treatment, "game_residual", best_layer, best_layer, 1.0)
    if treatment == "aligned_raw":
        return SteeringCondition(game, seed, rounds, block, target_trait, treatment, "action_aligned_raw", best_layer, best_layer, 1.0)
    if treatment == "aligned_common":
        return SteeringCondition(game, seed, rounds, block, target_trait, treatment, "action_aligned_common", best_layer, best_layer, 1.0)
    if treatment == "aligned_residual":
        return SteeringCondition(game, seed, rounds, block, target_trait, treatment, "action_aligned_residual", best_layer, best_layer, 1.0)
    if treatment == "random":
        return SteeringCondition(game, seed, rounds, block, target_trait, treatment, best_type, best_layer, best_layer, 1.0, random_control=True)
    if treatment == "wrong_layer":
        return SteeringCondition(
            game, seed, rounds, block, target_trait, treatment, best_type,
            best_layer, wrong_layer, 1.0, wrong_layer_control=True
        )
    raise ValueError(f"Unknown steering treatment '{treatment}'")


def build_conditions(args) -> list[SteeringCondition]:
    games = parse_csv(args.games)
    seeds = parse_csv(args.seeds, int)
    traits = parse_csv(args.target_traits)
    treatments = parse_csv(args.treatments)

    if args.blocks == "preflight":
        return [
            SteeringCondition("PdPd", 100, 3, "preflight", treatment="baseline", dose=0.0),
            build_treatment_condition("ShSh", 100, 3, "risk_aversion_heavy", "best", "preflight", args.wrong_layer),
            build_treatment_condition("ShSh", 100, 3, "risk_aversion_heavy", "random", "preflight", args.wrong_layer),
            build_treatment_condition("BaBa", 100, 3, "selfish_maximizer", "best", "preflight", args.wrong_layer),
        ]

    conditions: list[SteeringCondition] = []
    blocks = parse_csv(args.blocks)
    if "pilot" not in blocks and "dosepilot" not in blocks:
        raise ValueError("Only blocks supported: preflight, pilot, dosepilot")

    # Baseline is recorded once per game/seed and reused in analysis.
    for game in games:
        for seed in seeds:
            conditions.append(
                SteeringCondition(
                    game=game,
                    seed=seed,
                    rounds=args.rounds,
                    block="pilot",
                    treatment="baseline",
                    target_trait="none",
                )
            )

    if "pilot" in blocks:
        for game in games:
            for trait in traits:
                if trait not in BEST_CANDIDATES:
                    raise ValueError(f"No best candidate configured for target trait '{trait}'")
                for treatment in treatments:
                    for seed in seeds:
                        conditions.append(
                            build_treatment_condition(game, seed, args.rounds, trait, treatment, "pilot", args.wrong_layer)
                        )

    if "dosepilot" in blocks:
        dose_layers = parse_csv(args.dose_layers, int)
        doses = [d for d in parse_csv(args.doses, float) if d != 0.0]
        controls = parse_csv(args.dose_controls)
        for game in games:
            for trait in traits:
                if trait not in BEST_CANDIDATES:
                    raise ValueError(f"No best candidate configured for target trait '{trait}'")
                best_type, _ = BEST_CANDIDATES[trait]
                for layer in dose_layers:
                    if not (65 <= layer <= 79):
                        raise ValueError(f"dosepilot source layer must be in vector bank 65..79, got {layer}")
                    for dose in doses:
                        for control in controls:
                            for seed in seeds:
                                if control == "best":
                                    conditions.append(
                                        SteeringCondition(
                                            game=game, seed=seed, rounds=args.rounds, block="dosepilot",
                                            target_trait=trait, treatment="dose_best",
                                            vector_type=best_type, source_layer=layer, inject_layer=layer,
                                            dose=dose,
                                        )
                                    )
                                elif control == "random":
                                    conditions.append(
                                        SteeringCondition(
                                            game=game, seed=seed, rounds=args.rounds, block="dosepilot",
                                            target_trait=trait, treatment="dose_random",
                                            vector_type=best_type, source_layer=layer, inject_layer=layer,
                                            dose=dose, random_control=True,
                                        )
                                    )
                                elif control == "wrong_layer":
                                    conditions.append(
                                        SteeringCondition(
                                            game=game, seed=seed, rounds=args.rounds, block="dosepilot",
                                            target_trait=trait, treatment="dose_wrong_layer",
                                            vector_type=best_type, source_layer=layer, inject_layer=args.wrong_layer,
                                            dose=dose, wrong_layer_control=True,
                                        )
                                    )
                                elif control == "raw":
                                    conditions.append(
                                        SteeringCondition(
                                            game=game, seed=seed, rounds=args.rounds, block="dosepilot",
                                            target_trait=trait, treatment="dose_raw",
                                            vector_type="raw", source_layer=layer, inject_layer=layer,
                                            dose=dose,
                                        )
                                    )
                                else:
                                    raise ValueError(f"Unknown dosepilot control '{control}'")
    return conditions


def resolve_steering_vector(
    cond: SteeringCondition,
    metadata: pd.DataFrame,
    vectors: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray | None, str]:
    if cond.treatment == "baseline" or cond.dose == 0.0:
        return None, ""
    key = vector_key(metadata, cond.vector_type, cond.game, cond.target_trait, cond.source_layer)
    vec = np.asarray(vectors[key], dtype=np.float32)
    if cond.random_control:
        return make_random_like(vec, cond.game, cond.target_trait, cond.source_layer, cond.seed), f"random_norm_matched_to::{key}"
    return vec, key


def run_one_condition(
    model,
    tokenizer,
    game_code: str,
    base_vec: List[int],
    trait_none: Trait,
    cond: SteeringCondition,
    model_name: str,
    steering_vector: np.ndarray | None,
    vector_key_used: str,
    probe_prefix: str,
):
    random.seed(cond.seed)
    np.random.seed(cond.seed)
    torch.manual_seed(cond.seed)
    torch.cuda.manual_seed_all(cond.seed)

    game_vec = [v * cond.payoff_multiplier for v in base_vec]
    pair_id = f"{game_code}_steer_{cond.target_trait}_{cond.treatment}"

    agent_kwargs = dict(
        layer_spec="all",
        valid_moves=MOVE_LABELS,
        probe_prefix=probe_prefix,
        use_ln_f_all=True,
        max_history_arg=0,
        attn_layers="none",
        attn_topk=0,
        attn_dtype="float16",
        save_token_layers=False,
        decision_mode="sample_probe",
        generate_temperature=1.0,
        generate_max_tokens=48,
    )
    agent1 = SteeringAgent(
        model,
        tokenizer,
        **agent_kwargs,
        steering_vector=steering_vector,
        steering_layer=cond.inject_layer if steering_vector is not None else None,
        steering_dose=cond.dose,
    )
    agent2 = Agent(model, tokenizer, **agent_kwargs)

    history: list[Tuple[int, int, int, int]] = []
    rows: list[dict] = []
    activ: Dict[str, np.ndarray] = {}
    probs_npz: Dict[str, np.ndarray] = {}
    attn_npz: Dict[str, np.ndarray] = {}
    cum1 = cum2 = 0
    act0_count1 = act0_count2 = 0
    valid_rounds1 = valid_rounds2 = 0

    if cond.balanced_labels:
        label_sched_p1 = generate_balanced_label_schedule(cond.rounds)
        label_sched_p2 = generate_balanced_label_schedule(cond.rounds)
    if cond.balanced_display:
        sched_row_p1 = generate_balanced_swap_schedule(cond.rounds)
        sched_col_p1 = generate_balanced_swap_schedule(cond.rounds)
        sched_row_p2 = generate_balanced_swap_schedule(cond.rounds)
        sched_col_p2 = generate_balanced_swap_schedule(cond.rounds)

    for r in range(cond.rounds):
        if cond.balanced_labels:
            lta_p1, atl_p1 = label_sched_p1[r]
            lta_p2, atl_p2 = label_sched_p2[r]
        else:
            lta_p1, atl_p1 = generate_label_map()
            lta_p2, atl_p2 = generate_label_map()

        if cond.balanced_display:
            swap_rows_p1 = sched_row_p1[r]
            swap_cols_p1 = sched_col_p1[r]
            swap_rows_p2 = sched_row_p2[r]
            swap_cols_p2 = sched_col_p2[r]
        else:
            swap_rows_p1 = random.random() < 0.5
            swap_cols_p1 = random.random() < 0.5
            swap_rows_p2 = random.random() < 0.5
            swap_cols_p2 = random.random() < 0.5

        prompt1 = build_prompt(
            game_vec=game_vec,
            history=history,
            include_history=cond.history,
            valid_moves=MOVE_LABELS,
            player=1,
            cumulative_you=cum1,
            cumulative_opp=cum2,
            action_to_label=atl_p1,
            swap_rows=swap_rows_p1,
            swap_cols=swap_cols_p1,
            iv_prefix=trait_none.system_prefix,
            shuffle_valid_moves=True,
        )
        res1 = agent1.choose(prompt1)

        prompt2 = build_prompt(
            game_vec=game_vec,
            history=history,
            include_history=cond.history,
            valid_moves=MOVE_LABELS,
            player=2,
            cumulative_you=cum2,
            cumulative_opp=cum1,
            action_to_label=atl_p2,
            swap_rows=swap_rows_p2,
            swap_cols=swap_cols_p2,
            iv_prefix=trait_none.system_prefix,
            shuffle_valid_moves=True,
        )
        res2 = agent2.choose(prompt2)

        move1_label = res1["chosen_move"]
        move2_label = res2["chosen_move"]
        act1 = lta_p1[move1_label] if move1_label is not None else None
        act2 = lta_p2[move2_label] if move2_label is not None else None

        if act1 is not None and act2 is not None:
            A_mat = np.array(game_vec[:4]).reshape(2, 2)
            B_mat = np.array(game_vec[4:8]).reshape(2, 2)
            payoff1, payoff2 = int(A_mat[act1, act2]), int(B_mat[act1, act2])
            history.append((act1, act2, payoff1, payoff2))
            cum1 += payoff1
            cum2 += payoff2
            act0_count1 += int(act1 == 0)
            act0_count2 += int(act2 == 0)
            valid_rounds1 += 1
            valid_rounds2 += 1
        else:
            payoff1 = payoff2 = None

        for l, vec in res1["hidden_by_layer"].items():
            activ[f"r{r}_p1_l{l}"] = vec
        for l, vec in res2["hidden_by_layer"].items():
            activ[f"r{r}_p2_l{l}"] = vec
        activ[f"r{r}_p1_input_ids"] = res1["input_ids"]
        activ[f"r{r}_p2_input_ids"] = res2["input_ids"]
        activ[f"r{r}_p1_offsets"] = res1["offsets"]
        activ[f"r{r}_p2_offsets"] = res2["offsets"]
        activ[f"r{r}_p1_tokens"] = res1["tokens"]
        activ[f"r{r}_p2_tokens"] = res2["tokens"]
        activ[f"r{r}_p1_text"] = res1["text"]
        activ[f"r{r}_p2_text"] = res2["text"]
        activ[f"r{r}_p1_prompt_len"] = np.array([res1["prompt_len"]], dtype=np.int32)
        activ[f"r{r}_p2_prompt_len"] = np.array([res2["prompt_len"]], dtype=np.int32)
        activ[f"r{r}_p1_prefix_len"] = np.array([res1["prefix_len"]], dtype=np.int32)
        activ[f"r{r}_p2_prefix_len"] = np.array([res2["prefix_len"]], dtype=np.int32)

        for l, mv_map in res1["per_layer_probs"].items():
            for mv, p in mv_map.items():
                probs_npz[f"r{r}_p1_l{l}_{mv}"] = np.array([p], dtype=np.float32)
        for l, mv_map in res2["per_layer_probs"].items():
            for mv, p in mv_map.items():
                probs_npz[f"r{r}_p2_l{l}_{mv}"] = np.array([p], dtype=np.float32)

        prob_act0_p1 = res1["final_slot_probs"][atl_p1[0]]
        prob_act1_p1 = res1["final_slot_probs"][atl_p1[1]]
        prob_act0_p2 = res2["final_slot_probs"][atl_p2[0]]
        prob_act1_p2 = res2["final_slot_probs"][atl_p2[1]]
        den1 = prob_act0_p1 + prob_act1_p1
        pref0_p1 = prob_act0_p1 / den1 if den1 > 0 else np.nan

        rows.append(dict(
            pair_id=pair_id,
            game_code=game_code,
            round=r + 1,
            move1=act1,
            move2=act2,
            move1_label=move1_label,
            move2_label=move2_label,
            freq_act0_p1=act0_count1 / valid_rounds1 if valid_rounds1 else np.nan,
            freq_act0_p2=act0_count2 / valid_rounds2 if valid_rounds2 else np.nan,
            label_map_p1=atl_p1[0],
            label_map_p2=atl_p2[0],
            prompt1_seen=res1["prompt_used"],
            prompt2_seen=res2["prompt_used"],
            payoff1=payoff1,
            payoff2=payoff2,
            prob_A_p1=res1["final_slot_probs"]["A"],
            prob_B_p1=res1["final_slot_probs"]["B"],
            prob_A_p2=res2["final_slot_probs"]["A"],
            prob_B_p2=res2["final_slot_probs"]["B"],
            prob_act0_p1=prob_act0_p1,
            prob_act1_p1=prob_act1_p1,
            pref0_renorm_p1=pref0_p1,
            prob_act0_p2=prob_act0_p2,
            prob_act1_p2=prob_act1_p2,
            slot_argmax_p1=0 if prob_act0_p1 >= prob_act1_p1 else 1,
            slot_argmax_p2=0 if prob_act0_p2 >= prob_act1_p2 else 1,
            first_token_id_p1=res1["first_token_id"],
            first_token_id_p2=res2["first_token_id"],
            base_game=base_vec,
            game_matrix=game_vec,
            model_p1_name=model_name,
            model_p2_name=model_name,
            cross_model=False,
            trait_p1_id="none",
            trait_p1_name="",
            trait_p1_intensity="",
            trait_p2_id="none",
            trait_p2_name="",
            trait_p2_intensity="",
            row_first_p1=1 if swap_rows_p1 else 0,
            col_first_p1=1 if swap_cols_p1 else 0,
            row_first_p2=1 if swap_rows_p2 else 0,
            col_first_p2=1 if swap_cols_p2 else 0,
            prompt_format="counterbalanced_AB_matrix_rand",
            generated_text_p1="",
            generated_text_p2="",
            gen_parse_ok_p1=True,
            gen_parse_ok_p2=True,
            decision_mode="sample_probe",
            steering_enabled=bool(steering_vector is not None and cond.dose != 0.0),
            steering_target_trait=cond.target_trait,
            steering_treatment=cond.treatment,
            steering_vector_type=cond.vector_type,
            steering_vector_key=vector_key_used,
            steering_source_layer=cond.source_layer,
            steering_inject_layer=cond.inject_layer,
            steering_dose=cond.dose,
            steering_random_control=cond.random_control,
            steering_wrong_layer_control=cond.wrong_layer_control,
        ))

    return pd.DataFrame(rows), activ, probs_npz, attn_npz


def make_run_id(model_name: str, cond: SteeringCondition, exp_tag: str) -> str:
    model_short = model_name.split("/")[-1].lower().replace("-", "_")
    if cond.treatment == "baseline":
        steer = "baseline"
    else:
        steer = (
            f"target={cond.target_trait}_treat={cond.treatment}_"
            f"vtype={cond.vector_type}_srcL{cond.source_layer}_injL{cond.inject_layer}_dose={cond.dose:g}"
        )
    seed_tag = f"_s{cond.seed}" if cond.seed != 42 else ""
    return safe_name(f"{model_short}_STEER_{cond.game}_{steer}_{exp_tag}_{cond.block}{seed_tag}")


def save_outputs(df, activ, probs_npz, attn_npz, run_dir: pathlib.Path, activ_dir: pathlib.Path, config: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    pair_dir = activ_dir / df.pair_id.iloc[0]
    pair_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    if attn_npz:
        np.savez_compressed(pair_dir / "attn.npz", **attn_npz)
    np.savez_compressed(pair_dir / "acts.npz", **activ)
    np.savez_compressed(pair_dir / "moveprobs.npz", **probs_npz)
    df.to_parquet(run_dir / "results.parquet", index=False)
    return run_dir / "results.parquet"


class Logger:
    def __init__(self, exp_tag: str, log_root: pathlib.Path):
        self.log_dir = log_root / exp_tag
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.runtimes: list[dict] = []
        self.failures: list[dict] = []
        self.lines = [f"# Steering Pilot Log: {exp_tag}", f"Started: {datetime.datetime.now().isoformat()}", ""]

    def done(self, run_id: str, cond: SteeringCondition, elapsed: float):
        self.runtimes.append({**asdict(cond), "run_tag": run_id, "elapsed_seconds": round(elapsed, 1), "status": "ok"})
        self.lines.append(f"- OK `{run_id}` {elapsed:.1f}s")

    def fail(self, run_id: str, cond: SteeringCondition, elapsed: float, tb: str):
        self.failures.append({**asdict(cond), "run_tag": run_id, "elapsed_seconds": round(elapsed, 1), "status": "failed", "traceback": tb})
        self.lines.append(f"- FAILED `{run_id}` {elapsed:.1f}s")
        self.lines.append("```")
        self.lines.append(tb)
        self.lines.append("```")

    def write(self):
        pd.DataFrame(self.runtimes).to_csv(self.log_dir / "runtimes.tsv", sep="\t", index=False)
        if self.failures:
            pd.DataFrame(self.failures).to_csv(self.log_dir / "failed_runs.tsv", sep="\t", index=False)
        (self.log_dir / "STEERING_PILOT_LOG.md").write_text("\n".join(self.lines) + "\n")


def main():
    p = argparse.ArgumentParser(description="Qwen steering pilot batch runner")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-72B")
    p.add_argument("--exp_tag", required=True)
    p.add_argument("--blocks", default="pilot", help="pilot, dosepilot, or preflight")
    p.add_argument("--games", default=",".join(CORE_GAMES))
    p.add_argument("--target_traits", default=",".join(DEFAULT_TARGET_TRAITS))
    p.add_argument(
        "--treatments",
        default="best,best_x2,best_neg,raw,common,residual,aligned_raw,aligned_common,aligned_residual,random,wrong_layer",
    )
    p.add_argument("--dose_layers", default="65,70,75,78,79",
                   help="Source layers for dosepilot; must exist in vector bank")
    p.add_argument("--doses", default="-2,-1,-0.5,0.5,1,2",
                   help="Comma-separated nonzero steering doses for dosepilot")
    p.add_argument("--dose_controls", default="best,random,wrong_layer",
                   help="dosepilot controls: best,random,wrong_layer,raw")
    p.add_argument("--wrong_layer", type=int, default=30,
                   help="Layer used for wrong-layer injection control")
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--output_dir", default="output")
    p.add_argument("--log_root", default="validation_logs")
    p.add_argument("--vector_file", default=DEFAULT_VECTOR_FILE,
                   help="steering-vector .npz (REQUIRED; released sets live under "
                        "$SCA_DATA_ROOT/steering/directions/ — see docs/DATA.md)")
    p.add_argument("--metadata_file", default=DEFAULT_METADATA_FILE,
                   help="steering-vector metadata CSV (REQUIRED; see --vector_file)")
    p.add_argument("--probe_prefix", default="\nDecision: ")
    # NOTE (phase-4): this defaulted to `<package>/traits.json`, a file that has not existed
    # for some time — any run relying on the default crashed inside load_traits(). It is now
    # required. Repointing at the packaged traits_oneshot.json would turn that crash into a
    # successful run with a DIFFERENT trait set, which is a behaviour change, so it is not done.
    p.add_argument("--traits_file", default="",
                   help="trait-cue definition JSON (REQUIRED; the packaged set is "
                        "strategic_anatomy/traits_oneshot.json)")
    p.add_argument("--dry_run", action="store_true")
    # model loading compatibility
    p.add_argument("--load_8bit", action="store_true")
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--bnb_compute_dtype", default="bfloat16")
    p.add_argument("--attn_impl", default="sdpa")
    p.add_argument("--model_p2", default="")
    p.add_argument("--attn_layers", default="none")
    p.add_argument("--attn_topk", type=int, default=0)
    p.add_argument("--attn_dtype", choices=["float32", "float16"], default="float16")
    p.add_argument("--save_token_layers", action="store_true")
    p.add_argument("--max_history", type=int, default=0)
    p.add_argument("--generate_temperature", type=float, default=1.0)
    p.add_argument("--no_shuffle_valid_moves", action="store_true", default=False)
    args = p.parse_args()

    conditions = build_conditions(args)
    print(f"Steering pilot: {args.exp_tag}")
    print(f"Blocks: {args.blocks}")
    print(f"Conditions: {len(conditions)}")
    for cond in conditions[:20]:
        print(" ", cond)
    if len(conditions) > 20:
        print(f"  ... {len(conditions)-20} more")
    if args.dry_run:
        return

    resolve_vector_files(args.vector_file, args.metadata_file)
    if not args.traits_file:
        raise SystemExit(
            "--traits_file required — there is no default. The historical default named a "
            "`traits.json` that no longer exists. The packaged trait set is "
            "strategic_anatomy/traits_oneshot.json."
        )

    _disable_hf_allocator_warmup()
    _install_spark_cpu_first_bnb8_patch()

    metadata = pd.read_csv(args.metadata_file)
    vectors = np.load(args.vector_file)
    traits = load_traits(args.traits_file)
    trait_none = traits["none"]

    tokenizer = _load_tokenizer(args.model_name)
    for label in MOVE_LABELS:
        ids_with = tokenizer(args.probe_prefix + label, add_special_tokens=False).input_ids
        ids_without = tokenizer(args.probe_prefix, add_special_tokens=False).input_ids
        lcp = 0
        while lcp < min(len(ids_with), len(ids_without)) and ids_with[lcp] == ids_without[lcp]:
            lcp += 1
        n_new = len(ids_with) - lcp
        if n_new != 1:
            raise ValueError(f"Label {label} is not single token after probe prefix; got {n_new}")
    print("Labels verified.")

    qcfg = _build_quant_config(args)
    t0 = time.time()
    model = _load_model(args.model_name, args, qcfg)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    out_root = pathlib.Path(args.output_dir)
    logger = Logger(args.exp_tag, pathlib.Path(args.log_root))

    ok = failed = 0
    try:
        for i, cond in enumerate(conditions, 1):
            run_id = make_run_id(args.model_name, cond, args.exp_tag)
            print(f"\n[{i}/{len(conditions)}] {run_id}")
            print(
                f"  {cond.game} target={cond.target_trait} treatment={cond.treatment} "
                f"vtype={cond.vector_type} srcL={cond.source_layer} injL={cond.inject_layer} dose={cond.dose:g} "
                f"seed={cond.seed} R={cond.rounds}"
            )
            t_start = time.time()
            try:
                base_vec, *_ = bruns_games[cond.game]
                steering_vector, vector_key_used = resolve_steering_vector(cond, metadata, vectors)
                df, activ, probs_npz, attn_npz = run_one_condition(
                    model, tokenizer, cond.game, base_vec, trait_none, cond, args.model_name,
                    steering_vector, vector_key_used, args.probe_prefix,
                )
                config = {
                    "runner": "run_steering_pilot_batch.py",
                    "exp_tag": args.exp_tag,
                    "model_name": args.model_name,
                    "condition": asdict(cond),
                    "game": cond.game,
                    "base_vec": base_vec,
                    "game_vec": [v * cond.payoff_multiplier for v in base_vec],
                    "trait_p1": "none",
                    "trait_p2": "none",
                    "vector_key_used": vector_key_used,
                    "vector_file": args.vector_file,
                    "metadata_file": args.metadata_file,
                    "layer": "all",
                    "use_ln_f_all": True,
                    "probe_prefix": args.probe_prefix,
                    "decision_mode": "sample_probe",
                    "balanced_labels": cond.balanced_labels,
                    "balanced_display": cond.balanced_display,
                    "history": cond.history,
                    "payoff_multiplier": cond.payoff_multiplier,
                    "seed": cond.seed,
                    "load_8bit": args.load_8bit,
                    "load_4bit": args.load_4bit,
                }
                run_dir = out_root / run_id
                activ_dir = out_root / "activations" / run_id
                out_path = save_outputs(df, activ, probs_npz, attn_npz, run_dir, activ_dir, config)
                elapsed = time.time() - t_start
                p1_act0 = (df["move1"] == 0).mean()
                p1_pref = df["pref0_renorm_p1"].mean()
                print(f"  P1(act0)={p1_act0:.1%}  mean_pref0={p1_pref:.3f}  [{elapsed:.1f}s] -> {out_path}")
                logger.done(run_id, cond, elapsed)
                ok += 1
            except Exception:
                elapsed = time.time() - t_start
                tb = traceback.format_exc()
                print(f"  FAILED ({elapsed:.1f}s)")
                print(tb)
                logger.fail(run_id, cond, elapsed, tb)
                failed += 1
            _free_gpu_cache()
    finally:
        vectors.close()
        print(f"\nFreeing model {args.model_name}...")
        del model
        _free_gpu_cache()
        logger.write()

    print(f"\nSteering pilot complete: {ok} OK, {failed} FAILED")
    print(f"Logs: {logger.log_dir}")


if __name__ == "__main__":
    main()
