#!/usr/bin/env python3
"""Positive-control steering test for action-token directions.

This isolates the most basic question: can an activation intervention at the
decision slot move the normalized A/B action preference at all? It uses a
prompt-specific unembedding direction, W[label(action0)] - W[label(action1)],
scaled to a fraction of the residual norm at the injection layer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

from strategic_anatomy.games import bruns_games
from strategic_anatomy.runtime import (
    MOVE_LABELS,
    _build_quant_config,
    _disable_hf_allocator_warmup,
    _free_gpu_cache,
    _install_spark_cpu_first_bnb8_patch,
    _load_model,
    _load_tokenizer,
    build_move_token_map,
    build_prompt,
    generate_balanced_label_schedule,
    generate_balanced_swap_schedule,
    get_final_norm_module,
    resolve_model_input_device,
)
from strategic_anatomy.steering_hooks import (
    np_norm,
    probe_once,
    readout_from_label_probs,
)
from strategic_anatomy.steering_utils import make_random_like, stable_seed


def parse_csv(value: str, cast=str) -> list:
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    cols = list(view.columns)
    out = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def unembed_action_direction(lm_head, move_token_map: dict[str, int], action_to_label: dict[int, str]) -> np.ndarray:
    """Return W[label(action0)] - W[label(action1)] as a float32 numpy vector."""
    w = lm_head.weight.detach()
    tok0 = move_token_map[action_to_label[0]]
    tok1 = move_token_map[action_to_label[1]]
    vec = (w[tok0] - w[tok1]).to(torch.float32).cpu().numpy()
    norm = np_norm(vec)
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Invalid zero/non-finite unembedding action direction.")
    return (vec / norm).astype(np.float32)


def make_report(df: pd.DataFrame, out_dir: pathlib.Path, args) -> None:
    nonzero = df.loc[df["dose"] != 0].copy()
    if not nonzero.empty:
        nonzero["hook_delta_ratio"] = nonzero["actual_delta_norm"] / nonzero["dose_vector_norm"].replace(0, np.nan)
        nonzero["actual_delta_abs_cosine_with_vector"] = nonzero["actual_delta_cosine_with_vector"].abs()
        summary = (
            nonzero.groupby(["treatment", "inject_layer", "dose"], dropna=False)
            .agg(
                n=("round", "count"),
                mean_hook_delta_ratio=("hook_delta_ratio", "mean"),
                mean_abs_cosine=("actual_delta_abs_cosine_with_vector", "mean"),
                mean_hook_pref0_delta=("hook_pref0_delta", "mean"),
                mean_final_pref0_delta=("final_pref0_delta", "mean"),
                mean_abs_final_pref0_delta=("final_pref0_delta", lambda s: s.abs().mean()),
                argmax_flip_rate=("argmax_changed", "mean"),
                sampled_flip_rate=("sampled_action_changed", "mean"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(out_dir / "action_direction_summary.csv", index=False)

    corr_rows = []
    for (treatment, layer), grp in nonzero.groupby(["treatment", "inject_layer"], dropna=False):
        row = {"treatment": treatment, "inject_layer": layer}
        for metric in ["hook_pref0_delta", "final_pref0_delta"]:
            row[f"pearson_dose_{metric}"] = (
                float(grp["dose"].corr(grp[metric]))
                if grp["dose"].nunique() > 1 and grp[metric].nunique() > 1
                else np.nan
            )
        corr_rows.append(row)
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(out_dir / "action_direction_dose_correlations.csv", index=False)

    intended = nonzero.loc[nonzero["treatment"] == "action_unembed"]
    random_ctl = nonzero.loc[nonzero["treatment"] == "random_control"]
    intended_abs = float(intended["final_pref0_delta"].abs().mean()) if not intended.empty else np.nan
    random_abs = float(random_ctl["final_pref0_delta"].abs().mean()) if not random_ctl.empty else np.nan
    intended_flip = float(intended["argmax_changed"].mean()) if not intended.empty else np.nan
    random_flip = float(random_ctl["argmax_changed"].mean()) if not random_ctl.empty else np.nan

    lines = [
        f"# Action-Direction Positive Control: {args.exp_tag}",
        "",
        f"Created: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Design",
        "",
        f"- Game: `{args.game}`",
        f"- Seed: `{args.seed}`",
        f"- Rounds/prompts: `{args.rounds}`",
        f"- Layers: `{args.layers}`",
        f"- Doses are residual-norm fractions: `{args.doses}`",
        "- Direction: `W[label(action0)] - W[label(action1)]`, recomputed for each prompt label mapping.",
        "- Control: norm-matched random vector at the same layer and dose.",
        "",
        "## Verdict Inputs",
        "",
        f"- Mean |final preference shift|, action direction: `{intended_abs:.6f}`",
        f"- Mean |final preference shift|, random control: `{random_abs:.6f}`",
        f"- Argmax flip rate, action direction: `{intended_flip:.4f}`",
        f"- Argmax flip rate, random control: `{random_flip:.4f}`",
        "",
        "If action direction does not beat the random control, the current residual-stream steering site/scaling is not a reliable action-control intervention.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Dose Correlations",
        "",
        markdown_table(corr),
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'action_direction_results.csv'}`",
        f"- `{out_dir / 'action_direction_summary.csv'}`",
        f"- `{out_dir / 'action_direction_dose_correlations.csv'}`",
        f"- `{out_dir / 'config.json'}`",
    ]
    (out_dir / "ACTION_DIRECTION_CONTROL.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description="Action-direction positive-control steering diagnostic")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-72B")
    p.add_argument("--exp_tag", default="")
    p.add_argument("--game", default="ShSh")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--layers", default="30,78,79")
    p.add_argument("--doses", default="-0.25,-0.1,-0.05,0,0.05,0.1,0.25")
    p.add_argument("--payoff_multiplier", type=int, default=2)
    p.add_argument("--probe_prefix", default="\nDecision: ")
    p.add_argument("--output_dir", default="validation_logs")
    p.add_argument("--use_ln_f_all", action="store_true", default=True)
    p.add_argument("--dry_run", action="store_true")

    # Model-loading compatibility with run_sim_spark helpers.
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

    if not args.exp_tag:
        args.exp_tag = f"action_direction_control_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    layers = parse_csv(args.layers, int)
    doses = parse_csv(args.doses, float)
    treatments = ["action_unembed", "random_control"]
    n_rows = args.rounds * len(layers) * len(doses) * len(treatments)
    print(f"Action-direction control: {args.exp_tag}")
    print(f"Game={args.game} seed={args.seed} rounds={args.rounds}")
    print(f"Layers={layers}")
    print(f"Doses={doses}")
    print(f"Diagnostic paired rows={n_rows}")
    if args.dry_run:
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    _disable_hf_allocator_warmup()
    _install_spark_cpu_first_bnb8_patch()

    tokenizer = _load_tokenizer(args.model_name)
    move_token_map = build_move_token_map(tokenizer, args.probe_prefix, MOVE_LABELS)
    qcfg = _build_quant_config(args)
    t0 = time.time()
    model = _load_model(args.model_name, args, qcfg)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    final_norm = get_final_norm_module(model)
    lm_head = model.get_output_embeddings()
    input_device = resolve_model_input_device(model)

    base_vec, *_ = bruns_games[args.game]
    game_vec = [int(v) * int(args.payoff_multiplier) for v in base_vec]

    label_schedule = generate_balanced_label_schedule(args.rounds)
    row_schedule = generate_balanced_swap_schedule(args.rounds)
    col_schedule = generate_balanced_swap_schedule(args.rounds)

    out_dir = pathlib.Path(args.output_dir) / args.exp_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    try:
        for r in range(args.rounds):
            _lta, atl = label_schedule[r]
            prompt = build_prompt(
                game_vec=game_vec,
                history=[],
                include_history=False,
                valid_moves=MOVE_LABELS,
                player=1,
                cumulative_you=0,
                cumulative_opp=0,
                action_to_label=atl,
                swap_rows=row_schedule[r],
                swap_cols=col_schedule[r],
                iv_prefix="",
                shuffle_valid_moves=True,
            )
            sample_u = np.random.default_rng(stable_seed(args.exp_tag, args.game, args.seed, r)).random()
            action_unit = unembed_action_direction(lm_head, move_token_map, atl)

            baseline_by_layer = {}
            for layer in layers:
                baseline_by_layer[layer] = probe_once(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    probe_prefix=args.probe_prefix,
                    move_token_map=move_token_map,
                    final_norm=final_norm,
                    lm_head=lm_head,
                    input_device=input_device,
                    inject_layer=layer,
                    vector=None,
                    dose=0.0,
                    use_ln_f_all=args.use_ln_f_all,
                )

            for layer in layers:
                baseline = baseline_by_layer[layer]
                base_layer_readout = readout_from_label_probs(baseline["layer_probs"], atl)
                base_final_readout = readout_from_label_probs(baseline["final_probs"], atl)
                baseline_hidden_norm = np_norm(baseline["layer_hidden"])
                action_vec = (action_unit * baseline_hidden_norm).astype(np.float32)
                random_vec = make_random_like(action_vec, args.exp_tag, args.game, layer, r)

                for dose in doses:
                    for treatment, vec in [("action_unembed", action_vec), ("random_control", random_vec)]:
                        steered = probe_once(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            probe_prefix=args.probe_prefix,
                            move_token_map=move_token_map,
                            final_norm=final_norm,
                            lm_head=lm_head,
                            input_device=input_device,
                            inject_layer=layer,
                            vector=vec if dose != 0.0 else None,
                            dose=dose,
                            use_ln_f_all=args.use_ln_f_all,
                        )
                        hook_stats = steered["hook_stats"]
                        if all(f"hook_pre_prob_{label}" in hook_stats for label in MOVE_LABELS):
                            hook_pre_probs = {label: hook_stats[f"hook_pre_prob_{label}"] for label in MOVE_LABELS}
                            hook_post_probs = {label: hook_stats[f"hook_post_prob_{label}"] for label in MOVE_LABELS}
                            hook_pre_readout = readout_from_label_probs(hook_pre_probs, atl)
                            hook_post_readout = readout_from_label_probs(hook_post_probs, atl)
                        else:
                            hook_pre_readout = base_layer_readout
                            hook_post_readout = readout_from_label_probs(steered["layer_probs"], atl)
                        steered_final_readout = readout_from_label_probs(steered["final_probs"], atl)

                        sampled_before = 0 if sample_u < base_final_readout.pref0 else 1
                        sampled_after = 0 if sample_u < steered_final_readout.pref0 else 1
                        rows.append({
                            "game": args.game,
                            "seed": args.seed,
                            "round": r + 1,
                            "treatment": treatment,
                            "inject_layer": layer,
                            "dose": float(dose),
                            "label_map_p1": atl[0],
                            "row_first_p1": int(row_schedule[r]),
                            "col_first_p1": int(col_schedule[r]),
                            "sample_u": float(sample_u),
                            "baseline_hidden_norm": baseline_hidden_norm,
                            "vector_norm": np_norm(vec),
                            "dose_vector_norm": abs(float(dose)) * np_norm(vec),
                            "pre_hidden_norm": hook_stats.get("pre_hidden_norm", np.nan),
                            "post_hidden_norm": hook_stats.get("post_hidden_norm", np.nan),
                            "actual_delta_norm": hook_stats.get("actual_delta_norm", 0.0 if dose == 0.0 else np.nan),
                            "actual_delta_cosine_with_vector": hook_stats.get("actual_delta_cosine_with_vector", np.nan),
                            "hook_prob_act0_before": hook_pre_readout.prob_act0,
                            "hook_prob_act1_before": hook_pre_readout.prob_act1,
                            "hook_prob_act0_after": hook_post_readout.prob_act0,
                            "hook_prob_act1_after": hook_post_readout.prob_act1,
                            "hook_pref0_before": hook_pre_readout.pref0,
                            "hook_pref0_after": hook_post_readout.pref0,
                            "hook_pref0_delta": hook_post_readout.pref0 - hook_pre_readout.pref0,
                            "final_prob_act0_before": base_final_readout.prob_act0,
                            "final_prob_act1_before": base_final_readout.prob_act1,
                            "final_prob_act0_after": steered_final_readout.prob_act0,
                            "final_prob_act1_after": steered_final_readout.prob_act1,
                            "final_pref0_before": base_final_readout.pref0,
                            "final_pref0_after": steered_final_readout.pref0,
                            "final_pref0_delta": steered_final_readout.pref0 - base_final_readout.pref0,
                            "slot_argmax_before": base_final_readout.argmax_action,
                            "slot_argmax_after": steered_final_readout.argmax_action,
                            "argmax_changed": int(base_final_readout.argmax_action != steered_final_readout.argmax_action),
                            "sampled_action_before": sampled_before,
                            "sampled_action_after": sampled_after,
                            "sampled_action_changed": int(sampled_before != sampled_after),
                        })

            if (r + 1) % 2 == 0 or (r + 1) == args.rounds:
                print(f"  round {r + 1}/{args.rounds} complete")
    finally:
        print(f"Freeing model {args.model_name}...")
        del model
        _free_gpu_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "action_direction_results.csv", index=False)
    config = vars(args).copy()
    config["base_vec"] = base_vec
    config["game_vec"] = game_vec
    config["runner"] = "run_action_direction_control.py"
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    make_report(df, out_dir, args)
    print(f"Wrote: {out_dir / 'action_direction_results.csv'}")
    print(f"Wrote: {out_dir / 'ACTION_DIRECTION_CONTROL.md'}")


if __name__ == "__main__":
    main()
