#!/usr/bin/env python3
"""Minimal diagnostic for Experiment 1 steering mechanics.

This is not a broad steering sweep. It runs paired baseline/steered probe
passes on identical baseline prompts and records where an injected vector has
an effect: target-layer hidden state, target-layer A/B readout, final A/B
readout, and deterministic sample/argmax decisions.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch

from strategic_anatomy.games import bruns_games
from strategic_anatomy.runtime import (
    MOVE_LABELS,
    _build_quant_config,
    _disable_hf_allocator_warmup,
    _ensure_attention_mask,
    _free_gpu_cache,
    _install_spark_cpu_first_bnb8_patch,
    _load_model,
    _load_tokenizer,
    apply_final_norm_no_grad,
    build_move_token_map,
    build_prompt,
    generate_balanced_label_schedule,
    generate_balanced_swap_schedule,
    get_final_norm_module,
    resolve_model_input_device,
)
from strategic_anatomy.steering_utils import (
    DEFAULT_METADATA_FILE,
    DEFAULT_VECTOR_FILE,
    make_random_like,
    stable_seed,
    transformer_layers,
    vector_key,
)


DEFAULT_VECTOR_SPECS = (
    "inequity_aversion_light:action_aligned_residual:78,"
    "selfish_maximizer:action_aligned_raw:78"
)


@dataclass(frozen=True)
class VectorSpec:
    target_trait: str
    vector_type: str
    source_layer: int


@dataclass(frozen=True)
class ProbeReadout:
    prob_a: float
    prob_b: float
    prob_act0: float
    prob_act1: float
    pref0: float
    argmax_action: int


def parse_csv(value: str, cast=str) -> list:
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def parse_vector_specs(value: str) -> list[VectorSpec]:
    specs: list[VectorSpec] = []
    for item in parse_csv(value):
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Bad vector spec '{item}'. Expected trait:vector_type:source_layer"
            )
        specs.append(VectorSpec(parts[0], parts[1], int(parts[2])))
    return specs


def softmax_label_probs(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    final_norm: torch.nn.Module | None,
    move_token_map: Dict[str, int],
    use_ln_f_all: bool,
) -> Dict[str, float]:
    """Project one hidden vector through the logit lens for move labels."""
    h = hidden.detach().to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)
    if use_ln_f_all and final_norm is not None:
        h = apply_final_norm_no_grad(h, final_norm)
    h = h.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)
    logits = lm_head(h.clone().unsqueeze(0))[0]
    probs = torch.softmax(logits, dim=0)
    return {mv: float(probs[move_token_map[mv]].item()) for mv in MOVE_LABELS}


def final_label_probs(logits: torch.Tensor, move_token_map: Dict[str, int]) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=-1)
    return {mv: float(probs[move_token_map[mv]].item()) for mv in MOVE_LABELS}


def readout_from_label_probs(probs: Dict[str, float], action_to_label: Dict[int, str]) -> ProbeReadout:
    prob_a = float(probs["A"])
    prob_b = float(probs["B"])
    prob_act0 = float(probs[action_to_label[0]])
    prob_act1 = float(probs[action_to_label[1]])
    den = prob_act0 + prob_act1
    pref0 = float(prob_act0 / den) if den > 0 else math.nan
    argmax_action = 0 if prob_act0 >= prob_act1 else 1
    return ProbeReadout(prob_a, prob_b, prob_act0, prob_act1, pref0, argmax_action)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0 or not np.isfinite(den):
        return math.nan
    return float(np.dot(a, b) / den)


def np_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x.astype(np.float32)))


@contextlib.contextmanager
def optional_steering_hook(
    model,
    vector: np.ndarray | None,
    inject_layer: int | None,
    dose: float,
    hook_stats: dict,
    lm_head: torch.nn.Module | None = None,
    final_norm: torch.nn.Module | None = None,
    move_token_map: Dict[str, int] | None = None,
    use_ln_f_all: bool = True,
):
    """Add dose * vector to the final-token layer output and record hook stats."""
    if vector is None or inject_layer is None or dose == 0.0:
        yield
        return

    layers = transformer_layers(model)
    if not (0 <= int(inject_layer) < len(layers)):
        raise ValueError(f"Injection layer {inject_layer} out of range 0..{len(layers) - 1}")

    vec_np = np.asarray(vector, dtype=np.float32)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[-1] != vec_np.shape[0]:
            raise ValueError(f"Steering dim mismatch: hidden {hidden.shape[-1]} vs vector {vec_np.shape[0]}")

        vec = torch.as_tensor(vec_np, device=hidden.device, dtype=hidden.dtype)
        pre = hidden[:, -1, :].detach().to(torch.float32)
        hidden2 = hidden.clone()
        hidden2[:, -1, :] = hidden2[:, -1, :] + float(dose) * vec
        post = hidden2[:, -1, :].detach().to(torch.float32)
        delta = post - pre

        vec32 = vec.detach().to(torch.float32)
        hook_stats["pre_hidden_norm"] = float(torch.linalg.vector_norm(pre).item())
        hook_stats["post_hidden_norm"] = float(torch.linalg.vector_norm(post).item())
        hook_stats["actual_delta_norm"] = float(torch.linalg.vector_norm(delta).item())
        den = torch.linalg.vector_norm(delta) * torch.linalg.vector_norm(vec32)
        hook_stats["actual_delta_cosine_with_vector"] = (
            float(torch.sum(delta.squeeze(0) * vec32).item() / den.item())
            if den.item() > 0
            else math.nan
        )
        if lm_head is not None and move_token_map is not None:
            pre_probs = softmax_label_probs(
                pre.squeeze(0), lm_head, final_norm, move_token_map, use_ln_f_all
            )
            post_probs = softmax_label_probs(
                post.squeeze(0), lm_head, final_norm, move_token_map, use_ln_f_all
            )
            for label in MOVE_LABELS:
                hook_stats[f"hook_pre_prob_{label}"] = pre_probs[label]
                hook_stats[f"hook_post_prob_{label}"] = post_probs[label]

        if isinstance(output, tuple):
            return (hidden2,) + output[1:]
        return hidden2

    handle = layers[int(inject_layer)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def probe_once(
    model,
    tokenizer,
    prompt: str,
    probe_prefix: str,
    move_token_map: Dict[str, int],
    final_norm,
    lm_head,
    input_device: torch.device,
    inject_layer: int,
    vector: np.ndarray | None = None,
    dose: float = 0.0,
    use_ln_f_all: bool = True,
) -> dict:
    """Run one probe pass and return hidden/readout diagnostics."""
    text = prompt + probe_prefix
    enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(input_device)
    attention_mask = _ensure_attention_mask(input_ids, enc.get("attention_mask")).to(input_device)

    hook_stats: dict = {}
    with torch.inference_mode():
        with optional_steering_hook(
            model,
            vector,
            inject_layer,
            dose,
            hook_stats,
            lm_head=lm_head,
            final_norm=final_norm,
            move_token_map=move_token_map,
            use_ln_f_all=use_ln_f_all,
        ):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

    hs = out.hidden_states
    layer_hidden = hs[inject_layer + 1][0, -1, :].detach()
    layer_probs = softmax_label_probs(layer_hidden, lm_head, final_norm, move_token_map, use_ln_f_all)
    final_probs = final_label_probs(out.logits[0, -1, :].detach(), move_token_map)

    return {
        "layer_hidden": layer_hidden.to(torch.float32).cpu().numpy(),
        "layer_probs": layer_probs,
        "final_probs": final_probs,
        "hook_stats": hook_stats,
    }


def resolve_vector(
    spec: VectorSpec,
    game: str,
    metadata: pd.DataFrame,
    vectors: np.lib.npyio.NpzFile,
    treatment: str,
    seed: int,
) -> tuple[np.ndarray, str]:
    key = vector_key(metadata, spec.vector_type, game, spec.target_trait, spec.source_layer)
    vec = np.asarray(vectors[key], dtype=np.float32)
    if treatment == "random_control":
        return make_random_like(vec, game, spec.target_trait, spec.source_layer, seed), f"random_norm_matched_to::{key}"
    return vec, key


def make_report(df: pd.DataFrame, out_dir: pathlib.Path, args) -> str:
    nonzero = df.loc[df["dose"] != 0].copy()
    if not nonzero.empty:
        nonzero["hook_delta_ratio"] = nonzero["actual_delta_norm"] / nonzero["dose_vector_norm"].replace(0, np.nan)
        nonzero["actual_delta_abs_cosine_with_vector"] = nonzero["actual_delta_cosine_with_vector"].abs()
        nonzero["hook_mass_before"] = nonzero["hook_prob_act0_before"] + nonzero["hook_prob_act1_before"]
        nonzero["hook_mass_after"] = nonzero["hook_prob_act0_after"] + nonzero["hook_prob_act1_after"]
        nonzero["final_mass_before"] = nonzero["final_prob_act0_before"] + nonzero["final_prob_act1_before"]
        nonzero["final_mass_after"] = nonzero["final_prob_act0_after"] + nonzero["final_prob_act1_after"]
        summary = (
            nonzero.groupby(["target_trait", "treatment", "inject_layer", "dose"], dropna=False)
            .agg(
                n=("round", "count"),
                mean_hook_delta_ratio=("hook_delta_ratio", "mean"),
                mean_actual_delta_cosine=("actual_delta_cosine_with_vector", "mean"),
                mean_actual_delta_abs_cosine=("actual_delta_abs_cosine_with_vector", "mean"),
                mean_hook_mass_before=("hook_mass_before", "mean"),
                mean_hook_mass_after=("hook_mass_after", "mean"),
                mean_hook_pref0_delta=("hook_pref0_delta", "mean"),
                mean_inject_pref0_delta=("inject_layer_pref0_delta", "mean"),
                mean_final_mass_before=("final_mass_before", "mean"),
                mean_final_mass_after=("final_mass_after", "mean"),
                mean_final_pref0_delta=("final_pref0_delta", "mean"),
                argmax_flip_rate=("argmax_changed", "mean"),
                sampled_flip_rate=("sampled_action_changed", "mean"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame()

    summary_path = out_dir / "diagnostic_summary.csv"
    summary.to_csv(summary_path, index=False)

    def md_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
        if frame.empty:
            return "_No rows._"
        view = frame.head(max_rows).copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        cols = list(view.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    hook_ok = "inconclusive"
    if not nonzero.empty:
        ratio = (nonzero["actual_delta_norm"] / nonzero["dose_vector_norm"].replace(0, np.nan)).dropna()
        cosv = nonzero["actual_delta_cosine_with_vector"].abs().dropna()
        if len(ratio) and len(cosv):
            hook_ok = "yes" if abs(float(ratio.mean()) - 1.0) < 0.05 and float(cosv.mean()) > 0.95 else "no/partial"

    intended = nonzero.loc[nonzero["treatment"] == "intended"] if not nonzero.empty else pd.DataFrame()
    controls = nonzero.loc[nonzero["treatment"].isin(["random_control", "wrong_layer"])] if not nonzero.empty else pd.DataFrame()
    intended_final = float(intended["final_pref0_delta"].abs().mean()) if not intended.empty else math.nan
    control_final = float(controls["final_pref0_delta"].abs().mean()) if not controls.empty else math.nan

    lines = [
        f"# Steering Mechanism Diagnostic: {args.exp_tag}",
        "",
        f"Created: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Design",
        "",
        f"- Game: `{args.game}`",
        f"- Seed: `{args.seed}`",
        f"- Rounds/prompts: `{args.rounds}`",
        f"- Vector specs: `{args.vector_specs}`",
        f"- Doses: `{args.doses}`",
        f"- Wrong layer: `{args.wrong_layer}`",
        "- Prompts are baseline P1 prompts. No trait prompt is injected into the text.",
        "- Each row compares a baseline pass and a steered/control pass on the exact same prompt and label mapping.",
        "",
        "## Mechanism Checks",
        "",
        f"1. Hook changed target hidden state by intended norm: **{hook_ok}**.",
        "2. `hook_pref0_delta` is the immediate pre/post readout computed inside the hook.",
        "3. `inject_layer_pref0_delta` is a post-forward `output_hidden_states` sanity column; use `hook_pref0_delta` for immediate steering readout.",
        "4. Interpret normalized preference at layers with near-zero A/B mass cautiously.",
        f"5. Mean |final normalized preference shift|, intended vectors: `{intended_final:.6f}`.",
        f"6. Mean |final normalized preference shift|, controls: `{control_final:.6f}`.",
        "7. Action movement must be judged from `argmax_flip_rate` and `sampled_flip_rate`, not raw action-token mass.",
        "",
        "## Summary Table",
        "",
        md_table(summary),
        "",
        "## Output Files",
        "",
        f"- `{out_dir / 'diagnostic_results.csv'}`",
        f"- `{summary_path}`",
        f"- `{out_dir / 'config.json'}`",
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "STEERING_MECHANISM_DIAGNOSTIC.md").write_text(report)
    return report


def main():
    p = argparse.ArgumentParser(description="Minimal steering mechanism diagnostic")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-72B")
    p.add_argument("--exp_tag", default="")
    p.add_argument("--game", default="ShSh")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--vector_specs", default=DEFAULT_VECTOR_SPECS)
    p.add_argument("--doses", default="-16,-8,0,8,16")
    p.add_argument("--wrong_layer", type=int, default=30)
    p.add_argument("--treatments", default="intended,random_control,wrong_layer")
    p.add_argument("--payoff_multiplier", type=int, default=2)
    p.add_argument("--probe_prefix", default="\nDecision: ")
    p.add_argument("--output_dir", default="validation_logs")
    p.add_argument("--vector_file", default=DEFAULT_VECTOR_FILE)
    p.add_argument("--metadata_file", default=DEFAULT_METADATA_FILE)
    p.add_argument("--use_ln_f_all", action="store_true", default=True)
    p.add_argument("--dry_run", action="store_true")

    # Model-loading compatibility with run_sim_spark helpers.
    p.add_argument("--load_8bit", action="store_true")
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--bnb_compute_dtype", default="bfloat16")
    p.add_argument("--attn_impl", default="sdpa")
    p.add_argument("--gptoss_device_map", choices=["auto", "cuda0", "single_gpu"], default="auto")
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
        args.exp_tag = f"steering_mechanism_diagnostic_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    vector_specs = parse_vector_specs(args.vector_specs)
    doses = parse_csv(args.doses, float)
    treatments = parse_csv(args.treatments)

    total_rows = args.rounds * len(vector_specs) * len(doses) * len(treatments)
    print(f"Steering mechanism diagnostic: {args.exp_tag}")
    print(f"Game={args.game} seed={args.seed} rounds={args.rounds}")
    print(f"Vector specs={vector_specs}")
    print(f"Doses={doses}")
    print(f"Diagnostic paired rows={total_rows}")
    if args.dry_run:
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    _disable_hf_allocator_warmup()
    _install_spark_cpu_first_bnb8_patch()

    metadata = pd.read_csv(args.metadata_file)
    vectors = np.load(args.vector_file)

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
            lta, atl = label_schedule[r]
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

            # One baseline pass per prompt. Immediate layer readout is reused for
            # each candidate because the prompt is identical.
            baseline_by_layer: dict[int, dict] = {}
            needed_layers = sorted({spec.source_layer for spec in vector_specs} | {args.wrong_layer})
            for layer in needed_layers:
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

            for spec in vector_specs:
                for dose in doses:
                    for treatment in treatments:
                        inject_layer = args.wrong_layer if treatment == "wrong_layer" else spec.source_layer
                        baseline = baseline_by_layer[inject_layer]
                        base_layer_readout = readout_from_label_probs(baseline["layer_probs"], atl)
                        base_final_readout = readout_from_label_probs(baseline["final_probs"], atl)

                        if dose == 0.0:
                            vec = None
                            vector_key_used = ""
                        else:
                            vec, vector_key_used = resolve_vector(
                                spec, args.game, metadata, vectors, treatment, args.seed
                            )

                        steered = probe_once(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            probe_prefix=args.probe_prefix,
                            move_token_map=move_token_map,
                            final_norm=final_norm,
                            lm_head=lm_head,
                            input_device=input_device,
                            inject_layer=inject_layer,
                            vector=vec,
                            dose=dose,
                            use_ln_f_all=args.use_ln_f_all,
                        )
                        steered_layer_readout = readout_from_label_probs(steered["layer_probs"], atl)
                        steered_final_readout = readout_from_label_probs(steered["final_probs"], atl)

                        vec_norm = np_norm(vec) if vec is not None else 0.0
                        dose_vec_norm = abs(float(dose)) * vec_norm
                        layer_hidden_delta = steered["layer_hidden"] - baseline["layer_hidden"]
                        hook_stats = steered["hook_stats"]
                        if all(f"hook_pre_prob_{label}" in hook_stats for label in MOVE_LABELS):
                            hook_pre_probs = {label: hook_stats[f"hook_pre_prob_{label}"] for label in MOVE_LABELS}
                            hook_post_probs = {label: hook_stats[f"hook_post_prob_{label}"] for label in MOVE_LABELS}
                            hook_pre_readout = readout_from_label_probs(hook_pre_probs, atl)
                            hook_post_readout = readout_from_label_probs(hook_post_probs, atl)
                        else:
                            hook_pre_readout = base_layer_readout
                            hook_post_readout = steered_layer_readout

                        sampled_before = 0 if sample_u < base_final_readout.pref0 else 1
                        sampled_after = 0 if sample_u < steered_final_readout.pref0 else 1

                        rows.append({
                            "game": args.game,
                            "seed": args.seed,
                            "round": r + 1,
                            "target_trait": spec.target_trait,
                            "vector_type": spec.vector_type,
                            "source_layer": spec.source_layer,
                            "inject_layer": inject_layer,
                            "dose": float(dose),
                            "treatment": treatment,
                            "vector_key_used": vector_key_used,
                            "label_map_p1": atl[0],
                            "row_first_p1": int(row_schedule[r]),
                            "col_first_p1": int(col_schedule[r]),
                            "sample_u": float(sample_u),
                            "vector_norm": vec_norm,
                            "dose_vector_norm": dose_vec_norm,
                            "baseline_inject_hidden_norm": np_norm(baseline["layer_hidden"]),
                            "steered_inject_hidden_norm": np_norm(steered["layer_hidden"]),
                            "paired_layer_hidden_delta_norm": np_norm(layer_hidden_delta),
                            "paired_layer_hidden_delta_cosine_with_vector": (
                                cosine(layer_hidden_delta, vec) if vec is not None else math.nan
                            ),
                            "pre_hidden_norm": hook_stats.get("pre_hidden_norm", math.nan),
                            "post_hidden_norm": hook_stats.get("post_hidden_norm", math.nan),
                            "actual_delta_norm": hook_stats.get("actual_delta_norm", 0.0 if dose == 0.0 else math.nan),
                            "actual_delta_cosine_with_vector": hook_stats.get(
                                "actual_delta_cosine_with_vector", math.nan
                            ),
                            "hook_prob_A_before": hook_pre_readout.prob_a,
                            "hook_prob_B_before": hook_pre_readout.prob_b,
                            "hook_prob_A_after": hook_post_readout.prob_a,
                            "hook_prob_B_after": hook_post_readout.prob_b,
                            "hook_prob_act0_before": hook_pre_readout.prob_act0,
                            "hook_prob_act1_before": hook_pre_readout.prob_act1,
                            "hook_prob_act0_after": hook_post_readout.prob_act0,
                            "hook_prob_act1_after": hook_post_readout.prob_act1,
                            "hook_pref0_before": hook_pre_readout.pref0,
                            "hook_pref0_after": hook_post_readout.pref0,
                            "hook_pref0_delta": hook_post_readout.pref0 - hook_pre_readout.pref0,
                            "inject_layer_prob_A_before": base_layer_readout.prob_a,
                            "inject_layer_prob_B_before": base_layer_readout.prob_b,
                            "inject_layer_prob_A_after": steered_layer_readout.prob_a,
                            "inject_layer_prob_B_after": steered_layer_readout.prob_b,
                            "inject_layer_prob_act0_before": base_layer_readout.prob_act0,
                            "inject_layer_prob_act1_before": base_layer_readout.prob_act1,
                            "inject_layer_prob_act0_after": steered_layer_readout.prob_act0,
                            "inject_layer_prob_act1_after": steered_layer_readout.prob_act1,
                            "inject_layer_pref0_before": base_layer_readout.pref0,
                            "inject_layer_pref0_after": steered_layer_readout.pref0,
                            "inject_layer_pref0_delta": steered_layer_readout.pref0 - base_layer_readout.pref0,
                            "final_prob_A_before": base_final_readout.prob_a,
                            "final_prob_B_before": base_final_readout.prob_b,
                            "final_prob_A_after": steered_final_readout.prob_a,
                            "final_prob_B_after": steered_final_readout.prob_b,
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
        vectors.close()
        print(f"Freeing model {args.model_name}...")
        del model
        _free_gpu_cache()

    df = pd.DataFrame(rows)
    results_path = out_dir / "diagnostic_results.csv"
    df.to_csv(results_path, index=False)
    config = vars(args).copy()
    config["base_vec"] = base_vec
    config["game_vec"] = game_vec
    config["runner"] = "run_steering_mechanism_diagnostic.py"
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    make_report(df, out_dir, args)
    print(f"Wrote: {results_path}")
    print(f"Wrote: {out_dir / 'STEERING_MECHANISM_DIAGNOSTIC.md'}")


if __name__ == "__main__":
    main()
