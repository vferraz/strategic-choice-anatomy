#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_sim_spark.py — DGX Spark variant (single GPU, unified memory)

Simplified fork of run_sim.py for the NVIDIA DGX Spark (GB10, 128 GB unified
memory, single GPU). Removes multi-GPU device mapping, async-load workarounds,
and MPS fallback. Core experiment logic is identical.

Key design choices (cf. Akata et al., 2025):
  - Abstract neutral labels "A" and "B" with per-round randomized mapping
    to action indices (0, 1). This counterbalances token-prior bias in the
    decision-slot probe: any asymmetry in P(A) vs P(B) after the probe
    prefix becomes orthogonal to the action space and cancels in expectation.
  - Matrix-format payoff rules with symmetric "you" / "the other player"
  - Baseline games only (original matrix × 2, no stakes manipulation)
  - Traits loaded from JSON (inequity aversion / risk aversion × light / heavy)
  - Full activation capture + logit-lens layer probing
  - Cross-model support: P1 and P2 can use different models (--model_p2)
  - No in-prompt label demonstrations (avoids priming first-listed label)

Payoff encoding:
  vec = [a00, a01, a10, a11, b00, b01, b10, b11]
  Player 1 (row): action 0 → row 0, action 1 → row 1
  Player 2 (col): action 0 → col 0, action 1 → col 1
  Labels A/B are randomly mapped to actions 0/1 each round.
"""

from __future__ import annotations

import argparse, gc, json, os, pathlib, sys, datetime, random, re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import transformers.modeling_utils as _hf_modeling_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# --- local ---
from strategic_anatomy.games import bruns_games

# Prompt/label layer, moved to a torch-free module during the open-source migration.
# Re-exported here so every historical `from ...run_sim_spark import <name>` call site
# resolves unchanged.
from strategic_anatomy.prompting import (  # noqa: F401
    MOVE_LABELS,
    PAYOFF_MULTIPLIER,
    Trait,
    generate_label_map,
    generate_balanced_label_schedule,
    generate_balanced_swap_schedule,
    payoff_rules_symmetric,
    format_instruction,
    valid_moves_line,
    history_lines_symmetric,
    build_prompt,
    build_move_token_map,
    parse_generated_move,
    load_traits,
)


def _disable_hf_allocator_warmup():
    """Disable Transformers' caching allocator warmup on Spark unified memory."""
    try:
        _hf_modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    except Exception:
        pass


_SPARK_BNB8_LOAD_CTX = {
    "patched": False,
    "active": False,
    "model": None,
    "hf_quantizer": None,
    "device_map": None,
}


def _install_bnb_int8params_accelerate_compat_patch():
    """Drop HF/Accelerate private attrs before reconstructing bnb Int8Params.

    Accelerate 1.13 passes `Int8Params.__dict__` back into the constructor when
    attaching device/offload hooks. bitsandbytes 0.49.2 only accepts the real
    quantization constructor fields, so HF bookkeeping attrs such as
    `_is_hf_initialized` must not be forwarded.
    """
    try:
        import inspect
        import bitsandbytes as bnb
    except Exception:
        return

    cls = bnb.nn.Int8Params
    if getattr(cls, "_spark_accelerate_compat_patched", False):
        return

    orig_new = cls.__new__
    sig = inspect.signature(orig_new)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        cls._spark_accelerate_compat_patched = True
        return

    allowed_kwargs = {
        name for name, param in sig.parameters.items()
        if name != "cls" and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    def new_wrapper(cls_, *args, **kwargs):
        if kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}
        return orig_new(cls_, *args, **kwargs)

    cls.__new__ = staticmethod(new_wrapper)
    cls._spark_accelerate_compat_patched = True


def _install_spark_cpu_first_bnb8_patch():
    """Load bnb int8 weights on CPU first, then move only int8 params to GPU.

    On Spark unified memory, the default HF+bnb8 path materializes quantized
    tensors on GPU in bf16, bounces them back to CPU for Int8Params
    construction, and then moves the int8 result back to GPU. Redirecting the
    pre-quantization materialization to CPU avoids that wasteful triple-hop.
    """
    if _SPARK_BNB8_LOAD_CTX["patched"]:
        return

    try:
        import transformers.core_model_loading as _hf_core_loading
        import transformers.integrations.bitsandbytes as _hf_bnb
    except Exception:
        return

    orig_convert_and_load = _hf_core_loading.convert_and_load_state_dict_in_model
    orig_get_device = _hf_core_loading.get_device
    orig_quantize = _hf_bnb.Bnb8bitQuantize.convert

    def convert_and_load_wrapper(model, state_dict, load_config, tp_plan, disk_offload_index=None):
        hf_quantizer = load_config.hf_quantizer
        enable = type(hf_quantizer).__name__ == "Bnb8BitHfQuantizer"
        _SPARK_BNB8_LOAD_CTX.update(
            {
                "active": enable,
                "model": model if enable else None,
                "hf_quantizer": hf_quantizer if enable else None,
                "device_map": load_config.device_map if enable else None,
            }
        )
        try:
            return orig_convert_and_load(model, state_dict, load_config, tp_plan, disk_offload_index)
        finally:
            _SPARK_BNB8_LOAD_CTX.update(
                {
                    "active": False,
                    "model": None,
                    "hf_quantizer": None,
                    "device_map": None,
                }
            )

    def get_device_wrapper(device_map, key, valid_torch_device=False):
        if valid_torch_device and _SPARK_BNB8_LOAD_CTX["active"]:
            model = _SPARK_BNB8_LOAD_CTX["model"]
            hf_quantizer = _SPARK_BNB8_LOAD_CTX["hf_quantizer"]
            if model is not None and hf_quantizer is not None:
                try:
                    if hf_quantizer.param_needs_quantization(model, key):
                        return "cpu"
                except Exception:
                    pass
        return orig_get_device(device_map, key, valid_torch_device)

    def quantize_wrapper(self, input_dict, model=None, full_layer_name=None, **kwargs):
        if not _SPARK_BNB8_LOAD_CTX["active"] or _SPARK_BNB8_LOAD_CTX["device_map"] is None:
            return orig_quantize(self, input_dict, model=model, full_layer_name=full_layer_name, **kwargs)

        value = list(input_dict.values())[0]
        value = value[0] if isinstance(value, list) else value

        module, _ = _hf_bnb.get_module_from_name(model, full_layer_name)
        if issubclass(module.source_cls, _hf_bnb.Conv1D):
            value = value.T

        value_device = orig_get_device(
            _SPARK_BNB8_LOAD_CTX["device_map"], full_layer_name, valid_torch_device=True
        )
        bnb_kwargs = model.get_parameter_or_buffer(full_layer_name).__dict__.copy()
        bnb_kwargs.pop("SCB", None)
        new_value = _hf_bnb.bnb.nn.Int8Params(value.to("cpu"), requires_grad=False, **bnb_kwargs).to(value_device)
        return {full_layer_name: new_value}

    _hf_core_loading.convert_and_load_state_dict_in_model = convert_and_load_wrapper
    _hf_modeling_utils.convert_and_load_state_dict_in_model = convert_and_load_wrapper
    _hf_core_loading.get_device = get_device_wrapper
    _hf_bnb.Bnb8bitQuantize.convert = quantize_wrapper
    _SPARK_BNB8_LOAD_CTX["patched"] = True
    _install_bnb_int8params_accelerate_compat_patch()


def trim_to_tokens(prompt: str, tok: PreTrainedTokenizerBase, max_tokens: int = 1980) -> str:
    """Trim prompt to max_tokens, keeping HEADER (first section) + TAIL (recent history + decision).

    Strategy: split at "History so far:" marker.
      - If marker found: keep header up to marker + as much tail as fits.
      - If no marker: keep last max_tokens tokens (tail-priority).
    """
    ids = tok(prompt).input_ids
    if len(ids) <= max_tokens:
        return prompt

    # Try to find the history boundary
    marker = "History so far:"
    marker_pos = prompt.find(marker)

    if marker_pos > 0:
        header = prompt[:marker_pos]
        tail = prompt[marker_pos:]
        header_ids = tok(header).input_ids
        tail_ids = tok(tail).input_ids

        # Budget: keep full header, trim tail from the LEFT (drop oldest rounds)
        tail_budget = max_tokens - len(header_ids)
        if tail_budget > 0 and len(tail_ids) > tail_budget:
            # Keep the marker line + most recent rounds (tail end)
            marker_line_ids = tok(marker + "\n").input_ids
            remaining = tail_budget - len(marker_line_ids)
            if remaining > 0:
                trimmed_tail_ids = marker_line_ids + tail_ids[-remaining:]
            else:
                trimmed_tail_ids = tail_ids[-tail_budget:]
            return tok.decode(header_ids + trimmed_tail_ids)
        elif tail_budget > 0:
            return tok.decode(header_ids + tail_ids)
        else:
            # Header alone exceeds budget — keep tail priority
            return tok.decode(ids[-max_tokens:])
    else:
        # No marker — keep the tail (recent context + decision region)
        return tok.decode(ids[-max_tokens:])

# =============================================================================
# Model utilities
# =============================================================================

def ids_to_tokens(tok: PreTrainedTokenizerBase, ids: np.ndarray) -> List[str]:
    return tok.convert_ids_to_tokens(ids.tolist())


def get_num_layers(model) -> int:
    cfg = getattr(model, "config", None)
    n = getattr(cfg, "n_layer", None) or getattr(cfg, "num_hidden_layers", None)
    if n is None:
        raise NotImplementedError("Could not infer number of layers from model.config")
    return int(n)


def get_final_norm_module(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "model"):
        ln = getattr(model.model, "norm", None) or getattr(model.model, "ln_f", None)
        if ln is not None:
            return ln
        if hasattr(model.model, "decoder"):
            return getattr(model.model.decoder, "final_layer_norm", None)
    return None


def apply_final_norm_no_grad(x: torch.Tensor, norm) -> torch.Tensor:
    if norm is None:
        return x
    with torch.no_grad():
        y = norm(x.unsqueeze(0))
        if y.dim() == 2 and y.shape[0] == 1:
            y = y.squeeze(0)
        return y


def _is_gpt_oss_model(model_name: str) -> bool:
    return (model_name or "").strip().lower() == "openai/gpt-oss-120b"


def _normalize_model_device(device_like):
    if device_like is None:
        return None
    if isinstance(device_like, torch.device):
        return device_like
    if isinstance(device_like, int):
        return torch.device(f"cuda:{device_like}")

    device_str = str(device_like).strip().lower()
    if not device_str or device_str in {"disk", "meta"}:
        return None
    if device_str.isdigit():
        return torch.device(f"cuda:{device_str}")
    try:
        return torch.device(device_str)
    except (RuntimeError, ValueError):
        return None


def resolve_model_input_device(model) -> torch.device:
    """Pick the device where input tensors should be placed."""
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for target in hf_device_map.values():
            device = _normalize_model_device(target)
            if device is not None and device.type == "cuda":
                return device

    model_device = _normalize_model_device(getattr(model, "device", None))
    if model_device is not None:
        return model_device

    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def _ensure_attention_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is not None:
        return attention_mask
    return torch.ones_like(input_ids, dtype=torch.long)

# =============================================================================
# Attention layer spec parsing
# =============================================================================

def _parse_attn_layers(spec: str, n_layers: int) -> List[int]:
    """Parse --attn_layers spec into a sorted list of layer indices.

    Accepted formats:
      "none"          → []
      "all"           → [0, 1, ..., n_layers-1]
      "every:N"       → [0, N, 2N, ...] up to n_layers-1
      "0,10,20,79"    → explicit list (validated)
    """
    spec = spec.strip().lower()
    if spec == "none":
        return []
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("every:"):
        step = int(spec.split(":", 1)[1])
        if step < 1:
            raise ValueError(f"--attn_layers every:N requires N >= 1, got {step}")
        return list(range(0, n_layers, step))
    # comma-separated ints
    indices = sorted(set(int(x) for x in spec.split(",")))
    for idx in indices:
        if not (0 <= idx < n_layers):
            raise ValueError(f"--attn_layers index {idx} out of range 0..{n_layers - 1}")
    return indices


def _parse_hidden_layers(spec: str, n_layers: int) -> List[int]:
    """Parse --layer spec into hidden-state layer indices.

    Accepted formats:
      "none"          -> []
      "all"           -> [0, 1, ..., n_layers-1]
      "30"            -> [30]  (legacy behavior)
      "30,65,75,79"   -> explicit list
    """
    spec = str(spec).strip().lower()
    if spec == "none":
        return []
    if spec == "all":
        return list(range(n_layers))
    indices = sorted(set(int(x.strip()) for x in spec.split(",") if x.strip()))
    if not indices:
        raise ValueError(f"Empty --layer spec: {spec!r}")
    for idx in indices:
        if not (0 <= idx < n_layers):
            raise ValueError(f"Layer index {idx} out of range 0..{n_layers - 1}")
    return indices


def load_custom_games_json(path: str | os.PathLike) -> Dict[str, List]:
    """Load custom 2x2 games in run_sim_spark's bruns_games-compatible format.

    Accepted JSON shapes:
      [{"game_code": "SF_x1_y0", "vector": [...]}, ...]
      {"games": [{"game_code": "...", "vector": [...]}, ...]}
      {"SF_x1_y0": [...], ...}
    """
    p = pathlib.Path(path)
    raw = json.loads(p.read_text())
    if isinstance(raw, dict) and "games" in raw:
        raw = raw["games"]

    out: Dict[str, List] = {}
    if isinstance(raw, dict):
        iterable = []
        for code, value in raw.items():
            if isinstance(value, dict):
                rec = dict(value)
                rec.setdefault("game_code", code)
            else:
                rec = {"game_code": code, "vector": value}
            iterable.append(rec)
    elif isinstance(raw, list):
        iterable = raw
    else:
        raise ValueError(f"Unsupported custom games JSON shape in {p}")

    for rec in iterable:
        if not isinstance(rec, dict):
            raise ValueError(f"Custom game entry must be an object, got {type(rec).__name__}")
        code = str(rec.get("game_code") or rec.get("code") or "").strip()
        vec = rec.get("vector") or rec.get("payoff_vector") or rec.get("base_vec")
        if not code:
            raise ValueError(f"Custom game missing game_code: {rec}")
        if not isinstance(vec, list) or len(vec) != 8:
            raise ValueError(f"Custom game {code} must carry an 8-entry vector")
        vec = [int(v) for v in vec]
        name = str(rec.get("name") or rec.get("family") or "")
        out[code] = [vec, code, name]
    return out


# =============================================================================
# Agent
# =============================================================================

class Agent:
    def __init__(
        self,
        model: PreTrainedModel,
        tok: PreTrainedTokenizerBase,
        layer_spec: str,
        valid_moves: List[str],
        probe_prefix: str,
        use_ln_f_all: bool,
        max_history_arg: int,
        attn_layers: str = "none",
        attn_topk: int = 0,
        attn_dtype: str = "float16",
        save_token_layers: bool = False,
        decision_mode: str = "argmax",
        generate_temperature: float = 1.0,
        generate_max_tokens: int = 48,
    ):
        self.model = model
        self.tok = tok
        self.valid_moves = list(valid_moves)
        self.probe_prefix = probe_prefix
        self.use_ln_f_all = bool(use_ln_f_all)
        self.max_history_arg = int(max_history_arg)
        self.attn_topk = int(attn_topk)
        self.attn_dtype = np.float16 if attn_dtype == "float16" else np.float32
        self.save_token_layers = bool(save_token_layers)
        self.decision_mode = str(decision_mode)
        self.generate_temperature = float(generate_temperature)
        self.generate_max_tokens = int(generate_max_tokens)

        self.n_layers = get_num_layers(model)
        self.hook_layers = _parse_hidden_layers(layer_spec, self.n_layers)

        self.attn_layers = _parse_attn_layers(attn_layers, self.n_layers)
        self.save_attn = len(self.attn_layers) > 0

        self.final_norm = get_final_norm_module(model)
        self.lm_head = model.get_output_embeddings()
        self.input_device = resolve_model_input_device(model)

        self.move_token_map = build_move_token_map(self.tok, self.probe_prefix, self.valid_moves)

    def generate_move(self, prompt: str) -> Tuple[str, str, bool]:
        """
        Let the LLM actually generate text and parse the move from its output.
        Returns (chosen_move, generated_text, parse_ok).
        Falls back to None (caller applies sample_probe) if parsing fails.
        """
        enc = self.tok(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.input_device)
        attention_mask = _ensure_attention_mask(input_ids, enc.get("attention_mask"))
        attention_mask = attention_mask.to(self.input_device)

        gen_kwargs = dict(
            max_new_tokens=self.generate_max_tokens,
            do_sample=True,
            temperature=self.generate_temperature,
            top_p=0.95,
            pad_token_id=self.tok.eos_token_id,
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        new_ids = output_ids[0, input_ids.shape[1]:]
        generated_text = self.tok.decode(new_ids, skip_special_tokens=True)
        move_str = parse_generated_move(generated_text, self.valid_moves)
        return move_str, generated_text, move_str is not None

    def choose(self, prompt: str, max_history: int = 0):
        prompt_used = prompt if (max_history == 0 and self.max_history_arg == 0) \
            else trim_to_tokens(prompt, self.tok, max_tokens=1980)

        # --- Step 1: Get the actual move (generate pass, if needed) ---
        generated_text = ""
        gen_parse_ok = True  # True for argmax/sample_probe (no parsing needed)
        gen_move = None
        if self.decision_mode == "generate":
            gen_move, generated_text, gen_parse_ok = self.generate_move(prompt_used)

        # --- Step 2: Probe pass (activations + logit-lens, always runs) ---
        text = prompt_used + self.probe_prefix

        enc_fast = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = torch.tensor([enc_fast.input_ids], device=self.input_device)
        attention_mask = _ensure_attention_mask(input_ids, None).to(self.input_device)

        prompt_len = len(self.tok(prompt_used, add_special_tokens=False).input_ids)
        prefix_len = len(self.tok(self.probe_prefix, add_special_tokens=False).input_ids)

        seq_input_ids = np.asarray(enc_fast.input_ids, dtype=np.int32)
        seq_offsets = np.asarray(enc_fast.offset_mapping, dtype=np.int32)

        hidden_by_layer: Dict[int, torch.Tensor] = {}
        hidden_seq_by_layer: Dict[int, np.ndarray] = {}

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=self.save_attn,
                use_cache=False,
            )
            hs = out.hidden_states

        for l in self.hook_layers:
            h = hs[l + 1][0, -1, :].detach().to(self.lm_head.weight.device)
            hidden_by_layer[l] = h
            if self.save_token_layers:
                hidden_seq_by_layer[l] = hs[l + 1][0].detach().to(torch.float32).cpu().numpy()

        final_logits = out.logits[0, -1, :]
        final_probs = torch.softmax(final_logits, dim=-1)

        slot_move_probs = {
            mv: float(final_probs[self.move_token_map[mv]].item())
            for mv in self.valid_moves
        }

        # --- Step 3: Decide the move ---
        mv0, mv1 = self.valid_moves[0], self.valid_moves[1]

        if self.decision_mode == "generate":
            # No fallback to sample_probe — parse failure → None
            chosen_move = gen_move  # str label or None
        elif self.decision_mode == "sample_probe":
            # Sample from renormalized probe distribution
            p0 = slot_move_probs[mv0]
            p1 = slot_move_probs[mv1]
            den = p0 + p1
            if den > 0:
                chosen_move = mv0 if random.random() < (p0 / den) else mv1
            else:
                chosen_move = random.choice(self.valid_moves)
        else:
            # argmax (legacy, kept for reproducibility)
            chosen_move = max(slot_move_probs, key=slot_move_probs.get)

        if chosen_move is not None:
            first_token_id = int(self.move_token_map[chosen_move])
        else:
            first_token_id = -1

        # per-layer logit lens
        per_layer_probs: Dict[int, Dict[str, float]] = {}
        last_layer = self.n_layers - 1

        for l in sorted(hidden_by_layer.keys()):
            if l == last_layer:
                per_layer_probs[l] = {
                    mv: float(final_probs[self.move_token_map[mv]].item())
                    for mv in self.valid_moves
                }
                continue

            h = hidden_by_layer[l]
            if self.use_ln_f_all and self.final_norm is not None:
                h_proj = apply_final_norm_no_grad(h, self.final_norm)
            else:
                h_proj = h

            logits = self.lm_head(h_proj.clone().to(self.lm_head.weight.device).unsqueeze(0))[0]
            probs = torch.softmax(logits, dim=0)
            per_layer_probs[l] = {
                mv: float(probs[self.move_token_map[mv]].item())
                for mv in self.valid_moves
            }

        # optional attention
        attn_move_slot = {}
        if self.save_attn and getattr(out, "attentions", None) is not None:
            for l in self.attn_layers:
                A = out.attentions[l][0, :, -1, :]
                if self.attn_topk > 0:
                    k = min(self.attn_topk, A.shape[-1])
                    vals, idx = torch.topk(A, k, dim=-1)
                    attn_move_slot[l] = {
                        "idx": idx.cpu().numpy().astype(np.int32),
                        "val": vals.cpu().numpy().astype(self.attn_dtype),
                    }
                else:
                    attn_move_slot[l] = A.cpu().numpy().astype(self.attn_dtype)

        return dict(
            prompt_used=prompt_used,
            per_layer_probs=per_layer_probs,
            final_slot_probs=slot_move_probs,
            chosen_move=chosen_move,
            first_token_id=first_token_id,
            hidden_by_layer={l: v.to(torch.float32).cpu().numpy() for l, v in hidden_by_layer.items()},
            hidden_seq_by_layer=hidden_seq_by_layer if self.save_token_layers else {},
            attn_move_slot=attn_move_slot,
            input_ids=seq_input_ids,
            offsets=seq_offsets,
            tokens=np.array(ids_to_tokens(self.tok, seq_input_ids), dtype=object),
            text=np.array([text], dtype="U"),
            prompt_len=prompt_len,
            prefix_len=prefix_len,
            generated_text=generated_text,
            gen_parse_ok=gen_parse_ok,
        )

# =============================================================================
# Model loading helpers
# =============================================================================

def _build_quant_config(args):
    """Build BitsAndBytesConfig from CLI args, or return None."""
    if (_is_gpt_oss_model(args.model_name) or _is_gpt_oss_model(args.model_p2)) and (args.load_4bit or args.load_8bit):
        raise SystemExit("GPT-OSS does not support --load_4bit or --load_8bit in the Spark experiment path.")
    if not torch.cuda.is_available():
        if args.load_4bit or args.load_8bit:
            print("[warn] quantization requires CUDA; loading full precision.")
        return None
    from transformers import BitsAndBytesConfig
    if args.load_4bit:
        bnb_dt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[args.bnb_compute_dtype]
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bnb_dt,
        )
    if args.load_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            llm_int8_enable_fp32_cpu_offload=bool(getattr(args, "bnb_8bit_cpu_offload", False)),
        )
    return None


def _check_quantized(m) -> bool:
    """Return True if the model has int8/int4-quantized linear layers."""
    for name, module in m.named_modules():
        cls_name = type(module).__name__
        if 'Linear8bitLt' in cls_name or 'Int8Params' in cls_name or 'Linear4bit' in cls_name:
            print(f"  ✓ Found quantized module: {cls_name} at {name}")
            return True
    dtypes = {str(p.dtype) for p in m.parameters()}
    print(f"  Parameter dtypes found: {dtypes}")
    if 'torch.int8' in dtypes or 'torch.uint8' in dtypes:
        print(f"  ✓ Found quantized parameters")
        return True
    return False


def _resolve_load_device_map(device_map_arg: str, arg_name: str = "--hf_device_map"):
    device_map_arg = str(device_map_arg or "cuda0").strip().lower()
    if device_map_arg in {"cuda0", "single_gpu"}:
        return {"": 0}
    if device_map_arg == "auto":
        return "auto"
    raise ValueError(f"{arg_name} must be 'auto', 'cuda0', or 'single_gpu'")


def _parse_max_memory(max_memory):
    if not max_memory:
        return None
    if isinstance(max_memory, dict):
        return max_memory

    parsed = {}
    for item in str(max_memory).split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key.isdigit():
            key = int(key)
        parsed[key] = value.strip()
    return parsed or None


def _load_model(model_name: str, args, qcfg=None) -> PreTrainedModel:
    """Load a single model onto the DGX Spark GPU in eval mode.

    Simplified for single-GPU unified memory. When quantization is requested
    (qcfg is not None), does NOT pass torch_dtype — lets quantization_config
    control the dtype.
    """
    attn_impl = args.attn_impl
    if args.attn_layers != "none" and attn_impl in ("auto", "sdpa"):
        print(f"[INFO] Switching attn_impl from '{attn_impl}' to 'eager' "
              f"(required for --attn_layers output)")
        attn_impl = "eager"

    use_quant = qcfg is not None

    # --- Pre-load diagnostics ---
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free = props.total_memory - torch.cuda.memory_allocated(0)
        print(f"  GPU: {props.name} | {free / 1e9:.1f} GB free / {props.total_memory / 1e9:.1f} GB total")
    print(f"  quantization_config: {qcfg}")

    # Build kwargs for from_pretrained.
    if _is_gpt_oss_model(model_name):
        if use_quant:
            raise ValueError("GPT-OSS load path must not receive a quantization config.")
        gptoss_device_map = getattr(args, "gptoss_device_map", "auto")
        device_map = _resolve_load_device_map(gptoss_device_map, "--gptoss_device_map")
        load_kwargs = dict(
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
    else:
        hf_device_map = getattr(args, "hf_device_map", "cuda0")
        device_map = _resolve_load_device_map(hf_device_map)
        load_kwargs = dict(
            trust_remote_code=True,
            attn_implementation=attn_impl,
            device_map=device_map,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        if use_quant:
            load_kwargs["quantization_config"] = qcfg
            load_kwargs["dtype"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
        max_memory = _parse_max_memory(getattr(args, "hf_max_memory", None))
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        offload_folder = getattr(args, "hf_offload_folder", "")
        if offload_folder:
            load_kwargs["offload_folder"] = offload_folder
            load_kwargs["offload_buffers"] = bool(getattr(args, "hf_offload_buffers", False))

    print(f"  device_map: {load_kwargs['device_map']}")
    if "max_memory" in load_kwargs:
        print(f"  max_memory: {load_kwargs['max_memory']}")
    if "offload_folder" in load_kwargs:
        print(f"  offload_folder: {load_kwargs['offload_folder']}")
        print(f"  offload_buffers: {load_kwargs.get('offload_buffers')}")
    print(f"  dtype: {load_kwargs.get('dtype', load_kwargs.get('torch_dtype'))}")
    print(f"\n  Loading {model_name} ...")
    disable_cuda_cache = bool(getattr(args, "disable_cuda_cache_during_load", False))
    if disable_cuda_cache and torch.cuda.is_available():
        print("  cuda caching allocator during load: disabled")
        torch.cuda.empty_cache()
        torch.cuda.memory.caching_allocator_enable(False)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    finally:
        if disable_cuda_cache and torch.cuda.is_available():
            torch.cuda.memory.caching_allocator_enable(True)
            torch.cuda.empty_cache()
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        device_counts: Dict[str, int] = {}
        for target in hf_device_map.values():
            device_counts[str(target)] = device_counts.get(str(target), 0) + 1
        print(f"  resolved hf_device_map targets: {device_counts}")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1e9
        print(f"  GPU memory after load: {alloc:.1f} GB")

    if use_quant:
        if _check_quantized(model):
            print("  ✓ Quantization verified.")
        else:
            alloc = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
            print(f"\n  ✗ QUANTIZATION FAILED — loaded in full precision ({alloc:.0f} GB).")

    return model.eval()


def _load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load and configure a tokenizer."""
    if _is_gpt_oss_model(model_name):
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    else:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def _free_gpu_cache():
    """Run garbage collection and free CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()

# =============================================================================
# Main simulation loop
# =============================================================================

def run_one_match(
    game_code: str,
    base_vec: List[int],
    trait_p1: Trait,
    trait_p2: Trait,
    args,
    tok_p1: PreTrainedTokenizerBase,
    tok_p2: PreTrainedTokenizerBase,
    qcfg,
):
    """
    Run one full match (all rounds). Loads/unloads models internally.
    For cross-model (args.model_p2 differs from args.model_name), swaps
    models on/off GPU each round via delete-and-reload.
    """
    # Scale payoffs
    multiplier = getattr(args, 'payoff_multiplier', PAYOFF_MULTIPLIER)
    game_vec = [v * multiplier for v in base_vec]

    cross_model = bool(args.model_p2) and args.model_p2 != args.model_name
    model_name_p1 = args.model_name
    model_name_p2 = args.model_p2 if cross_model else args.model_name

    pair_id = f"{game_code}_{trait_p1.id}_v_{trait_p2.id}"
    attn_npz: Dict[str, np.ndarray] = {}
    cum1 = cum2 = 0
    total_rounds = args.rounds

    # Shared kwargs for Agent construction
    agent_kwargs = dict(
        layer_spec=args.layer,
        valid_moves=MOVE_LABELS,
        probe_prefix=args.probe_prefix,
        use_ln_f_all=args.use_ln_f_all,
        max_history_arg=args.max_history,
        attn_layers=args.attn_layers,
        attn_topk=args.attn_topk,
        attn_dtype=args.attn_dtype,
        save_token_layers=args.save_token_layers,
        decision_mode=args.decision_mode,
        generate_temperature=args.generate_temperature,
        generate_max_tokens=args.generate_max_tokens,
    )

    # --- Load initial model (P1's model) ---
    print(f"  Loading {model_name_p1}...")
    current_model = _load_model(model_name_p1, args, qcfg)
    current_is_p1 = True

    # For same-model: create agents once (they share the model)
    agent1 = agent2 = None
    if not cross_model:
        agent1 = Agent(current_model, tok_p1, **agent_kwargs)
        agent2 = Agent(current_model, tok_p2, **agent_kwargs)

    # history stores (action1_idx, action2_idx, payoff1, payoff2) in action space
    history: List[Tuple[int, int, int, int]] = []
    rows: List[Dict] = []
    activ: Dict[str, np.ndarray] = {}
    probs_npz: Dict[str, np.ndarray] = {}

    act0_count1 = act0_count2 = 0   # frequency of action 0
    valid_rounds1 = valid_rounds2 = 0

    # Pre-compute balanced schedules if requested
    _bal_labels = getattr(args, 'balanced_labels', False)
    _bal_display = getattr(args, 'balanced_display', False)
    if _bal_labels:
        _label_sched_p1 = generate_balanced_label_schedule(total_rounds)
        _label_sched_p2 = generate_balanced_label_schedule(total_rounds)
    if _bal_display:
        _sched_row_p1 = generate_balanced_swap_schedule(total_rounds)
        _sched_col_p1 = generate_balanced_swap_schedule(total_rounds)
        _sched_row_p2 = generate_balanced_swap_schedule(total_rounds)
        _sched_col_p2 = generate_balanced_swap_schedule(total_rounds)

    for r in range(total_rounds):
        # --- Independent rounds: reset all sequence state ---
        if getattr(args, 'independent_rounds', False):
            history = []
            cum1 = cum2 = 0
            act0_count1 = act0_count2 = 0
            valid_rounds1 = valid_rounds2 = 0

        # --- Generate label mappings for each player this round ---
        if _bal_labels:
            lta_p1, atl_p1 = _label_sched_p1[r]
            lta_p2, atl_p2 = _label_sched_p2[r]
        else:
            lta_p1, atl_p1 = generate_label_map()
            lta_p2, atl_p2 = generate_label_map()

        # --- Generate matrix display order for each player ---
        if _bal_display:
            swap_rows_p1 = _sched_row_p1[r]
            swap_cols_p1 = _sched_col_p1[r]
            swap_rows_p2 = _sched_row_p2[r]
            swap_cols_p2 = _sched_col_p2[r]
        else:
            swap_rows_p1 = random.random() < 0.5
            swap_cols_p1 = random.random() < 0.5
            swap_rows_p2 = random.random() < 0.5
            swap_cols_p2 = random.random() < 0.5

        prompt1 = build_prompt(
            game_vec=game_vec,
            history=history,
            include_history=args.history,
            valid_moves=MOVE_LABELS,
            player=1,
            cumulative_you=cum1,
            cumulative_opp=cum2,
            action_to_label=atl_p1,
            swap_rows=swap_rows_p1,
            swap_cols=swap_cols_p1,
            iv_prefix=trait_p1.system_prefix,
            max_history=args.max_history,
            shuffle_valid_moves=not args.no_shuffle_valid_moves,
        )

        # --- P1 inference (swap if needed) ---
        if cross_model:
            if not current_is_p1:
                agent2 = None
                current_model = None
                _free_gpu_cache()
                current_model = _load_model(model_name_p1, args, qcfg)
                current_is_p1 = True
                if (r + 1) <= 3 or (r + 1) % 20 == 0:
                    print(f"    [round {r+1}] swapped → {model_name_p1} (P1)")
            agent1 = Agent(current_model, tok_p1, **agent_kwargs)

        res1 = agent1.choose(prompt1, max_history=args.max_history)

        prompt2 = build_prompt(
            game_vec=game_vec,
            history=history,
            include_history=args.history,
            valid_moves=MOVE_LABELS,
            player=2,
            cumulative_you=cum2,
            cumulative_opp=cum1,
            action_to_label=atl_p2,
            swap_rows=swap_rows_p2,
            swap_cols=swap_cols_p2,
            iv_prefix=trait_p2.system_prefix,
            max_history=args.max_history,
            shuffle_valid_moves=not args.no_shuffle_valid_moves,
        )

        # --- P2 inference (swap if needed) ---
        if cross_model:
            agent1 = None
            current_model = None
            _free_gpu_cache()
            current_model = _load_model(model_name_p2, args, qcfg)
            current_is_p1 = False
            if (r + 1) <= 3 or (r + 1) % 20 == 0:
                print(f"    [round {r+1}] swapped → {model_name_p2} (P2)")
            agent2 = Agent(current_model, tok_p2, **agent_kwargs)

        res2 = agent2.choose(prompt2, max_history=args.max_history)

        # --- progress for cross-model ---
        if cross_model and ((r + 1) % 10 == 0 or r == 0):
            print(f"    round {r+1}/{total_rounds} done")

        # attention saving
        if args.attn_layers != "none":
            for l, arr in res1["attn_move_slot"].items():
                if isinstance(arr, dict):
                    attn_npz[f"r{r}_p1_l{l}_attn_idx"] = arr["idx"]
                    attn_npz[f"r{r}_p1_l{l}_attn_val"] = arr["val"]
                else:
                    attn_npz[f"r{r}_p1_l{l}_attn"] = arr
            for l, arr in res2["attn_move_slot"].items():
                if isinstance(arr, dict):
                    attn_npz[f"r{r}_p2_l{l}_attn_idx"] = arr["idx"]
                    attn_npz[f"r{r}_p2_l{l}_attn_val"] = arr["val"]
                else:
                    attn_npz[f"r{r}_p2_l{l}_attn"] = arr

        # --- Map labels to action indices ---
        move1_label = res1["chosen_move"]   # "A", "B", or None
        move2_label = res2["chosen_move"]

        if move1_label is not None:
            act1 = lta_p1[move1_label]      # action index for P1 (row)
        else:
            act1 = None
        if move2_label is not None:
            act2 = lta_p2[move2_label]      # action index for P2 (col)
        else:
            act2 = None

        # --- Compute payoffs (only if both moves are valid) ---
        if act1 is not None and act2 is not None:
            A_mat = np.array(game_vec[:4]).reshape(2, 2)
            B_mat = np.array(game_vec[4:8]).reshape(2, 2)
            payoff1, payoff2 = int(A_mat[act1, act2]), int(B_mat[act1, act2])

            # sanity check
            assert payoff1 == game_vec[act1 * 2 + act2] and \
                   payoff2 == game_vec[4 + act1 * 2 + act2], (
                f"Payoff mismatch! actions=({act1},{act2}): "
                f"computed=({payoff1},{payoff2}), "
                f"expected=({game_vec[act1 * 2 + act2]},{game_vec[4 + act1 * 2 + act2]})"
            )

            # update history and cumulative (only complete rounds)
            history.append((act1, act2, payoff1, payoff2))
            cum1 += payoff1
            cum2 += payoff2

            act0_count1 += int(act1 == 0)
            valid_rounds1 += 1
            act0_count2 += int(act2 == 0)
            valid_rounds2 += 1
        else:
            payoff1 = payoff2 = None

        freq_act0_p1 = act0_count1 / valid_rounds1 if valid_rounds1 > 0 else float("nan")
        freq_act0_p2 = act0_count2 / valid_rounds2 if valid_rounds2 > 0 else float("nan")

        # save activations (always, even on parse failure — probe data is still valid)
        for l, vec in res1["hidden_by_layer"].items():
            activ[f"r{r}_p1_l{l}"] = vec
            if args.save_token_layers and l in res1["hidden_seq_by_layer"]:
                activ[f"r{r}_p1_l{l}_seq"] = res1["hidden_seq_by_layer"][l]
        for l, vec in res2["hidden_by_layer"].items():
            activ[f"r{r}_p2_l{l}"] = vec
            if args.save_token_layers and l in res2["hidden_seq_by_layer"]:
                activ[f"r{r}_p2_l{l}_seq"] = res2["hidden_seq_by_layer"][l]

        # metadata
        activ[f"r{r}_p1_input_ids"] = res1["input_ids"]
        activ[f"r{r}_p2_input_ids"] = res2["input_ids"]
        activ[f"r{r}_p1_offsets"] = res1["offsets"]
        activ[f"r{r}_p2_offsets"] = res2["offsets"]
        activ[f"r{r}_p1_tokens"] = res1["tokens"]
        activ[f"r{r}_p2_tokens"] = res2["tokens"]
        activ[f"r{r}_p1_text"] = res1["text"]
        activ[f"r{r}_p2_text"] = res2["text"]
        activ[f"r{r}_p1_prompt_len"] = np.array([res1["prompt_len"]], dtype=np.int32)
        activ[f"r{r}_p1_prefix_len"] = np.array([res1["prefix_len"]], dtype=np.int32)
        activ[f"r{r}_p2_prompt_len"] = np.array([res2["prompt_len"]], dtype=np.int32)
        activ[f"r{r}_p2_prefix_len"] = np.array([res2["prefix_len"]], dtype=np.int32)

        # per-layer move probs (in label space — analysis maps via stored label_map)
        for l, mv_map in res1["per_layer_probs"].items():
            for mv, p in mv_map.items():
                probs_npz[f"r{r}_p1_l{l}_{mv}"] = np.array([p], dtype=np.float32)
        for l, mv_map in res2["per_layer_probs"].items():
            for mv, p in mv_map.items():
                probs_npz[f"r{r}_p2_l{l}_{mv}"] = np.array([p], dtype=np.float32)

        # --- Map probe probs to action space ---
        prob_act0_p1 = res1["final_slot_probs"][atl_p1[0]]  # prob of action 0's label
        prob_act1_p1 = res1["final_slot_probs"][atl_p1[1]]
        prob_act0_p2 = res2["final_slot_probs"][atl_p2[0]]
        prob_act1_p2 = res2["final_slot_probs"][atl_p2[1]]

        rows.append(dict(
            pair_id=pair_id,
            game_code=game_code,
            round=r + 1,
            # Action-space moves (canonical)
            move1=act1, move2=act2,
            move1_label=move1_label, move2_label=move2_label,
            freq_act0_p1=freq_act0_p1, freq_act0_p2=freq_act0_p2,
            # Label mapping for this round (which label is action 0)
            label_map_p1=atl_p1[0],  # e.g. "A" means A→action0
            label_map_p2=atl_p2[0],
            prompt1_seen=res1["prompt_used"],
            prompt2_seen=res2["prompt_used"],
            payoff1=payoff1, payoff2=payoff2,
            # Raw label-space probs
            prob_A_p1=res1["final_slot_probs"]["A"],
            prob_B_p1=res1["final_slot_probs"]["B"],
            prob_A_p2=res2["final_slot_probs"]["A"],
            prob_B_p2=res2["final_slot_probs"]["B"],
            # Action-space probs (mapped)
            prob_act0_p1=prob_act0_p1, prob_act1_p1=prob_act1_p1,
            prob_act0_p2=prob_act0_p2, prob_act1_p2=prob_act1_p2,
            slot_argmax_p1=0 if prob_act0_p1 >= prob_act1_p1 else 1,
            slot_argmax_p2=0 if prob_act0_p2 >= prob_act1_p2 else 1,
            first_token_id_p1=res1["first_token_id"],
            first_token_id_p2=res2["first_token_id"],
            base_game=base_vec,
            game_matrix=game_vec,
            model_p1_name=model_name_p1,
            model_p2_name=model_name_p2,
            cross_model=cross_model,
            trait_p1_id=trait_p1.id,
            trait_p1_name=trait_p1.trait,
            trait_p1_intensity=trait_p1.intensity,
            trait_p2_id=trait_p2.id,
            trait_p2_name=trait_p2.trait,
            trait_p2_intensity=trait_p2.intensity,
            # Matrix display order: which action index appears first in rows/cols
            row_first_p1=1 if swap_rows_p1 else 0,
            col_first_p1=1 if swap_cols_p1 else 0,
            row_first_p2=1 if swap_rows_p2 else 0,
            col_first_p2=1 if swap_cols_p2 else 0,
            prompt_format="counterbalanced_AB_matrix_rand",
            generated_text_p1=res1.get("generated_text", ""),
            generated_text_p2=res2.get("generated_text", ""),
            gen_parse_ok_p1=res1.get("gen_parse_ok", True),
            gen_parse_ok_p2=res2.get("gen_parse_ok", True),
            decision_mode=args.decision_mode,
        ))

    # --- Cleanup: free the last model ---
    agent1 = agent2 = None
    current_model = None
    _free_gpu_cache()

    return pd.DataFrame(rows), activ, probs_npz, attn_npz

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Experiment v2: counterbalanced A/B prompts + trait injection")
    p.add_argument("--game", required=True, help="Game code (e.g. PdPd, ChCh, ShSh)")
    p.add_argument("--custom_games_json", default="",
                   help="Optional JSON with custom games; entries need game_code and 8-entry vector")
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--history", action="store_true")
    p.add_argument("--max_history", type=int, default=0)
    p.add_argument("--layer", default="all",
                   help='Hidden-state layers to save: "none", "all", one int, or comma-separated ints')
    p.add_argument("--model_name", default="Qwen/Qwen2.5-32B",
                    help="Model for P1 (and P2 if --model_p2 not set)")
    p.add_argument("--model_p2", default="",
                    help="Model for P2. Empty or omitted = same as --model_name")
    p.add_argument("--output_dir", default="output")
    p.add_argument("--use_ln_f_all", action="store_true")
    p.add_argument("--probe_prefix", default='\nDecision: ')
    p.add_argument("--attn_layers", default="none",
                    help='Which layers save attention: "none", "all", "every:N", or comma-separated ints')
    p.add_argument("--attn_topk", type=int, default=0)
    p.add_argument("--attn_dtype", choices=["float32", "float16"], default="float16")
    p.add_argument("--save_token_layers", action="store_true")
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--load_8bit", action="store_true")
    p.add_argument("--bnb_compute_dtype", choices=["float16","bfloat16","float32"], default="bfloat16")
    p.add_argument("--bnb_8bit_cpu_offload", action="store_true",
                   help="Allow bnb int8 auto device maps to keep spilled modules on CPU in fp32")
    p.add_argument("--hf_device_map", choices=["auto", "cuda0", "single_gpu"], default="cuda0",
                   help="Device map for non-GPT-OSS HF models; default preserves the historical Spark path")
    p.add_argument("--hf_max_memory", default="",
                   help='Optional Accelerate max_memory for non-GPT-OSS auto maps, e.g. "0=70GiB,cpu=45GiB"')
    p.add_argument("--hf_offload_folder", default="",
                   help="Optional Accelerate disk offload folder for auto device maps")
    p.add_argument("--hf_offload_buffers", action="store_true",
                   help="Offload buffers with Accelerate disk offload")
    p.add_argument("--disable_cuda_cache_during_load", action="store_true",
                   help="Disable PyTorch CUDA caching allocator while from_pretrained runs")
    p.add_argument("--attn_impl", choices=["auto","eager","sdpa"], default="sdpa")
    p.add_argument("--no_shuffle_valid_moves", action="store_true", default=False,
                   help="Disable shuffling of valid moves order in prompts (default: shuffle is ON)")
    p.add_argument("--decision_mode", choices=["argmax", "sample_probe", "generate"],
                    default="sample_probe",
                    help="How to pick the behavioral move: argmax (legacy), sample_probe, or generate")
    p.add_argument("--generate_temperature", type=float, default=1.0,
                    help="Sampling temperature for generate mode")
    p.add_argument("--generate_max_tokens", type=int, default=48,
                    help="Max new tokens for generate mode")
    p.add_argument("--run_tag", default="")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed (Python, NumPy, PyTorch)")

    # Experiment design flags (non-breaking: defaults preserve existing behavior)
    p.add_argument("--balanced_labels", action="store_true",
                   help="Use deterministic balanced A/B label schedule instead of random")
    p.add_argument("--balanced_display", action="store_true",
                   help="Use deterministic balanced row/col display order (4 independent schedules)")
    p.add_argument("--independent_rounds", action="store_true",
                   help="Reset all sequence state each round (true one-shot: no history, cumulative=0)")
    p.add_argument("--payoff_multiplier", type=int, default=2,
                   help="Payoff multiplier applied to base game vector (default: 2)")

    # Trait system — each player gets their own trait
    # NOTE (phase-4): this defaulted to `<package>/traits.json`, a file that has not existed
    # for some time — any run relying on the default crashed inside load_traits(). It is now
    # required. Repointing at the packaged traits_oneshot.json would turn that crash into a
    # successful run with a DIFFERENT trait set, which is a behaviour change, so it is not done.
    p.add_argument("--traits_file", default="",
                   help="Path to traits JSON (REQUIRED; the packaged set is "
                        "strategic_anatomy/traits_oneshot.json)")
    p.add_argument("--trait_p1", default="none", help="Trait ID for player 1, or 'none'")
    p.add_argument("--trait_p2", default="none", help="Trait ID for player 2, or 'none'")
    return p.parse_args()

# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    _disable_hf_allocator_warmup()
    _install_spark_cpu_first_bnb8_patch()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # --- load traits ---
    if not args.traits_file:
        sys.exit(
            "--traits_file required — there is no default. The historical default named a "
            "`traits.json` that no longer exists, so a bare run crashed inside load_traits(). "
            "The packaged trait set is strategic_anatomy/traits_oneshot.json."
        )
    traits = load_traits(args.traits_file)
    for tid in [args.trait_p1, args.trait_p2]:
        if tid not in traits:
            sys.exit(f"Unknown trait '{tid}'. Available: {list(traits.keys())}")

    trait_p1 = traits[args.trait_p1]
    trait_p2 = traits[args.trait_p2]

    # --- cross-model detection ---
    cross_model = bool(args.model_p2) and args.model_p2 != args.model_name

    # --- output dirs ---
    out_root = pathlib.Path(args.output_dir)
    model_short_p1 = args.model_name.split("/")[-1].lower().replace("-", "_")
    if cross_model:
        model_short_p2 = args.model_p2.split("/")[-1].lower().replace("-", "_")
        model_tag = f"{model_short_p1}_x_{model_short_p2}"
    else:
        model_tag = model_short_p1
    ts = args.run_tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_tag = f"_s{args.seed}" if args.seed != 42 else ""
    run_id = f"{model_tag}_{args.game}_p1={args.trait_p1}_p2={args.trait_p2}_{ts}{seed_tag}"

    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    activ_dir = out_root / "activations" / run_id
    activ_dir.mkdir(parents=True, exist_ok=True)

    # --- load tokenizers ---
    tok_p1 = _load_tokenizer(args.model_name)
    tok_p2 = _load_tokenizer(args.model_p2) if cross_model else tok_p1

    # --- verify A and B are single tokens in all tokenizers ---
    # Use the same LCP logic as build_move_token_map: BPE merging means
    # len(tok(prefix + label)) may equal len(tok(prefix)) when the trailing
    # space absorbs into the label token.  The correct check is that exactly
    # one token diverges at the LCP boundary.
    for label in MOVE_LABELS:
        for tok_name, tok_obj in [("P1", tok_p1), ("P2", tok_p2)]:
            ids_with = tok_obj(args.probe_prefix + label, add_special_tokens=False).input_ids
            ids_without = tok_obj(args.probe_prefix, add_special_tokens=False).input_ids
            lcp = 0
            max_lcp = min(len(ids_with), len(ids_without))
            while lcp < max_lcp and ids_with[lcp] == ids_without[lcp]:
                lcp += 1
            n_new = len(ids_with) - lcp
            assert n_new == 1, (
                f"Label '{label}' is not a single token after probe prefix "
                f"in {tok_name} tokenizer: got {n_new} new tokens after LCP={lcp}. "
                f"ids_with={ids_with}, ids_without={ids_without}"
            )
        print(f"  [ok] Label '{label}' is a single token in all tokenizers")

    # --- build quantization config ---
    qcfg = _build_quant_config(args)

    # --- validate game ---
    game_code = args.game
    available_games = dict(bruns_games)
    if getattr(args, "custom_games_json", ""):
        available_games.update(load_custom_games_json(args.custom_games_json))
    if game_code not in available_games:
        sys.exit(f"Unknown game '{game_code}'. Available: {list(available_games.keys())[:20]}...")
    base_vec, *_ = available_games[game_code]

    # --- save config ---
    multiplier = getattr(args, 'payoff_multiplier', PAYOFF_MULTIPLIER)
    config = {
        **vars(args),
        "trait_p1_obj": trait_p1.__dict__,
        "trait_p2_obj": trait_p2.__dict__,
        "base_vec": base_vec,
        "game_vec": [v * multiplier for v in base_vec],
        "payoff_multiplier": multiplier,
        "cross_model": cross_model,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    # --- run ---
    label = f"{args.model_name}" + (f" × {args.model_p2}" if cross_model else "")
    print(f"Running: {game_code} | P1={args.trait_p1} vs P2={args.trait_p2} | {label}")

    df, activ, probs_npz, attn_npz = run_one_match(
        game_code, base_vec, trait_p1, trait_p2, args,
        tok_p1, tok_p2, qcfg,
    )

    # --- save ---
    pair_dir = activ_dir / df.pair_id.iloc[0]
    pair_dir.mkdir(parents=True, exist_ok=True)

    if attn_npz:
        np.savez_compressed(pair_dir / "attn.npz", **attn_npz)
    np.savez_compressed(pair_dir / "acts.npz", **activ)
    np.savez_compressed(pair_dir / "moveprobs.npz", **probs_npz)

    out_file = run_dir / f"results.parquet"
    df.to_parquet(out_file, index=False)

    # quick summary (in action space)
    valid_p1 = df["move1"].notna()
    valid_p2 = df["move2"].notna()
    p1_act0 = (df.loc[valid_p1, "move1"] == 0).mean() if valid_p1.any() else float("nan")
    p2_act0 = (df.loc[valid_p2, "move2"] == 0).mean() if valid_p2.any() else float("nan")
    parse_ok_p1 = df["gen_parse_ok_p1"].mean() if "gen_parse_ok_p1" in df else 1.0
    parse_ok_p2 = df["gen_parse_ok_p2"].mean() if "gen_parse_ok_p2" in df else 1.0
    print(f"  P1(act0)={p1_act0:.1%}  P2(act0)={p2_act0:.1%}  "
          f"parse_p1={parse_ok_p1:.0%}  parse_p2={parse_ok_p2:.0%}  [{args.decision_mode}]")
    print(f"  Saved: {out_file}")
    print(f"  Activations: {pair_dir}")


if __name__ == "__main__":
    main()
