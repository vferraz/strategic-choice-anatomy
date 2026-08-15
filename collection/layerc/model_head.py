"""Unembedding-head loading and per-token logit-difference scoring for Layer C.

Extracted verbatim from ``analysis/block_c/probe_token_attribution.py`` (private repo,
commit 1f47050): ``LOG`` (L46), ``letter_token_id`` (L130), ``ModelHead`` (L147),
``load_model_head`` (L154), ``free_model_head`` (L212), ``apply_rmsnorm`` (L224) and
``per_token_score_diff_opt0_minus_opt1`` (L232).

The originating module is 1611 lines of A/B-era probe tooling that plan §5 excludes; it
also imports ``analysis.block_a.layer1_lambda_delta1`` and
``analysis.block_c.probe_activation_manifolds``, both likewise excluded. Only these seven
symbols are live dependencies of the two shipped Layer C collectors.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

LOG = logging.getLogger("probe_token_attribution")


class ModelHead:
    lm_head_weight: torch.Tensor  # (vocab, d_model), bf16
    norm_weight: torch.Tensor  # (d_model,), bf16
    rms_norm_eps: float
    device: torch.device


def letter_token_id(tokenizer, letter: str) -> int:
    """Return the single-token ID for ``letter``, preferring the leading-space
    variant since most tokenizers emit letters with a preceding space after a
    newline."""
    for variant in (" " + letter, letter):
        ids = tokenizer(variant, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return ids[0]
    return tokenizer(letter, add_special_tokens=False).input_ids[0]


def load_model_head(model_hf_id: str, device: str = "cuda") -> ModelHead:
    """Load only ``lm_head.weight`` and ``model.norm.weight`` from local cache.

    Fully offline (``local_files_only``): the weights are necessarily cached
    (data collection used them), and gated repos (e.g. Meta-Llama) would 401 on
    an unauthenticated network HEAD otherwise."""
    cfg = AutoConfig.from_pretrained(model_hf_id, local_files_only=True)
    eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
    local = Path(
        snapshot_download(
            model_hf_id,
            allow_patterns=["*.json", "*.safetensors"],
            local_files_only=True,
        )
    )
    index_path = local / "model.safetensors.index.json"
    target_keys = ("lm_head.weight", "model.norm.weight")
    weights: dict[str, torch.Tensor] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
        shards = {weight_map[k] for k in target_keys if k in weight_map}
        for shard in shards:
            sd = load_file(local / shard)
            for k in target_keys:
                if k in sd:
                    weights[k] = sd[k]
    else:
        sd = load_file(local / "model.safetensors")
        for k in target_keys:
            if k in sd:
                weights[k] = sd[k]
    if "lm_head.weight" not in weights:
        # Tied embeddings: lm_head shares model.embed_tokens.weight
        embed_key = "model.embed_tokens.weight"
        for shard_name in weights.get("__shards__", {}):
            sd = load_file(local / shard_name)
            if embed_key in sd:
                weights["lm_head.weight"] = sd[embed_key]
                break
    if "lm_head.weight" not in weights or "model.norm.weight" not in weights:
        raise RuntimeError(
            f"missing head weights for {model_hf_id}: got keys {list(weights)}"
        )
    dev = torch.device(device)
    lm = weights["lm_head.weight"].to(dev, dtype=torch.bfloat16)
    norm = weights["model.norm.weight"].to(dev, dtype=torch.bfloat16)
    LOG.info(
        "loaded head for %s: lm_head=%s norm=%s eps=%s device=%s",
        model_hf_id,
        tuple(lm.shape),
        tuple(norm.shape),
        eps,
        dev,
    )
    return ModelHead(lm_head_weight=lm, norm_weight=norm, rms_norm_eps=eps, device=dev)


def free_model_head(head: ModelHead) -> None:
    del head.lm_head_weight
    del head.norm_weight
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_rmsnorm(h: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * gamma."""
    h_f32 = h.to(torch.float32)
    rms = torch.rsqrt(h_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = h_f32 * rms
    return (out.to(gamma.dtype) * gamma).contiguous()


def per_token_score_diff_opt0_minus_opt1(
    *,
    seq_residual: np.ndarray,
    head: ModelHead,
    opt0_id: int,
    opt1_id: int,
) -> np.ndarray:
    """Project ``(T, D)`` residuals through final_norm + lm_head and return
    ``logits[:, opt0_id] - logits[:, opt1_id]`` as ``(T,) float32``."""
    h = torch.from_numpy(np.asarray(seq_residual)).to(head.device)
    h = h.to(dtype=head.lm_head_weight.dtype)
    h_norm = apply_rmsnorm(h, head.norm_weight, head.rms_norm_eps)
    logits = h_norm @ head.lm_head_weight.T
    diff = (logits[:, opt0_id] - logits[:, opt1_id]).to(torch.float32).cpu().numpy()
    del h, h_norm, logits
    return diff
