"""Model loading and crash-safe persistence primitives shared by the collectors.

Extracted verbatim from ``analysis/block_b/generate_oneshot_substrate.py`` (private repo,
commit 8d8370e): ``_git_commit``, ``_now``, ``_sha256_file``, ``_fsync_dir``,
``resolve_cue_list`` and ``_setup_model``. That module is the superseded A/B-matrix
generator and is not part of the release (plan §4.3: extract the reused helpers, drop the
generator).

``ROUTER_LAYERS_DEFAULT`` is extracted from
``analysis/block_b/generate_oneshot_moe_commit_capture.py:74`` (same commit) for the same
reason — plan §5 excludes the ``generate_oneshot_moe_*`` family, but the three GPT-OSS
collectors need this one constant.

``CONFIG_PATH`` and ``MANIFEST_DIR`` now resolve through
:mod:`strategic_anatomy.config` instead of the private repo's
``analysis/block_b/tables/causal_oneshot/manifests`` layout.

Note ``_setup_model`` imports torch lazily, inside the function — importing this module
does not require the ``[gpu]`` extra.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from strategic_anatomy.config import manifests_root, repo_root
from strategic_anatomy.prompting import MOVE_LABELS, build_move_token_map
from collection.oneshot_common import PROBE_PREFIX

ROOT = repo_root()
MANIFEST_DIR = manifests_root()
CONFIG_PATH = MANIFEST_DIR / "oneshot_config.json"

# Verbatim from analysis/block_b/generate_oneshot_moe_commit_capture.py:74.
ROUTER_LAYERS_DEFAULT = [1, 3, 6, 9, 12, 15, 18, 21, 22, 24, 27, 30, 33, 35]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False, cwd=str(ROOT)).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def resolve_cue_list(config: dict) -> list[str]:
    """Expanded cue list (fixes the silent-placebo bug, §4)."""
    cues = list(config["traits"])
    if config.get("include_placebo"):
        cues.append("length_match_null")
    if config.get("include_levelk"):
        cues += ["level1", "level2"]
    return cues


def _setup_model(args, model_cfg):
    import torch  # noqa
    from strategic_anatomy.runtime import (
        _build_quant_config, _disable_hf_allocator_warmup, _ensure_attention_mask,
        _install_spark_cpu_first_bnb8_patch, _is_gpt_oss_model, _load_model,
        _load_tokenizer, resolve_model_input_device,
    )
    model_name = model_cfg["model_name"]
    args.model_name = model_name
    if model_cfg.get("load_8bit") and not getattr(args, "load_8bit", False):
        args.load_8bit = True
    # Compat attributes read by strategic_anatomy.runtime loaders (defensive across entry points).
    for attr, default in (("model_p2", ""), ("load_4bit", False), ("attn_layers", "none"),
                          ("attn_topk", 0), ("attn_dtype", "float16"), ("save_token_layers", False),
                          ("max_history", 0), ("generate_temperature", 1.0),
                          ("no_shuffle_valid_moves", False)):
        if not hasattr(args, attr):
            setattr(args, attr, default)
    _disable_hf_allocator_warmup()
    _install_spark_cpu_first_bnb8_patch()
    tokenizer = _load_tokenizer(model_name)
    move_token_map = build_move_token_map(tokenizer, PROBE_PREFIX, MOVE_LABELS)
    if _is_gpt_oss_model(model_name) and (getattr(args, "load_4bit", False) or getattr(args, "load_8bit", False)):
        raise SystemExit("GPT-OSS forbids quantization flags.")
    qcfg = _build_quant_config(args)
    t0 = time.time()
    model = _load_model(model_name, args, qcfg)
    load_s = time.time() - t0
    input_device = resolve_model_input_device(model)
    return (torch, model, tokenizer, move_token_map, input_device, _ensure_attention_mask, load_s)
