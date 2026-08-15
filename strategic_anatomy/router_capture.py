#!/usr/bin/env python3
"""Capture gpt-oss MoE router signals by deterministic replay of saved decision inputs.

For each captured DESIGN_V2 gpt-oss match this re-feeds the decision-slot
``input_ids`` already stored in ``acts.npz`` back through the model and records, at
each target MoE layer: the router **gate logits** (pre-softmax), the **top-k selected
expert indices**, and the **top-k routing weights** — at the decision slot (all
target layers) and across sequence positions (deepest layers, a few rounds). Output
is written to ``router.npz`` alongside the existing ``acts.npz``.

This is a pure ``acts.npz -> router.npz`` transform. The saved ``input_ids`` ARE the
exact wrapped/truncated decision input that produced the captured residual
(``acts.npz[r{t}_p{P}_l{L}] == hidden_states[L+1][0, -1, :]`` of a
``model(decision_input, output_hidden_states=True)`` forward; see
``src/run_akata_replication.py:_capture_decision_activations``). So replay is
deterministic and needs no prompt construction, counterbalancing, or RNG — reuse is
by *artifact*, not by re-running generation. ``--self-check`` re-derives the saved
residual and asserts it matches, which proves the captured router signals correspond
to the same forward computation.

Router weights convention: gpt-oss routes with topk-then-softmax
(``GptOssTopKRouter``: ``topk(logits, k)`` then ``softmax`` over the top-k values).
We capture the gate logits and **derive** ``topk_idx`` / ``weights`` from them under
that convention, which is robust to the exact router-module return signature
(mxfp4 kernel path vs. native path). The chosen ``capture_mode`` is stamped into the
npz.

CLAUDE.md #7: run with ``.venv_gptoss``. The generic ``.venv`` lacks the ``kernels``
package and MXFP4-fallbacks, which changes the router code path; this script
hard-errors if ``kernels`` is missing (override with ``--allow-no-kernels``).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

from strategic_anatomy.runtime import (  # noqa: E402
    _build_quant_config,
    _disable_hf_allocator_warmup,
    _free_gpu_cache,
    _install_spark_cpu_first_bnb8_patch,
    _is_gpt_oss_model,
    _load_model,
    _load_tokenizer,
    resolve_model_input_device,
)
from strategic_anatomy.steering_utils import transformer_layers  # noqa: E402

LOG = logging.getLogger("run_router_capture")

ROUTER_SCHEMA_VERSION = "router_v1"
EXPECTED_PROMPT_TEMPLATE_VERSION = "design_v2_v1"
DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
DEFAULT_DATA_ROOT = Path("output/design_v2_main")
DEFAULT_GAMES_CSV = Path("scripts/experiment1/phase2_game_list.csv")
DEFAULT_SEEDS = (100, 200, 300)
DEFAULT_LAYERS = (1, 6, 9, 22, 24, 35)
DEFAULT_SEQ_LAYERS = (24, 35)
DEFAULT_SEQUENCE_ROUNDS = (1, 5, 10)
N_ROUNDS = 10
PLAYERS = (1, 2)

# 19 DESIGN_V2 cells (baseline + 9 cues x {typeA, typeB}); mirrors
# scripts/experiment1/run_design_v2_phase2.sh ALL_CELLS.
_CUES = (
    "level0_naive", "level1", "level2", "maximin", "selfish_maximizer",
    "risk_aversion", "inequity_aversion", "loss_aversion", "length_match_null",
)
DEFAULT_CELLS = ("baseline",) + tuple(
    f"{cue}_{ab}" for cue in _CUES for ab in ("typeA", "typeB")
)


# ──────────────────────────────────────────────────────────────────────
# Environment / model loading
# ──────────────────────────────────────────────────────────────────────

def _kernels_available() -> bool:
    try:
        import kernels  # noqa: F401
        return True
    except Exception:
        return False


def _make_load_args(model_name: str, gptoss_device_map: str = "cuda0") -> argparse.Namespace:
    """Minimal Namespace satisfying run_sim_spark._build_quant_config / _load_model."""
    return argparse.Namespace(
        model_name=model_name,
        model_p2="",
        load_4bit=False,
        load_8bit=False,
        bnb_compute_dtype="bfloat16",
        attn_impl="sdpa",
        attn_layers="none",
        gptoss_device_map=gptoss_device_map,
    )


def load_gptoss(model_name: str, gptoss_device_map: str = "cuda0"):
    args = _make_load_args(model_name, gptoss_device_map)
    _disable_hf_allocator_warmup()
    _install_spark_cpu_first_bnb8_patch()
    tokenizer = _load_tokenizer(model_name)
    qcfg = _build_quant_config(args)  # None for gpt-oss
    t0 = time.time()
    model = _load_model(model_name, args, qcfg)
    LOG.info("model loaded in %.1fs", time.time() - t0)
    return model, tokenizer


def model_topk(model, override: int | None) -> int:
    if override is not None and override > 0:
        return int(override)
    cfg = model.config
    for attr in ("num_experts_per_tok", "experts_per_token", "top_k"):
        v = getattr(cfg, attr, None)
        if v:
            return int(v)
    return 4


# ──────────────────────────────────────────────────────────────────────
# Router capture
# ──────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def capture_moe_router(model, layers: Iterable[int]):
    """Capture full-sequence router **gate logits** ``(T, E)`` per target layer.

    Re-implements the hook structure of ``_legacy/src/run_gptoss_moe_router_diagnostic``
    (read-only history per CLAUDE.md #9 — pattern-matched, not imported), with two
    differences: it captures the FULL sequence tensor (not just the last token) and
    stores only the gate logits (top-k / weights are derived downstream under the
    documented topk-then-softmax convention).

    The mxfp4 path returns ``(routed_out, router_logits)`` from ``block.mlp``; the
    native path exposes full logits as ``output[0]`` of ``block.mlp.router``. Both
    hooks are registered; whichever yields a ``(*, num_experts)`` tensor wins.
    """
    blocks = transformer_layers(model)
    capture: dict[int, np.ndarray] = {}
    handles = []

    def _as_seq_TE(t: torch.Tensor) -> np.ndarray:
        a = t.detach()
        if a.ndim == 3:        # (B, T, E) -> (T, E)
            a = a[0]
        elif a.ndim != 2:      # expect (T, E)
            return None  # type: ignore[return-value]
        return a.to(torch.float32).cpu().numpy()

    for layer in layers:
        layer = int(layer)
        if not (0 <= layer < len(blocks)):
            raise ValueError(f"layer {layer} out of range 0..{len(blocks) - 1}")
        block = blocks[layer]
        router = getattr(block.mlp, "router", None)
        num_experts = int(
            getattr(router, "num_experts", 0)
            or getattr(model.config, "num_local_experts", 0)
            or getattr(model.config, "num_experts", 0)
        )

        def make_mlp_hook(idx: int, n_exp: int):
            def hook(_module, _inputs, output):
                if idx in capture:  # router hook (nested, fires first) already gave raw logits
                    return
                if not (isinstance(output, tuple) and len(output) >= 2):
                    return
                raw = output[1]
                if not torch.is_tensor(raw):
                    return
                arr = _as_seq_TE(raw)
                if arr is not None and (n_exp == 0 or arr.shape[-1] == n_exp):
                    capture[idx] = arr  # prefer full logits
            return hook

        handles.append(block.mlp.register_forward_hook(make_mlp_hook(layer, num_experts)))

        if router is not None:
            def make_router_hook(idx: int, n_exp: int):
                def hook(_module, _inputs, output):
                    if idx in capture:  # mlp hook already gave full logits
                        return
                    cand = output[0] if isinstance(output, tuple) and output else output
                    if not torch.is_tensor(cand):
                        return
                    arr = _as_seq_TE(cand)
                    if arr is not None and (n_exp == 0 or arr.shape[-1] == n_exp):
                        capture[idx] = arr
                return hook

            handles.append(router.register_forward_hook(make_router_hook(layer, num_experts)))

    try:
        yield capture
    finally:
        for h in handles:
            h.remove()


def detect_router_api(model, layers, input_ids, input_device) -> str:
    """Decide capture mode once. Returns "native" if ``output_router_logits=True``
    populates ``out.router_logits`` with usable per-layer tensors AND hooks also fire;
    otherwise "hooks". We capture via hooks regardless (layer-explicit); ``native`` is
    recorded as available for provenance / cross-check."""
    ids = input_ids.to(input_device).unsqueeze(0)
    native_ok = False
    try:
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=False, output_router_logits=True)
        rl = getattr(out, "router_logits", None)
        native_ok = isinstance(rl, (tuple, list)) and len(rl) > max(layers) and torch.is_tensor(rl[max(layers)])
    except TypeError:
        native_ok = False
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("native router-logit probe failed: %s", exc)
        native_ok = False

    with torch.inference_mode():
        with capture_moe_router(model, layers) as cap:
            model(input_ids=ids, use_cache=False)
    hooks_ok = all(int(l) in cap for l in layers)
    if not hooks_ok:
        missing = [int(l) for l in layers if int(l) not in cap]
        raise RuntimeError(
            f"router hooks captured nothing at layers {missing}; "
            "wrong env (need .venv_gptoss/kernels) or unexpected MoE layout"
        )
    mode = "native+hooks" if native_ok else "hooks"
    LOG.info("capture_mode=%s (native_router_logits=%s)", mode, native_ok)
    return mode


def _topk_idx_weights(gate_logits: np.ndarray, top_k: int):
    """gpt-oss convention: topk over experts, softmax over the top-k logit values.

    ``gate_logits`` is (T, E). Returns (topk_idx (T, k) int32, weights (T, k) float32).
    """
    g = torch.from_numpy(np.asarray(gate_logits, dtype=np.float32))
    vals, idx = torch.topk(g, k=int(top_k), dim=-1)
    w = torch.softmax(vals, dim=-1)
    return idx.to(torch.int32).numpy(), w.to(torch.float32).numpy()


def router_signals_for_input_ids(model, input_ids, layers, input_device):
    """Return {layer: gate_logits (T, E) float32} for one decision input."""
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=input_device).unsqueeze(0)
    with torch.inference_mode():
        with capture_moe_router(model, layers) as cap:
            model(input_ids=ids, use_cache=False)
    return {int(l): cap[int(l)] for l in layers}


# ──────────────────────────────────────────────────────────────────────
# Per-match processing
# ──────────────────────────────────────────────────────────────────────

def _read_match_config(match_dir: Path) -> dict:
    cfg_path = match_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def process_match_dir(
    model, match_dir: Path, layers, seq_layers, sequence_rounds, top_k,
    capture_mode: str, input_device, skip_existing: bool = True,
) -> str:
    acts_path = match_dir / "acts.npz"
    if not acts_path.exists():
        return "no_acts"
    router_path = match_dir / "router.npz"
    if skip_existing and router_path.exists():
        return "skipped"

    cfg = _read_match_config(match_dir)
    ptv = str(cfg.get("prompt_template_version", "unknown"))
    if ptv != EXPECTED_PROMPT_TEMPLATE_VERSION:
        LOG.warning("%s: prompt_template_version=%s (expected %s)", match_dir.name, ptv,
                    EXPECTED_PROMPT_TEMPLATE_VERSION)

    seq_layers = {int(x) for x in seq_layers}
    sequence_rounds = {int(x) for x in sequence_rounds}

    out: dict[str, np.ndarray] = {}
    num_experts = None
    with np.load(acts_path, allow_pickle=True) as z:
        keys = set(z.files)
        for r in range(1, N_ROUNDS + 1):
            for p in PLAYERS:
                ids_key = f"r{r}_p{p}_input_ids"
                if ids_key not in keys:
                    continue
                input_ids = np.asarray(z[ids_key]).astype(np.int64)
                sig = router_signals_for_input_ids(model, input_ids, layers, input_device)
                for L in layers:
                    L = int(L)
                    gl = sig[L]  # (T, E) float32
                    if num_experts is None:
                        num_experts = int(gl.shape[-1])
                    idx, w = _topk_idx_weights(gl, top_k)  # (T,k),(T,k)
                    base = f"r{r}_p{p}_l{L}"
                    # decision slot = last position
                    out[f"{base}_gate_logits"] = gl[-1, :].astype(np.float32)
                    out[f"{base}_topk_idx"] = idx[-1, :].astype(np.int32)
                    out[f"{base}_weights"] = w[-1, :].astype(np.float32)
                    if L in seq_layers and r in sequence_rounds:
                        out[f"{base}_gate_logits_seq"] = gl.astype(np.float16)
                        out[f"{base}_topk_idx_seq"] = idx.astype(np.int32)
                        out[f"{base}_weights_seq"] = w.astype(np.float16)

    if not out:
        return "empty"

    # Metadata (numpy-serializable scalars/strings).
    out["schema_version"] = np.asarray(ROUTER_SCHEMA_VERSION, dtype="U")
    out["capture_mode"] = np.asarray(capture_mode, dtype="U")
    out["prompt_template_version"] = np.asarray(ptv, dtype="U")
    out["acts_schema_version"] = np.asarray(str(cfg.get("acts_schema_version", "unknown")), dtype="U")
    out["num_experts"] = np.asarray([int(num_experts or 0)], dtype=np.int32)
    out["top_k"] = np.asarray([int(top_k)], dtype=np.int32)
    out["layers"] = np.asarray([int(x) for x in layers], dtype=np.int32)
    out["seq_layers"] = np.asarray(sorted(seq_layers), dtype=np.int32)
    out["sequence_rounds"] = np.asarray(sorted(sequence_rounds), dtype=np.int32)
    np.savez_compressed(router_path, **out)
    return "ok"


def iter_target_match_dirs(
    data_root: Path, model_kind: str, games, seeds, cells,
) -> Iterator[Path]:
    base = Path(data_root) / model_kind
    for game in games:
        for seed in seeds:
            for cell in cells:
                d = base / f"{game}_s{seed}_{cell}"
                if d.is_dir():
                    yield d


def load_games_csv(path: Path) -> list[str]:
    import csv
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        col = "game_code" if reader.fieldnames and "game_code" in reader.fieldnames else (reader.fieldnames or [None])[0]
        return [row[col] for row in reader if row.get(col)]


# ──────────────────────────────────────────────────────────────────────
# Self-check
# ──────────────────────────────────────────────────────────────────────

def self_check(model, match_dir: Path, layers, input_device) -> None:
    """Assert that replaying saved input_ids reproduces the saved decision-slot
    residual (cosine > 0.999), proving router signals match the captured forward."""
    acts_path = match_dir / "acts.npz"
    with np.load(acts_path, allow_pickle=True) as z:
        keys = set(z.files)
        # pick a layer that has a stored decision slot
        L = next(int(l) for l in layers if f"r1_p1_l{int(l)}" in keys)
        saved = np.asarray(z[f"r1_p1_l{L}"], dtype=np.float32)
        input_ids = np.asarray(z["r1_p1_input_ids"]).astype(np.int64)
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=input_device).unsqueeze(0)
    with torch.inference_mode():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    repl = out.hidden_states[L + 1][0, -1, :].to(torch.float32).cpu().numpy()
    cos = float(np.dot(saved, repl) / (np.linalg.norm(saved) * np.linalg.norm(repl) + 1e-12))
    max_abs = float(np.max(np.abs(saved - repl)))
    LOG.info("self-check %s L%d: cosine=%.6f max_abs_diff=%.4g (T=%d)",
             match_dir.name, L, cos, max_abs, input_ids.shape[0])
    if cos < 0.999:
        raise AssertionError(f"replay mismatch: cosine={cos:.6f} < 0.999 at L{L}")
    LOG.info("self-check PASS")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--model_kind", default="gptoss")
    p.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--games-csv", default=str(DEFAULT_GAMES_CSV))
    p.add_argument("--games", nargs="*", default=None, help="explicit game codes (override CSV)")
    p.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--cells", nargs="*", default=list(DEFAULT_CELLS))
    p.add_argument("--layers", nargs="*", type=int, default=list(DEFAULT_LAYERS))
    p.add_argument("--seq-layers", nargs="*", type=int, default=list(DEFAULT_SEQ_LAYERS))
    p.add_argument("--sequence-rounds", nargs="*", type=int, default=list(DEFAULT_SEQUENCE_ROUNDS))
    p.add_argument("--top-k", type=int, default=None, help="default: model config num_experts_per_tok")
    p.add_argument("--gptoss_device_map", choices=["auto", "cuda0"], default="cuda0")
    p.add_argument("--skip_existing", dest="skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument("--limit", type=int, default=0, help="process at most N match dirs (0 = all)")
    p.add_argument("--self-check", action="store_true", help="run replay assertion on first match and exit")
    p.add_argument("--allow-no-kernels", action="store_true", help="bypass the .venv_gptoss kernels guard")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if _is_gpt_oss_model(args.model_name) and not _kernels_available():
        msg = ("`kernels` package not importable — running under generic .venv would "
               "MXFP4-fallback and change the router path. Use .venv_gptoss.")
        if args.allow_no_kernels:
            LOG.warning("%s (continuing due to --allow-no-kernels)", msg)
        else:
            LOG.error(msg)
            return 2

    games = args.games if args.games else load_games_csv(Path(args.games_csv))
    match_dirs = list(iter_target_match_dirs(Path(args.data_root), args.model_kind, games, args.seeds, args.cells))
    if args.limit:
        match_dirs = match_dirs[: args.limit]
    LOG.info("target matches: %d (games=%d seeds=%d cells=%d)",
             len(match_dirs), len(games), len(args.seeds), len(args.cells))
    if args.dry_run:
        for d in match_dirs[:10]:
            LOG.info("  %s", d)
        return 0
    if not match_dirs:
        LOG.error("no target match dirs found under %s/%s", args.data_root, args.model_kind)
        return 1

    model, _tok = load_gptoss(args.model_name, args.gptoss_device_map)
    input_device = resolve_model_input_device(model)
    top_k = model_topk(model, args.top_k)
    LOG.info("top_k=%d", top_k)

    try:
        # determine capture_mode + run self-check on the first usable match
        probe_dir = next((d for d in match_dirs if (d / "acts.npz").exists()), None)
        if probe_dir is None:
            LOG.error("no acts.npz found in any target dir")
            return 1
        with np.load(probe_dir / "acts.npz", allow_pickle=True) as z:
            probe_ids = torch.as_tensor(np.asarray(z["r1_p1_input_ids"]).astype(np.int64), dtype=torch.long)
        capture_mode = detect_router_api(model, args.layers, probe_ids, input_device)
        self_check(model, probe_dir, args.layers, input_device)
        if args.self_check:
            LOG.info("self-check mode: capture_mode=%s; exiting before capture", capture_mode)
            return 0

        counts = {"ok": 0, "skipped": 0, "no_acts": 0, "empty": 0}
        t0 = time.time()
        for i, d in enumerate(match_dirs, 1):
            status = process_match_dir(
                model, d, args.layers, args.seq_layers, args.sequence_rounds,
                top_k, capture_mode, input_device, skip_existing=args.skip_existing,
            )
            counts[status] = counts.get(status, 0) + 1
            if i % 50 == 0 or i == len(match_dirs):
                LOG.info("[%d/%d] %s | counts=%s | %.0fs", i, len(match_dirs), d.name, counts, time.time() - t0)
        LOG.info("done at %s; counts=%s", dt.datetime.now().isoformat(timespec="seconds"), counts)
        return 0
    finally:
        del model
        _free_gpu_cache()


if __name__ == "__main__":
    raise SystemExit(main())
