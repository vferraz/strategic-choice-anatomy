"""Steering hook readout helpers.

``_signed_cos`` is extracted verbatim from
``analysis/block_b/run_causal_protocol_oneshot.py`` (private repo, commit 1f47050, line 129).
That module is not part of the release: it imports ``run_causal_protocol_v2``, which in
turn imports ``src.run_scripted_opponent_batch`` — a module that exists only under
``_legacy/`` and therefore breaks at import in the private repo today.

``make_slot_hook``, ``scaled_vector`` and ``dual_readout`` were appended in phase 2 from
``analysis/block_b/akata_steer_core.py`` (same commit), verbatim: core injection plus the
DUAL READOUT (Option A — inject at the decision slot during prefill, then generate freely
and read the REALIZED J/P decision; the slot pref is kept as a labeled diagnostic).
"""
from __future__ import annotations

import numpy as np
import torch


def _signed_cos(hook_stats: dict, x) -> float:
    """Cosine with the SIGNED requested delta (assert_hook_integrity expects this).
    The hook injects dose*vector, so its cosine-with-vector == sign(dose); multiply by
    sign(dose) to get +1 for a faithful hook at either sign. dose==0 rows are excluded."""
    x = float(x) if x != "" else 0.0
    if x == 0.0:
        return 0.0
    c = float(hook_stats.get("actual_delta_cosine_with_vector", 1.0))
    return (1.0 if x > 0 else -1.0) * c


def make_slot_hook(layer_module, vector_t, *, prefill_only: bool):
    """register_forward_hook that ADDS vector_t to the last-token hidden state.
    prefill_only=True -> fires only on the multi-token prefill pass (Option A: inject once at the slot,
    not on each generated token). prefill_only=False -> every pass (used for the single diagnostic forward)."""
    def hook(mod, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        if prefill_only and hidden.shape[1] <= 1:        # decode step (cached) -> do NOT inject
            return out
        hidden[:, -1, :] = hidden[:, -1, :] + vector_t.to(hidden.dtype)
        return (hidden,) + tuple(out[1:]) if isinstance(out, tuple) else hidden
    return layer_module.register_forward_hook(hook)


def scaled_vector(unit, residual_norm, dose, torch_device):
    """dose * (||h|| * unit) — the injected delta, matched to the captured residual norm (A5 convention)."""
    v = (float(dose) * float(residual_norm)) * np.asarray(unit, dtype=np.float32)
    return torch.tensor(v, device=torch_device)


def dual_readout(model, tok, layers, inject_layer, ids, attn, jp_ids, pad, vector_t, dev,
                 gen_tokens=12):
    """Return (slot_pref_J, realized_letter). slot = injected single forward at the slot; realized =
    generate under the prefill-only injection -> parse the committed J/P."""
    lm = layers[int(inject_layer)]
    # (1) slot diagnostic: inject on the forward, read J/P logits at the slot.
    h = make_slot_hook(lm, vector_t, prefill_only=False)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    h.remove()
    probs = torch.softmax(out.logits[0, -1, :].detach(), dim=-1)
    pj, pp = float(probs[jp_ids["J"]]), float(probs[jp_ids["P"]])
    slot_pref_J = pj / (pj + pp + 1e-9)
    # (2) realized decision: inject at the slot during prefill only, generate, parse.
    h = make_slot_hook(lm, vector_t, prefill_only=True)
    with torch.inference_mode():
        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=gen_tokens,
                             do_sample=False, pad_token_id=pad)
    h.remove()
    from collection.akata_common import parse_akata_move
    realized = parse_akata_move(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True))
    return slot_pref_J, realized
