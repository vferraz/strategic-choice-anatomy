"""Locked Layer C capture constants and condition ordering.

Extracted verbatim from ``analysis/block_c/generate_oneshot_layerc.py`` (private repo,
commit 8d8370e): ``LAYERS_DEFAULT`` (L62), ``DENSE_MODELS`` (L63), ``CUE_TO_TRAITCOL``
(L65), ``ROW_COLS`` (L70) and ``_conditions`` (L91). The originating module is the
A/B-matrix Layer C generator, excluded by plan §5; the Akata generator reuses only these
locked constants and the condition-ordering helper.
"""
from __future__ import annotations

LAYERS_DEFAULT = [79, 40]                 # locked: deepest (C1) + one mid layer
DENSE_MODELS = ("qwen", "qwen_instruct", "llama31_instruct")

CUE_TO_TRAITCOL = {
    "risk_aversion": "risk", "loss_aversion": "loss", "inequity_aversion": "inequity",
    "maximin": "maximin", "selfish_maximizer": "selfish",
}

ROW_COLS = [
    "model", "game_code", "cb_id", "condition", "layer", "token_index", "token_str",
    "char_start", "char_end", "region", "is_canonical_row", "score_canonical",
    "score_trait_target", "canonical_action_letter", "prompt_hash",
]


def _conditions(cue_objs):
    """(key_prefix, iv_prefix, condition, cue_id) — P1 baseline + one per cue."""
    out = [("p1_baseline", "", "baseline", None)]
    for cid, cobj in cue_objs:
        out.append((f"p1_cue_{cid}", cobj.system_prefix, f"cue_{cid}", cid))
    return out
