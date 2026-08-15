"""Unit tests for the Layer C one-shot region map — the single highest-risk piece.

The headline test asserts the own/opponent payoff split is correct under ALL 16
counterbalance cells (row/col swaps + label map), independent of any tokenizer: it
compares the numbers tagged own/opponent against an independent reconstruction of what
``payoff_rules_symmetric`` renders. A second (tokenizer-level) test confirms that real
tokens overlapping each number/letter char-span get the right region.
"""
import pandas as pd
import pytest

from strategic_anatomy.config import game_features_csv
from collection.oneshot_common import (
    build_oneshot_prompt, counterbalance_grid, load_game_vec, PROBE_PREFIX,
)
from collection.layerc.region_map_oneshot import (
    number_and_letter_spans, build_region_map,
)

_GF = pd.read_csv(game_features_csv())
CANON = _GF.set_index("game_code")["canonical_action_p1"].astype(int).to_dict()
GAMES = ["AsAs", "PdBa", "NcCm", "DlHa", "ChCh"]


def _expected_own_opp(vec, cb):
    """Independent reconstruction of payoff_rules_symmetric's P1 display order (own, opp)."""
    a00, a01, a10, a11, b00, b01, b10, b11 = [int(x) for x in vec]
    c00, c01, c10, c11 = (a00, b00), (a01, b01), (a10, b10), (a11, b11)
    if cb["row_swap"]:
        c00, c01, c10, c11 = c10, c11, c00, c01
    if cb["col_swap"]:
        c00, c01, c10, c11 = c01, c00, c11, c10
    disp = [c00, c01, c10, c11]
    return [c[0] for c in disp], [c[1] for c in disp]


@pytest.mark.parametrize("game", GAMES)
def test_own_opponent_spans_swap_invariant(game):
    vec = [int(x) for x in load_game_vec(game)]
    canon = int(CANON[game])
    for cb in counterbalance_grid():
        prompt = build_oneshot_prompt(vec, cb, player=1, iv_prefix="")
        text = prompt + PROBE_PREFIX
        own_s, opp_s, lab_s, canon_span, _sec = number_and_letter_spans(text, prompt, cb, canon)
        own = [int(text[s:e]) for s, e in own_s]
        opp = [int(text[s:e]) for s, e in opp_s]
        exp_own, exp_opp = _expected_own_opp(vec, cb)
        cbid = cb["counterbalance_id"]
        assert own == exp_own, (game, cbid, "own", own, exp_own)
        assert opp == exp_opp, (game, cbid, "opp", opp, exp_opp)
        # exactly 6 option letters (2 Other= + 2 You= + 2 valid-moves), all A/B, both present
        labs = [text[s:e] for s, e in lab_s]
        assert len(labs) == 6 and set(labs) == {"A", "B"}, (game, cbid, labs)
        # canonical row line is the displayed You=<canonical letter> row
        canon_letter = cb["action_to_label"][canon]
        assert canon_span is not None, (game, cbid, "no canon row")
        assert text[canon_span[0]:canon_span[1]].startswith(f"You={canon_letter}"), (game, cbid)


def test_own_opponent_survives_cue_prefix():
    """A non-empty cue prefix shifts every token; own/opponent tagging must still be right."""
    game = "AsAs"
    vec = [int(x) for x in load_game_vec(game)]
    canon = int(CANON[game])
    prefix = "From now on you are extremely risk averse and avoid worst-case outcomes."
    for cb in counterbalance_grid()[:4]:
        prompt = build_oneshot_prompt(vec, cb, player=1, iv_prefix=prefix)
        text = prompt + PROBE_PREFIX
        own_s, opp_s, lab_s, canon_span, sec = number_and_letter_spans(text, prompt, cb, canon)
        own = [int(text[s:e]) for s, e in own_s]
        opp = [int(text[s:e]) for s, e in opp_s]
        exp_own, exp_opp = _expected_own_opp(vec, cb)
        cbid = cb["counterbalance_id"]
        assert own == exp_own and opp == exp_opp, (cbid, own, exp_own)
        assert len(lab_s) == 6
        assert sec["cue_prefix"][1] > sec["cue_prefix"][0], "cue_prefix span should be non-empty"
        assert prompt[:sec["cue_prefix"][1]].startswith(prefix)


def _qwen_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B", use_fast=True, local_files_only=True)
    except Exception:
        return None


def _chat_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct", use_fast=True, local_files_only=True)
        return tok if tok.chat_template else None
    except Exception:
        return None


def test_chat_wrapped_own_opponent_correct():
    """The chat template shifts every token; own/opponent tagging + sections must still hold,
    chat header -> cue_prefix, trailing 'Decision: ' -> answer_prefix."""
    tok = _chat_tokenizer()
    if tok is None:
        pytest.skip("instruct tokenizer with chat_template not available")
    from collection.oneshot_chat import build_oneshot_chat_text, ASSISTANT_DECISION_PREFIX
    from collection.layerc.region_map_oneshot import build_region_map_chat
    game = "AsAs"
    vec = [int(x) for x in load_game_vec(game)]
    canon = int(CANON[game])
    for cb in counterbalance_grid()[:4]:
        text = build_oneshot_chat_text(tok, vec, cb, player=1)
        prompt = text[: -len(ASSISTANT_DECISION_PREFIX)]
        own_s, opp_s, lab_s, canon_span, sec = number_and_letter_spans(text, prompt, cb, canon)
        own = [int(text[s:e]) for s, e in own_s]
        opp = [int(text[s:e]) for s, e in opp_s]
        exp_own, exp_opp = _expected_own_opp(vec, cb)
        assert own == exp_own and opp == exp_opp, (cb["counterbalance_id"], own, exp_own)
        assert len(lab_s) == 6
        assert sec["cue_prefix"][1] > sec["cue_prefix"][0], "chat header should occupy cue_prefix"
        assert sec["answer_prefix"][1] > sec["answer_prefix"][0], "trailing 'Decision: '"
        rm = build_region_map_chat(vec, cb, canonical_action_p1=canon, tokenizer=tok)
        assert any(r["region"] == "answer_prefix" for r in rm)
        assert any(r["region"] == "own_payoff" for r in rm)


def test_token_tagging_consistent_with_spans():
    tok = _qwen_tokenizer()
    if tok is None:
        pytest.skip("qwen fast tokenizer not available (needs HF cache)")
    game = "AsAs"
    vec = [int(x) for x in load_game_vec(game)]
    canon = int(CANON[game])

    def regions_over(rm, s, e):
        return [r["region"] for r in rm
                if r["char_end"] > r["char_start"] and r["char_start"] < e and r["char_end"] > s]

    for cb in counterbalance_grid()[:4]:
        prompt = build_oneshot_prompt(vec, cb, player=1, iv_prefix="")
        text = prompt + PROBE_PREFIX
        own_s, opp_s, lab_s, _c, _sec = number_and_letter_spans(text, prompt, cb, canon)
        rm = build_region_map(vec, cb, canonical_action_p1=canon, tokenizer=tok)
        for s, e in own_s:
            regs = regions_over(rm, s, e)
            assert regs and all(x == "own_payoff" for x in regs), (cb["counterbalance_id"], "own", regs)
        for s, e in opp_s:
            regs = regions_over(rm, s, e)
            assert regs and all(x == "opponent_payoff" for x in regs), (cb["counterbalance_id"], "opp", regs)
        for s, e in lab_s:
            regs = regions_over(rm, s, e)
            assert regs and all(x == "label_token" for x in regs), (cb["counterbalance_id"], "label", regs)
        assert any(r["region"] == "answer_prefix" for r in rm), cb["counterbalance_id"]
        assert any(r["region"] == "own_payoff" for r in rm) and any(r["region"] == "opponent_payoff" for r in rm)
