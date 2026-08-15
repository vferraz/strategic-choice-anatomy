"""Layer C region map (one-shot) — map every token of a rendered one-shot prompt to a
prompt region, counterbalance-aware. P1 only (attribution is P1's decision).

THE HEADLINE-CRITICAL PIECE is the own/opponent payoff split, and it is **swap-invariant**:
``strategic_anatomy.prompting.payoff_rules_symmetric`` always renders each payoff cell as
``(your payoff, other player's payoff)`` — the FIRST number in every ``(x, y)`` pair is the
prompt-player's own payoff, the SECOND is the opponent's. ``swap_rows``/``swap_cols`` only
reorder which cell is displayed where; they never flip the order inside a cell. So:
    own_payoff      = the first number of each (x, y) pair
    opponent_payoff = the second number
No swap reverse-engineering is needed for own-vs-opponent (that was the fragile path).
``is_canonical_row`` *does* use the cb label map (which displayed ``You=<letter>`` row is
the canonical action's row), but that is a simple letter match, not a payoff remap.

Regions: intro, payoff_struct, valid_moves, question, answer_prefix, label_token,
own_payoff, opponent_payoff.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


from collection.oneshot_common import build_oneshot_prompt, PROBE_PREFIX  # noqa: E402

# Literal section anchors (must match oneshot_common / run_sim_spark renderers verbatim).
_INTRO_ANCHOR = "You are about to play one round of a two-player game."
_PAYOFF_ANCHOR = "Game rules (your payoff, other player's payoff):"
_VALID_NOTE_ANCHOR = "Note: The order of 'Valid moves' below is arbitrary"
_CHOOSE_ANCHOR = "Choose your move based on your preferences and the game rules above."

_PAIR_RE = re.compile(r"\((\d+),\s*(\d+)\)")          # (own, opponent)
_HEADER_LETTER_RE = re.compile(r"(?:Other|You)=([AB])")  # the A/B in You=/Other=
_VM_LETTER_RE = re.compile(r"(?<![A-Za-z])([AB])(?![A-Za-z])")  # standalone A/B in valid-moves line
_YOU_ROW_RE = re.compile(r"You=([AB])\b[^\n]*")          # a full You=<letter> row line


def _section_spans(prompt: str, full_len: int) -> dict:
    """Char spans for the five linear sections over [0, full_len) (full_len includes PROBE_PREFIX)."""
    i_intro = prompt.index(_INTRO_ANCHOR)
    i_payoff = prompt.index(_PAYOFF_ANCHOR)
    i_note = prompt.index(_VALID_NOTE_ANCHOR)
    i_choose = prompt.index(_CHOOSE_ANCHOR)
    return {
        "cue_prefix": (0, i_intro),              # iv_prefix tokens (empty span for baseline)
        "intro": (i_intro, i_payoff),
        "payoff_struct": (i_payoff, i_note),     # generic payoff text; numbers/letters override below
        "valid_moves": (i_note, i_choose),
        "question": (i_choose, len(prompt)),
        "answer_prefix": (len(prompt), full_len),  # the appended PROBE_PREFIX
    }


def number_and_letter_spans(text: str, prompt: str, cb: dict, canonical_action_p1: int):
    """Return (own_spans, opp_spans, label_spans, canon_row_span) as char (start,end) tuples.

    own/opp = the two numbers of each (x,y) pair (swap-invariant: first=own, second=opp).
    label   = the A/B letters in You=/Other= headers + the valid-moves option letters.
    canon_row_span = char span of the displayed ``You=<canonical_letter>`` row line.
    """
    sec = _section_spans(prompt, len(text))
    p0, p1 = sec["payoff_struct"]
    payoff_text = text[p0:p1]
    own_spans, opp_spans, label_spans = [], [], []
    for m in _PAIR_RE.finditer(payoff_text):
        own_spans.append((p0 + m.start(1), p0 + m.end(1)))
        opp_spans.append((p0 + m.start(2), p0 + m.end(2)))
    for m in _HEADER_LETTER_RE.finditer(payoff_text):
        label_spans.append((p0 + m.start(1), p0 + m.end(1)))
    v0, v1 = sec["valid_moves"]
    # the only standalone A/B in the valid-moves region are the option letters in
    # "Valid moves (unordered): A, B." (the Note line has no standalone A/B).
    for m in _VM_LETTER_RE.finditer(text[v0:v1]):
        label_spans.append((v0 + m.start(1), v0 + m.end(1)))
    canon_letter = cb["action_to_label"][int(canonical_action_p1)]
    canon_row_span = None
    for m in _YOU_ROW_RE.finditer(payoff_text):
        if m.group(1) == canon_letter:
            canon_row_span = (p0 + m.start(), p0 + m.end())
            break
    return own_spans, opp_spans, label_spans, canon_row_span, sec


def _overlaps(a, b, s, e):
    return a < e and b > s


def build_region_map(game_vec, cb: dict, *, canonical_action_p1: int, tokenizer, iv_prefix: str = ""):
    """Return a list of per-token dicts for the one-shot prompt (P1) + PROBE_PREFIX.

    ``iv_prefix`` is the cue/intervention system prefix for cued conditions (empty for
    baseline); its tokens get region ``cue_prefix``. Each dict: token_index, token_str,
    char_start, char_end, region, is_canonical_row, canonical_action_letter.
    """
    prompt = build_oneshot_prompt(list(game_vec), cb, player=1, iv_prefix=iv_prefix)
    return build_region_map_from_text(prompt + PROBE_PREFIX, prompt, cb,
                                      canonical_action_p1=canonical_action_p1, tokenizer=tokenizer)


def build_region_map_chat(game_vec, cb: dict, *, canonical_action_p1: int, tokenizer, iv_prefix: str = ""):
    """CHAT-wrapped region map (llama needs its chat template). Chat header tokens fall under
    ``cue_prefix``; the trailing assistant 'Decision: ' under ``answer_prefix``. Same swap-
    invariant own/opponent + section logic as the raw path."""
    from collection.oneshot_chat import build_oneshot_chat_text, ASSISTANT_DECISION_PREFIX
    text = build_oneshot_chat_text(tokenizer, list(game_vec), cb, player=1, iv_prefix=iv_prefix)
    prompt = text[: -len(ASSISTANT_DECISION_PREFIX)]
    return build_region_map_from_text(text, prompt, cb,
                                      canonical_action_p1=canonical_action_p1, tokenizer=tokenizer)


def build_region_map_from_text(text: str, prompt: str, cb: dict, *, canonical_action_p1: int, tokenizer):
    """Shared core: region map for an arbitrary rendered ``(text, prompt)`` pair, where
    ``prompt`` = ``text`` minus the answer prefix (raw: ``PROBE_PREFIX``; chat: the assistant
    'Decision: '). Used by both the raw and chat builders so the logic is shared."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    ids = enc["input_ids"]
    own_spans, opp_spans, label_spans, canon_row_span, sec = number_and_letter_spans(
        text, prompt, cb, canonical_action_p1)
    canon_letter = cb["action_to_label"][int(canonical_action_p1)]
    specific = (("own_payoff", own_spans), ("opponent_payoff", opp_spans), ("label_token", label_spans))
    section_order = ("cue_prefix", "intro", "payoff_struct", "valid_moves", "question", "answer_prefix")

    rows = []
    for i, (tid, (a, b)) in enumerate(zip(ids, offsets)):
        if a == b:  # zero-width (special) token — skip from regions but keep a row
            region = "special"
        else:
            region = None
            for name, spans in specific:
                if any(_overlaps(a, b, s, e) for (s, e) in spans):
                    region = name
                    break
            if region is None:
                for name in section_order:
                    s, e = sec[name]
                    if s <= a < e:
                        region = name
                        break
            region = region or "other"
        is_canon = bool(canon_row_span and a < canon_row_span[1] and b > canon_row_span[0]
                        and region in ("own_payoff", "opponent_payoff", "label_token"))
        rows.append({
            "token_index": i,
            "token_str": text[a:b] if b > a else tokenizer.decode([tid]),
            "char_start": int(a), "char_end": int(b),
            "region": region,
            "is_canonical_row": is_canon,
            "canonical_action_letter": canon_letter,
        })
    return rows
