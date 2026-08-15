"""One-shot **Akata sentence-form** prompt + 4-cell counterbalance + **J/P** decode.

LOCKED SPEC: docs/AKATA_ONESHOT_RECOLLECTION.md (PI-approved 2026-06-24). This is the CORRECTED
substrate prompt — it replaces the invalid A/B-matrix `oneshot_common.build_oneshot_prompt`.
ADDITIVE: this module does not touch oneshot_common.py / causal_v2_common.py.

Prompt = the 10-round Akata prompt with ONLY repeated-game material removed (`repeatedly`, the
"10 rounds" sentence, the round-number line, history) + grammar fixed ("1 point" not "1 points").
Ends with "A: Option" — the decision slot (the next token is the committed Option J/P).
"""
from __future__ import annotations
import hashlib
from typing import Sequence

# Move labels for THIS substrate: Option J / Option P  (NOT A/B).
MOVE_LABELS: tuple[str, str] = ("J", "P")
ANSWER_PREFIX = "A: Option"            # prompt ends here; model generates " J" / " P"


def akata_cb_grid() -> list[dict]:
    """The final 4-cell counterbalance grid (locked). 2 axes: label-map x question-order.

    cb0: J=act0,P=act1, question J/P   cb1: J=act0,P=act1, question P/J
    cb2: J=act1,P=act0, question J/P   cb3: J=act1,P=act0, question P/J
    No seeds, no row/col swaps, no valid-moves line, no rule-order shuffle.
    """
    rows: list[dict] = []
    cbid = 0
    for label_map in ("J=act0", "J=act1"):          # which canonical action Option J maps to
        for q_order in ("JP", "PJ"):                # order options appear in the question
            j_act = 0 if label_map == "J=act0" else 1
            rows.append({
                "counterbalance_id": cbid,
                "label_map": label_map,
                "q_order": q_order,
                "letter_to_action": {"J": j_act, "P": 1 - j_act},
                "action_to_letter": {j_act: "J", 1 - j_act: "P"},
            })
            cbid += 1
    assert len(rows) == 4, f"akata grid must have 4 cells, got {len(rows)}"
    return rows


def _points(x: int) -> str:
    return f"{x} point" + ("" if int(x) == 1 else "s")   # grammar fixed


def build_akata_oneshot_prompt(game_vec: Sequence[int], cb: dict, *, player: int = 1,
                               iv_prefix: str = "") -> str:
    """Single-shot Akata sentence-form prompt body (raw; ends with 'A: Option').

    game_vec = [a00,a01,a10,a11,b00,b01,b10,b11]; a=P1 payoff, b=P2 payoff, indexed (p1_act, p2_act).
    Rule order is FIXED J-J, J-P, P-J, P-P (no shuffle). For each rule (my,other) with
    ma=opt[my], oa=opt[other]:  P1 -> you=a[(ma,oa)], other=b[(ma,oa)];  P2 -> you=b[(oa,ma)],
    other=a[(oa,ma)] (perspective swap). Question option-order varies per cb; intro is always
    "between Option J and Option P".
    """
    v = [int(x) for x in game_vec]
    a = {(0, 0): v[0], (0, 1): v[1], (1, 0): v[2], (1, 1): v[3]}
    b = {(0, 0): v[4], (0, 1): v[5], (1, 0): v[6], (1, 1): v[7]}
    opt = cb["letter_to_action"]                    # {'J': act, 'P': act}
    rules = []
    for my, oth in (("J", "J"), ("J", "P"), ("P", "J"), ("P", "P")):
        ma, oa = opt[my], opt[oth]
        if int(player) == 1:
            you, other = a[(ma, oa)], b[(ma, oa)]
        else:                                       # P2: 'you'=P2, my action = column, other = row
            you, other = b[(oa, ma)], a[(oa, ma)]
        rules.append(f"If you choose Option {my} and the other player chooses Option {oth}, "
                     f"then you win {_points(you)} and the other player wins {_points(other)}.")
    q1, q2 = ("J", "P") if cb["q_order"] == "JP" else ("P", "J")
    body = ("You are playing a game with another player. In this game, you can choose between "
            "Option J and Option P. The rules of the game are as follows:\n"
            + "\n".join(rules)
            + f"\n\nQ: Which Option do you choose, Option {q1} or Option {q2}?\n{ANSWER_PREFIX}")
    return (iv_prefix + "\n\n" + body) if iv_prefix else body


def akata_user_question(game_vec: Sequence[int], cb: dict, *, player: int = 1,
                        iv_prefix: str = "") -> str:
    """The same prompt WITHOUT the trailing 'A: Option' — the user turn for chat models (llama).
    The assistant turn then starts with ANSWER_PREFIX ('A: Option')."""
    full = build_akata_oneshot_prompt(game_vec, cb, player=player, iv_prefix=iv_prefix)
    assert full.endswith("\n" + ANSWER_PREFIX)
    return full[: -(len(ANSWER_PREFIX) + 1)].rstrip("\n")


def parse_akata_move(text: str) -> str | None:
    """Parse the committed Option from a generation. Returns 'J' / 'P' / None (store None as a
    parse failure; NO argmax fallback). Reads only J/P after an 'Option' / 'A:' anchor, else a
    bare standalone J or P. Normalizes gpt-oss final-channel formatting first (markdown emphasis,
    zero-width chars, narrow/nbsp unicode spaces) so e.g. 'Option **J**' or '**​J​**'
    are read as the letter the model actually committed."""
    import re
    t = re.sub(r"[​‌‍﻿]", "", text)        # zero-width chars -> drop
    t = re.sub(r"[*_`]", " ", t)                               # markdown emphasis -> space
    t = re.sub(r"[       ]", " ", t)  # narrow/nbsp spaces -> ascii
    t = re.sub(r"[ \t]+", " ", t).strip()
    m = re.search(r"Option\s+([JP])\b", t)          # "Option J" / "Option P"
    if m:
        return m.group(1)
    m = re.search(r"\bA:\s*(?:Option\s+)?([JP])\b", t)
    if m:
        return m.group(1)
    m = re.match(r"\s*([JP])\b", t)                  # bare leading J/P (model continues after slot)
    if m:
        return m.group(1)
    return None


def prompt_sha256(p: str) -> str:
    return hashlib.sha256(p.encode("utf-8")).hexdigest()
