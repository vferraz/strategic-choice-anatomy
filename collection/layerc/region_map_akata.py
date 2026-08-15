"""Per-token region map for the **Akata sentence-form** one-shot prompt (Layer C token attribution).
Faithful analog of region_map_oneshot.py, adapted from the A/B-matrix to the sentence structure.

Regions: cue_prefix, intro, rule (the 4 'If you choose...' sentences), own_payoff (the 'you win X
point(s)' number), opponent_payoff (the 'the other player wins Y point(s)' number), label_token (the
'Option J'/'Option P' letters), question, answer_prefix. Every token is mapped (no 'other').
is_canonical_row = own/opp/label tokens inside the rule sentences where the player's OWN action
(the 'you choose Option M') equals the canonical action.
"""
from __future__ import annotations
import re
from collection.akata_common import build_akata_oneshot_prompt, akata_user_question, ANSWER_PREFIX

_RULE_RE = re.compile(
    r"If you choose Option (?P<my>[JP]) and the other player chooses Option (?P<oth>[JP]), "
    r"then you win (?P<own>\d+) points? and the other player wins (?P<opp>\d+) points?\.")


def _overlaps(a, b, s, e):
    return a < e and b > s


def _spans_from_text(prompt_text: str, cb: dict, canonical_action_p1: int):
    """Return (specific_spans, section_spans, canon_letter). specific_spans: list of (region, (s,e),
    is_canon) for own/opp payoff numbers + Option letters; section_spans: ordered coarse regions."""
    l2a = cb["letter_to_action"]
    a2l = cb["action_to_letter"]
    canon_letter = a2l[int(canonical_action_p1)]
    specific = []
    # intro = start .. first rule; rules region spans the rule block; question; answer_prefix
    first = _RULE_RE.search(prompt_text)
    rules_start = first.start() if first else 0
    q = re.search(r"\nQ: Which Option", prompt_text)
    q_start = q.start() if q else len(prompt_text)
    ap = prompt_text.rfind(ANSWER_PREFIX)
    sections = [
        ("intro", (0, rules_start)),
        ("rule", (rules_start, q_start)),
        ("question", (q_start, ap if ap >= 0 else len(prompt_text))),
        ("answer_prefix", (ap, len(prompt_text)) if ap >= 0 else (len(prompt_text), len(prompt_text))),
    ]
    for m in _RULE_RE.finditer(prompt_text):
        my = m.group("my")
        is_canon = (l2a[my] == int(canonical_action_p1))
        specific.append(("own_payoff", m.span("own"), is_canon))
        specific.append(("opponent_payoff", m.span("opp"), is_canon))
        # the two 'Option X' letters in this rule (my + other)
        for g in ("my", "oth"):
            specific.append(("label_token", m.span(g), is_canon))
    return specific, sections, canon_letter


def build_region_map_from_text(text: str, prompt_text: str, cb: dict, *, canonical_action_p1: int,
                               tokenizer, cue_end: int = 0):
    """Per-token dicts over `text` (prompt incl. ANSWER_PREFIX, possibly chat-wrapped). cue_end = char
    index where the cue/system+header prefix ends (tokens before it -> cue_prefix)."""
    specific, sections, canon_letter = _spans_from_text(prompt_text, cb, canonical_action_p1)
    # shift prompt-relative spans into `text` coordinates
    base = text.find(prompt_text)
    if base < 0:
        base = 0
    spshift = [(r, (s + base, e + base), c) for (r, (s, e), c) in specific]
    secshift = [(r, (s + base, e + base)) for (r, (s, e)) in sections]
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    rows = []
    for i, (tid, (a, b)) in enumerate(zip(enc["input_ids"], enc["offset_mapping"])):
        if a == b:
            region, is_canon = "special", False
        elif b <= cue_end:
            region, is_canon = "cue_prefix", False
        else:
            region, is_canon = None, False
            for (r, (s, e), c) in spshift:                      # specific spans first (payoffs, letters)
                if _overlaps(a, b, s, e):
                    region, is_canon = r, c
                    break
            if region is None:
                for (r, (s, e)) in secshift:                    # coarse sections
                    if _overlaps(a, b, s, e):
                        region = r
                        break
            region = region or "answer_prefix"                  # trailing tokens -> answer slot
        rows.append({
            "token_index": i, "token_str": tokenizer.decode([tid]),
            "char_start": int(a), "char_end": int(b),
            "region": region, "is_canonical_row": bool(is_canon),
            "canonical_action_letter": canon_letter,
        })
    return rows


def build_region_map(game_vec, cb: dict, *, canonical_action_p1: int, tokenizer, iv_prefix: str = ""):
    """Raw path (qwen/qwen_instruct): prompt ends with 'A: Option'."""
    prompt = build_akata_oneshot_prompt(list(game_vec), cb, player=1, iv_prefix=iv_prefix)
    cue_end = (prompt.find("\n\n") + 2) if iv_prefix else 0     # iv_prefix is prepended + '\n\n'
    return build_region_map_from_text(prompt, prompt, cb, canonical_action_p1=canonical_action_p1,
                                      tokenizer=tokenizer, cue_end=cue_end)


def build_region_map_chat(game_vec, cb: dict, *, canonical_action_p1: int, tokenizer, iv_prefix: str = ""):
    """Chat path (llama): user turn = the body w/o 'A: Option'; assistant turn starts with it.
    Chat header/system tokens (everything before the game body) -> cue_prefix."""
    user = akata_user_question(list(game_vec), cb, player=1, iv_prefix=iv_prefix)
    chat = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                         add_generation_prompt=True)
    text = chat + ANSWER_PREFIX
    # The user BODY (game text, no 'A: Option') is contiguous in `text`; the assistant header sits
    # between it and the trailing 'A: Option'. Span-find on `user`; everything before it = cue_prefix,
    # everything after (assistant header + 'A: Option') -> answer_prefix default.
    cue_end = text.find(user)
    return build_region_map_from_text(text, user, cb, canonical_action_p1=canonical_action_p1,
                                      tokenizer=tokenizer, cue_end=max(cue_end, 0))
