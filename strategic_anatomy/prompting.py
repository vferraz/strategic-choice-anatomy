"""Prompt construction, counterbalancing schedules, and move parsing.

Moved verbatim out of ``src/run_sim_spark.py`` (private repo, commit 1f47050) during
the open-source migration. Nothing in this module imports ``torch`` or ``transformers``,
which is what lets the analysis layers, the collection-side prompt builders, and the test
suite run in an ``[analysis]``-only environment. ``strategic_anatomy.runtime`` re-exports
every name defined here, so ``from strategic_anatomy.runtime import build_prompt`` and
friends keep resolving exactly as before.

The counterbalancing schedules use the global :mod:`random` module, seeded by
``runtime.main()``; moving them to this module does not change the RNG object they draw
from, so schedules are bit-for-bit unchanged.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


# =============================================================================
# Constants
# =============================================================================

MOVE_LABELS = ["A", "B"]           # abstract counterbalanced labels
PAYOFF_MULTIPLIER = 2              # original matrix × 2


@dataclass
class Trait:
    id: str = "none"
    trait: str = ""
    intensity: str = ""
    system_prefix: str = ""


def generate_label_map() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Generate a random mapping between labels (A, B) and action indices (0, 1).

    Returns (label_to_action, action_to_label).
    Each player gets an independent mapping per round so that any token-prior
    bias in P(A) vs P(B) is randomized out across rounds.
    """
    if random.random() < 0.5:
        label_to_action = {"A": 0, "B": 1}
    else:
        label_to_action = {"A": 1, "B": 0}
    action_to_label = {v: k for k, v in label_to_action.items()}
    return label_to_action, action_to_label


def generate_balanced_label_schedule(n_rounds: int):
    """Pre-compute a label schedule with exact 50/50 balance across rounds.

    Returns a list of (label_to_action, action_to_label) tuples, shuffled.
    """
    half = n_rounds // 2
    schedule = []
    for i in range(n_rounds):
        if i < half:
            lta = {"A": 0, "B": 1}
        else:
            lta = {"A": 1, "B": 0}
        atl = {v: k for k, v in lta.items()}
        schedule.append((lta, atl))
    random.shuffle(schedule)
    return schedule


def generate_balanced_swap_schedule(n_rounds: int) -> List[bool]:
    """Pre-compute a balanced True/False schedule for display-order swaps.

    Returns a shuffled list with exact 50/50 balance.
    Call once per dimension (swap_rows_p1, swap_cols_p1, swap_rows_p2, swap_cols_p2).
    """
    half = n_rounds // 2
    swaps = [False] * half + [True] * (n_rounds - half)
    random.shuffle(swaps)
    return swaps


# =============================================================================
# Prompt construction
# =============================================================================

def payoff_rules_symmetric(vec: List[int], player: int,
                           action_to_label: Dict[int, str],
                           swap_rows: bool = False,
                           swap_cols: bool = False) -> str:
    """
    Matrix-format payoff rules with symmetric framing.
    Each cell shows (your payoff, other player's payoff).
    Labels come from action_to_label (randomized per round).
    Within one player's prompt, the same mapping labels both "you" and "other" actions.

    swap_rows / swap_cols randomize the *display* order of rows (your actions)
    and columns (other player's actions) so that act0 is not always first.
    Payoff values follow the swap — only the presentation order changes.
    """
    a00, a01, a10, a11, b00, b01, b10, b11 = vec
    lab0 = action_to_label[0]   # label for action index 0
    lab1 = action_to_label[1]   # label for action index 1

    if player == 1:
        # P1 is row player: action 0 = row 0, action 1 = row 1
        # Columns (other player): action 0 = col 0, action 1 = col 1
        c00 = (a00, b00); c01 = (a01, b01); c10 = (a10, b10); c11 = (a11, b11)
    else:
        # P2 is column player: you_action 0 = col 0, you_action 1 = col 1
        # Other (P1): action 0 = row 0, action 1 = row 1
        c00 = (b00, a00); c01 = (b10, a10); c10 = (b01, a01); c11 = (b11, a11)

    # --- Apply presentational swaps ---
    # swap_rows: swap your-action display order (row 0 ↔ row 1)
    if swap_rows:
        c00, c01, c10, c11 = c10, c11, c00, c01

    # swap_cols: swap other-player-action display order (col 0 ↔ col 1)
    if swap_cols:
        c00, c01, c10, c11 = c01, c00, c11, c10

    # Display labels track the swaps
    row_lab0, row_lab1 = (lab1, lab0) if swap_rows else (lab0, lab1)
    col_lab0, col_lab1 = (lab1, lab0) if swap_cols else (lab0, lab1)

    lines = [
        "Game rules (your payoff, other player's payoff):",
        f"            Other={col_lab0}        Other={col_lab1}",
        f"You={row_lab0}     ({c00[0]}, {c00[1]})          ({c01[0]}, {c01[1]})",
        f"You={row_lab1}     ({c10[0]}, {c10[1]})          ({c11[0]}, {c11[1]})",
    ]
    return "\n".join(lines)


def format_instruction() -> str:
    return (
        "Decide your move now.\n"
        "Output ONLY one line starting with Decision: followed by your chosen move."
    )


def valid_moves_line(valid_moves: List[str], shuffle: bool = True) -> str:
    """Always shuffled by default — order should never leak positional signal."""
    vm = random.sample(valid_moves, k=len(valid_moves)) if shuffle else list(valid_moves)
    return f"Valid moves (unordered): {', '.join(vm)}."


def history_lines_symmetric(
    history: List[Tuple[int, int, int, int]],
    player: int,
    action_to_label: Dict[int, str],
    max_history: int = 0,
) -> List[str]:
    """
    Format history with symmetric language.
    history entries: (action1, action2, payoff1, payoff2)
      where action1/action2 are action indices (0 or 1).
    Labels are translated using the CURRENT round's action_to_label mapping
    so that history is always consistent with the current prompt's label space.
    """
    if not history:
        return []
    hist = history[-max_history:] if max_history > 0 else history
    lines = []
    start_idx = len(history) - len(hist) + 1
    for i, (a1, a2, p1, p2) in enumerate(hist, start=start_idx):
        if player == 1:
            you_lab = action_to_label[a1]
            oth_lab = action_to_label[a2]
            lines.append(f"Round {i}: you chose {you_lab}, the other player chose {oth_lab} (you = {p1}, other = {p2})")
        else:
            you_lab = action_to_label[a2]
            oth_lab = action_to_label[a1]
            lines.append(f"Round {i}: you chose {you_lab}, the other player chose {oth_lab} (you = {p2}, other = {p1})")
    return lines


def build_prompt(
    game_vec: List[int],
    history: List[Tuple[int, int, int, int]],
    include_history: bool,
    valid_moves: List[str],
    player: int,
    cumulative_you: int,
    cumulative_opp: int,
    action_to_label: Dict[int, str],
    swap_rows: bool = False,
    swap_cols: bool = False,
    iv_prefix: str = "",
    max_history: int = 0,
    shuffle_valid_moves: bool = True,
) -> str:
    chunks: List[str] = []
    if iv_prefix:
        chunks.append(iv_prefix)

    chunks.append("You are about to play one round of a two-player game.")
    chunks.append(payoff_rules_symmetric(game_vec, player, action_to_label,
                                         swap_rows=swap_rows, swap_cols=swap_cols))

    chunks.append(f"Cumulative payoff so far: you = {cumulative_you}, the other player = {cumulative_opp}.")

    if include_history and history:
        lines = history_lines_symmetric(history, player, action_to_label,
                                        max_history=max_history)
        chunks.append("History so far:\n" + "\n".join(lines))

    chunks.append("Note: The order of 'Valid moves' below is arbitrary; do not infer preference from it.")
    chunks.append(valid_moves_line(valid_moves, shuffle_valid_moves))

    chunks.append(
        "Choose your move based on your preferences and the game rules above."
    )

    chunks.append(format_instruction())
    return "\n\n".join(chunks)

def build_move_token_map(tok, probe_prefix: str, valid_moves):
    """
    Robustly map each move (e.g. "A","B") to the token id the model will emit
    immediately after the probe_prefix.

    With BPE/SentencePiece tokenizers, tokenization at the boundary can change
    depending on what comes next. So `tok(probe_prefix)` and `tok(probe_prefix+"F")`
    may not share the same prefix length. This implementation uses the longest
    common prefix (LCP) to find the first token that differs.
    """
    prefix_ids = tok(probe_prefix, add_special_tokens=False).input_ids

    move_token_map = {}
    for mv in valid_moves:
        full_ids = tok(probe_prefix + mv, add_special_tokens=False).input_ids

        # Longest common prefix length (LCP) between prefix_ids and full_ids
        lcp = 0
        max_lcp = min(len(prefix_ids), len(full_ids))
        while lcp < max_lcp and prefix_ids[lcp] == full_ids[lcp]:
            lcp += 1

        # The first token after the shared prefix is the move token
        if lcp >= len(full_ids):
            raise ValueError(
                f"Tokenization failed for move {mv}: full_ids ended early. "
                f"prefix_len={len(prefix_ids)} full_len={len(full_ids)} lcp={lcp}"
            )

        move_token_map[mv] = full_ids[lcp]

    return move_token_map


def parse_generated_move(generated_text: str, valid_moves: List[str]):
    """Parse a generated move label from free-form model output."""
    moves_re = "".join(re.escape(mv) for mv in valid_moves)

    match = re.search(
        r'(?i)\bdecision\b[\s:*_-]*["\'*]*\b([' + moves_re + r'])\b',
        generated_text,
    )
    if match:
        move_str = match.group(1).upper()
        if move_str in valid_moves:
            return move_str

    match = re.search(
        r'(?i)\b(?:choose|choosing|pick|picking|select|selecting|suggest|go with)\b'
        r'\s*["\'*]*\b([' + moves_re + r'])\b',
        generated_text,
    )
    if match:
        move_str = match.group(1).upper()
        if move_str in valid_moves:
            return move_str

    bare = generated_text.strip().upper().strip("*\"'")
    if bare in valid_moves:
        return bare

    return None

def load_traits(path: str) -> Dict[str, Trait]:
    with open(path) as f:
        raw = json.load(f)
    traits = {"none": Trait()}
    for entry in raw:
        traits[entry["id"]] = Trait(
            id=entry["id"],
            trait=entry["trait"],
            intensity=entry["intensity"],
            system_prefix=entry["system_prefix"],
        )
    return traits
