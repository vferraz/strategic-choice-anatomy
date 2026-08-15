"""Chat-template prompt wrapping for one-shot capture (additive; no core file edited).

llama-3.1-instruct hard-depends on its chat template to engage the task (validated: raw ->
off-task board-game hallucination; chat-wrapped -> clean "Decision: A", 8/8). The other
models tolerate raw input. This wraps the EXACT one-shot game body as the user turn and
starts the assistant turn with "Decision: " so the decision-slot readout + residual sit at
the same A/B-prediction position as the raw path — just inside the model's native chat
scaffolding. The chat template inserts header tokens, so the slot moves vs the raw substrate:
chat-wrapped models must be (re)captured fresh, not joined to raw residuals.
"""
from __future__ import annotations

import sys
from pathlib import Path


from collection.oneshot_common import build_oneshot_prompt  # noqa: E402

ASSISTANT_DECISION_PREFIX = "Decision: "  # chat-mode probe prefix (assistant turn start)


def build_oneshot_chat_text(tokenizer, game_vec, cb, *, player: int = 1, iv_prefix: str = "") -> str:
    """Full chat-wrapped text ending in 'Decision: ' (the decision slot).

    Body = the exact raw one-shot prompt (`build_oneshot_prompt`); wrapping = the model's own
    chat template (user turn + assistant generation header). Readout/residual are taken at the
    last token (the space after 'Decision: '), as in the raw path.
    """
    body = build_oneshot_prompt(list(game_vec), cb, player=player, iv_prefix=iv_prefix)
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": body}], add_generation_prompt=True, tokenize=False)
    return chat + ASSISTANT_DECISION_PREFIX
