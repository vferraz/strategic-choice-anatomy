"""Generation-side helpers shared by the collection entry points.

Extracted verbatim from ``analysis/block_b/validate_readout_oneshot.py`` (private repo,
commit 8d8370e): ``HARMONY_FINAL_MARKER``, ``HARMONY_END_RE``, ``STANDALONE_LETTER_RE`` and
``_extract_move``. The originating module is an A/B-era validator that is not part of the
release (plan §4.3 keeps only the reused functions).

``_make_move_stopper`` (L229-246) and ``_generate`` (L250-265) were appended in phase 2,
completing the extraction plan §4.3 specifies for this file.
"""
from __future__ import annotations

import re

from strategic_anatomy.prompting import MOVE_LABELS, parse_generated_move
from collection.oneshot_common import PROBE_PREFIX

# Harmony final-channel marker — mirrors src/run_akata_replication.py:128 (HARMONY_FINAL_MARKER).
# Defined locally to avoid importing that heavy off-limits runner just for one constant string.
HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"

HARMONY_END_RE = re.compile(r"<\|[^|]*\|>")

STANDALONE_LETTER_RE = re.compile(r"\b([AB])\b")  # UPPERCASE only — never the article "a"/"b"


def _extract_move(withspec_text: str, is_gptoss: bool):
    """Return (parsed_label|None, parse_source, final_channel_present) from the RAW with-specials
    generation text. Takes only the with-specials text so it can be re-applied identically at
    analyze time from the stored raw_text_trunc (keeps the parser uniform across all 4 models
    regardless of when each was generated).

    Order (each step only RESCUES what the prior left unparsed — so models that already answer
    cleanly are unaffected):
      1. parse_generated_move on the channel-stripped text ("Decision:/choose" anchors).
      2. decision_recon: reparse PROBE_PREFIX + text (the model is completing the Decision line).
      3. narrative single-letter: if exactly ONE distinct UPPERCASE standalone letter is present
         (e.g. llama 'You chose to play as A', 'You=A'), use it. Ambiguous text mentioning both
         A and B (reasoning) stays unparsed — we never guess on reasoning. 'A1'-style confabulated
         squares are not standalone letters, so they are correctly ignored.
    For gptoss the move is read ONLY from the harmony final channel; analysis-channel reasoning is
    never parsed (it discusses both A and B without deciding, so any letter there is spurious). If
    gptoss has not emitted a final channel yet, it has not answered -> (None, 'no_final_channel').
    """
    if is_gptoss:
        if HARMONY_FINAL_MARKER not in (withspec_text or ""):
            return None, "no_final_channel", False  # still reasoning — not an answer
        final_present = True
        target = HARMONY_END_RE.sub("", (withspec_text or "").split(HARMONY_FINAL_MARKER)[-1])
        source = "final_channel"
    else:
        final_present = False
        target = HARMONY_END_RE.sub("", withspec_text or "")  # strip any <|...|> leakage
        source = "whole_text"
    label = parse_generated_move(target, list(MOVE_LABELS))
    if label is None:
        label = parse_generated_move(PROBE_PREFIX + target, list(MOVE_LABELS))
        if label is not None:
            source += "+decision_recon"
    if label is None:
        letters = set(STANDALONE_LETTER_RE.findall(target))
        if len(letters) == 1:
            label = letters.pop()
            source += "+single_letter"
    return label, source, final_present


def _make_move_stopper(torch, tokenizer, input_len, is_gptoss, min_new=3, stride=16):
    """StoppingCriteria that halts generation as soon as a move is decodable. Budget becomes a
    ceiling, not the actual length — captures the move without paying for the rest. For verbose
    dense models (llama 'You chose to play as A') it stops at the letter; for gptoss it stops only
    once the harmony FINAL channel has emitted the move (analysis-channel reasoning never triggers
    it — see _extract_move). Checked every `stride` tokens (not every step) to bound decode cost on
    gptoss's long reasoning; overshoot <= stride tokens is harmless (same decode prefix)."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _MoveStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            n = int(input_ids.shape[1] - input_len)
            if n < min_new or (n % stride) != 0:
                return False
            withspec = tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=False)
            label, _, _ = _extract_move(withspec, is_gptoss)
            return label is not None

    return StoppingCriteriaList([_MoveStop()])


def _generate(model, tokenizer, input_ids, attention_mask, max_new_tokens, do_sample, torch,
              stopping_criteria=None):
    kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                      else tokenizer.eos_token_id),
    )
    if do_sample:
        kwargs.update(do_sample=True, temperature=1.0, top_p=0.95)
    else:
        kwargs.update(do_sample=False)
    if stopping_criteria is not None:
        kwargs["stopping_criteria"] = stopping_criteria
    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    return out[0, input_ids.shape[1]:].detach()
