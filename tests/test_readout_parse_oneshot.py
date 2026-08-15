"""Unit tests for the uniform generation move parser (`validate_readout_oneshot._extract_move`).

Locks the two failure modes that matter for the corrected (generation) decoder:
  - gpt-oss: the move is read ONLY from the harmony final channel; analysis-channel
    reasoning (which discusses both A and B) never yields a move.
  - dense: a single standalone uppercase letter is rescued ("you chose to play as A"),
    but confabulated squares ("A1") and ambiguous both-letter reasoning are NOT parsed.
"""
from collection.genutils import _extract_move, HARMONY_FINAL_MARKER


def test_gptoss_reads_only_final_channel():
    txt = ("We must weigh A versus B given the payoffs..." + HARMONY_FINAL_MARKER + "B")
    label, source, final_present = _extract_move(txt, is_gptoss=True)
    assert label == "B"
    assert final_present is True
    assert "final_channel" in source


def test_gptoss_no_final_channel_is_not_an_answer():
    # analysis-channel reasoning only (mentions both A and B) -> not an answer
    txt = "Let me reason: option A gives more, but B is safer. Hmm, A or B?"
    label, source, final_present = _extract_move(txt, is_gptoss=True)
    assert label is None
    assert source == "no_final_channel"
    assert final_present is False


def test_dense_single_letter_rescue():
    label, _src, _f = _extract_move("\n\nYou chose to play as A.\n\n", is_gptoss=False)
    assert label == "A"


def test_dense_confabulated_square_not_parsed():
    # 'A1' is a confabulated board square, not the action letter -> must not yield 'A'
    label, _src, _f = _extract_move("\n\nYou chose to move your pawn to A1.\n\n", is_gptoss=False)
    assert label is None


def test_dense_ambiguous_both_letters_not_parsed():
    label, _src, _f = _extract_move("I could play A or B depending on the opponent.", is_gptoss=False)
    assert label is None


def test_dense_clean_decision_line():
    label, _src, _f = _extract_move(" B", is_gptoss=False)
    assert label == "B"
