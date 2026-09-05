"""Unit tests for the steer hook's cosine sign discipline (negative-dose / reversed control).

`_signed_cos` multiplies the raw hook cosine-with-vector by sign(dose) so a FAITHFUL hook
reads +1 at either dose sign, while a reversed direction reads -1. `assert_hook_integrity`
checks the recorded (signed) cosine stays >= min_cosine and the norm ratio stays in band.
All pure-CPU (no model / no GPU).
"""
import pandas as pd
import pytest

# The assertions here are pure CPU arithmetic, but `steering.steer_core` imports torch at
# module scope (it also defines the real forward hooks). Skip the module rather than fail
# collection in a Tier-1/Tier-2 environment, which installs `[analysis,dev]` and no torch.
pytest.importorskip("torch", reason="steering.steer_core imports torch; install the [gpu] extra")

from steering.steer_core import _signed_cos  # noqa: E402
from steering.causal_common import assert_hook_integrity  # noqa: E402


def test_signed_cos_multiplies_by_dose_sign():
    assert _signed_cos({"actual_delta_cosine_with_vector": 0.9}, 2.0) == pytest.approx(0.9)
    assert _signed_cos({"actual_delta_cosine_with_vector": 0.9}, -2.0) == pytest.approx(-0.9)
    assert _signed_cos({"actual_delta_cosine_with_vector": 1.0}, 0.0) == 0.0  # dose-0 excluded


def test_signed_cos_faithful_hook_is_plus_one_at_both_signs():
    # a faithful hook injects dose*vec, so raw cosine(delta, vec) == sign(dose)
    for dose in (1.0, 8.0, -1.0, -8.0):
        raw = 1.0 if dose > 0 else -1.0
        assert _signed_cos({"actual_delta_cosine_with_vector": raw}, dose) == pytest.approx(1.0)


def test_signed_cos_reversed_direction_is_minus_one():
    # reversed control: actual delta points opposite the intended direction at +dose
    assert _signed_cos({"actual_delta_cosine_with_vector": -1.0}, 1.0) == pytest.approx(-1.0)


def _hook_df(cos, ratio=1.0):
    return pd.DataFrame([
        {"dose": 1.0, "actual_delta_norm": ratio, "requested_delta_norm": 1.0,
         "actual_delta_cosine_with_requested_delta": cos, "model": "qwen", "layer": 79},
        {"dose": -1.0, "actual_delta_norm": ratio, "requested_delta_norm": 1.0,
         "actual_delta_cosine_with_requested_delta": cos, "model": "qwen", "layer": 79},
        {"dose": 0.0, "actual_delta_norm": 0.0, "requested_delta_norm": 0.0,
         "actual_delta_cosine_with_requested_delta": 0.0, "model": "qwen", "layer": 79},
    ])


def test_assert_hook_integrity_passes_faithful():
    assert_hook_integrity(_hook_df(cos=0.99, ratio=1.0))  # no raise


def test_assert_hook_integrity_flags_low_cosine():
    with pytest.raises(AssertionError):
        assert_hook_integrity(_hook_df(cos=0.50, ratio=1.0))


def test_assert_hook_integrity_flags_bad_norm_ratio():
    with pytest.raises(AssertionError):
        assert_hook_integrity(_hook_df(cos=0.99, ratio=2.0))
