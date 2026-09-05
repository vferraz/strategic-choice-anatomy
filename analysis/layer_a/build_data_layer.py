#!/usr/bin/env python3
"""Compatibility entry point for the corrected Layer-A data layer.

The paper-final data layer lives in ``analysis.layer_a.src.shared_data`` and
reads the corrected integrated substrate:

* ``$SCA_DATA_ROOT/substrate/{model}/{game}/results.parquet``;
* dense models: ``decoded_action`` gated by ``parse_ok``;
* GPT-OSS: ``realized_action`` gated by ``commit_type != 'none'``; mixed rows are
  already resolved to 0/1 and included as behaviour.

This wrapper preserves the historical command
``python analysis/layer_a/build_data_layer.py`` while preventing any future
rebuild from recreating the stale slot cache. It also writes
``oneshot_unified_pairs.parquet`` as a corrected legacy alias of ``fig1_panel.parquet``;
new code should read ``fig1_panel.parquet`` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from analysis.layer_a.src import shared_data as SD  # noqa: E402


LEGACY_UNIFIED = SD.DATA / "oneshot_unified_pairs.parquet"


def _write_legacy_alias() -> None:
    panel = pd.read_parquet(SD.DATA / "fig1_panel.parquet")
    panel.to_parquet(LEGACY_UNIFIED, index=False)
    print(f"wrote corrected legacy alias {LEGACY_UNIFIED} "
          f"({len(panel)} rows; source=fig1_panel.parquet)")


def _canary() -> None:
    gl = pd.read_parquet(SD.DATA / "layerA_game_level.parquet")
    means = gl.groupby("model")["p_canon"].mean()
    gpt = float(means.loc["gptoss"])
    llama = float(means.loc["llama31_instruct"])
    if gpt < 0.85:
        raise AssertionError(f"GPT-OSS p_canon={gpt:.3f}; expected corrected-root high conformity")
    if llama < 0.70:
        raise AssertionError(f"Llama p_canon={llama:.3f}; expected corrected chat/generated source")
    print("[canary] corrected behavioural cache: "
          f"GPT-OSS p_canon={gpt:.3f}, Llama p_canon={llama:.3f}")


def main() -> int:
    rc = SD.main()
    _write_legacy_alias()
    _canary()
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
