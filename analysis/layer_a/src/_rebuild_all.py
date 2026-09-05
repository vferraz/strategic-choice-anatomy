#!/usr/bin/env python3
"""Rebuild ALL Layer A artifacts on the corrected generate()/commit decisions, in
dependency order. Run after any change to the decision source. CPU only.

Order: shared_data (decisions + audit) -> attribution (partition/SHAP/GLMM, feeds Fig2c)
-> fig3 (human λ, λ table, model-selection CV, fingerprint) -> trait_effects -> fig2 -> fig1.
"""
from __future__ import annotations
import sys
from pathlib import Path

from analysis.layer_a.src import shared_data as SD
from analysis.layer_a.src import attribution as A
from analysis.layer_a.src import fig3_behavioral_model as F3
from analysis.layer_a.src import trait_effects as TE
from analysis.layer_a.src import fig2_trait_steering as F2
from analysis.layer_a.src import fig1_rationality as F1


def main() -> int:
    print("\n########## 1/6 shared_data (decisions + audit) ##########")
    SD.main()
    print("\n########## 2/6 attribution (partition/SHAP/GLMM) ##########")
    A.main(force=True)
    print("\n########## 3/6 fig3 (λ, model-selection CV, fingerprint) ##########")
    F3.main(force=True)
    print("\n########## 4/6 trait_effects ##########")
    TE.main()
    print("\n########## 5/6 fig2 ##########")
    F2.main()
    print("\n########## 6/6 fig1 ##########")
    F1.main()
    print("\n########## REBUILD COMPLETE ##########")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
