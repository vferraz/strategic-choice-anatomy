#!/usr/bin/env python3
"""Compare rebuilt figures against the committed reference renders.

PDF bytes never match: matplotlib stamps a creation date, and different matplotlib
versions lay text out differently. So this rasterises both sides and compares pixels.

    uv run python scripts/dev/compare_figures.py            # the 6 Tier-1 figures
    uv run python scripts/dev/compare_figures.py --all      # all 11
    uv run python scripts/dev/compare_figures.py --dpi 150

Requires `pdftoppm` (poppler). Without it the script says so and exits 0 rather than
failing a gate for a missing optional tool.

**Read the threshold with the renderer in mind.** The committed reference PDFs were
rendered with **matplotlib 3.10.8**. On that version, four of the six Tier-1 figures come
back pixel-identical. On a different minor version expect a few percent of pixels to
differ from text metrics and canvas rounding alone — that is a rendering difference, not a
numeric one. `make verify-data` is what proves the underlying tables are unchanged.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REFERENCE_MATPLOTLIB = "3.10.8"

TIER1 = [
    "fig2_trait_steering",
    "fig_layerB_main_v2",
    "fig_fusion_combined",
    "fig_nullspread_geometry_nhb",
    "fig_steering_causal",
    "fig_layerC_paper",
]
HEAVY = [
    "figS_attribution",
    "fig_router_bottleneck",
    "fig_token_heatmap_3panel",
    "fig_behaviour_merged_2x3",
    "figS_model_selection",
]


def rasterise(pdf: Path, out_stem: Path, dpi: int) -> Path | None:
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(pdf), str(out_stem)],
                   check=False, capture_output=True)
    png = out_stem.with_suffix(".png")
    return png if png.exists() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="include the 5 figures that need the deposit")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="per-channel difference counted as a differing pixel (default 0.02)")
    args = ap.parse_args(argv)

    if shutil.which("pdftoppm") is None:
        print("pdftoppm not found (install poppler) — skipping the visual comparison.")
        return 0

    import numpy as np
    from matplotlib import image as mpimg
    import matplotlib

    root = Path(__file__).resolve().parents[2]
    ref_dir = root / "data" / "results" / "figures_reference"
    names = TIER1 + (HEAVY if args.all else [])

    if matplotlib.__version__ != REFERENCE_MATPLOTLIB:
        print(f"note: reference PDFs were rendered with matplotlib {REFERENCE_MATPLOTLIB}; "
              f"this environment has {matplotlib.__version__}.")
        print("      Expect a few percent of pixels to differ from text layout alone.\n")

    tmp = Path(tempfile.mkdtemp(prefix="figdiff-"))
    print(f"{'figure':32s} {'canvas':>10s} {'mean|Δ|':>9s} {'>thr px':>8s}   verdict")
    print("-" * 78)

    identical = differing = missing = 0
    for name in names:
        ref = ref_dir / f"{name}.pdf"
        hits = sorted(root.glob(f"analysis/**/figures/{name}.pdf"))
        if not ref.exists() or not hits:
            print(f"{name:32s} {'—':>10s} {'—':>9s} {'—':>8s}   "
                  f"{'no reference' if not ref.exists() else 'not rebuilt'}")
            missing += 1
            continue
        a_png = rasterise(ref, tmp / f"{name}_ref", args.dpi)
        b_png = rasterise(hits[0], tmp / f"{name}_new", args.dpi)
        if not a_png or not b_png:
            print(f"{name:32s} rasterisation failed")
            missing += 1
            continue
        a = mpimg.imread(a_png)[..., :3]
        b = mpimg.imread(b_png)[..., :3]
        canvas = "same" if a.shape == b.shape else "DIFFERENT"
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        d = np.abs(a[:h, :w] - b[:h, :w])
        frac = float((d.max(axis=2) > args.threshold).mean())
        if d.max() == 0:
            verdict, identical = "IDENTICAL", identical + 1
        elif frac < 0.001:
            verdict, identical = "visually identical", identical + 1
        else:
            verdict, differing = "differs", differing + 1
        print(f"{name:32s} {canvas:>10s} {d.mean():9.5f} {frac*100:7.3f}%   {verdict}")

    print(f"\n{identical} identical, {differing} differing, {missing} unavailable")
    print(f"rasters kept in {tmp}")
    if differing:
        print("\nA difference here is not automatically a problem — check the renderer version "
              "first, then `make verify-data` for the underlying tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
