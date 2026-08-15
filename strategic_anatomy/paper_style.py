"""Shared visual standard for the main paper figures.

Imported by the *final* figure scripts so they read as one source:
  - fig_main_trait_steering.py
  - fig_main_rationality.py

This sets fonts (with editable Type-42 text in the PDF), a small shared colour
palette, and a common type scale + panel-label convention "(a) / (b)". It does
NOT change any figure's layout or structure — only the visual language.

For the current NHB manuscript figures, use the locked geometry/type constants
below. The paper figures are intentionally caption-led: no in-figure master
titles or subtitles; only panel letters/titles remain inside the graphic.
"""
from __future__ import annotations

import matplotlib as mpl

# --- shared semantic colours (reconciled across the existing figures) -------
INK = "#222222"          # axis lines, headings
SUBTLE = "#444444"       # subtitles
FAINT = "#666666"        # footnotes
GREY = "#7a7a7a"         # neutral series / random-play references
GOOD_GREEN = "#0b7a45"   # "on target" / equilibrium / maximin
REF_RED = "#c83b31"      # human / random reference, "away"
SOFT_BLUE = "#1f77b4"    # net-steering series

# region / class background bands (kept consistent between figures)
BAND_DOMINANCE = "#e8eef6"
BAND_COORDINATION = "#e6f4e6"
BAND_NOPE = "#fceadb"

# --- shared type scale (points) ---------------------------------------------
FS_SUPTITLE = 13.0
FS_SUBTITLE = 8.0
FS_PANEL = 9.5
FS_AXIS = 8.0
FS_TICK = 7.0
FS_LEGEND = 7.0
FS_FOOT = 6.2

SEP = "  ·  "             # subtitle separator convention

# --- locked NHB main-figure production geometry -----------------------------
# These are the final page-tested dimensions used by Figs. 2-3 in
# paper/paper_NHB/paper_NHB_v2.pdf. Future main figures should start here unless
# the manuscript format changes.
NHB_TEXTWIDTH_IN = 7.1
NHB_TWO_ROW_HEIGHT_IN = 5.75
NHB_TWO_PANEL_HEIGHT_IN = 3.75
NHB_LEGEND_Y = 0.985

# Caption-led main-figure type scale. These values have been checked at
# width=\textwidth in the manuscript PDF.
NHB_FS_PANEL = 8.7
NHB_FS_AXIS = 8.0
NHB_FS_AXIS_LARGE = 8.3
NHB_FS_TICK = 7.0
NHB_FS_TICK_SMALL = 6.8
NHB_FS_LEGEND = 6.6
NHB_FS_FOOT = 6.0
NHB_FS_MINI_TITLE = 7.0
NHB_FS_INSET_STAT = 5.2

# Standard panel-title placement for caption-led figures.
NHB_PANEL_LETTER_X = -0.075
NHB_PANEL_TITLE_X = 0.055
NHB_PANEL_TITLE_Y = 1.055

# Page-tested layout recipes.
NHB_FIG2_TOP = 0.875
NHB_FIG2_LEFT = 0.105
NHB_FIG2_RIGHT = 0.985
NHB_FIG2_BOTTOM = 0.135
NHB_FIG3_TOP = 0.815
NHB_FIG3_LEFT = 0.075
NHB_FIG3_RIGHT = 0.985
NHB_FIG3_BOTTOM = 0.170


def apply() -> None:
    """Apply the shared rcParams. Call once at import time in each figure script."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,        # embed editable TrueType text in PDF (journal-friendly)
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": True,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def panel_title(ax, letter: str, text: str, *, pad: float = 6.0, fontsize: float = FS_PANEL) -> None:
    """Left-aligned bold panel label using the shared "(a)  Title" convention."""
    ax.set_title(f"({letter})  {text}", fontsize=fontsize, loc="left",
                 fontweight="bold", color=INK, pad=pad)


def nhb_panel_title(ax, letter: str, text: str, *, title_x: float = NHB_PANEL_TITLE_X,
                    y: float = NHB_PANEL_TITLE_Y, letter_x: float = NHB_PANEL_LETTER_X,
                    fontsize: float = NHB_FS_PANEL) -> None:
    """Panel label/title placement for caption-led NHB main figures."""
    ax.text(letter_x, y, f"({letter})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=fontsize, fontweight="bold", color=INK,
            clip_on=False)
    ax.text(title_x, y, text, transform=ax.transAxes, ha="left",
            va="bottom", fontsize=fontsize, fontweight="bold", color=INK,
            clip_on=False)


# physical placement of the figure title block (inches from the top), shared by every
# figure so the title<->subtitle gap is identical regardless of figure height.
TITLE_TOP_IN = 0.24      # title centre, inches below the top edge
SUBTITLE_TOP_IN = 0.50   # subtitle centre, inches below the top edge  (=> 0.26in constant gap)


def figure_titles(fig, title: str, subtitle: str) -> float:
    """Place the figure-level title + subtitle at identical physical spacing across figures.

    Uses absolute inch offsets from the top so the title/subtitle gap is the same in every
    figure no matter its height. Returns the subtitle y (figure fraction) for callers that
    need to lay out content beneath it.
    """
    h = fig.get_figheight()
    y_title = 1.0 - TITLE_TOP_IN / h
    y_sub = 1.0 - SUBTITLE_TOP_IN / h
    fig.text(0.5, y_title, title, ha="center", va="center",
             fontsize=FS_SUPTITLE, fontweight="bold", color=INK)
    fig.text(0.5, y_sub, subtitle, ha="center", va="center",
             fontsize=FS_SUBTITLE, color=SUBTLE)
    return y_sub
