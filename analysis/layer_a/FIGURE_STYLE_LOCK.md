# NHB Figure Style Lock

This is the locked production standard for main-text figures in
`paper/paper_NHB/paper_NHB_v2.tex`. Do not solve figure readability in LaTeX:
main figures are imported at `width=\textwidth`, and layout/font fixes belong in
the figure generator.

## Global Rule

- Main figures are caption-led: no in-figure master title, subtitle, or
  convention line.
- Keep only panel letters and short panel titles inside the graphic.
- Put narrative interpretation, panel descriptions, and caveats in the caption.
- Export vector PDF plus 600 dpi PNG for inspection.
- Render the compiled manuscript page before accepting a figure.

## Locked Sizes

- Text-width figure width: `7.1 in`.
- Two-row / four-panel main figure: `7.1 x 5.75 in`.
- One-row / two-panel main figure: `7.1 x 3.75 in`.
- Manuscript include: `\includegraphics[width=\textwidth]{...}`.

## Locked Type Scale

- Panel title: `8.7 pt`, bold.
- Axis label: `8.0 pt`; use `8.3 pt` only for the two-panel Fig. 3 geometry.
- Tick labels: `7.0 pt`; use `6.8 pt` only where dense mini-panels require it.
- Shared legend: `6.6 pt`, single line whenever possible.
- Footnote / small annotation: `6.0 pt`.
- Mini-panel title: `7.0 pt`.
- Inset statistic labels: `5.2 pt`, only inside mini-panels.

## Locked Layout Conventions

- Shared top legend anchor: `bbox_to_anchor=(0.5, 0.985)`.
- Panel-title position: letter at `x=-0.075`, title at `x=0.055`,
  `y=1.055` in axes coordinates, unless a specific panel requires a tiny
  horizontal adjustment.
- Fig. 2 layout margins: `left=0.105`, `right=0.985`, `bottom=0.135`,
  `top=0.875`.
- Fig. 3 layout margins: `left=0.075`, `right=0.985`, `bottom=0.170`,
  `top=0.815`.
- Fig. 3 panel widths: `[1.38, 1.0]`, `wspace=0.16`.
- Fig. 3 panel B is a true `2 x 2` scatter grid with no spacer row; the full
  mini-panel block must be the same height as panel A.

## Source Of Truth

The machine-readable constants live in
`analysis/_shared/_paper_style.py` under the `NHB_*` names. New main-text figure
scripts should import those constants rather than hard-coding dimensions.

## Acceptance Check

Before presenting a figure:

1. Regenerate the source PDF/PNG.
2. Copy the PDF into `paper/paper_NHB/images/` if it is a manuscript figure.
3. Compile `paper_NHB_v2.tex`.
4. Render the manuscript page at high resolution.
5. Check for clipped text, font mismatch, legend-title crowding, uneven panel
   heights, stray annotations, and caption collisions.
