#!/usr/bin/env bash
# tier1_figures.sh — rebuild the paper figures and report which reach for the released
# data deposit versus which run from the committed tables alone.
#
# Usage:
#   scripts/dev/tier1_figures.sh              rebuild all 11
#   NO_HEAVY=1 scripts/dev/tier1_figures.sh   hide the deposit -> only the 6 clone-only
#                                             figures may build; the other 5 MUST fail
#   TIER1_ONLY=1 scripts/dev/tier1_figures.sh build just the 6 clone-only figures
#   PY="python" scripts/dev/tier1_figures.sh  override the interpreter
#
# Exit status is 0 only if every figure the mode expects to build actually built AND no
# figure that was expected to fail unexpectedly succeeded.
set -u
# Repo root without requiring git: a Zenodo software archive or a plain tarball has no
# .git, and every path below is repo-relative. Prefer git when it is available.
ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "${ROOT_DIR:-}" ] || ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$ROOT_DIR"
PY="${PY:-uv run python}"
NO_HEAVY="${NO_HEAVY:-0}"
TIER1_ONLY="${TIER1_ONLY:-0}"

# Portable mtime. BSD stat spells modification time `-f %m`; on GNU stat `-f` means
# --file-system and prints filesystem info instead, so the freshness comparison below fed
# `[` a string like "Inodes: Total: ..." and every figure was misreported as RAN-NO-PDF on
# Linux CI. Try GNU first, fall back to BSD, and yield 0 for anything unreadable.
mtime() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0; }

# Newest mtime among analysis/**/<name>, or 0 when nothing matches -- same semantics the
# `find ... -exec stat ... | sort -rn | head -1` pipeline had, minus the BSD dependency.
newest_mtime() {
  local newest=0 f m
  while IFS= read -r f; do
    m=$(mtime "$f")
    [ "$m" -gt "$newest" ] && newest="$m"
  done < <(find analysis -name "$1" -type f 2>/dev/null)
  printf '%s\n' "$newest"
}

# figure pdf | producer | tier
#   tier1 = committed tables only (rebuilds from a clean clone)
#   heavy = additionally needs the released deposit under $SCA_DATA_ROOT
#   human = additionally needs the user-supplied raw human data (see data/human_refs/raw/)
PRODUCERS=(
  "fig2_trait_steering.pdf|analysis/layer_a/figscripts/fig2_trait_steering.py|tier1"
  "fig_layerB_main_v2.pdf|analysis/layer_b/fig_layerB_main_v2.py|tier1"
  "fig_fusion_combined.pdf|analysis/layer_b/fusion_associative/fig_fusion_combined.py|tier1"
  "fig_nullspread_geometry_nhb.pdf|analysis/layer_b/fusion_associative/fig_nullspread_nhb.py|tier1"
  "fig_steering_causal.pdf|analysis/steering/fig_steering_causal_paper.py|tier1"
  "fig_layerC_paper.pdf|analysis/layer_c/fig_layerC_paper.py|tier1"
  "figS_attribution.pdf|analysis/layer_a/build_attribution.py|heavy"
  "fig_router_bottleneck.pdf|analysis/layer_b/router_bottleneck_analysis.py|heavy"
  "fig_token_heatmap_3panel.pdf|analysis/layer_c/fig_token_heatmap_3panel.py|heavy"
  "fig_behaviour_merged_2x3.pdf|analysis/layer_a/figscripts/fig_behaviour_merged_2x3.py|human"
  "figS_model_selection.pdf|analysis/layer_a/figscripts/fig3_qre_to_levelk.py|human"
)

LOG=$(mktemp -d)

# Four of the producers below (build_attribution.py, router_bottleneck_analysis.py,
# fig3_qre_to_levelk.py, fig2_trait_steering.py) are Tier-2 BUILDERS: they write tables into
# results_root(). Running them against the committed tree silently overwrites released data --
# which is exactly what happened before this guard existed. Point results_root() at a scratch
# copy so the gate can never mutate data/results/, and verify that afterwards.
RESULTS_SRC="$ROOT_DIR/data/results"
RESULTS_SCRATCH="$LOG/results"
cp -R "$RESULTS_SRC" "$RESULTS_SCRATCH"
export SCA_RESULTS_ROOT="$RESULTS_SCRATCH"
RESULTS_BEFORE=$(find "$RESULTS_SRC" -type f -exec shasum -a 256 {} \; | sort | shasum -a 256 | cut -d" " -f1)

# NO_HEAVY hides the deposit by pointing SCA_DATA_ROOT at an empty directory, which is what
# config.data_root() resolves everything through. Without this the variable was documented
# but never read, so the mode silently did nothing.
if [ "$NO_HEAVY" = "1" ]; then
  EMPTY_ROOT="$LOG/empty_data_root"
  mkdir -p "$EMPTY_ROOT"
  export SCA_DATA_ROOT="$EMPTY_ROOT"
  echo "NO_HEAVY=1 -> SCA_DATA_ROOT=$EMPTY_ROOT (deposit hidden)"
  echo
fi

ok=0; bad=0; expected_fail=0; unexpected_ok=0

for entry in "${PRODUCERS[@]}"; do
  IFS='|' read -r pdf script tier <<<"$entry"

  if [ "$TIER1_ONLY" = "1" ] && [ "$tier" != "tier1" ]; then
    continue
  fi

  # Expect a failure only when the data the figure needs has been hidden.
  expect_fail=0
  if [ "$NO_HEAVY" = "1" ] && [ "$tier" != "tier1" ]; then
    expect_fail=1
  fi

  # Record the pre-existing output so a stale PDF from an earlier run cannot be counted as a
  # fresh build. The original script computed this and then never used it.
  before=$(newest_mtime "$pdf")

  logfile="$LOG/$(basename "$script").log"
  if $PY "$script" >"$logfile" 2>&1; then rc=0; else rc=1; fi

  after=$(newest_mtime "$pdf")
  built=0
  [ "$rc" -eq 0 ] && [ "$after" -gt "$before" ] && built=1

  if [ "$expect_fail" = "1" ]; then
    if [ "$built" = "1" ]; then
      printf '  UNEXPECTED-OK  %-32s (%s: built with the deposit hidden)\n' "$pdf" "$tier"
      unexpected_ok=$((unexpected_ok+1))
    else
      printf '  EXPECTED-FAIL  %-32s (%s) %s\n' "$pdf" "$tier" \
             "$(tail -1 "$logfile" | cut -c1-58)"
      expected_fail=$((expected_fail+1))
    fi
    continue
  fi

  if [ "$built" = "1" ]; then
    hit=$(find analysis -name "$pdf" -type f 2>/dev/null | head -1)
    printf '  BUILT          %-32s -> %s\n' "$pdf" "$hit"
    ok=$((ok+1))
  elif [ "$rc" -eq 0 ]; then
    printf '  RAN-NO-PDF     %-32s (%s wrote no fresh %s)\n' "$pdf" "$script" "$pdf"
    bad=$((bad+1))
  else
    printf '  FAIL           %-32s %s\n' "$pdf" "$(tail -1 "$logfile" | cut -c1-70)"
    bad=$((bad+1))
  fi
done

# The committed tables must be untouched by a figure rebuild.
RESULTS_AFTER=$(find "$RESULTS_SRC" -type f -exec shasum -a 256 {} \; | sort | shasum -a 256 | cut -d" " -f1)
if [ "$RESULTS_BEFORE" != "$RESULTS_AFTER" ]; then
  echo
  echo "  ERROR: data/results/ changed during the figure rebuild — a producer wrote through"
  echo "         the SCA_RESULTS_ROOT override. Restore with: git checkout -- data/results"
  bad=$((bad+1))
fi

echo
if [ "$NO_HEAVY" = "1" ]; then
  echo "tier-1 figures: $ok built, $bad failed, $expected_fail correctly refused without the deposit, $unexpected_ok unexpectedly built"
else
  echo "tier-1 figures: $ok built, $bad failed"
fi
echo "logs in $LOG"
exit $(( bad > 0 || unexpected_ok > 0 ))
