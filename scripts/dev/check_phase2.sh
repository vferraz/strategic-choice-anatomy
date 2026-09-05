#!/usr/bin/env bash
# check_phase2.sh — the PHASE 2 acceptance gate, re-runnable by later phases.
#
# Verifies that the collection/ and steering/ pipeline surface is standalone:
#   1. every module imports cleanly against the installed strategic_anatomy package
#   2. every argparse entry point prints --help
#   3. no PYTHONPATH convention, machine-specific path, or heavy-data literal survives
#
# Entry points WITHOUT argparse (preflights, build_perm_directions) are import-checked
# only — running them with --help would execute the real job, which loads a model.
#
# Usage: scripts/dev/check_phase2.sh      (from anywhere inside a clone)
set -u
# Repo root without requiring git: a Zenodo software archive or a plain tarball has no
# .git, and every path below is repo-relative. Prefer git when it is available.
ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "${ROOT_DIR:-}" ] || ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-uv run python}"
fail=0
pass=0

note() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { pass=$((pass+1)); printf '  PASS  %s\n' "$*"; }
bad()  { fail=$((fail+1)); printf '  FAIL  %s\n' "$*"; }

# ---------------------------------------------------------------- 1. module imports
note "1. module imports"
while IFS= read -r f; do
  mod="${f%.py}"; mod="${mod//\//.}"
  if $PY -c "import $mod" >/dev/null 2>&1; then ok "import $mod"; else
    bad "import $mod"; $PY -c "import $mod" 2>&1 | tail -3 | sed 's/^/        /'
  fi
done < <(find collection steering -name '*.py' | sort)

# ------------------------------------------------------------------- 2. --help works
note "2. argparse entry points respond to --help"
while IFS= read -r f; do
  if grep -q "ArgumentParser" "$f"; then
    if $PY "$f" --help 2>&1 | grep -q "^usage:"; then ok "--help $f"; else bad "--help $f"; fi
  fi
done < <(find collection steering -name '*.py' | sort)

# -------------------------------------------------------------------- 3. hygiene greps
note "3. hygiene greps"

check_empty() {  # check_empty <label> <grep-output>
  if [ -z "$2" ]; then ok "$1"; else bad "$1"; printf '%s\n' "$2" | sed 's/^/        /'; fi
}

# This script names the very patterns it forbids, so it filters itself out of every grep.
EXCL="--exclude-dir=.venv --exclude-dir=.git --exclude-dir=__pycache__"
# Both dev gates necessarily spell out the patterns they forbid, so both are excluded from
# every grep here. check_hygiene.sh is the authority on machine paths and secrets and does the
# same for itself; this gate keeps a lighter copy so it stays useful standalone.
drop_self() { grep -v -e 'scripts/dev/check_phase2.sh' -e 'scripts/dev/check_hygiene.sh' || true; }

check_empty "no PYTHONPATH convention" \
  "$(grep -rn $EXCL 'PYTHONPATH' scripts/ collection/ steering/ 2>/dev/null | drop_self)"

check_empty "no machine-specific paths" \
  "$(grep -rnE $EXCL '/home/vferraz|/Users/viniciusferraz' . \
      --include='*.py' --include='*.sh' --include='*.md' --include='*.toml' 2>/dev/null | drop_self)"

# Scope per PHASE2 gate: the released heavy-data roots. `output/oneshot_main` (the
# superseded A/B substrate) is deliberately out of scope — it survives only as the
# --substrate-root default in steering/extract_directions.py, flagged for sign-off
# because repointing it would change a CLI default.
# Widened after Tier 2 (report §2.2): the original pattern only forbade `oneshot_akata*`
# literals, so `rebuild_lib.py`'s surviving `datasets/processed/game_features.csv` -- a
# PRIVATE-layout path -- passed this gate and only failed at table-regeneration time.
#
# The private-layout alternatives are matched only when QUOTED, i.e. in path construction
# (`os.path.join(ROOT, "datasets", "processed")`, `ROOT / "datasets" / "processed"`,
# `"analysis/layerB_final/..."`). Bare mentions in prose are NOT matched, because playbook
# rule 5 requires every extracted function to name its private origin file in a header
# comment -- flagging those would make the gate fight the convention it exists to serve.
# Comment-only lines are dropped for the same reason. `data/`, the RELEASED layout, is
# deliberately absent from the pattern.
PRIVATE_LAYOUT='["'"'"']datasets["'"'"']|["'"'"'][^"'"'"']*(datasets/processed|analysis/_shared/datasets|analysis/layer[ABC]|analysis/block_[abc])'
check_empty "no released heavy-data literals outside config.py" \
  "$(grep -rln $EXCL 'oneshot_akata_main_dl\|oneshot_akata_gptoss_recap\|output/oneshot_akata' \
      --include='*.py' . 2>/dev/null \
      | grep -v 'strategic_anatomy/config.py' | grep -v '^./docs/' || true)"

check_empty "no private-layout path literals" \
  "$(grep -rnE $EXCL "$PRIVATE_LAYOUT" --include='*.py' . 2>/dev/null \
      | grep -v 'strategic_anatomy/config.py' | grep -v '^./docs/' \
      | grep -vE '^[^:]*:[0-9]+:[[:space:]]*#' || true)"

check_empty "no legacy import forms" \
  "$(grep -rnE $EXCL 'sys\.path|from src\.|import src\.|from analysis\.block_[abc]' \
      --include='*.py' . 2>/dev/null || true)"

# ------------------------------------------------------------------------- 4. launchers
note "4. launcher syntax"
for f in scripts/launchers/*.sh scripts/setup/*.sh; do
  if bash -n "$f" 2>/dev/null; then ok "bash -n $f"; else bad "bash -n $f"; fi
done

note "phase-2 gate: $pass passed, $fail failed"
exit $(( fail > 0 ))
