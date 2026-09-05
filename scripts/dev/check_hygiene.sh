#!/usr/bin/env bash
# check_hygiene.sh — no machine-specific paths or secrets in anything git tracks.
#
# Scoped to what `git add -A` would stage — tracked files PLUS untracked-but-not-ignored
# ones — because most of the tree is not committed yet and the point is to catch a leak
# BEFORE it enters history. A working tree normally holds a gitignored
# data_heavy/ full of absolute symlinks into wherever the deposit lives; those are a
# developer's business and must not fail the gate. What must never happen is one of them
# becoming *tracked*, which is exactly how an absolute local path reached a tracked tree
# once before (data/human_refs/raw/{nagel,griffiths}).
#
# Usage: scripts/dev/check_hygiene.sh
set -u
# Repo root without requiring git: a Zenodo software archive or a plain tarball has no
# .git, and every path below is repo-relative. Prefer git when it is available.
ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "${ROOT_DIR:-}" ] || ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$ROOT_DIR"

fail=0
note() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  PASS  %s\n' "$*"; }
bad()  { fail=1; printf '  FAIL  %s\n' "$*"; }

# This script names the patterns it forbids, so it excludes itself from every scan.
SELF="scripts/dev/check_hygiene.sh"
GATE="scripts/dev/check_phase2.sh"

note "1. tracked file contents"

scan() {  # scan <label> <extended-regex>
  local label="$1" pattern="$2" hits
  hits=$(git ls-files -z --cached --others --exclude-standard \
         | grep -zZv -e "^${SELF}$" -e "^${GATE}$" \
         | xargs -0 grep -InE "$pattern" 2>/dev/null)
  if [ -z "$hits" ]; then ok "$label"; else bad "$label"; printf '%s\n' "$hits" | sed 's/^/        /'; fi
}

scan "no machine-specific home paths" '/home/vferraz|/Users/[a-z]'
scan "no personal handles"            'visferraz'
scan "no API tokens"                  'HF_TOKEN|hf_[A-Za-z0-9]{20}|sk-[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16}'
scan "no .env contents"               '^[A-Z_]+_(SECRET|PASSWORD|APIKEY|API_KEY)='

note "2. tracked symlinks"
# Any tracked symlink at all is suspicious here; an absolute one is a hard failure.
links=$(git ls-files --cached --others --exclude-standard | while read -r f; do
          [ -L "$f" ] && echo "$f"
        done)
if [ -z "$links" ]; then
  ok "no tracked symlinks"
else
  absolute=$(printf '%s\n' "$links" | while read -r l; do
    t=$(readlink "$l" 2>/dev/null || true)
    if [ "${t#/}" != "$t" ]; then echo "$l -> $t"; fi
  done)
  if [ -z "$absolute" ]; then
    ok "tracked symlinks are all relative"
    printf '%s\n' "$links" | sed 's/^/        (relative) /'
  else
    bad "tracked symlink points at an absolute local path"
    printf '%s\n' "$absolute" | sed 's/^/        /'
  fi
fi

note "3. heavy data is not tracked"
heavy=$(git ls-files --cached --others --exclude-standard | grep -E '^data_heavy/|^data/human_refs/raw/(nagel|griffiths)/' || true)
if [ -z "$heavy" ]; then ok "no deposit or raw human data tracked"; else
  bad "heavy or non-redistributable data is tracked"; printf '%s\n' "$heavy" | sed 's/^/        /'; fi

note "4. no oversized tracked files"
big=$(git ls-files --cached --others --exclude-standard | while read -r f; do
        [ -f "$f" ] || continue
        s=$(wc -c <"$f" 2>/dev/null || echo 0)
        [ "$s" -gt 52428800 ] && printf '%10d  %s\n' "$s" "$f"
      done)
if [ -z "$big" ]; then ok "no tracked file over 50 MB (GitHub's warning threshold)"; else
  bad "tracked file over 50 MB"; printf '%s\n' "$big" | sed 's/^/        /'; fi

echo
if [ "$fail" -eq 0 ]; then echo "hygiene: PASS"; else echo "hygiene: FAIL"; fi
exit "$fail"
