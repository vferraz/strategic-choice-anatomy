# oss_migration — PROGRESS

Migration bookkeeping for `neural-llms-gametheory` → `strategic-choice-anatomy`.
**Internal.** Release plan §5 excludes operational/PI-workflow docs from the public repo — remove or
exclude this directory in the Phase 6 hygiene pass before going public.

---

## 2026-09-07 — Phase 6 Part B (GPU machine): q05 port, env pins, smokes, artifact sync

Branch **`phase6-spark`**. Executed on the DGX Spark against the private working copy at `4a05270`
(read-only source). Full detail in `oss_migration/GPU_VERIFICATION_REPORT.md`.

### Headline: `PHASE6_gpu_machine.md` Part A was wrong on its central claim

Part A concluded the q05 producing code was "not recoverable from git history" and that "the Spark
working tree is the only source. Do not delete or reset it before diffing." Verified on the Spark
after `git fetch origin`:

* `8d8370e` (the commit the q05 manifests record) **is `HEAD~2`** — an ancestor, contrary to Part A.
* It does contain no `q05` string (Part A correct); `eaded0c` contains 7.
* The manifests were written `2026-08-22T17:01:42Z`; `eaded0c` was committed `2026-08-22T17:44:37Z`
  — **42 min 55 s later** — and pushed.

The directions were extracted from a dirty tree, and the edits were committed 43 minutes afterwards.
Part A was reasoning from the Mac, which had not fetched those two commits. **The q05 tools were
never at risk**, the recovery-before-reset urgency was unfounded, and the PI-signed reimplementation
fallback was not needed.

**Action for the Mac:** amend `oss_migration/PHASE6_gpu_machine.md` — it is the canonical copy and
still carries the falsified conclusion.

### What was done

1. **Evidence snapshot + reconciliation** → `~/phase6_docs/spark_state/` (git status/stash/diff/
   ahead-behind, `ls-remote`, the Part A proof, both env freezes, `RECONCILIATION.md`).
   The working tree was **clean**; no stash; `main` == `origin/main` == `4a05270`.
2. **Branch `spark-q05-tools`** (`f18d651`) pushed to the private repo — the one untracked source
   file (`analysis/block_b/run_akata_trait_steer_test.py`) plus `SPARK_ONLY_ARTIFACTS.md`, a sha256
   inventory of the gitignored Spark-only artifacts. No tool code: none was missing.
3. **q05 port** of `eaded0c` into `steering/` (5 files, 1 new). Proof passed (§2 of the report).
4. **Env pins**: `[gpu]`/`[gptoss]` floors → exact pins; `uv.lock` regenerated; both envs rebuilt
   from the new pins and reproduce the live collection environments exactly.
5. **Artifact sync**: all plan §8.1 items present and staged to `_deposit/` with checksums; the
   empirical direction manifests added to tracked `data/manifests/directions/`.
6. **Tests**: 54/54 base suite; tier-2 **7 passed / 2 skipped** (the 2 need the Layer-B residual
   cache, which was then built — 5.2 GB — but the re-run that would have exercised them was killed);
   `--help` sweep 26/26.
7. **Bounded GPU smokes a–e: NOT RUN.** The session was stopped externally while smoke (a) was
   loading Qwen's weights, before any assertion. **The gate condition "Smokes a–c PASS" is
   therefore unmet**, and the 18/18 small-dose dose-0 bit-identity bar is unverified here. Nothing
   partial was written. Everything needed to resume is in place (both GPU envs built, all four
   checkpoints cached, symlink farm and `layer_b_cache` ready); the commands are in §6 of the
   report.

### Deviations from the written instructions

| Instruction | What was done | Why |
|---|---|---|
| PHASE6 §3: `uv venv --python 3.11` | **3.12** for `[gpu]`/`[gptoss]` | the pinned `torch 2.11.0` is a `cp312` wheel here; 3.11 cannot install the versions the science ran on. `.python-version` stays 3.11 for the analysis arm. |
| PHASE6 §7: "leave everything uncommitted", branch `gpu-verification` | committed and pushed on **`phase6-spark`** | superseded by the session brief, which explicitly authorised commits and pushes of new branches and named this branch. |
| STEP 1 brief: commit the uncommitted q05 tool edits | committed the untracked test file + artifact inventory instead | there were no uncommitted tool edits — see the headline. The briefed commit message would have been false. |
| Release plan §4.4: "no separate perp file to hunt for" | ported `build_perp_directions.py` as a new file | `eaded0c` made ℓ-orthogonalization standalone for the q05 arm; §4.4 predates it. |

### Proposed commit message (this branch)

```
phase-6: gpu verification + env pins + spark artifact sync
```

### Push commands (already run by this session)

```bash
# private repo — the preserved Spark state
git -C <old repo> push -u origin spark-q05-tools     # f18d651

# release repo — this work
git -C <this repo> push -u origin phase6-spark
```

`main` was not touched on either repository.

### Still open

* **Smokes a–e and the tier-2 re-run** — the only substantive gap in this phase. See §6 of the
  report for the exact resume commands; expect ~1.5–2 h of GPU time.

* **D1** (`SPRINT_q05_gap.md`): ship both arms' heavy data, or q05 heavy data + empirical tables
  only. Both arms are staged, so either is available. Vinicius's call.
* The Mac's Part A note needs the correction above.
* `c902bcd`, named in the session brief as the Mac tip, resolves in no reachable repository.
  Convergence commands for the Mac: `~/phase6_docs/spark_state/RECONCILIATION.md`.
* Phase 7 (Zenodo) unstarted.
