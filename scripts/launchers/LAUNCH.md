# DESIGN_V2 Phase 2 Launch

This launcher runs the remaining 76 strategic-type games into `output/design_v2_main/` and skips complete Phase 1 matches through `prompt_template_version=design_v2_v1`.

## Preflight

- Confirm `.venv/bin/python3` and `.venv_gptoss/bin/python3` exist.
- Confirm the GPU is free for a long run.
- Confirm available disk is above 300 GB; saved activations are large.
- Run `bash -n scripts/experiment1/run_design_v2_phase2.sh`.
- Run `.venv/bin/python3 scripts/experiment1/build_phase2_game_list.py` and confirm it prints 76 comma-separated game codes.

## Launch

```bash
mkdir -p validation_logs
nohup setsid scripts/experiment1/run_design_v2_phase2.sh \
    > _legacy/validation_logs/design_v2_phase2_launcher.out 2>&1 < /dev/null &
disown
```

The script creates `_legacy/validation_logs/design_v2_main_phase2_<YYYYMMDD>/` at launch time.

## Monitor

```bash
tail -f _legacy/validation_logs/design_v2_main_phase2_<YYYYMMDD>/sweep_*.log
```

Each model writes one sweep log and one `format_decisions_<model>.csv`. The capability axis lands under `_legacy/validation_logs/design_v2_main_phase2_<YYYYMMDD>/capability/<model>/capability_axis.parquet`.

## Resume

Rerun the same command. Existing complete matches with `prompt_template_version=design_v2_v1` and complete `acts.npz` archives are skipped. Template-version mismatches are refused unless the runner is invoked manually with `--force_overwrite_template_mismatch`.
