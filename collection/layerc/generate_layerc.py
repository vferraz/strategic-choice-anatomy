"""Layer C token-attribution capture on the CORRECTED **Akata** substrate (dense only, J/P labels).

Faithful port of generate_oneshot_layerc.py: same per-token logit-lens score (final-norm + lm_head,
canonical re-scored per cb) at the locked lens layers, only adapted to the Akata sentence-form prompt
+ J/P letters + 4-cell cb + region_map_akata. gpt-oss excluded (lens does not read its commit token).
Output: $SCA_DATA_ROOT/layerc/{model}/{game}/{tokens.parquet, config.json, _DONE}. Run under .venv.
"""
from __future__ import annotations
import argparse, json, shutil, socket, sys, time, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd

from collection.model_setup import (  # noqa: E402
    _setup_model, _git_commit, _now, _sha256_file, _fsync_dir, resolve_cue_list, CONFIG_PATH, MANIFEST_DIR)
from collection.layerc.model_head import (  # noqa: E402
    load_model_head, free_model_head, letter_token_id, per_token_score_diff_opt0_minus_opt1)
from collection.layerc.layerc_spec import (  # noqa: E402  (reuse locked constants/structure)
    LAYERS_DEFAULT, DENSE_MODELS, CUE_TO_TRAITCOL, ROW_COLS, _conditions)
from collection.layerc.region_map_akata import build_region_map, build_region_map_chat  # noqa: E402
from collection.akata_common import (  # noqa: E402
    build_akata_oneshot_prompt, akata_user_question, akata_cb_grid, prompt_sha256, ANSWER_PREFIX)
from collection.oneshot_common import load_game_vec  # noqa: E402
from analysis.layer_a.trait_steering_proper import trait_targets  # noqa: E402
from strategic_anatomy.config import data_root, game_features_csv, layerc_root, repo_root, traits_path

ROOT = repo_root()
OUT_ROOT_DEFAULT = layerc_root()
LOG_ROOT = data_root() / "run_logs" / "oneshot_akata_layerc"
GAME_FEATURES = game_features_csv()


def _letter_ids(tokenizer):
    return {"J": letter_token_id(tokenizer, "J"), "P": letter_token_id(tokenizer, "P")}


def run_game(*, model, tokenizer, head, input_device, torch, model_key, game_code, game_vec,
             canon, conditions, layers, tt_row, letter_ids, chat=False):
    rows = []
    for cb in akata_cb_grid():
        cbid = cb["counterbalance_id"]
        a2l = cb["action_to_letter"]
        canon_letter, noncanon_letter = a2l[int(canon)], a2l[1 - int(canon)]
        for key_prefix, iv_prefix, condition, cid in conditions:
            if chat:
                user = akata_user_question(list(game_vec), cb, player=1, iv_prefix=iv_prefix)
                text = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                     add_generation_prompt=True) + ANSWER_PREFIX
                region_rows = build_region_map_chat(game_vec, cb, canonical_action_p1=int(canon),
                                                    tokenizer=tokenizer, iv_prefix=iv_prefix)
            else:
                text = build_akata_oneshot_prompt(list(game_vec), cb, player=1, iv_prefix=iv_prefix)
                region_rows = build_region_map(game_vec, cb, canonical_action_p1=int(canon),
                                               tokenizer=tokenizer, iv_prefix=iv_prefix)
            ph = prompt_sha256(text)
            enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")
            input_ids = enc["input_ids"].to(input_device)
            attn = enc.get("attention_mask")
            attn = attn.to(input_device) if attn is not None else torch.ones_like(input_ids)
            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
            seq_len = input_ids.shape[1]
            assert len(region_rows) == seq_len, \
                f"region/forward mismatch {len(region_rows)}!={seq_len} ({game_code} cb{cbid} {condition})"
            tt_target_id = tt_other_id = None
            if cid in CUE_TO_TRAITCOL and tt_row is not None:
                raw_ta = tt_row[f"a_{CUE_TO_TRAITCOL[cid]}"]
                if pd.notna(raw_ta):
                    ta = int(raw_ta)
                    tt_target_id, tt_other_id = letter_ids[a2l[ta]], letter_ids[a2l[1 - ta]]
            for L in layers:
                h_seq = out.hidden_states[L + 1][0, :, :].detach().to(torch.float32).cpu().numpy()
                score_can = per_token_score_diff_opt0_minus_opt1(
                    seq_residual=h_seq, head=head,
                    opt0_id=letter_ids[canon_letter], opt1_id=letter_ids[noncanon_letter])
                score_tt = (per_token_score_diff_opt0_minus_opt1(seq_residual=h_seq, head=head,
                            opt0_id=tt_target_id, opt1_id=tt_other_id)
                            if tt_target_id is not None else np.full(seq_len, np.nan, dtype=np.float32))
                for i, rr in enumerate(region_rows):
                    rows.append({
                        "model": model_key, "game_code": game_code, "cb_id": cbid, "condition": condition,
                        "layer": int(L), "token_index": i, "token_str": rr["token_str"],
                        "char_start": rr["char_start"], "char_end": rr["char_end"], "region": rr["region"],
                        "is_canonical_row": bool(rr["is_canonical_row"]),
                        "score_canonical": float(score_can[i]), "score_trait_target": float(score_tt[i]),
                        "canonical_action_letter": canon_letter, "prompt_hash": ph})
            del out
    return pd.DataFrame(rows)


def persist_game(out_root, game_code, df, layers, n_conditions, provenance):
    final_dir, tmp_dir = out_root / game_code, out_root / "_tmp" / game_code
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pq, cfgp = tmp_dir / "tokens.parquet", tmp_dir / "config.json"
    assert set(ROW_COLS).issubset(df.columns), f"missing cols {set(ROW_COLS) - set(df.columns)}"
    assert df["score_canonical"].notna().all(), "null score_canonical"
    n_groups = df.groupby(["cb_id", "condition", "layer"]).ngroups
    assert n_groups == 4 * n_conditions * len(layers), f"groups {n_groups} != {4*n_conditions*len(layers)}"
    assert (df["region"] != "other").all(), "unmapped tokens (region=other)"
    df.to_parquet(pq, index=False)
    cfgp.write_text(json.dumps({**provenance, "game_code": game_code, "n_rows": int(len(df)),
                                "sha256": {pq.name: _sha256_file(pq)}}, indent=2, default=str))
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tokens.parquet", "config.json"):
        (tmp_dir / name).replace(final_dir / name)
    _fsync_dir(final_dir)
    (final_dir / "_DONE").write_text(_now())
    _fsync_dir(final_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"status": "ok", "game_code": game_code, "n_rows": int(len(df))}


def main():
    import types as _t
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=DENSE_MODELS)
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--games-from-manifest", default=str(MANIFEST_DIR / "game_universe_oneshot.csv"))
    p.add_argument("--out-root", default=str(OUT_ROOT_DEFAULT))
    p.add_argument("--layers", default=",".join(map(str, LAYERS_DEFAULT)))
    p.add_argument("--max-games", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    layers = [int(x) for x in str(args.layers).split(",") if x != ""]

    config = json.loads(Path(args.config).read_text())
    model_cfg = config["models"][args.model]
    cue_list = resolve_cue_list(config)
    universe = pd.read_csv(args.games_from_manifest)
    gf = pd.read_csv(GAME_FEATURES).set_index("game_code")
    canon = gf["canonical_action_p1"].astype(int).to_dict()
    games = universe[(universe["include_h0"]) & (universe[f"in_{args.model}"])]["game_code"].tolist()
    games = [g for g in games if canon.get(g, -1) in (0, 1)]
    if args.max_games > 0:
        games = games[: args.max_games]
    from strategic_anatomy.prompting import load_traits
    traits_all = load_traits(str(traits_path(config["traits_file"])))
    cue_objs = [(cid, traits_all[cid]) for cid in cue_list]
    conditions = _conditions(cue_objs)
    tt = trait_targets().set_index("game_code") if callable(trait_targets) else None

    if args.dry_run:
        print(json.dumps({"model": args.model, "n_games": len(games), "layers": layers,
                          "n_conditions": len(conditions), "cb": 4,
                          "rows_per_game_estimate": "4*conditions*layers*seq_len", "chat": args.chat,
                          "out_root": str(Path(args.out_root) / args.model)}, indent=2))
        return

    out_root = Path(args.out_root) / args.model
    out_root.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    margs = _t.SimpleNamespace(attn_impl="sdpa")
    torch, model, tok, _mtm, dev, _ens, load_s = _setup_model(margs, model_cfg)
    model.eval()
    head = load_model_head(model_cfg["model_name"], device=str(dev))
    letter_ids = _letter_ids(tok)
    print(f"[akata layerc] {args.model} loaded {load_s:.0f}s; J/P ids={letter_ids}; layers={layers}; "
          f"{len(games)} games; conditions={len(conditions)} chat={args.chat}", flush=True)
    provenance = {"substrate": "akata_oneshot_layerc", "model": args.model, "git_commit": _git_commit(),
                  "built_at": _now(), "host": socket.gethostname(), "layers": layers, "chat": bool(args.chat),
                  "lens": "final_norm+lm_head_per_token_canonical_JP"}
    stats = []
    for i, g in enumerate(games, 1):
        if (out_root / g / "_DONE").exists() and args.skip_existing:
            stats.append({"status": "skipped", "game_code": g}); continue
        t0 = time.time()
        tt_row = tt.loc[g] if (tt is not None and g in tt.index) else None
        df = run_game(model=model, tokenizer=tok, head=head, input_device=dev, torch=torch,
                      model_key=args.model, game_code=g, game_vec=[int(x) for x in load_game_vec(g)],
                      canon=canon[g], conditions=conditions, layers=layers, tt_row=tt_row,
                      letter_ids=letter_ids, chat=args.chat)
        s = persist_game(out_root, g, df, layers, len(conditions), provenance)
        s["elapsed_s"] = round(time.time() - t0, 1); stats.append(s)
        print(f"[{i}/{len(games)}] ok {g} ({s['elapsed_s']}s) rows={s['n_rows']}", flush=True)
    free_model_head(head)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(stats).to_csv(LOG_ROOT / f"akata_layerc_{args.model}_{ts}.csv", index=False)
    print(f"[akata layerc] {args.model}: {sum(1 for s in stats if s.get('status')=='ok')} done", flush=True)
    print("AKATA_LAYERC_DONE", flush=True)


if __name__ == "__main__":
    main()
