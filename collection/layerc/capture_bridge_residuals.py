"""Layer B<->C bridge re-capture — AKATA substrate variant (dense only).

Akata-ported peer of analysis/layerC_final/capture_residuals_bridge.py: identical purpose (store the
RESIDUAL VECTORS at payoff/answer/verb positions, baseline only, so a free-direction probe can be
compared against the output-direction lens), but built on the CORRECTED Akata prompt / 4-cell cb /
region_map_akata / J-P letters — so the residuals align with the shipped lens data in
$SCA_DATA_ROOT/layerc/. Run under .venv; llama needs --chat. GPT-OSS excluded.

    python analysis/block_c/capture_residuals_bridge_akata.py --model qwen          --load_8bit
    python analysis/block_c/capture_residuals_bridge_akata.py --model qwen_instruct --load_8bit
    python analysis/block_c/capture_residuals_bridge_akata.py --model llama31_instruct --load_8bit --chat
    (CPU plumbing check, no model: add --dry-run)

Output per game: $SCA_DATA_ROOT/layerc_bridge_residuals/{model}/{game}/{resid.npy, meta.parquet, config.json, _DONE}
  resid.npy : float16 [n_rows, hidden]; meta.parquet aligns row-for-row (same columns as the A/B variant).
"""
from __future__ import annotations
import argparse, gc, json, os, shutil, socket, sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd


from collection.akata_common import (  # noqa: E402  (AKATA prompt/cb, not A/B)
    build_akata_oneshot_prompt, akata_user_question, akata_cb_grid, ANSWER_PREFIX, prompt_sha256)
from collection.oneshot_common import load_game_vec  # noqa: E402
from collection.model_setup import _setup_model, _git_commit, _now, _fsync_dir  # noqa: E402
from collection.layerc.region_map_akata import build_region_map, build_region_map_chat  # noqa: E402  (AKATA region map)
from collection.layerc.model_head import (  # noqa: E402
    load_model_head, free_model_head, letter_token_id, per_token_score_diff_opt0_minus_opt1)
from strategic_anatomy.config import game_features_csv, layerc_bridge_root, layerc_root, manifests_root, repo_root

ROOT = repo_root()
GAME_FEATURES = game_features_csv()
MANIFEST_DIR = manifests_root()
CONFIG_PATH = MANIFEST_DIR / "oneshot_config.json"
OUT_DEFAULT = layerc_bridge_root()
LAYERS_DEFAULT = [79, 40]
DENSE_MODELS = ("qwen", "qwen_instruct", "llama31_instruct")
TARGET_REGIONS = {"own_payoff", "opponent_payoff", "answer_prefix"}   # where we want residuals
CONTROL_TOKENS = {"win", "wins"}                                     # framing-verb control position


def delta1c_from_vec(vec, canon):
    """Signed canonical level-1 incentive (q=0.5), vec = [p1 4 cells, p2 4 cells]."""
    p1 = np.array(vec[:4]).reshape(2, 2)
    d1 = p1[0].mean() - p1[1].mean()
    return float(d1 if int(canon) == 0 else -d1)


def select_positions(region_rows):
    return [i for i, rr in enumerate(region_rows)
            if rr["region"] in TARGET_REGIONS or str(rr["token_str"]).strip() in CONTROL_TOKENS]


def load_games(args):
    config = json.loads(Path(args.config).read_text())
    model_cfg = config["models"][args.model]
    universe = pd.read_csv(args.games_from_manifest)
    canon = pd.read_csv(GAME_FEATURES).set_index("game_code")["canonical_action_p1"].astype(int).to_dict()
    games = universe[(universe["include_h0"]) & (universe[f"in_{args.model}"])]["game_code"].tolist()
    games = [g for g in games if canon.get(g, -1) in (0, 1)]
    if args.max_games > 0:
        games = games[: args.max_games]
    return config, model_cfg, games, canon


def run_game_resid(*, model, tokenizer, head, dev, torch, model_key, game_code, vec, canon, layers,
                   letter_ids, chat):
    Xs, metas = [], []
    d1c = delta1c_from_vec(vec, canon)
    for cb in akata_cb_grid():
        cbid, a2l = cb["counterbalance_id"], cb["action_to_letter"]
        canon_letter, noncanon_letter = a2l[int(canon)], a2l[1 - int(canon)]
        if chat:
            user = akata_user_question(list(vec), cb, player=1, iv_prefix="")
            text = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                 add_generation_prompt=True) + ANSWER_PREFIX
            region_rows = build_region_map_chat(vec, cb, canonical_action_p1=int(canon),
                                                tokenizer=tokenizer, iv_prefix="")
        else:
            text = build_akata_oneshot_prompt(list(vec), cb, player=1, iv_prefix="")
            region_rows = build_region_map(vec, cb, canonical_action_p1=int(canon),
                                           tokenizer=tokenizer, iv_prefix="")
        enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)
        attn = enc.get("attention_mask")
        attn = attn.to(dev) if attn is not None else torch.ones_like(input_ids)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
        seq_len = input_ids.shape[1]
        assert len(region_rows) == seq_len, f"region/forward mismatch {len(region_rows)}!={seq_len} {game_code} cb{cbid}"
        sel = select_positions(region_rows)
        for L in layers:
            h = out.hidden_states[L + 1][0, :, :].detach().to(torch.float32).cpu().numpy()
            score_can = per_token_score_diff_opt0_minus_opt1(
                seq_residual=h, head=head, opt0_id=letter_ids[canon_letter], opt1_id=letter_ids[noncanon_letter])
            for i in sel:
                rr = region_rows[i]; ts = str(rr["token_str"]).strip()
                Xs.append(h[i].astype(np.float16))
                metas.append(dict(model=model_key, game_code=game_code, cb_id=cbid, layer=int(L),
                                  token_index=i, token_str=rr["token_str"], region=rr["region"],
                                  is_canonical_row=bool(rr["is_canonical_row"]),
                                  is_final=bool(i == seq_len - 1),    # final pre-choice token (primary readout)
                                  digit=int(ts) if ts.isdigit() else -1, score_canonical=float(score_can[i]),
                                  canonical_action=int(canon), delta1c=d1c, incentive_sign=int(np.sign(d1c))))
        del out
    return np.vstack(Xs).astype(np.float16), pd.DataFrame(metas)


def persist(out_root, game_code, X, meta, provenance):
    final = out_root / game_code
    tmp = out_root / "_tmp" / game_code
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "resid.npy", X)
    meta.to_parquet(tmp / "meta.parquet", index=False)
    (tmp / "config.json").write_text(json.dumps(
        {**provenance, "game_code": game_code, "n_rows": int(len(meta)), "hidden": int(X.shape[1])},
        indent=2, default=str))
    if final.exists():
        shutil.rmtree(final)
    final.mkdir(parents=True, exist_ok=True)
    for name in ("resid.npy", "meta.parquet", "config.json"):
        os.replace(tmp / name, final / name)
    _fsync_dir(final); (final / "_DONE").write_text(_now()); _fsync_dir(final)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"game_code": game_code, "n_rows": int(len(meta))}


def dry_run(args, model_cfg, games, canon, layers):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_cfg["model_name"], use_fast=True, local_files_only=True)
    g0 = games[0]; vec = [int(x) for x in load_game_vec(g0)]; cb0 = akata_cb_grid()[0]
    if args.chat:
        rm = build_region_map_chat(vec, cb0, canonical_action_p1=int(canon[g0]), tokenizer=tok, iv_prefix="")
    else:
        rm = build_region_map(vec, cb0, canonical_action_p1=int(canon[g0]), tokenizer=tok, iv_prefix="")
    sel = select_positions(rm)
    info = dict(mode="dry_run", model=args.model, chat=bool(args.chat), n_games=len(games), layers=layers,
                cb_cells=len(akata_cb_grid()), example_game=g0, n_tokens=len(rm), n_selected_per_cb=len(sel),
                selected_regions=pd.Series([rm[i]["region"] for i in sel]).value_counts().to_dict(),
                rows_per_game=len(sel) * len(akata_cb_grid()) * len(layers),
                delta1c_example=round(delta1c_from_vec(vec, canon[g0]), 3))
    print(json.dumps(info, indent=2, default=str)); print(f"[bridge-akata dry-run] OK ({args.model})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=DENSE_MODELS)
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--games-from-manifest", default=str(MANIFEST_DIR / "game_universe_oneshot.csv"))
    p.add_argument("--out-root", default=str(OUT_DEFAULT))
    p.add_argument("--layers", default=",".join(map(str, LAYERS_DEFAULT)))
    p.add_argument("--max-games", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--force-replace", action="store_true")
    p.add_argument("--chat", action="store_true", help="llama-3.1-instruct requires its chat template")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--load_8bit", action="store_true")
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--bnb_compute_dtype", default="bfloat16")
    p.add_argument("--attn_impl", default="sdpa")
    p.add_argument("--use_ln_f_all", action="store_true", default=True)
    args = p.parse_args()

    layers = [int(x) for x in str(args.layers).split(",") if x != ""]
    config, model_cfg, games, canon = load_games(args)
    if not games:
        raise SystemExit("No eligible games.")
    if args.dry_run:
        dry_run(args, model_cfg, games, canon, layers); return

    out_root = Path(args.out_root) / args.model
    out_root.mkdir(parents=True, exist_ok=True)
    from strategic_anatomy.runtime import _free_gpu_cache
    torch, model, tokenizer, _mtm, dev, _mask, load_s = _setup_model(args, model_cfg)
    head = load_model_head(model_cfg["model_name"], device=str(dev))
    letter_ids = {"J": letter_token_id(tokenizer, "J"), "P": letter_token_id(tokenizer, "P")}
    provenance = dict(model=args.model, model_name=model_cfg["model_name"], git_commit=_git_commit(),
                      built_at=_now(), host=socket.gethostname(), lens_layers=layers, substrate="akata",
                      condition="baseline", target_regions=sorted(TARGET_REGIONS),
                      control_tokens=sorted(CONTROL_TOKENS), chat=bool(args.chat),
                      purpose="layerBC_bridge_probe_vs_lens_AKATA")
    print(f"[bridge-akata] {args.model} loaded {load_s:.1f}s; J/P={letter_ids}; layers={layers}; n_games={len(games)}")
    stats, errors = [], []
    try:
        for i, g in enumerate(games, 1):
            if (out_root / g / "_DONE").exists() and args.skip_existing and not args.force_replace:
                print(f"[{i}/{len(games)}] SKIP {g}"); continue
            t0 = time.time()
            try:
                vec = [int(x) for x in load_game_vec(g)]
                X, meta = run_game_resid(model=model, tokenizer=tokenizer, head=head, dev=dev, torch=torch,
                                         model_key=args.model, game_code=g, vec=vec, canon=int(canon[g]),
                                         layers=layers, letter_ids=letter_ids, chat=args.chat)
                s = persist(out_root, g, X, meta, provenance); s["elapsed_s"] = round(time.time() - t0, 1)
                stats.append(s); print(f"[{i}/{len(games)}] ok {g} ({s['elapsed_s']}s) rows={s['n_rows']}")
            except Exception as exc:
                errors.append(dict(game_code=g, error=str(exc), traceback=traceback.format_exc()))
                print(f"[{i}/{len(games)}] FAILED {g}: {exc}")
            gc.collect(); _free_gpu_cache()
    finally:
        free_model_head(head); del model; _free_gpu_cache()
    if errors:
        (out_root / "_errors.json").write_text(json.dumps(errors, indent=2)); sys.exit(1)
    print(f"[bridge-akata] done: {len(stats)} games -> {out_root}")
    print("BRIDGE_AKATA_DONE")


if __name__ == "__main__":
    main()
