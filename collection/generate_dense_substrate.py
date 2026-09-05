"""DENSE Akata one-shot substrate capture (qwen / qwen_instruct / llama31_instruct).

ADDITIVE peer of generate_oneshot_substrate.py, repointed to the LOCKED corrected substrate
(docs/METHODS.md): Akata sentence-form prompt + 4-cell cb + **generate->parse
J/P** decode (the realized decision; slot readout demoted to a diagnostic `pref0`) + **ALL-layer
residual capture incl L0**. The original A/B-matrix entrypoint is untouched.

Per (model, game) over the 4-cell cb x 8 conditions (p1_baseline + 5 traits + placebo + p2_baseline):
  - forward (output_hidden_states) -> residual at EVERY layer hidden_states[0..n] (l0 = embedding)
  - generate(max_new_tokens, do_sample=False) -> parse_akata_move -> realized decoded_action
Outputs: $SCA_DATA_ROOT/substrate/{model}/{game}/{results.parquet, acts.npz, config.json, _DONE}
  acts keys: {key_prefix}_cb{cb}_l{L}  (L = 0..n_layers, l0 = embedding)

Crash-safe (tmp -> validate -> os.replace -> _DONE). gptoss is NOT handled here (harmony +
mixed-strategy handling lives in the gpt-oss entrypoint). Run under .venv.
"""
from __future__ import annotations
import argparse, datetime as dt, gc, json, os, shutil, socket, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd


from collection.akata_common import (  # noqa: E402
    MOVE_LABELS, ANSWER_PREFIX, akata_cb_grid, build_akata_oneshot_prompt, akata_user_question,
    parse_akata_move, prompt_sha256,
)
from collection.model_setup import (  # noqa: E402  (reuse validated helpers)
    _git_commit, _now, _sha256_file, _fsync_dir, _setup_model, resolve_cue_list,
    CONFIG_PATH, MANIFEST_DIR,
)
from collection.oneshot_common import load_game_vec  # noqa: E402
from strategic_anatomy.config import data_root, repo_root, substrate_root, traits_path

ROOT = repo_root()
LOG_ROOT = data_root() / "run_logs" / "oneshot_akata"
DEFAULT_OUT_ROOT = substrate_root()

CORE_COLS = {"model", "game_code", "player", "condition", "counterbalance_id", "label_map", "q_order",
             "prompt_hash", "move_letter", "parse_ok", "decoded_action", "decoded_label",
             "pref_J", "pref_P", "prob_act0", "prob_act1", "pref0"}


def _jp_token_ids(tok):
    return {mv: tok(" " + mv, add_special_tokens=False)["input_ids"][-1] for mv in MOVE_LABELS}


def _capture_and_decode(model, tok, dev, ens_mask, torch, jp_ids, pad, text, gen_tokens):
    """ALL-layer last-token residual + slot pref(J/P) + generate->parse the committed J/P."""
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"].to(dev)
    attn = ens_mask(ids, enc.get("attention_mask")).to(dev)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    hidden = {i: out.hidden_states[i][0, -1, :].detach().to(torch.float32).cpu().numpy()
              for i in range(len(out.hidden_states))}           # i=0 -> embedding (L0)
    probs = torch.softmax(out.logits[0, -1, :].detach(), dim=-1)
    pref = {mv: float(probs[jp_ids[mv]].item()) for mv in MOVE_LABELS}
    del out, probs
    with torch.inference_mode():
        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=gen_tokens,
                             do_sample=False, pad_token_id=pad)
    gen_txt = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
    return hidden, pref, gen_txt


def run_match(*, model, tok, dev, ens_mask, torch, jp_ids, pad, model_key, game_code, game_vec,
              cue_objs, chat, gen_tokens):
    conditions = [("p1_baseline", 1, "", "baseline")]
    conditions += [(f"p1_cue_{cid}", 1, cobj.system_prefix, f"cue_{cid}") for cid, cobj in cue_objs]
    conditions += [("p2_baseline", 2, "", "baseline")]
    rows, activ = [], {}
    for cb in akata_cb_grid():
        for key_prefix, player, iv_prefix, condition in conditions:
            if chat:
                user = akata_user_question(game_vec, cb, player=player, iv_prefix=iv_prefix)
                text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                               add_generation_prompt=True) + ANSWER_PREFIX
            else:
                text = build_akata_oneshot_prompt(game_vec, cb, player=player, iv_prefix=iv_prefix)
            hidden, pref, gen_txt = _capture_and_decode(model, tok, dev, ens_mask, torch, jp_ids,
                                                        pad, text, gen_tokens)
            mv = parse_akata_move(gen_txt)
            realized = cb["letter_to_action"][mv] if mv is not None else None
            a0, a1 = cb["action_to_letter"][0], cb["action_to_letter"][1]   # slot DIAGNOSTIC mapping
            p_act0, p_act1 = pref[a0], pref[a1]
            denom = p_act0 + p_act1
            for i, h in hidden.items():
                activ[f"{key_prefix}_cb{cb['counterbalance_id']}_l{i}"] = h.astype(np.float32)
            rows.append({
                "model": model_key, "game_code": game_code, "player": player, "condition": condition,
                "counterbalance_id": cb["counterbalance_id"], "label_map": cb["label_map"],
                "q_order": cb["q_order"], f"label_map_p{player}": cb["label_map"],
                "prompt_hash": prompt_sha256(text),
                "move_letter": (mv or ""), "parse_ok": int(mv is not None),
                "decoded_action": (int(realized) if realized is not None else -1),
                "decoded_label": (mv or ""),
                "gen_text": gen_txt.replace("\n", "\\n")[:48],
                "pref_J": pref["J"], "pref_P": pref["P"],
                "prob_act0": p_act0, "prob_act1": p_act1,
                "pref0": (p_act0 / denom) if denom > 0 else float("nan"),   # slot readout = DIAGNOSTIC
            })
    return pd.DataFrame(rows), activ


def validate_match(df, activ, n_cues):
    exp_conditions = 2 + n_cues
    assert len(df) == 4 * exp_conditions, f"rows {len(df)} != {4*exp_conditions}"
    assert not (CORE_COLS - set(df.columns)), f"missing cols {CORE_COLS - set(df.columns)}"
    assert df["decoded_action"].isin([0, 1, -1]).all(), "decoded_action not in {0,1,-1}"
    assert df["player"].isin([1, 2]).all()
    vc = df.groupby("counterbalance_id").size()
    assert len(vc) == 4 and (vc == exp_conditions).all(), f"cb balance off: {vc.to_dict()}"
    layers = {int(k.rsplit("_l", 1)[1]) for k in activ}
    assert len(activ) == 4 * exp_conditions * len(layers), \
        f"acts {len(activ)} != {4*exp_conditions*len(layers)} (layers={len(layers)})"
    return len(layers)


def persist_match(out_root, game_code, df, activ, n_cues, provenance):
    final_dir, tmp_dir = out_root / game_code, out_root / "_tmp" / game_code
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pq, npz, cfgp = tmp_dir / "results.parquet", tmp_dir / "acts.npz", tmp_dir / "config.json"
    df.to_parquet(pq, index=False)
    np.savez_compressed(npz, **activ)
    n_layers = validate_match(df, activ, n_cues)               # raises before promote on drift
    shas = {pq.name: _sha256_file(pq), npz.name: _sha256_file(npz)}
    cfgp.write_text(json.dumps({**provenance, "game_code": game_code, "n_rows": int(len(df)),
                                "n_acts_keys": int(len(activ)), "n_layers_incl_l0": int(n_layers),
                                "sha256": shas}, indent=2, default=str))
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in ("results.parquet", "acts.npz", "config.json"):
        os.replace(tmp_dir / name, final_dir / name)
    _fsync_dir(final_dir)
    (final_dir / "_DONE").write_text(_now())
    _fsync_dir(final_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"status": "ok", "game_code": game_code, "n_rows": int(len(df)),
            "n_acts_keys": int(len(activ)), "n_layers": int(n_layers)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("qwen", "qwen_instruct", "llama31_instruct"))
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--games-from-manifest", default=str(MANIFEST_DIR / "game_universe_oneshot.csv"))
    p.add_argument("--out-root", default="")          # default DEFAULT_OUT_ROOT/{model}
    p.add_argument("--traits-file", default="")
    p.add_argument("--n-games", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--force-replace", action="store_true")
    p.add_argument("--chat", action="store_true", help="llama31_instruct requires this")
    p.add_argument("--gen-tokens", type=int, default=12)
    p.add_argument("--attn_impl", default="sdpa")
    args = p.parse_args()

    config = json.loads(Path(args.config).read_text())
    model_cfg = config["models"][args.model]
    cue_list = resolve_cue_list(config)
    universe = pd.read_csv(args.games_from_manifest)
    games = universe[(universe["include_h0"]) & (universe[f"in_{args.model}"])]["game_code"].tolist()
    if args.n_games > 0:
        games = games[: args.n_games]
    if not games:
        raise SystemExit("No eligible games (include_h0 & in_{model}).")
    pm = int(config.get("payoff_multiplier", 1))
    traits_file = args.traits_file or str(traits_path(config["traits_file"]))
    out_root = Path(args.out_root) if args.out_root else (DEFAULT_OUT_ROOT / args.model)

    from strategic_anatomy.prompting import load_traits

    from strategic_anatomy.runtime import _free_gpu_cache
    traits_all = load_traits(traits_file)
    cue_objs = []
    for cid in cue_list:
        if cid not in traits_all:
            raise SystemExit(f"Cue '{cid}' not in {traits_file}")
        cue_objs.append((cid, traits_all[cid]))
    n_cues = len(cue_objs)

    out_root.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    torch, model, tok, _ab_mtm, dev, ens_mask, load_s = _setup_model(args, model_cfg)
    model.eval()
    jp_ids = _jp_token_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"[akata dense] model={args.model} loaded in {load_s:.0f}s; J/P ids={jp_ids}; "
          f"n_games={len(games)} n_cues={n_cues} cb=4 pm={pm} chat={args.chat} out={out_root}",
          flush=True)
    provenance = {
        "substrate": "akata_oneshot", "model": args.model, "model_name": model_cfg["model_name"],
        "git_commit": _git_commit(), "built_at": _now(), "host": socket.gethostname(),
        "cue_list": cue_list, "payoff_multiplier": pm, "chat": bool(args.chat),
        "decoder": "generate_parse_JP", "answer_prefix": ANSWER_PREFIX, "cb": "akata_4cell",
        "capture": "all_layers_incl_l0", "gen_tokens": int(args.gen_tokens),
        "spec": "docs/AKATA_ONESHOT_RECOLLECTION.md",
    }
    stats, errors = [], []
    try:
        for i, game_code in enumerate(games, 1):
            done = out_root / game_code / "_DONE"
            if done.exists() and args.skip_existing and not args.force_replace:
                print(f"[{i}/{len(games)}] SKIP {game_code} (_DONE)")
                stats.append({"status": "skipped", "game_code": game_code}); continue
            t0 = time.time()
            try:
                vec = [int(x) * pm for x in load_game_vec(game_code)]
                df, activ = run_match(model=model, tok=tok, dev=dev, ens_mask=ens_mask, torch=torch,
                                      jp_ids=jp_ids, pad=pad, model_key=args.model,
                                      game_code=game_code, game_vec=vec, cue_objs=cue_objs,
                                      chat=args.chat, gen_tokens=args.gen_tokens)
                s = persist_match(out_root, game_code, df, activ, n_cues, provenance)
                s["elapsed_s"] = round(time.time() - t0, 1)
                s["parse_rate"] = round(float((df["parse_ok"] == 1).mean()), 4)
                stats.append(s)
                print(f"[{i}/{len(games)}] ok {game_code} ({s['elapsed_s']}s) rows={s['n_rows']} "
                      f"acts={s['n_acts_keys']} layers={s['n_layers']} parse={s['parse_rate']}", flush=True)
            except Exception as exc:
                errors.append({"game_code": game_code, "error": str(exc),
                               "traceback": traceback.format_exc()})
                stats.append({"status": "error", "game_code": game_code})
                print(f"[{i}/{len(games)}] FAILED {game_code}: {exc}", flush=True)
            gc.collect(); _free_gpu_cache()
    finally:
        del model; _free_gpu_cache()

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(stats).to_csv(LOG_ROOT / f"akata_capture_{args.model}_{ts}.csv", index=False)
    if errors:
        (LOG_ROOT / f"akata_errors_{args.model}_{ts}.json").write_text(json.dumps(errors, indent=2))
    inv = {"substrate": "akata_oneshot", "model": args.model,
           "status": "READY" if not errors else "READY_WITH_ERRORS", "n_games": len(games),
           "n_done": sum(1 for s in stats if s.get("status") == "ok"),
           "n_skipped": sum(1 for s in stats if s.get("status") == "skipped"),
           "n_errors": len(errors), "out_root": str(out_root), "built_at": _now()}
    (MANIFEST_DIR / f"oneshot_akata_substrate_{args.model}.json").write_text(json.dumps(inv, indent=2))
    print(f"[akata dense] {inv['status']}: {inv['n_done']} done, {inv['n_skipped']} skipped, "
          f"{inv['n_errors']} errors -> {out_root}", flush=True)
    print("AKATA_DENSE_DONE", flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
