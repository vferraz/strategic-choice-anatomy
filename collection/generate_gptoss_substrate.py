"""PRODUCTION gpt-oss Akata substrate capture (harmony arm).

Per docs/AKATA_ONESHOT_RECOLLECTION.md + the locked gpt-oss handling (AKATA_PREFLIGHT_HANDOVER.md):
  - harmony CHAT TEMPLATE (game text verbatim in the user turn) + PURE GREEDY (no penalty) -> NO loop
  - per-cell classify the harmony final channel: pure / mixed (strategic non-commit) / none
  - capture position: pure -> commit letter; mixed -> analysis->final TRANSITION; none -> no capture
  - all-layer residual (incl L0) + MoE router (capture_moe_router) at the capture token
  - mixed: stated_p(J) -> SEEDED realized action; raw final-channel text ALWAYS stored
Outputs: $SCA_DATA_ROOT/substrate/gptoss/{game}/{results.parquet, acts.npz, router.npz, config.json, _DONE}
Run under .venv_gptoss (kernels). gpt-oss only.
"""
from __future__ import annotations
import argparse, datetime as dt, gc, hashlib, json, os, re, shutil, socket, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd


from collection.akata_common import (  # noqa: E402
    MOVE_LABELS, ANSWER_PREFIX, akata_cb_grid, build_akata_oneshot_prompt, parse_akata_move, prompt_sha256,
)
from collection.preflight_gptoss import HARMONY_TAG  # noqa: E402
from collection.genutils import _generate, HARMONY_FINAL_MARKER  # noqa: E402
from collection.model_setup import (  # noqa: E402
    _git_commit, _now, _sha256_file, _fsync_dir, _setup_model, resolve_cue_list, CONFIG_PATH, MANIFEST_DIR,
)
from collection.model_setup import ROUTER_LAYERS_DEFAULT  # noqa: E402
from collection.oneshot_common import load_game_vec  # noqa: E402
from strategic_anatomy.router_capture import capture_moe_router, _topk_idx_weights, model_topk, _kernels_available  # noqa: E402
from strategic_anatomy.config import data_root, repo_root, substrate_root, traits_path

ROOT = repo_root()
LOG_ROOT = data_root() / "run_logs" / "oneshot_akata"
DEFAULT_OUT_ROOT = substrate_root()

# broadened: explicit mixing OR strategic non-commitment (gpt-oss + Qwen3.6 probe phrasings)
NONCOMMIT_RE = re.compile(
    r"randomi[sz]e|\bmix(?:ed|ing)?\b|each with|with probability|% of the time|half the time|"
    r"50\s*[/\-]\s*50|fifty[\s-]?fifty|p\(\s*[JP]\s*\)|equal probab|"
    r"no pure|no dominant|neither (?:option |is )?(?:strictly )?(?:better|dominant|optimal)|"
    r"depends on (?:the |what )?(?:other|opponent)|anti-?coordinat|indifferen|"
    r"there is no (?:single |one )?(?:best|right|correct|dominant)", re.I)


def extract_stated_p_J(text: str):
    """Best-effort stated P(Option J). 0.5 for equal/50-50; X% near 'Option J'; else None."""
    if re.search(r"50\s*[/\-]\s*50|fifty[\s-]?fifty|equal probab|half the time", text, re.I):
        return 0.5
    m = re.search(r"(\d{1,3})\s*%[^.]{0,25}?Option\s*J", text, re.I) or \
        re.search(r"Option\s*J[^.]{0,25}?(\d{1,3})\s*%", text, re.I)
    if m:
        v = int(m.group(1)) / 100.0
        return v if 0.0 <= v <= 1.0 else None
    return None


def classify(final_text: str):
    """(commit_type, move_letter, stated_p_J) — non-commit checked FIRST (stops false letter-grab)."""
    if NONCOMMIT_RE.search(final_text):
        return "mixed", None, extract_stated_p_J(final_text)
    mv = parse_akata_move(final_text)
    return ("pure", mv, None) if mv else ("none", None, None)


_JP_RE = {"J": re.compile(r"(?<![A-Za-z])J(?![A-Za-z])"), "P": re.compile(r"(?<![A-Za-z])P(?![A-Za-z])")}


def find_commit_index(new_ids, tok, label):
    text, seen, rx = "", False, _JP_RE.get(label)
    for i, tid in enumerate(new_ids.tolist()):
        piece = tok.decode([int(tid)], skip_special_tokens=False)
        text += piece
        if not seen and HARMONY_FINAL_MARKER in text:
            seen = True
        if seen and rx is not None and rx.search(piece):
            return i
    return None


def find_transition_index(new_ids, tok):
    """Token index at which the analysis->final HARMONY marker completes (mixed capture position)."""
    text = ""
    for i, tid in enumerate(new_ids.tolist()):
        text += tok.decode([int(tid)], skip_special_tokens=False)
        if HARMONY_FINAL_MARKER in text:
            return i
    return None


def _seeded_action(game, cb_id, player, p_act0):
    seed = int(hashlib.sha256(f"{game}_{cb_id}_{player}".encode()).hexdigest()[:8], 16)
    return 0 if np.random.RandomState(seed).random() < p_act0 else 1


def _capture(model, tok, dev, ens_mask, torch, full_ids, router_layers, jp_ids, top_k):
    ids = full_ids.unsqueeze(0).to(dev) if full_ids.ndim == 1 else full_ids.to(dev)
    attn = ens_mask(ids, None).to(dev) if ens_mask is not None else torch.ones_like(ids)
    with torch.inference_mode():
        with capture_moe_router(model, router_layers) as cap:
            out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    resid = {i: out.hidden_states[i][0, -1, :].detach().to(torch.float32).cpu().numpy()
             for i in range(len(out.hidden_states))}                  # all layers incl L0
    router = {}
    for L in router_layers:
        L = int(L)
        gl = np.asarray(cap[L][-1, :], dtype=np.float32)
        idx, w = _topk_idx_weights(gl[None, :], top_k)
        router[L] = (gl, idx[0].astype(np.int32), w[0].astype(np.float32))
    probs = torch.softmax(out.logits[0, -1, :].detach(), dim=-1)
    pref = {mv: float(probs[jp_ids[mv]].item()) for mv in MOVE_LABELS}
    return resid, router, pref


def run_match(*, model, tok, dev, ens_mask, torch, jp_ids, top_k, model_key, game_code, game_vec,
              cue_objs, router_layers, max_new_tokens):
    conditions = [("p1_baseline", 1, "", "baseline")]
    conditions += [(f"p1_cue_{cid}", 1, cobj.system_prefix, f"cue_{cid}") for cid, cobj in cue_objs]
    conditions += [("p2_baseline", 2, "", "baseline")]
    rows, resid_store, router_store = [], {}, {}
    for cb in akata_cb_grid():
        cbid = cb["counterbalance_id"]
        for key_prefix, player, iv_prefix, condition in conditions:
            raw = build_akata_oneshot_prompt(game_vec, cb, player=player, iv_prefix=iv_prefix)
            prompt = tok.apply_chat_template([{"role": "user", "content": raw}], tokenize=False,
                                             add_generation_prompt=True)               # harmony framing
            enc = tok(prompt, add_special_tokens=False, return_tensors="pt")
            ids = enc["input_ids"].to(dev)
            attn = ens_mask(ids, enc.get("attention_mask")).to(dev) if ens_mask is not None else torch.ones_like(ids)
            t0 = time.time()
            new_ids = _generate(model, tok, ids, attn, max_new_tokens, False, torch)   # PURE GREEDY, no stopper
            withspec = tok.decode(new_ids, skip_special_tokens=False)
            final_present = HARMONY_FINAL_MARKER in withspec
            final_text = HARMONY_TAG.sub("", withspec.split(HARMONY_FINAL_MARKER)[-1]) if final_present else ""
            ctype, mv, p_J = classify(final_text) if final_present else ("none", None, None)
            if ctype == "pure" and mv is not None:
                cidx = find_commit_index(new_ids, tok, mv)
                cap_ids = None if cidx is None else torch.cat([ids[0], new_ids[:cidx]], dim=0)
            elif ctype == "mixed":
                tidx = find_transition_index(new_ids, tok)
                cidx = tidx
                cap_ids = None if tidx is None else torch.cat([ids[0], new_ids[:tidx + 1]], dim=0)
            else:
                cidx, cap_ids = None, None
            captured = cap_ids is not None
            if captured:
                resid, router, pref = _capture(model, tok, dev, ens_mask, torch, cap_ids,
                                               router_layers, jp_ids, top_k)
                for i, h in resid.items():
                    resid_store[f"{key_prefix}_cb{cbid}_l{i}"] = h.astype(np.float32)
                for L, (gl, idx, w) in router.items():
                    router_store[f"{key_prefix}_cb{cbid}_l{L}_gate"] = gl
                    router_store[f"{key_prefix}_cb{cbid}_l{L}_idx"] = idx
                    router_store[f"{key_prefix}_cb{cbid}_l{L}_w"] = w
                pref_J = pref["J"]
            else:
                pref_J = float("nan")
            # realized action
            if ctype == "pure" and mv is not None:
                realized, prob_source, stated_p0 = cb["letter_to_action"][mv], "pure", float("nan")
            elif ctype == "mixed":
                pJ = p_J if p_J is not None else 0.5
                p_act0 = pJ if cb["letter_to_action"]["J"] == 0 else (1.0 - pJ)
                realized = _seeded_action(game_code, cbid, player, p_act0)
                prob_source, stated_p0 = ("stated" if p_J is not None else "default_uniform"), p_act0
            else:
                realized, prob_source, stated_p0 = -1, "NA", float("nan")
            rows.append({
                "model": model_key, "game_code": game_code, "player": player, "condition": condition,
                "counterbalance_id": cbid, "label_map": cb["label_map"], "q_order": cb["q_order"],
                f"label_map_p{player}": cb["label_map"], "prompt_hash": prompt_sha256(prompt),
                "commit_type": ctype, "move_letter": (mv or ""),
                "decoded_action": int(realized), "realized_action": int(realized),
                "stated_p_act0": stated_p0, "prob_source": prob_source,
                "capture_idx": (int(ids.shape[1] + cidx) if cidx is not None else -1),
                "captured": int(captured), "final_marker": int(final_present),
                "n_new_tokens": int(new_ids.shape[0]), "slot_pref_J": pref_J,
                "final_text": final_text.replace("\n", "\\n")[:600],   # RAW answer (re-classifiable)
                "gen_s": round(time.time() - t0, 1),
            })
            print(f"    {game_code} cb{cbid} {key_prefix}: {ctype}{('/'+mv) if mv else ''} "
                  f"act={realized} cap={captured} ntok={int(new_ids.shape[0])} {round(time.time()-t0,0)}s",
                  flush=True)
    return pd.DataFrame(rows), resid_store, router_store


def validate_match(df, resid, n_cues):
    exp = 4 * (2 + n_cues)
    assert len(df) == exp, f"rows {len(df)} != {exp}"
    assert df["decoded_action"].isin([0, 1, -1]).all()
    assert df["player"].isin([1, 2]).all()
    vc = df.groupby("counterbalance_id").size()
    assert len(vc) == 4 and (vc == (2 + n_cues)).all(), f"cb balance {vc.to_dict()}"
    layers = {int(k.rsplit("_l", 1)[1]) for k in resid} if resid else set()
    return len(layers)


def persist_match(out_root, game_code, df, resid, router, n_cues, provenance):
    final_dir, tmp_dir = out_root / game_code, out_root / "_tmp" / game_code
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pq, npz, rnpz, cfgp = (tmp_dir / "results.parquet", tmp_dir / "acts.npz",
                           tmp_dir / "router.npz", tmp_dir / "config.json")
    df.to_parquet(pq, index=False)
    np.savez_compressed(npz, **resid)
    np.savez_compressed(rnpz, **router)
    n_layers = validate_match(df, resid, n_cues)
    shas = {pq.name: _sha256_file(pq), npz.name: _sha256_file(npz), rnpz.name: _sha256_file(rnpz)}
    cfgp.write_text(json.dumps({**provenance, "game_code": game_code, "n_rows": int(len(df)),
                                "n_resid_keys": int(len(resid)), "n_router_keys": int(len(router)),
                                "n_layers_incl_l0": int(n_layers),
                                "n_captured": int((df["captured"] == 1).sum()), "sha256": shas},
                               indent=2, default=str))
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in ("results.parquet", "acts.npz", "router.npz", "config.json"):
        os.replace(tmp_dir / name, final_dir / name)
    _fsync_dir(final_dir)
    (final_dir / "_DONE").write_text(_now())
    _fsync_dir(final_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"status": "ok", "game_code": game_code, "n_rows": int(len(df)),
            "n_resid_keys": int(len(resid)), "n_router_keys": int(len(router)),
            "n_captured": int((df["captured"] == 1).sum()),
            "commit_types": df["commit_type"].value_counts().to_dict()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--games-from-manifest", default=str(MANIFEST_DIR / "game_universe_oneshot.csv"))
    p.add_argument("--out-root", default="")
    p.add_argument("--traits-file", default="")
    p.add_argument("--n-games", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--force-replace", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--router-layers", default=",".join(str(x) for x in ROUTER_LAYERS_DEFAULT))
    p.add_argument("--attn_impl", default="sdpa")
    args = p.parse_args()
    if not _kernels_available():
        raise SystemExit("kernels missing — run under .venv_gptoss (CLAUDE.md #7).")

    config = json.loads(Path(args.config).read_text())
    model_cfg = config["models"]["gptoss"]
    cue_list = resolve_cue_list(config)
    universe = pd.read_csv(args.games_from_manifest)
    games = universe[(universe["include_h0"]) & (universe["in_gptoss"])]["game_code"].tolist()
    if args.n_games > 0:
        games = games[: args.n_games]
    if not games:
        raise SystemExit("No eligible games (include_h0 & in_gptoss).")
    pm = int(config.get("payoff_multiplier", 1))
    router_layers = [int(x) for x in str(args.router_layers).split(",") if x != ""]
    traits_file = args.traits_file or str(traits_path(config["traits_file"]))
    out_root = Path(args.out_root) if args.out_root else (DEFAULT_OUT_ROOT / "gptoss")

    from strategic_anatomy.prompting import load_traits

    from strategic_anatomy.runtime import _free_gpu_cache
    traits_all = load_traits(traits_file)
    cue_objs = []
    for cid in cue_list:
        if cid not in traits_all:
            raise SystemExit(f"Cue '{cid}' not in {traits_file}")
        cue_objs.append((cid, traits_all[cid]))
    n_cues = len(cue_objs)

    args.load_8bit = False
    args.load_4bit = False
    out_root.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    torch, model, tok, _ab, dev, ens_mask, load_s = _setup_model(args, model_cfg)
    model.eval()
    top_k = model_topk(model, None)
    jp_ids = {mv: tok(" " + mv, add_special_tokens=False)["input_ids"][-1] for mv in MOVE_LABELS}
    print(f"[akata gptoss] loaded {load_s:.0f}s; J/P ids={jp_ids}; n_games={len(games)} n_cues={n_cues} "
          f"router_layers={len(router_layers)} top_k={top_k} max_new_tokens={args.max_new_tokens} out={out_root}",
          flush=True)
    provenance = {
        "substrate": "akata_oneshot", "model": "gptoss", "model_name": model_cfg["model_name"],
        "git_commit": _git_commit(), "built_at": _now(), "host": socket.gethostname(),
        "cue_list": cue_list, "payoff_multiplier": pm, "feeding": "harmony_chat_template_pure_greedy",
        "decoder": "generate_parse_JP_per_cell_pure_mixed_none", "answer_prefix": ANSWER_PREFIX,
        "cb": "akata_4cell", "capture": "all_layers_incl_l0", "router_layers": router_layers,
        "max_new_tokens": int(args.max_new_tokens), "spec": "docs/AKATA_ONESHOT_RECOLLECTION.md",
    }
    stats, errors = [], []
    try:
        for i, game_code in enumerate(games, 1):
            done = out_root / game_code / "_DONE"
            if done.exists() and args.skip_existing and not args.force_replace:
                print(f"[{i}/{len(games)}] SKIP {game_code}"); stats.append({"status": "skipped", "game_code": game_code}); continue
            t0 = time.time()
            try:
                vec = [int(x) * pm for x in load_game_vec(game_code)]
                df, resid, router = run_match(model=model, tok=tok, dev=dev, ens_mask=ens_mask, torch=torch,
                                              jp_ids=jp_ids, top_k=top_k, model_key="gptoss",
                                              game_code=game_code, game_vec=vec, cue_objs=cue_objs,
                                              router_layers=router_layers, max_new_tokens=args.max_new_tokens)
                s = persist_match(out_root, game_code, df, resid, router, n_cues, provenance)
                s["elapsed_s"] = round(time.time() - t0, 1)
                stats.append(s)
                print(f"[{i}/{len(games)}] ok {game_code} ({s['elapsed_s']}s) rows={s['n_rows']} "
                      f"captured={s['n_captured']} types={s['commit_types']}", flush=True)
            except Exception as exc:
                errors.append({"game_code": game_code, "error": str(exc), "traceback": traceback.format_exc()})
                stats.append({"status": "error", "game_code": game_code})
                print(f"[{i}/{len(games)}] FAILED {game_code}: {exc}", flush=True)
            gc.collect(); _free_gpu_cache()
    finally:
        del model; _free_gpu_cache()

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(stats).to_csv(LOG_ROOT / f"akata_gptoss_{ts}.csv", index=False)
    if errors:
        (LOG_ROOT / f"akata_gptoss_errors_{ts}.json").write_text(json.dumps(errors, indent=2))
    inv = {"substrate": "akata_oneshot", "model": "gptoss",
           "status": "READY" if not errors else "READY_WITH_ERRORS", "n_games": len(games),
           "n_done": sum(1 for s in stats if s.get("status") == "ok"),
           "n_skipped": sum(1 for s in stats if s.get("status") == "skipped"),
           "n_errors": len(errors), "out_root": str(out_root), "built_at": _now()}
    (MANIFEST_DIR / "oneshot_akata_substrate_gptoss.json").write_text(json.dumps(inv, indent=2))
    print(f"[akata gptoss] {inv['status']}: {inv['n_done']} done, {inv['n_skipped']} skipped, "
          f"{inv['n_errors']} errors", flush=True)
    print("AKATA_GPTOSS_DONE", flush=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
