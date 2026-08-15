"""PRODUCTION Akata dense steering — directional modes on the locked 54-game stratified sample.

Reuses the SMOKE-VALIDATED core (akata_steer_core: Option-A inject + dual readout). Per
(mode, game[wave order], cb, steer-layer, direction-variant, dose): inject the scaled direction at the
'A: Option' slot and record BOTH readouts — headline=regenerated J/P decision, diagnostic=slot pref.
Modes (each with random + wrong_layer controls):
  h0       d_gate (unembedding action dir, per-prompt)   apparatus check, small doses
  h1_dinc  d_inc          incentive axis
  h2_choice d_choice_perp choice axis (⟂ d_inc)
  h3_oppinc d_opp         opponent-incentive axis
NOT_ESTIMABLE directions (zero-norm) are skipped (recorded as status). Resumable per (mode) parquet.
Run under .venv. Cost lever = the 54-game wave-ordered sample (precision-stop, docs/STEER_SAMPLE_RULE.md).

SUPERSEDED GRID — kept for provenance. This is the saturated-dose steering arm; the
paper's causal claims rest on the small-dose linear-regime arm (steering/run_smalldose.py).
The letter-saturation finding reported in Methods §11 comes from this runner.
"""
from __future__ import annotations
import argparse, json, sys, time, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, torch
from collection.akata_common import (build_akata_oneshot_prompt, akata_cb_grid,
                                                   akata_user_question, ANSWER_PREFIX)
from steering.causal_common import load_game_vec
from collection.model_setup import _setup_model
from steering.steer_core import scaled_vector, dual_readout
from strategic_anatomy.action_direction import unembed_action_direction
from strategic_anatomy.config import game_features_csv, manifests_root, repo_root, steering_root

ROOT = repo_root()
CFG = json.loads((manifests_root() / "oneshot_config.json").read_text())
DIRS_ROOT = steering_root() / "directions" / "akata"
SAMPLE = manifests_root() / "steer_sample_54.csv"
OUT_ROOT = steering_root() / "saturated"
FEATS = pd.read_csv(game_features_csv())
CANON = {r.game_code: int(r.canonical_action_p1) for r in FEATS.itertuples() if pd.notna(r.canonical_action_p1)}
DOSES_1D = [-1.0, 0.0, 0.5, 1.0, 2.0]
DOSES_H0 = [-0.25, -0.1, 0.0, 0.1, 0.25]
MODE_DIR = {"h1_dinc": "d_inc", "h2_choice": "d_choice_perp", "h3_oppinc": "d_opp"}  # h0 = d_gate per-prompt


def _wrong_layer(steer_layers, L):
    cands = [x for x in steer_layers if x != L]
    return cands[len(cands) // 2] if cands else L


def run(model_key, modes, n_games, out_root):
    model_cfg = CFG["models"][model_key]
    steer_layers = list(model_cfg["steer_layers"])
    z = np.load(DIRS_ROOT / model_key / "directions.npz")
    have = set(z.files)
    sample = pd.read_csv(SAMPLE)
    games = sample.sort_values("wave")["game_code"].tolist()
    if n_games > 0:
        games = games[:n_games]
    chat = bool(model_cfg.get("chat") or model_key == "llama31_instruct")
    import types as _t
    args = _t.SimpleNamespace(attn_impl="sdpa")
    torch_, model, tok, ab_mtm, dev, ens, load_s = _setup_model(args, model_cfg)
    model.eval()
    layers = model.model.layers
    jp = {mv: tok(" " + mv, add_special_tokens=False)["input_ids"][-1] for mv in ("J", "P")}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[steer] {model_key} loaded {load_s:.0f}s; layers={steer_layers}; {len(games)} games; modes={modes}", flush=True)

    import glob as _glob
    for mode in modes:
        if _glob.glob(str(out_root / f"steer_{model_key}_{mode}_*.parquet")):
            print(f"[steer] {model_key}/{mode}: SKIP (parquet exists)", flush=True)
            continue
        rows = []
        doses = DOSES_H0 if mode == "h0" else DOSES_1D
        t0 = time.time()
        for gi, g in enumerate(games, 1):
            vec = [int(x) for x in load_game_vec(g)]
            canon = CANON.get(g)
            for cb in akata_cb_grid():
                cbid = cb["counterbalance_id"]
                if chat:
                    user = akata_user_question(vec, cb, player=1)
                    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                     add_generation_prompt=True) + ANSWER_PREFIX
                else:
                    prompt = build_akata_oneshot_prompt(vec, cb, player=1)
                enc = tok(prompt, add_special_tokens=False, return_tensors="pt")
                ids = enc["input_ids"].to(dev); attn = ens(ids, enc.get("attention_mask")).to(dev)
                with torch.inference_mode():
                    out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
                for L in steer_layers:
                    hnorm = float(np.linalg.norm(out.hidden_states[L + 1][0, -1, :].float().cpu().numpy()))
                    # resolve the direction unit for this mode/layer
                    if mode == "h0":
                        unit = unembed_action_direction(model.get_output_embeddings(), jp, cb["action_to_letter"])
                        unit = np.asarray(unit, dtype=np.float32); status = "OK"
                    else:
                        k = f"{MODE_DIR[mode]}_l{L}"
                        unit = z[k] if k in have else None
                        status = "OK" if (unit is not None and np.linalg.norm(unit) > 0) else "NOT_ESTIMABLE"
                    rnd = z[f"d_random_l{L}"] if f"d_random_l{L}" in have else None
                    wl = _wrong_layer(steer_layers, L)
                    # variants: main, random, wrong_layer
                    variants = [("main", unit, L), ("random", rnd, L), ("wrong_layer", unit, wl)]
                    for vname, vunit, vL in variants:
                        if vunit is None or np.linalg.norm(vunit) == 0:
                            continue
                        wln = float(np.linalg.norm(out.hidden_states[vL + 1][0, -1, :].float().cpu().numpy()))
                        for dose in doses:
                            vt = scaled_vector(vunit, wln, dose, dev)
                            sp, mv = dual_readout(model, tok, layers, vL, ids, attn, jp, pad, vt, dev)
                            act = cb["letter_to_action"].get(mv) if mv else None
                            rows.append({
                                "model": model_key, "mode": mode, "game_code": g, "wave": int(sample[sample.game_code==g].wave.iloc[0]),
                                "counterbalance_id": cbid, "steer_layer": L, "variant": vname,
                                "inject_layer": vL, "dose": dose, "status": status,
                                "slot_pref_J": round(sp, 4), "realized_letter": (mv or ""),
                                "realized_action": (int(act) if act is not None else -1),
                                "canonical_action_p1": (canon if canon is not None else -1),
                                "realized_canonical": (None if act is None or canon is None else int(act == canon)),
                            })
            if gi % 6 == 0 or gi == len(games):
                print(f"  [{mode}] {gi}/{len(games)} games ({time.time()-t0:.0f}s, {len(rows)} rows)", flush=True)
        df = pd.DataFrame(rows)
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        fp = out_root / f"steer_{model_key}_{mode}_{ts}.parquet"
        df.to_parquet(fp, index=False)
        print(f"[steer] {model_key}/{mode}: wrote {len(df)} rows -> {fp}", flush=True)
    del model
    print("AKATA_STEER_DONE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("qwen", "qwen_instruct", "llama31_instruct"))
    p.add_argument("--modes", default="h0,h1_dinc,h2_choice,h3_oppinc")
    p.add_argument("--n-games", type=int, default=0)
    p.add_argument("--out-root", default=str(OUT_ROOT))
    a = p.parse_args()
    run(a.model, a.modes.split(","), a.n_games, Path(a.out_root))


if __name__ == "__main__":
    main()
