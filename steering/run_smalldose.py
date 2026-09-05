"""SMALL-DOSE Akata dense steering — the linear-regime arm (locked 2026-07-02).

The saturated-grid production run (run_akata_steer.py, |dose| ∈ {0.5,1,2}) proved the apparatus
(h0: dose-0 bit-identity, decisions flip) but its h1/h2 doses live ENTIRELY in a letter-hijack
saturated regime: at every collected dose the injection slams the next token to Option J (+dose)
or Option P (−dose) with 100% letter coherence, which the 4-cell counterbalance cancels on the
strategic axes — every slope statistic reads ≈0 (docs/METHODS.md, 2026-07-02 REVISION).
This arm asks the OPEN question — does pushing d_inc / d_choice_perp move the decision on the
CANONICAL axis — at doses inside the regime where h0 was validated (|dose| ≤ 0.25).

LOCKED SPEC (2026-07-02; the main_perp variant was added the same day):
  doses    [-0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25]
  modes    h1_dinc (d_inc) + h2_choice (d_choice_perp)
  layers   65, 79
  variants main + random + main_perp (the direction with its component along the letter readout
           axis ell = norm.weight*(W_U[' J']-W_U[' P']) projected out; built by the perp-npz step,
           see directions_akata_perp/{model}/manifest.json — cos removed 0.005..0.181)
  models   qwen -> qwen_instruct -> llama31_instruct
  out      $SCA_DATA_ROOT/steering/smalldose/
Injection convention IDENTICAL to run_akata_steer.py: hook on model.model.layers[L] (same
off-by-one as the collected runs, kept deliberately for comparability), delta = dose*||h_site||*unit,
dual readout (slot pref diagnostic + regenerated J/P decision), mode-complete parquet writes,
per-(model,mode) resume via glob INSIDE the out-root. Run under .venv.
"""
from __future__ import annotations
import argparse, datetime as dt, glob as _glob, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

from collection.akata_common import (build_akata_oneshot_prompt, akata_cb_grid,
                                                   akata_user_question, ANSWER_PREFIX)
from steering.causal_common import load_game_vec
from collection.model_setup import _setup_model
from steering.steer_core import scaled_vector, dual_readout
# Reuse the production runner's resolved config/paths so both arms are byte-identically anchored.
from steering.run_saturated import CFG, DIRS_ROOT, SAMPLE, CANON, MODE_DIR
from strategic_anatomy.config import repo_root, steering_root

DOSES_SD = [-0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25]
LAYERS_SD = [65, 79]
MODES_SD = ["h1_dinc", "h2_choice"]
ROOT = repo_root()
OUT_ROOT_SD = steering_root() / "smalldose"
DIRS_PERP = steering_root() / "directions" / "akata_perp"


def run(model_key, modes, layers, n_games, out_root):
    model_cfg = CFG["models"][model_key]
    z = np.load(DIRS_ROOT / model_key / "directions.npz")
    have = set(z.files)
    zp = np.load(DIRS_PERP / model_key / "directions.npz")
    havep = set(zp.files)
    sample = pd.read_csv(SAMPLE)
    games = sample.sort_values("wave")["game_code"].tolist()
    if n_games > 0:
        games = games[:n_games]
    chat = bool(model_cfg.get("chat") or model_key == "llama31_instruct")
    import types as _t
    args = _t.SimpleNamespace(attn_impl="sdpa")
    torch_, model, tok, ab_mtm, dev, ens, load_s = _setup_model(args, model_cfg)
    model.eval()
    lyr = model.model.layers
    jp = {mv: tok(" " + mv, add_special_tokens=False)["input_ids"][-1] for mv in ("J", "P")}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[steer-sd] {model_key} loaded {load_s:.0f}s; layers={layers}; doses={DOSES_SD}; "
          f"{len(games)} games; modes={modes}", flush=True)

    for mode in modes:
        if _glob.glob(str(out_root / f"steer_{model_key}_{mode}_*.parquet")):
            print(f"[steer-sd] {model_key}/{mode}: SKIP (parquet exists)", flush=True)
            continue
        rows = []
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
                for L in layers:
                    k = f"{MODE_DIR[mode]}_l{L}"
                    unit = z[k] if k in have else None
                    status = "OK" if (unit is not None and np.linalg.norm(unit) > 0) else "NOT_ESTIMABLE"
                    rnd = z[f"d_random_l{L}"] if f"d_random_l{L}" in have else None
                    unitp = zp[k] if k in havep else None
                    hnorm = float(np.linalg.norm(out.hidden_states[L + 1][0, -1, :].float().cpu().numpy()))
                    for vname, vunit in (("main", unit), ("random", rnd), ("main_perp", unitp)):
                        if vunit is None or np.linalg.norm(vunit) == 0:
                            continue
                        for dose in DOSES_SD:
                            vt = scaled_vector(vunit, hnorm, dose, dev)
                            sp, mv = dual_readout(model, tok, lyr, L, ids, attn, jp, pad, vt, dev)
                            act = cb["letter_to_action"].get(mv) if mv else None
                            rows.append({
                                "model": model_key, "mode": mode, "game_code": g,
                                "wave": int(sample[sample.game_code == g].wave.iloc[0]),
                                "counterbalance_id": cbid, "steer_layer": L, "variant": vname,
                                "inject_layer": L, "dose": dose, "status": status,
                                "slot_pref_J": round(sp, 4), "realized_letter": (mv or ""),
                                "realized_action": (int(act) if act is not None else -1),
                                "canonical_action_p1": (canon if canon is not None else -1),
                                "realized_canonical": (None if act is None or canon is None else int(act == canon)),
                            })
            if gi % 6 == 0 or gi == len(games):
                print(f"  [sd:{mode}] {gi}/{len(games)} games ({time.time()-t0:.0f}s, {len(rows)} rows)", flush=True)
        df = pd.DataFrame(rows)
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        fp = out_root / f"steer_{model_key}_{mode}_{ts}.parquet"
        df.to_parquet(fp, index=False)
        print(f"[steer-sd] {model_key}/{mode}: wrote {len(df)} rows -> {fp}", flush=True)
    del model
    print("AKATA_STEER_SMALLDOSE_DONE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("qwen", "qwen_instruct", "llama31_instruct"))
    p.add_argument("--modes", default=",".join(MODES_SD))
    p.add_argument("--layers", default=",".join(str(x) for x in LAYERS_SD))
    p.add_argument("--n-games", type=int, default=0)
    p.add_argument("--out-root", default=str(OUT_ROOT_SD))
    a = p.parse_args()
    run(a.model, a.modes.split(","), [int(x) for x in a.layers.split(",")], a.n_games, Path(a.out_root))


if __name__ == "__main__":
    main()
