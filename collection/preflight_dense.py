"""Akata one-shot DENSE preflight (qwen, qwen_instruct, llama31_instruct).
Per spec docs/AKATA_ONESHOT_RECOLLECTION.md §7 — the mechanics gate, NOT a full run:
render -> generate(do_sample=False) -> parse J/P -> capture residual at the `A: Option` slot at
ALL layers (incl L0). Writes a table for PI approval.

Loads via the VALIDATED `_setup_model` (Spark CPU-first bnb8 patch) — a raw from_pretrained OOM'd.
Run under .venv."""
from __future__ import annotations
import sys, json, time, types
from pathlib import Path
import torch
import pandas as pd
from collection.akata_common import (
    akata_cb_grid, build_akata_oneshot_prompt, akata_user_question, parse_akata_move,
    prompt_sha256, ANSWER_PREFIX,
)
from steering.causal_common import load_game_vec
from collection.model_setup import _setup_model
from strategic_anatomy.config import data_root, game_features_csv, manifests_root

CONFIG = json.loads((manifests_root() / "oneshot_config.json").read_text())
MODELS = [("qwen", False), ("qwen_instruct", False), ("llama31_instruct", True)]   # (config key, chat)
GAMES = ["AsBa", "CmDl"]
CB = akata_cb_grid()
feats = pd.read_csv(game_features_csv())
CANON = {r.game_code: int(r.canonical_action_p1) for r in feats.itertuples() if pd.notna(r.canonical_action_p1)}


def build_prompt(tok, vec, cb, chat):
    if not chat:
        return build_akata_oneshot_prompt(vec, cb, player=1)
    user = akata_user_question(vec, cb, player=1)
    chat_txt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True)
    return chat_txt + ANSWER_PREFIX


def jp_token_ids(tok):
    return {mv: tok(" " + mv, add_special_tokens=False)["input_ids"][-1] for mv in ("J", "P")}


def run_model(model_key, chat):
    print(f"\n##### {model_key} #####", flush=True)
    model_cfg = CONFIG["models"][model_key]
    args = types.SimpleNamespace(attn_impl="sdpa")
    _torch, model, tok, _mtm_ab, dev, ens_mask, load_s = _setup_model(args, model_cfg)
    model.eval()
    mtm = jp_token_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"  loaded {model_cfg['model_name']} in {load_s:.0f}s; J/P tokens {mtm}", flush=True)
    rows = []
    for g in GAMES:
        vec = [int(x) for x in load_game_vec(g)]
        for cb in CB:
            prompt = build_prompt(tok, vec, cb, chat)
            enc = tok(prompt, add_special_tokens=False, return_tensors="pt").to(dev)
            with torch.inference_mode():
                out = model(**enc, output_hidden_states=True, use_cache=False)
            n_layers = len(out.hidden_states)                      # incl. embedding (L0)
            logits = out.logits[0, -1, :].float()
            pj, pp = torch.softmax(logits[[mtm["J"], mtm["P"]]], dim=-1).tolist()
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=pad)
            gen_txt = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            mv = parse_akata_move(gen_txt)
            act = cb["letter_to_action"].get(mv) if mv else None
            canon = CANON.get(g)
            rows.append({
                "model": model_key, "game": g, "cb": cb["counterbalance_id"], "label_map": cb["label_map"],
                "q_order": cb["q_order"], "prompt_sha": prompt_sha256(prompt)[:10],
                "gen": gen_txt.replace("\n", "\\n")[:36], "parsed": mv, "action": act,
                "canonical_action_p1": canon,
                "aligned_canonical": (None if act is None or canon is None else int(act == canon)),
                "slot_pref_J": round(pj, 3), "n_resid_layers": n_layers,
            })
            print(f"  {g} cb{cb['counterbalance_id']} {cb['label_map']:7s} {cb['q_order']}: "
                  f"gen={gen_txt[:14]!r} parse={mv} act={act} canon={canon} prefJ={pj:.2f} "
                  f"resid_layers={n_layers}", flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    allrows = []
    for key, chat in MODELS:
        try:
            allrows += run_model(key, chat)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{key}] FAILED: {e}", flush=True)
    df = pd.DataFrame(allrows)
    outdir = data_root() / "akata_preflight"; outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "dense_preflight.csv", index=False)
    print("\n================ DENSE PREFLIGHT TABLE ================")
    print(df.to_string(index=False))
    print(f"\nwrote {outdir/'dense_preflight.csv'}")
    print("PREFLIGHT_DENSE_DONE", flush=True)


if __name__ == "__main__":
    main()
