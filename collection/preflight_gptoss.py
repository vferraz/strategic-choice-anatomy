"""Akata one-shot GPT-OSS preflight — harmony final-channel COMMIT decode (J/P) + commit-token
residual + MoE router. Per spec docs/AKATA_ONESHOT_RECOLLECTION.md §7. RUN UNDER .venv_gptoss.

Loads via validated `_setup_model`; reuses validated `capture_moe_router` / `_topk_idx_weights` /
`_generate` / `HARMONY_FINAL_MARKER`. Only the prompt (Akata) + the letter (J/P) are new.
2 games x 4 cb x P1 baseline. NOT a full run."""
from __future__ import annotations
import sys, json, time, types, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from collection.akata_common import (
    akata_cb_grid, build_akata_oneshot_prompt, parse_akata_move, prompt_sha256, MOVE_LABELS,
)
from steering.causal_common import load_game_vec
from collection.model_setup import _setup_model
from collection.genutils import _generate, HARMONY_FINAL_MARKER
from collection.model_setup import ROUTER_LAYERS_DEFAULT
from strategic_anatomy.router_capture import capture_moe_router, _topk_idx_weights, model_topk, _kernels_available
from strategic_anatomy.config import data_root, game_features_csv, manifests_root

CONFIG = json.loads((manifests_root() / "oneshot_config.json").read_text())
GAMES = ["AsBa", "CmDl"]
CB = akata_cb_grid()
feats = pd.read_csv(game_features_csv())
CANON = {r.game_code: int(r.canonical_action_p1) for r in feats.itertuples() if pd.notna(r.canonical_action_p1)}
HARMONY_TAG = re.compile(r"<\|[^|]*\|>")
_JP_RE = {"J": re.compile(r"(?<![A-Za-z])J(?![A-Za-z])"), "P": re.compile(r"(?<![A-Za-z])P(?![A-Za-z])")}


def extract_move_jp(withspec_text: str):
    """(move, final_present) — move read ONLY from the harmony final channel; J/P."""
    if HARMONY_FINAL_MARKER not in withspec_text:
        return None, False
    final = withspec_text.split(HARMONY_FINAL_MARKER)[-1]
    return parse_akata_move(HARMONY_TAG.sub("", final)), True


def find_commit_index_jp(new_ids, tok, label):
    """Index in new_ids of the first standalone J/P letter AFTER the harmony final marker."""
    text, seen, rx = "", False, _JP_RE.get(label)
    for i, tid in enumerate(new_ids.tolist()):
        piece = tok.decode([int(tid)], skip_special_tokens=False)
        text += piece
        if not seen and HARMONY_FINAL_MARKER in text:
            seen = True
        if seen and rx is not None and rx.search(piece):
            return i
    return None


class JPCommitStopper(StoppingCriteria):
    def __init__(self, tok, input_len, stride=16, min_new=3):
        self.tok, self.input_len, self.stride, self.min_new = tok, input_len, stride, min_new

    def __call__(self, input_ids, scores, **kw):
        gl = input_ids.shape[1] - self.input_len
        if gl < self.min_new or gl % self.stride != 0:
            return False
        mv, present = extract_move_jp(self.tok.decode(input_ids[0, self.input_len:], skip_special_tokens=False))
        return bool(present and mv)


def refeed_jp(model, tok, dev, ens_mask, full_ids, router_layers, residual_layers, mtm, top_k):
    ids = full_ids.to(dev).unsqueeze(0) if full_ids.ndim == 1 else full_ids.to(dev)
    attn = ens_mask(ids, None).to(dev) if ens_mask is not None else torch.ones_like(ids)
    with torch.inference_mode():
        with capture_moe_router(model, router_layers) as cap:
            out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    router = {}
    for L in router_layers:
        L = int(L)
        gl = np.asarray(cap[L][-1, :], dtype=np.float32)
        idx, w = _topk_idx_weights(gl[None, :], top_k)
        router[L] = (gl, idx[0], w[0])
    resid = {int(L): out.hidden_states[int(L) + 1][0, -1, :].detach().to(torch.float32).cpu().numpy()
             for L in residual_layers}
    probs = torch.softmax(out.logits[0, -1, :].detach(), dim=-1)
    fp = {mv: float(probs[mtm[mv]].item()) for mv in MOVE_LABELS}
    return router, resid, fp


def main():
    if not _kernels_available():
        raise SystemExit("kernels package missing — run under .venv_gptoss (CLAUDE.md #7).")
    model_cfg = CONFIG["models"]["gptoss"]
    residual_layers = list(model_cfg["capture_layers"])            # 0..35
    router_layers = list(ROUTER_LAYERS_DEFAULT)
    args = types.SimpleNamespace(attn_impl="sdpa", load_8bit=False, load_4bit=False)  # gpt-oss = MXFP4, no quant
    _t, model, tok, _ab, dev, ens_mask, load_s = _setup_model(args, model_cfg)
    model.eval()
    top_k = model_topk(model, None)
    mtm = {mv: tok(" " + mv, add_special_tokens=False)["input_ids"][-1] for mv in ("J", "P")}
    print(f"loaded gptoss in {load_s:.0f}s; J/P tokens {mtm}; router {len(router_layers)} layers; "
          f"residual {len(residual_layers)} layers; top_k {top_k}", flush=True)
    rows = []
    for g in GAMES:
        vec = [int(x) for x in load_game_vec(g)]
        for cb in CB:
            raw = build_akata_oneshot_prompt(vec, cb, player=1)        # GAME PRESENTATION UNCHANGED
            prompt = tok.apply_chat_template([{"role": "user", "content": raw}], tokenize=False,
                                             add_generation_prompt=True)   # native harmony framing
            assert raw in prompt, "game text not verbatim inside the harmony template"
            enc = tok(prompt, add_special_tokens=False, return_tensors="pt")
            ids = enc["input_ids"].to(dev)
            attn = (ens_mask(ids, enc.get("attention_mask")).to(dev) if ens_mask is not None
                    else torch.ones_like(ids))
            stopper = StoppingCriteriaList([JPCommitStopper(tok, ids.shape[1])])
            t0 = time.time()
            new_ids = _generate(model, tok, ids, attn, 4096, False, torch, stopping_criteria=stopper)
            withspec = tok.decode(new_ids, skip_special_tokens=False)
            mv, final_present = extract_move_jp(withspec)
            cidx = find_commit_index_jp(new_ids, tok, mv) if (final_present and mv) else None
            commit_present = cidx is not None
            prefJ, nresid, nrouter = None, 0, 0
            if commit_present:
                full = torch.cat([ids[0], new_ids[:cidx]], dim=0)
                router, resid, fp = refeed_jp(model, tok, dev, ens_mask, full, router_layers,
                                              residual_layers, mtm, top_k)
                prefJ = round(fp["J"] / (fp["J"] + fp["P"] + 1e-9), 3)
                nresid, nrouter = len(resid), len(router)
            act = cb["letter_to_action"].get(mv) if mv else None
            canon = CANON.get(g)
            rows.append({
                "model": "gptoss", "game": g, "cb": cb["counterbalance_id"], "label_map": cb["label_map"],
                "q_order": cb["q_order"], "prompt_sha": prompt_sha256(prompt)[:10],
                "n_new_tokens": int(new_ids.shape[0]), "commit_present": commit_present,
                "parsed": mv, "action": act, "canonical_action_p1": canon,
                "aligned_canonical": (None if act is None or canon is None else int(act == canon)),
                "commit_idx": (int(ids.shape[1] + cidx) if cidx is not None else -1),
                "slot_pref_J": prefJ, "n_resid_layers": nresid, "n_router_layers": nrouter,
                "gen_s": round(time.time() - t0, 1),
            })
            print(f"  {g} cb{cb['counterbalance_id']} {cb['label_map']:7s} {cb['q_order']}: "
                  f"commit={commit_present} parse={mv} act={act} canon={canon} prefJ={prefJ} "
                  f"resid={nresid} router={nrouter} ntok={int(new_ids.shape[0])} {round(time.time()-t0,0)}s",
                  flush=True)
    df = pd.DataFrame(rows)
    outdir = data_root() / "akata_preflight"; outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "gptoss_preflight.csv", index=False)
    print("\n================ GPT-OSS PREFLIGHT TABLE ================")
    print(df.to_string(index=False))
    print(f"\nwrote {outdir/'gptoss_preflight.csv'}")
    print("PREFLIGHT_GPTOSS_DONE", flush=True)


if __name__ == "__main__":
    main()
