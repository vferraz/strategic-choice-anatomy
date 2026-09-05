"""GPT-OSS RECAPTURE — uniform capture site (analysis->final transition) for ALL P1-baseline rows.

Fixes the fatal defect in the original collector: capture position was chosen by the model's own output
(pure -> commit letter, mixed -> transition) = two sites = unidentifiable cross-row geometry.

PRIMARY SITE (identical for every row, never conditioned on output type):
  the token at which the harmony analysis->final marker COMPLETES (find_transition_index) — present in
  every stream, pure or mixed. Residual = all layers incl L0; router = gate/idx/w at all router layers.
SECONDARY (validation only, pure rows): the pre-letter position, checked against the stored acts.npz.
POLICY TARGET (neural): pure -> 1/0 on canonical; mixed -> stated_p_act0 -> P(canonical).
  default_uniform rows (harness-imputed 0.5) are FLAGGED and excluded from the neural target.
REPRODUCTION GATE: regenerate greedily w/ the original config; compare move_letter/commit_type/
  n_new_tokens against the stored row. Mismatch -> flag the row, never silently substitute.
PROVENANCE: full generated token ids, uncropped final text, prompt hash, transition idx, letter idx.
Population: 144 games x 4 cb, P1 baseline = 576 rows. Out: $SCA_DATA_ROOT/gptoss_recap/. .venv_gptoss.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, socket, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch

from collection.akata_common import (
    MOVE_LABELS, akata_cb_grid, build_akata_oneshot_prompt, prompt_sha256)
from collection.preflight_gptoss import HARMONY_TAG
from collection.genutils import _generate, HARMONY_FINAL_MARKER
from collection.model_setup import (
    _git_commit, _now, _sha256_file, _fsync_dir, _setup_model, CONFIG_PATH, MANIFEST_DIR)
from collection.model_setup import ROUTER_LAYERS_DEFAULT
from collection.oneshot_common import load_game_vec
from collection.generate_gptoss_substrate import (
    classify, find_commit_index, find_transition_index, _capture)
from strategic_anatomy.router_capture import model_topk, _kernels_available
from strategic_anatomy.config import game_features_csv, gptoss_recap_root, repo_root, substrate_root

ROOT = repo_root()
OUT_DEFAULT = gptoss_recap_root()
SRC_ROOT = substrate_root() / "gptoss"                            # the original (defective) capture
GAME_FEATURES = game_features_csv()
COS_TOL = 0.999                                                    # residual reproduction tolerance

# The harmony chat template injects a LIVE "Current date:" line, so a prompt built today differs from
# the one the original run saw (collected 2026-06-25/26, crossing UTC midnight). We pin each row's date
# by brute-forcing the candidates until prompt_sha256 == the row's STORED prompt_hash — i.e. the prompt
# is provably byte-identical to the original before we regenerate. No match -> the row FAILS, never runs.
_DATE_RE = re.compile(r"Current date: [\d-]+")
DATE_CANDIDATES = ["2026-06-25", "2026-06-26", "2026-06-24", "2026-06-27", "2026-06-23", "2026-06-28"]


def pin_prompt_to_stored(prompt: str, stored_hash: str):
    """(prompt_with_original_date, date) if a candidate reproduces stored_hash, else (None, None)."""
    for d in DATE_CANDIDATES:
        cand = _DATE_RE.sub(f"Current date: {d}", prompt)
        if prompt_sha256(cand) == stored_hash:
            return cand, d
    return None, None


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def run_game(*, model, tok, dev, ens_mask, jp_ids, top_k, router_layers, game_code, game_vec,
             canon, max_new_tokens, src_df, src_acts):
    rows, resid_store, router_store, prov = [], {}, {}, {}
    for cb in akata_cb_grid():
        cbid = cb["counterbalance_id"]
        raw = build_akata_oneshot_prompt(game_vec, cb, player=1, iv_prefix="")
        prompt = tok.apply_chat_template([{"role": "user", "content": raw}], tokenize=False,
                                         add_generation_prompt=True)
        # ---- PIN the harmony injected date to the ORIGINAL run's, verified by prompt_hash ----
        s0 = src_df[(src_df.counterbalance_id == cbid) & (src_df.player == 1) &
                    (src_df.condition == "baseline")]
        if len(s0) != 1:
            print(f"    {game_code} cb{cbid}: SKIP — no unique source row", flush=True)
            continue
        stored_hash = str(s0.iloc[0].prompt_hash)
        pinned, pin_date = pin_prompt_to_stored(prompt, stored_hash)
        if pinned is None:
            print(f"    {game_code} cb{cbid}: FAIL — prompt not reproducible (no date candidate matches "
                  f"stored hash {stored_hash[:12]}); row NOT regenerated", flush=True)
            rows.append({"model": "gptoss", "game_code": game_code, "player": 1, "condition": "baseline",
                         "counterbalance_id": cbid, "prompt_hash": prompt_sha256(prompt),
                         "commit_type": "PROMPT_UNREPRODUCIBLE", "captured": 0, "repro_ok": 0,
                         "repro_mismatch": "prompt_pin_failed", "val_ok": 0, "val_cos": float("nan"),
                         "use_in_neural_target": 0, "p_canonical": float("nan")})
            continue
        prompt = pinned                                   # provably byte-identical to the original
        enc = tok(prompt, add_special_tokens=False, return_tensors="pt")
        ids = enc["input_ids"].to(dev)
        attn = ens_mask(ids, enc.get("attention_mask")).to(dev) if ens_mask is not None else torch.ones_like(ids)
        t0 = time.time()
        new_ids = _generate(model, tok, ids, attn, max_new_tokens, False, torch)     # PURE GREEDY (orig cfg)
        withspec = tok.decode(new_ids, skip_special_tokens=False)
        final_present = HARMONY_FINAL_MARKER in withspec
        final_text = HARMONY_TAG.sub("", withspec.split(HARMONY_FINAL_MARKER)[-1]) if final_present else ""
        ctype, mv, p_J = classify(final_text) if final_present else ("none", None, None)

        # ---- PRIMARY: uniform transition site, for EVERY row ----
        tidx = find_transition_index(new_ids, tok)
        captured = tidx is not None
        if captured:
            cap_ids = torch.cat([ids[0], new_ids[:tidx + 1]], dim=0)
            resid, router, pref = _capture(model, tok, dev, ens_mask, torch, cap_ids,
                                           router_layers, jp_ids, top_k)
            for i, h in resid.items():
                resid_store[f"p1_baseline_cb{cbid}_l{i}"] = h.astype(np.float32)
            for L, (gl, idx, w) in router.items():
                router_store[f"p1_baseline_cb{cbid}_l{L}_gate"] = gl
                router_store[f"p1_baseline_cb{cbid}_l{L}_idx"] = idx
                router_store[f"p1_baseline_cb{cbid}_l{L}_w"] = w
            pref_J = pref["J"]
        else:
            pref_J = float("nan")

        # ---- VALIDATION vs the ORIGINAL stored acts, at each row's ORIGINAL site ----
        # MIXED rows: their original site WAS the transition == the new primary site -> the primary
        #   capture must reproduce acts.npz exactly. This is the strongest available anchor.
        # PURE rows: their original site was the pre-letter token -> re-capture it and compare.
        val_cos, val_site = float("nan"), ""
        cidx_letter = find_commit_index(new_ids, tok, mv) if (ctype == "pure" and mv is not None) else None
        if src_acts is not None:
            ks = [kk for kk in src_acts.files if kk.startswith(f"p1_baseline_cb{cbid}_l")]
            if ks:
                Ls = sorted(int(kk.rsplit("_l", 1)[1]) for kk in ks)
                Lm = Ls[len(Ls) // 2]                       # mid layer: real signal, not the shared L0
                ref = src_acts[f"p1_baseline_cb{cbid}_l{Lm}"]
                if ctype == "mixed" and captured:
                    val_cos, val_site = _cos(resid[Lm], ref), "transition(same_site)"
                elif ctype == "pure" and cidx_letter is not None:
                    lids = torch.cat([ids[0], new_ids[:cidx_letter]], dim=0)
                    r2, _, _ = _capture(model, tok, dev, ens_mask, torch, lids, router_layers, jp_ids, top_k)
                    val_cos, val_site = _cos(r2[Lm], ref), "letter(re-captured)"
        val_ok = bool(np.isfinite(val_cos) and val_cos >= COS_TOL)   # ENFORCED tolerance

        # ---- REPRODUCTION GATE vs the stored row ----
        s = src_df[(src_df.counterbalance_id == cbid) & (src_df.player == 1) &
                   (src_df.condition == "baseline")]
        repro_ok, repro_note = True, ""
        if len(s) == 1:
            s = s.iloc[0]
            checks = {"commit_type": (ctype, s.commit_type), "move_letter": (mv or "", s.move_letter),
                      "n_new_tokens": (int(new_ids.shape[0]), int(s.n_new_tokens)),
                      "prompt_hash": (prompt_sha256(prompt), s.prompt_hash)}   # catches prompt drift
            bad = [k for k, (a, b) in checks.items() if a != b]
            repro_ok = not bad
            repro_note = ",".join(bad)
        else:
            repro_ok, repro_note = False, "no_source_row"

        # ---- POLICY TARGET (neural): P(canonical action) ----
        l2a = cb["letter_to_action"]
        if ctype == "pure" and mv is not None:
            p_act0 = 1.0 if l2a[mv] == 0 else 0.0
            psrc = "pure"
        elif ctype == "mixed" and p_J is not None:
            p_act0 = p_J if l2a["J"] == 0 else (1.0 - p_J)
            psrc = "stated"
        elif ctype == "mixed":
            p_act0, psrc = 0.5, "default_uniform"        # FLAGGED: excluded from neural target
        else:
            p_act0, psrc = float("nan"), "NA"
        p_canon = p_act0 if int(canon) == 0 else (1.0 - p_act0)
        use_neural = psrc in ("pure", "stated")           # hardening #3

        prov[f"p1_baseline_cb{cbid}_genids"] = new_ids.detach().cpu().numpy().astype(np.int32)
        rows.append({
            "model": "gptoss", "game_code": game_code, "player": 1, "condition": "baseline",
            "counterbalance_id": cbid, "label_map": cb["label_map"], "q_order": cb["q_order"],
            "prompt_hash": prompt_sha256(prompt), "commit_type": ctype, "move_letter": (mv or ""),
            "capture_site": "analysis_final_transition", "pin_date": pin_date, "transition_idx": (-1 if tidx is None else int(tidx)),
            "letter_idx": (-1 if cidx_letter is None else int(cidx_letter)),   # explicit None (idx 0 safe)
            "stated_p_J": (float(p_J) if p_J is not None else float("nan")),
            "p_act0": p_act0, "p_canonical": p_canon, "prob_source": psrc, "use_in_neural_target": int(use_neural),
            "canonical_action_p1": int(canon), "captured": int(captured), "final_marker": int(final_present),
            "n_new_tokens": int(new_ids.shape[0]), "slot_pref_J": pref_J,
            "repro_ok": int(repro_ok), "repro_mismatch": repro_note,
            "val_cos": val_cos, "val_site": val_site, "val_ok": int(val_ok),
            "final_text_full": final_text.replace("\n", "\\n"),     # UNCROPPED
            "gen_s": round(time.time() - t0, 1)})
        print(f"    {game_code} cb{cbid}: {ctype}{('/'+mv) if mv else ''} p_canon={p_canon:.2f} ({psrc}) "
              f"tidx={tidx} repro={'OK' if repro_ok else 'MISMATCH:'+repro_note} "
              f"valcos={val_cos:.5f}[{val_site}]{'' if val_ok else ' <TOL!'} "
              f"{round(time.time()-t0)}s", flush=True)
    return pd.DataFrame(rows), resid_store, router_store, prov


GAME_FILES = ("_DONE", "results.parquet", "acts.npz", "router.npz", "genids.npz", "config.json")


def _game_complete(d):
    """A game counts as done only if every artifact exists AND is non-empty. `_DONE` existing is not
    enough: a crash mid-flush can publish 0-byte files under a 0-byte _DONE, which --skip-existing
    would then skip forever (ChHr, 2026-07-12)."""
    return all((d / n).exists() and (d / n).stat().st_size > 0 for n in GAME_FILES)


def persist(out_root, game_code, df, resid, router, prov, provenance):
    fin, tmp = out_root / game_code, out_root / "_tmp" / game_code
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    assert len(df) == 4, f"rows {len(df)} != 4 (a cb failed to pin/regenerate)"
    df.to_parquet(tmp / "results.parquet", index=False)
    np.savez_compressed(tmp / "acts.npz", **resid)
    np.savez_compressed(tmp / "router.npz", **router)
    np.savez_compressed(tmp / "genids.npz", **prov)
    (tmp / "config.json").write_text(json.dumps(
        {**provenance, "game_code": game_code, "n_rows": int(len(df)),
         "n_captured": int(df.captured.sum()), "n_repro_ok": int(df.repro_ok.sum()), "n_val_ok": int(df.val_ok.sum()),
         "sha256": {"results.parquet": _sha256_file(tmp / "results.parquet")}}, indent=2, default=str))
    if fin.exists():
        shutil.rmtree(fin)
    fin.mkdir(parents=True, exist_ok=True)
    # fsync each FILE's contents before publishing it. _fsync_dir alone only durably records the
    # NAME: a hard-lock then leaves a 0-byte file whose _DONE makes --skip-existing skip it forever
    # (this is exactly what happened to ChHr on 2026-07-12).
    for n in ("results.parquet", "acts.npz", "router.npz", "genids.npz", "config.json"):
        with open(tmp / n, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp / n, fin / n)
    _fsync_dir(fin); (fin / "_DONE").write_text(_now()); _fsync_dir(fin)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"game_code": game_code, "n_captured": int(df.captured.sum()),
            "n_repro_ok": int(df.repro_ok.sum()), "n_val_ok": int(df.val_ok.sum()),
            "commit_types": df.commit_type.value_counts().to_dict()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--games-from-manifest", default=str(MANIFEST_DIR / "game_universe_oneshot.csv"))
    p.add_argument("--out-root", default=str(OUT_DEFAULT))
    p.add_argument("--games", default="")               # explicit game list (preflight)
    p.add_argument("--n-games", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--router-layers", default=",".join(str(x) for x in ROUTER_LAYERS_DEFAULT))
    p.add_argument("--attn_impl", default="sdpa")
    a = p.parse_args()
    if not _kernels_available():
        raise SystemExit("run under .venv_gptoss")

    cfg = json.loads(Path(a.config).read_text())
    mcfg = cfg["models"]["gptoss"]
    canon = pd.read_csv(GAME_FEATURES).set_index("game_code")["canonical_action_p1"].astype(int).to_dict()
    if a.games:
        games = [g for g in a.games.split(",") if g]
    else:
        u = pd.read_csv(a.games_from_manifest)
        games = u[(u["include_h0"]) & (u["in_gptoss"])]["game_code"].tolist()
        if a.n_games > 0:
            games = games[: a.n_games]
    router_layers = [int(x) for x in a.router_layers.split(",") if x]
    out_root = Path(a.out_root); out_root.mkdir(parents=True, exist_ok=True)

    import types as _t
    args = _t.SimpleNamespace(attn_impl=a.attn_impl, load_8bit=False, load_4bit=False)
    torch_, model, tok, _m, dev, ens, load_s = _setup_model(args, mcfg)
    model.eval()
    jp = {mv: tok(" " + mv, add_special_tokens=False)["input_ids"][-1] for mv in MOVE_LABELS}
    top_k = model_topk(model, None)
    prov = {"model": "gptoss", "substrate": "akata_oneshot_RECAPTURE", "capture_site": "analysis_final_transition",
            "population": "P1_baseline_only", "git_commit": _git_commit(), "built_at": _now(),
            "host": socket.gethostname(), "max_new_tokens": a.max_new_tokens, "cos_tol": COS_TOL,
            "router_layers": router_layers}
    print(f"[recap] gptoss loaded {load_s:.0f}s; jp={jp}; top_k={top_k}; {len(games)} games -> {out_root}", flush=True)

    stats = []
    for i, g in enumerate(games, 1):
        if a.skip_existing and _game_complete(out_root / g):
            continue
        src_df = pd.read_parquet(SRC_ROOT / g / "results.parquet")
        src_acts = np.load(SRC_ROOT / g / "acts.npz") if (SRC_ROOT / g / "acts.npz").exists() else None
        t0 = time.time()
        df, resid, router, gp = run_game(model=model, tok=tok, dev=dev, ens_mask=ens, jp_ids=jp,
                                         top_k=top_k, router_layers=router_layers, game_code=g,
                                         game_vec=[int(x) for x in load_game_vec(g)], canon=canon[g],
                                         max_new_tokens=a.max_new_tokens, src_df=src_df, src_acts=src_acts)
        s = persist(out_root, g, df, resid, router, gp, prov)
        s["elapsed_s"] = round(time.time() - t0)
        stats.append(s)
        print(f"[{i}/{len(games)}] ok {g} ({s['elapsed_s']}s) cap={s['n_captured']}/4 "
              f"repro_ok={s['n_repro_ok']}/4 val_ok={s['n_val_ok']}/4 {s['commit_types']}", flush=True)
    print(f"[recap] done: {len(stats)} games", flush=True)
    print("GPTOSS_RECAP_DONE", flush=True)


if __name__ == "__main__":
    main()
