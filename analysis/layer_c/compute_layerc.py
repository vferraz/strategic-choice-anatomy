#!/usr/bin/env python3
"""
Layer C — aggregation pipeline (one-shot Akata logit-lens).

Reads $SCA_DATA_ROOT/layerc/{model}/{game}/tokens.parquet and writes two tidy tables:
  tables/decision_readouts.csv  -- per (model,layer,game,cb,condition): answer-prefix and
                                   cue-prefix lens readouts on canonical + trait-target axes.
  tables/region_contrib.csv     -- per (model,layer,game,cb,condition,region): signed increment
                                   contribution to the decision signal + mean |score|.
  tables/incentive_delta1c.csv  -- per game: signed canonical level-1 incentive Δ1^c (q=0.5).

Conventions (hard constraints respected):
  * score_canonical / score_trait_target are already re-scored per cb to the canonical/target
    LETTER at capture (docs/METHODS.md HC-2) -> safe to aggregate across games on these axes.
  * Decision signal D = score_canonical at the FINAL pre-choice token (the last `" Option"`,
    identical across models -> model-comparable). The answer_prefix MEAN is kept as `ap_*` robustness
    (it dilutes Llama, whose answer_prefix region includes chat-scaffolding tokens).
  * Region contribution = sum of per-token increments Δs[t]=s[t]-s[t-1] within a region (first token
    of a sequence -> 0); a telescoping additive decomposition (direct-logit-attribution style).
  * GPT-OSS is excluded by design (MoE/harmony) -- not in the data root.
"""
import pandas as pd, numpy as np, glob, json, os, time
from strategic_anatomy.config import layerc_root, results_root

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = str(layerc_root())
OUT  = str(results_root() / 'layer_c')
os.makedirs(OUT, exist_ok=True)

MODELS = ['qwen', 'qwen_instruct', 'llama31_instruct']
NEED = ['game_code', 'cb_id', 'condition', 'layer', 'token_index',
        'region', 'score_canonical', 'score_trait_target']

# ---- signed canonical level-1 incentive Δ1^c (q=0.5 uniform belief), from raw payoffs ----
up = pd.read_parquet(os.path.join(REPO, 'data/human_refs/unified_pairs.parquet')) \
       .drop_duplicates('game_code')
def delta1c(row):
    v = json.loads(row['matrix_8vec'])          # [p1: 4 cells, p2: 4 cells], cells (0,0)(0,1)(1,0)(1,1)
    p1 = np.array(v[:4]).reshape(2, 2)          # rows = own action, cols = opp action
    d1 = p1[0].mean() - p1[1].mean()            # E[own|act0] - E[own|act1]
    return d1 if row['canonical_action_p1'] == 0 else -d1   # + = toward canonical
INC = {gc: delta1c(r) for gc, r in up.set_index('game_code').iterrows()}
pd.Series(INC, name='delta1c').rename_axis('game_code').to_csv(os.path.join(OUT, 'incentive_delta1c.csv'))
print(f'Δ1^c: {len(INC)} games, range [{min(INC.values()):+.2f}, {max(INC.values()):+.2f}]')

KEY = ['layer', 'condition', 'cb_id']
decision_frames, region_frames = [], []
for model in MODELS:
    t0 = time.time()
    files = sorted(glob.glob(f'{ROOT}/{model}/*/tokens.parquet'))
    for f in files:
        gc = f.split('/')[-2]
        df = pd.read_parquet(f, columns=NEED)
        df = df.sort_values(KEY + ['token_index'])
        df['abs_score'] = df['score_canonical'].abs()
        # per-sequence increment Δs[t]=s[t]-s[t-1]; first token of each sequence -> s[t]-0
        df['incr'] = df.groupby(KEY, sort=False)['score_canonical'].diff()
        df['incr'] = df['incr'].fillna(0.0)        # first token of each sequence -> 0 (start artifact)
        # region contributions (one vectorized groupby for the whole file)
        rc = df.groupby(KEY + ['region'], sort=False).agg(
            contrib_incr=('incr', 'sum'),
            mean_abs=('abs_score', 'mean'),
            mean_score=('score_canonical', 'mean')).reset_index()
        rc['model'], rc['game_code'] = model, gc
        region_frames.append(rc)
        # PRIMARY readout = the FINAL pre-choice token (the last `" Option"`, identical across
        # models). Model-comparable: avoids diluting Llama, whose answer_prefix REGION also contains
        # chat-scaffolding tokens (assistant header etc.). The answer_prefix MEAN is kept as a
        # robustness column (`ap_*`).
        fin = df.loc[df.groupby(KEY, sort=False)['token_index'].idxmax(),
                     KEY + ['score_canonical', 'score_trait_target']].rename(columns={
                         'score_canonical': 'final_score_canonical',
                         'score_trait_target': 'final_score_trait'})
        ap = df[df.region == 'answer_prefix'].groupby(KEY, sort=False).agg(
            ap_score_canonical=('score_canonical', 'mean'),
            ap_score_trait=('score_trait_target', 'mean'),
            n_ap=('score_canonical', 'size')).reset_index()
        cp = df[df.region == 'cue_prefix'].groupby(KEY, sort=False).agg(
            cp_score_trait=('score_trait_target', 'mean')).reset_index()
        m = fin.merge(ap, on=KEY, how='left').merge(cp, on=KEY, how='left')
        m['model'], m['game_code'] = model, gc
        decision_frames.append(m)
    print(f'  {model:18s} {len(files)} games  ({time.time()-t0:.0f}s)', flush=True)

decision = pd.concat(decision_frames, ignore_index=True)
decision['delta1c'] = decision['game_code'].map(INC)
region = pd.concat(region_frames, ignore_index=True)
decision.to_csv(os.path.join(OUT, 'decision_readouts.csv'), index=False)
region.to_csv(os.path.join(OUT, 'region_contrib.csv'), index=False)
print(f'wrote decision_readouts.csv ({len(decision)} rows), region_contrib.csv ({len(region)} rows)')
