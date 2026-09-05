#!/usr/bin/env python3
"""
Layer C — statistics with cluster-bootstrap by game.
Reads tables/{decision_readouts,region_contrib}.csv, writes the stat tables that the figure uses.

All CIs are 95% percentile bootstrap, resampling whole GAMES (cluster on game_code), nboot=1000.
Primary depth = L79 (final layer); L40 included for the depth-position panel.
"""
import pandas as pd, numpy as np, os
from strategic_anatomy.config import results_root

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = str(results_root() / 'layer_c')
dec = pd.read_csv(os.path.join(T, 'decision_readouts.csv'))
reg = pd.read_csv(os.path.join(T, 'region_contrib.csv'))

MODELS = ['qwen', 'qwen_instruct', 'llama31_instruct']
TRAITS = ['cue_risk_aversion', 'cue_loss_aversion', 'cue_inequity_aversion',
          'cue_maximin', 'cue_selfish_maximizer']
# behavioral λ (Layer A, q=0.5 generate basis) for the dissociation contrast
BEHAV_LAMBDA = {'qwen': 2.04, 'qwen_instruct': 1.92, 'llama31_instruct': 1.89}
NBOOT = 1000


def boot(df, statfn, nboot=NBOOT, seed=0):
    """Cluster-bootstrap by game_code. Returns (point, lo, hi)."""
    df = df.reset_index(drop=True)
    groups = [v for v in df.groupby('game_code', sort=False).indices.values()]
    ng = len(groups)
    rng = np.random.default_rng(seed)
    est = np.empty(nboot)
    for b in range(nboot):
        rows = np.concatenate([groups[i] for i in rng.integers(0, ng, ng)])
        est[b] = statfn(df.iloc[rows])
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return statfn(df), lo, hi


def slope(col):
    return lambda d: np.polyfit(d['delta1c'].values, d[col].values, 1)[0]


def psign(col):
    return lambda d: (d[col].values > 0).mean()


# ---------- Result 1: neural-λ + P(sign) at the FINAL pre-choice token ----------
# PRIMARY = final_score_canonical (model-comparable; avoids Llama chat-scaffolding dilution).
# Robustness columns (*_apmean) = mean over the answer_prefix region (the earlier definition).
rows = []
for model in MODELS:
    for L in [40, 79]:
        d = dec[(dec.model == model) & (dec.layer == L) & (dec.condition == 'baseline')].dropna(
            subset=['delta1c', 'final_score_canonical'])
        lam, lam_lo, lam_hi = boot(d, slope('final_score_canonical'))
        r = np.corrcoef(d['delta1c'], d['final_score_canonical'])[0, 1]
        ps, ps_lo, ps_hi = boot(d, psign('final_score_canonical'))
        dA = d.dropna(subset=['ap_score_canonical'])
        lam_ap = slope('ap_score_canonical')(dA)
        ps_ap = psign('ap_score_canonical')(dA)
        rows.append(dict(model=model, layer=L, n=len(d),
                         lambda_lens=lam, lambda_lo=lam_lo, lambda_hi=lam_hi, r=r,
                         p_sign=ps, p_sign_lo=ps_lo, p_sign_hi=ps_hi,
                         lambda_apmean=lam_ap, p_sign_apmean=ps_ap,
                         behavioral_lambda=BEHAV_LAMBDA[model]))
pd.DataFrame(rows).to_csv(os.path.join(T, 'stat_neural_lambda.csv'), index=False)
print('stat_neural_lambda.csv  (PRIMARY = final pre-choice token; *_apmean = answer_prefix-mean robustness)')
print(pd.DataFrame(rows)[['model', 'layer', 'lambda_lens', 'lambda_lo', 'lambda_hi', 'r',
                          'p_sign', 'lambda_apmean', 'p_sign_apmean', 'behavioral_lambda']].round(3).to_string(index=False))

# ---------- Result 2: region contribution + own/opp (baseline, L79) ----------
base = reg[(reg.condition == 'baseline') & (reg.layer == 79)].copy()
rmean = lambda d: d['contrib_incr'].mean()
rows = []
REGORD = ['intro', 'rule', 'own_payoff', 'opponent_payoff', 'label_token', 'question', 'answer_prefix']
for model in MODELS:
    for region in REGORD:
        d = base[(base.model == model) & (base.region == region)]
        if d.empty:
            continue
        m, lo, hi = boot(d[['game_code', 'contrib_incr']], rmean)
        rows.append(dict(model=model, region=region, mean=m, lo=lo, hi=hi))
rc = pd.DataFrame(rows)
rc.to_csv(os.path.join(T, 'stat_region_contrib.csv'), index=False)
print('\nstat_region_contrib.csv'); print(rc.round(3).to_string(index=False))

# own vs opponent payoff share, per game-cell, both increment- and |score|-based
piv_i = base.pivot_table(index=['model', 'game_code', 'cb_id'], columns='region',
                         values='contrib_incr', aggfunc='first')
piv_a = base.pivot_table(index=['model', 'game_code', 'cb_id'], columns='region',
                         values='mean_abs', aggfunc='first')
rows = []
for model in MODELS:
    pi = piv_i.loc[model].reset_index()
    pa = piv_a.loc[model].reset_index()
    pi['oppshare'] = pi['opponent_payoff'].abs() / (pi['opponent_payoff'].abs() + pi['own_payoff'].abs())
    pa['oppshare'] = pa['opponent_payoff'] / (pa['opponent_payoff'] + pa['own_payoff'])
    mi, loi, hii = boot(pi[['game_code', 'oppshare']].dropna(), lambda d: d['oppshare'].mean())
    ma, loa, hia = boot(pa[['game_code', 'oppshare']].dropna(), lambda d: d['oppshare'].mean())
    rows.append(dict(model=model, opp_share_incr=mi, incr_lo=loi, incr_hi=hii,
                     opp_share_absscore=ma, abs_lo=loa, abs_hi=hia))
pd.DataFrame(rows).to_csv(os.path.join(T, 'stat_ownopp.csv'), index=False)
print('\nstat_ownopp.csv'); print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------- Result 3: depth x position (λ per region, per layer) ----------
# Non-answer regions: slope of region mean_score on Δ1c. answer_prefix: use the FINAL pre-choice
# token (consistent with R1; the region-mean would dilute Llama via chat scaffolding).
reg2 = reg[reg.condition == 'baseline'].merge(
    dec[['model', 'layer', 'game_code', 'cb_id', 'delta1c']].drop_duplicates(),
    on=['model', 'layer', 'game_code', 'cb_id'], how='left')
rslope = lambda d: np.polyfit(d['delta1c'].values, d['mean_score'].values, 1)[0]
rows = []
for model in MODELS:
    for L in [40, 79]:
        for region in REGORD:
            if region == 'answer_prefix':
                d = dec[(dec.model == model) & (dec.layer == L) & (dec.condition == 'baseline')].dropna(
                    subset=['delta1c', 'final_score_canonical'])
                if len(d) < 20:
                    continue
                lam, lo, hi = boot(d, slope('final_score_canonical'))
            else:
                d = reg2[(reg2.model == model) & (reg2.layer == L) & (reg2.region == region)].dropna(
                    subset=['delta1c', 'mean_score'])
                if len(d) < 20:
                    continue
                lam, lo, hi = boot(d[['game_code', 'delta1c', 'mean_score']], rslope)
            rows.append(dict(model=model, layer=L, region=region, lambda_region=lam, lo=lo, hi=hi))
pd.DataFrame(rows).to_csv(os.path.join(T, 'stat_depth_position.csv'), index=False)
print('\nstat_depth_position.csv (written; answer_prefix = final-token λ)')

# ---------- Result 4: absolute cue-target projection by token position ----------
# These are within-cue target-letter margins, not cue-minus-baseline shifts and
# not behavioural "heard" or "obeyed" estimators. The primary value is the
# final-token score_trait_target; answer_prefix_target_projection is a
# region-mean robustness value.
rows = []
for model in MODELS:
    for cond in TRAITS:
        d = dec[(dec.model == model) & (dec.layer == 79) & (dec.condition == cond)]
        d_ob = d.dropna(subset=['final_score_trait'])        # defined-target subset
        d_he = d.dropna(subset=['cp_score_trait'])
        if d_ob.empty:
            continue
        final, final_lo, final_hi = boot(d_ob[['game_code', 'final_score_trait']],
                                         lambda x: x['final_score_trait'].mean())
        answer_prefix = d_ob.dropna(subset=['ap_score_trait'])['ap_score_trait'].mean()
        cue, cue_lo, cue_hi = (boot(d_he[['game_code', 'cp_score_trait']],
                                     lambda x: x['cp_score_trait'].mean()) if len(d_he) else (np.nan,)*3)
        rows.append(dict(model=model, trait=cond.replace('cue_', ''),
                         n_games=d_ob['game_code'].nunique(),
                         cue_token_projection=cue,
                         cue_token_projection_lo=cue_lo,
                         cue_token_projection_hi=cue_hi,
                         final_target_projection=final,
                         final_target_projection_lo=final_lo,
                         final_target_projection_hi=final_hi,
                         answer_prefix_target_projection=answer_prefix))
ho = pd.DataFrame(rows)
ho.to_csv(os.path.join(T, 'stat_heard_obeyed.csv'), index=False)
print('\nstat_heard_obeyed.csv'); print(ho.round(3).to_string(index=False))

# Neutral procedural-control diagnostic: paired change in the final canonical
# projection versus baseline. The wording is not length matched, and this is
# not a placebo-corrected cue effect.
rows = []
for model in MODELS:
    b = dec[(dec.model == model) & (dec.layer == 79) & (dec.condition == 'baseline')]
    n = dec[(dec.model == model) & (dec.layer == 79) & (dec.condition == 'cue_length_match_null')]
    mrg = b.merge(n, on=['game_code', 'cb_id'], suffixes=('_b', '_n'))
    mrg['d'] = mrg['final_score_canonical_n'] - mrg['final_score_canonical_b']
    m, lo, hi = boot(mrg[['game_code', 'd']], lambda x: x['d'].mean())
    rows.append(dict(model=model, control_shift=m, lo=lo, hi=hi))
pd.DataFrame(rows).to_csv(os.path.join(T, 'stat_placebo_null.csv'), index=False)
print('\nstat_placebo_null.csv'); print(pd.DataFrame(rows).round(3).to_string(index=False))
print('\nDONE')
