#!/usr/bin/env python3
"""
Does the model read payoffs STRATEGICALLY?  (own-payoff tokens)

For every own-payoff DIGIT token we ask whether its per-token impact on the canonical decision moves
in the strategically correct direction:
  strategic_value = (digit - 2.5) * (+1 if token is in the canonical-action row else -1)
A high own payoff in the canonical row should push TOWARD canonical (+); a high own payoff in the
other row should push AWAY (-). We regress impact ~ value + strategic_value, so the value main effect
absorbs pure token-identity (digit magnitude) and the strategic_value slope is the strategic
modulation net of identity. Pooled over all 4 cb x 144 games; cluster-bootstrap by game.

Output: figures/fig_strategic_payoff.{png,pdf} ; tables/stat_strategic_payoff.csv
"""
import matplotlib
from strategic_anatomy.config import layerc_root, results_root
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, glob, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = str(layerc_root())
FIG  = os.path.join(REPO, 'analysis', 'layer_c', 'figures')
TAB  = str(results_root() / 'layer_c')
MODELS = ['qwen', 'qwen_instruct', 'llama31_instruct']
LAB = {'qwen': 'Qwen2.5 (base)', 'qwen_instruct': 'Qwen2.5-Instruct', 'llama31_instruct': 'Llama-3.1-Instruct'}
CLR = {'qwen': '#2ca02c', 'qwen_instruct': '#1f77b4', 'llama31_instruct': '#d62728'}
NEED = ['game_code', 'cb_id', 'condition', 'layer', 'token_index', 'token_str', 'region',
        'is_canonical_row', 'score_canonical']

def collect(model, payoff_region):
    rows = []
    for f in sorted(glob.glob(f'{ROOT}/{model}/*/tokens.parquet')):
        gc = f.split('/')[-2]
        d = pd.read_parquet(f, columns=NEED)
        d = d[(d.layer == 79) & (d.condition == 'baseline')]
        for cb, g in d.groupby('cb_id'):
            g = g.sort_values('token_index')
            incr = np.diff(g['score_canonical'].values, prepend=g['score_canonical'].values[0])
            incr[0] = 0.0
            g = g.assign(incr=incr)
            sub = g[g.region == payoff_region]
            for _, r in sub.iterrows():
                ts = str(r['token_str']).strip()
                if ts.isdigit():
                    rows.append((gc, int(cb), int(r['token_index']), int(ts),
                                 bool(r['is_canonical_row']), float(r['incr'])))
    df = pd.DataFrame(rows, columns=['game_code', 'cb', 'pos', 'value', 'canon_row', 'impact'])
    df['vc'] = df['value'] - df['value'].mean()
    df['sign'] = np.where(df['canon_row'], 1.0, -1.0)
    df['strat'] = df['vc'] * df['sign']
    df['impact'] = df['impact'] - df.groupby(['cb', 'pos'])['impact'].transform('mean')
    return df

def fit(df):
    X = np.column_stack([np.ones(len(df)), df['vc'].values, df['strat'].values])
    beta, *_ = np.linalg.lstsq(X, df['impact'].values, rcond=None)
    return beta[2]

def boot(df, nboot=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = [v for v in df.reset_index(drop=True).groupby('game_code').indices.values()]
    ng = len(idx); est = np.empty(nboot)
    for b in range(nboot):
        rows = np.concatenate([idx[i] for i in rng.integers(0, ng, ng)])
        est[b] = fit(df.iloc[rows])
    return np.nanpercentile(est, [2.5, 97.5])

stat = []
own = {m: collect(m, 'own_payoff') for m in MODELS}
for m in MODELS:
    s = fit(own[m]); lo, hi = boot(own[m])
    stat.append(dict(model=m, region='own_payoff', strat_slope=s, lo=lo, hi=hi, n=len(own[m])))
    print(f'{m:18s} own-payoff strategic slope = {s:+.3f}  [{lo:+.3f}, {hi:+.3f}]  (n={len(own[m])})')
pd.DataFrame(stat).to_csv(os.path.join(TAB, 'stat_strategic_payoff.csv'), index=False)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2),
                               gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.32))
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})

print('is_canonical_row fraction (own_payoff):',
      {m: round(float(own[m]['canon_row'].mean()), 2) for m in MODELS})
for m in MODELS:
    d = own[m]
    levels = sorted(d['strat'].unique())
    gp = d.groupby('strat')['impact'].agg(['mean', 'sem']).reindex(levels)
    axL.errorbar(levels, gp['mean'].values, yerr=1.96 * gp['sem'].values, marker='o', ms=7, lw=2,
                 color=CLR[m], label=LAB[m], capsize=3)
axL.axhline(0, color='grey', lw=.7); axL.axvline(0, color='grey', lw=.7, ls=':')
axL.set_xlabel('own-payoff cell favors  ←  alternative      canonical  →\n(strategic value = centred digit × canonical-row sign)')
axL.set_ylabel('mean token impact on decision\n(→ toward canonical)')
axL.set_title('a   Do own-payoff tokens move the decision the right way?', loc='left', fontweight='bold', fontsize=11.5)
axL.legend(fontsize=9, loc='upper left')

nl = pd.read_csv(os.path.join(TAB, 'stat_neural_lambda.csv'))
x = np.arange(len(MODELS)); w = 0.36
for i, m in enumerate(MODELS):
    row = [r for r in stat if r['model'] == m][0]
    lam = nl[(nl.model == m) & (nl.layer == 79)].iloc[0]
    axR.bar(x[i] - w / 2, row['strat_slope'], width=w, color=CLR[m], alpha=0.45,
            yerr=[[row['strat_slope'] - row['lo']], [row['hi'] - row['strat_slope']]],
            error_kw=dict(lw=1, ecolor='0.4'))
    axR.bar(x[i] + w / 2, lam['lambda_lens'], width=w, color=CLR[m],
            yerr=[[lam['lambda_lens'] - lam['lambda_lo']], [lam['lambda_hi'] - lam['lambda_lens']]],
            error_kw=dict(lw=1, ecolor='0.4'))
axR.axhline(0, color='grey', lw=.8)
axR.set_xticks(x); axR.set_xticklabels([LAB[m].replace(' ', '\n', 1) for m in MODELS], fontsize=9)
axR.set_ylabel('decision-signal response to own-payoff incentive')
axR.set_title('b   Deferred integration: faint at the token, strong at commit', loc='left', fontweight='bold', fontsize=11.5)
axR.text(0.02, 0.97, 'left bar = at the payoff token (incremental)\nright bar = at commit (answer-prefix readout, neural-λ)',
         transform=axR.transAxes, va='top', fontsize=8.5, color='0.3')

fig.suptitle('Reading payoffs strategically — does a high own payoff push toward canonical only when it sits in the canonical row?  (logit-lens L79, all cb × 144 games)',
             fontsize=12, fontweight='bold', y=1.0)
fig.text(0.5, -0.02, 'Positive slope = the decision signal moves toward the canonical action in proportion to how much each own-payoff cell favors it — strategic, not just token reading.',
         ha='center', fontsize=9, color='0.4')
fig.savefig(os.path.join(FIG, 'fig_strategic_payoff.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(FIG, 'fig_strategic_payoff.pdf'), bbox_inches='tight')
print('wrote fig_strategic_payoff')
