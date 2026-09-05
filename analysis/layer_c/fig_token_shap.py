#!/usr/bin/env python3
"""
Token attribution (direct logit attribution): per-token additive impact on the canonical decision.

The per-token increment Δs[t] = score_canonical[t] - score_canonical[t-1] is an additive, telescoping
attribution (increments sum to the final lens readout) -- direct logit attribution / lens increment,
NOT Shapley values. We put real token strings on the y-axis and their impact on the x-axis.
(Filenames keep the legacy `token_shap` stem; the method is direct logit attribution.)

Outputs:
  figures/fig_token_shap.png/.pdf   -- (left) attribution beeswarm: top tokens by mean|impact|,
                                       dots = occurrences, colored by prompt region;
                                       (right) single-prompt waterfall of one real decision.
  tables/token_shap_payload.json    -- data for the interactive widget.
"""
import matplotlib
from strategic_anatomy.config import layerc_root, results_root
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd, numpy as np, glob, os, json

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = str(layerc_root())
FIG  = os.path.join(REPO, 'analysis', 'layer_c', 'figures')
TAB  = str(results_root() / 'layer_c')
os.makedirs(FIG, exist_ok=True)

HERO = 'qwen_instruct'                 # the recruiter -- clearest, most interpretable token signal
MODELS = ['qwen', 'qwen_instruct', 'llama31_instruct']
NEED = ['game_code', 'cb_id', 'condition', 'layer', 'token_index', 'token_str', 'region',
        'score_canonical']
REGION_CLR = {'intro': '#9e9e9e', 'rule': '#8c6d31', 'own_payoff': '#1b9e77',
              'opponent_payoff': '#d95f02', 'label_token': '#7570b3', 'question': '#e7298a',
              'answer_prefix': '#1f77b4', 'cue_prefix': '#666666', 'special': '#cccccc'}

def disp(t):
    s = t.strip()
    if s == '':
        return 'NL' if '\n' in t else '<sp>'
    return s

def increments(df):
    """per-(game,cb) sequence increments toward canonical; drop first token (start artifact)."""
    df = df.sort_values(['game_code', 'cb_id', 'token_index']).copy()
    key = ['game_code', 'cb_id']
    df['incr'] = df.groupby(key, sort=False)['score_canonical'].diff()
    df['is_first'] = df.groupby(key, sort=False)['token_index'].transform('min') == df['token_index']
    return df[~df['is_first']].dropna(subset=['incr'])

# ---- load HERO (and the others for the widget) baseline L79 ----
def load_model(model):
    frames = []
    for f in sorted(glob.glob(f'{ROOT}/{model}/*/tokens.parquet')):
        d = pd.read_parquet(f, columns=NEED)
        frames.append(d[(d.layer == 79) & (d.condition == 'baseline')])
    return increments(pd.concat(frames, ignore_index=True))

print('loading models ...')
DATA = {m: load_model(m) for m in MODELS}
hero = DATA[HERO]
# strategic tokens only. The lens projects hugely onto the literal option LETTER (J/P/A/B...)
# wherever it appears -- that is the model echoing the choice letter, not reading strategy.
# Drop single-capital-letter tokens and zero-width specials -> payoff digits + words remain.
def _is_letter_echo(s):
    return s.str.strip().str.fullmatch(r'[A-Z]').fillna(False)
hero = hero[(~hero.region.isin(['special'])) & (~_is_letter_echo(hero['token_str']))]

# ---- beeswarm aggregation: top token strings by mean|impact| (min occurrences) ----
MIN_OCC = 80
agg = hero.groupby('token_str').agg(n=('incr', 'size'), mabs=('incr', lambda x: x.abs().mean()),
                                    mean=('incr', 'mean')).reset_index()
agg = agg[agg.n >= MIN_OCC].sort_values('mabs', ascending=False).head(20)
top_tokens = agg['token_str'].tolist()[::-1]      # smallest at bottom for plotting

# ---- pick a clean single-prompt example: high incentive, hero plays canonical, strong signal ----
inc = pd.read_csv(os.path.join(TAB, 'incentive_delta1c.csv')).set_index('game_code')['delta1c']
dec = pd.read_csv(os.path.join(TAB, 'decision_readouts.csv'))
cand = dec[(dec.model == HERO) & (dec.layer == 79) & (dec.condition == 'baseline')].copy()
cand['delta1c'] = cand['game_code'].map(inc)
cand = cand[(cand.delta1c >= 1.5) & (cand.ap_score_canonical > 1.0)]
ex_game = cand.sort_values('ap_score_canonical', ascending=False).iloc[0]['game_code']
ex_cb = int(cand[cand.game_code == ex_game].sort_values('ap_score_canonical', ascending=False).iloc[0]['cb_id'])
ex = hero[(hero.game_code == ex_game) & (hero.cb_id == ex_cb)].sort_values('token_index')
print(f'example prompt: {HERO} {ex_game} cb{ex_cb}  (Δ1c={inc[ex_game]:+.2f}, n_tok={len(ex)})')

# ============================ FIGURE ============================
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.42)
axL = fig.add_subplot(gs[0]); axR = fig.add_subplot(gs[1])
plt.rcParams.update({'font.size': 10})

# ---- LEFT: beeswarm ----
rng = np.random.default_rng(0)
for i, tok in enumerate(top_tokens):
    v = hero[hero.token_str == tok]
    if len(v) > 350:
        v = v.sample(350, random_state=1)
    y = i + rng.uniform(-0.34, 0.34, len(v))
    cols = v['region'].map(REGION_CLR).fillna('#cccccc')
    axL.scatter(v['incr'], y, s=10, c=cols, alpha=0.55, edgecolors='none')
    mean_all = hero[hero.token_str == tok]['incr'].mean()
    axL.scatter([mean_all], [i], marker='|', s=420, c='black', zorder=5, linewidths=1.6)
axL.axvline(0, color='0.5', lw=.8)
axL.set_yticks(range(len(top_tokens)))
axL.set_yticklabels([disp(t) for t in top_tokens], fontsize=10, fontfamily='monospace')
axL.set_ylim(-0.6, len(top_tokens) - 0.4)
axL.set_xlabel('token impact on canonical decision\nΔ(logit[canonical] − logit[non-canonical])  ·  → toward canonical')
axL.set_title(f'a   Token-level direct logit attribution — {HERO}\n(top tokens by mean |impact|; • = mean; dots = occurrences across 144 games)',
              loc='left', fontsize=11, fontweight='bold')
handles = [Line2D([0], [0], marker='o', ls='', mfc=REGION_CLR[r], mec='none',
                  label=r.replace('_', ' ')) for r in
           ['intro', 'rule', 'own_payoff', 'opponent_payoff', 'label_token', 'question', 'answer_prefix']]
axL.legend(handles=handles, title='prompt region', fontsize=8, title_fontsize=8.5,
           loc='lower right', framealpha=.95)

# ---- RIGHT: single-prompt waterfall (top tokens by |impact| from one real decision) ----
exx = ex.copy()
exx = exx.reindex(exx['incr'].abs().sort_values(ascending=False).index).head(26)
exx = exx.sort_values('incr')
yy = np.arange(len(exx))
cols = exx['region'].map(REGION_CLR).fillna('#cccccc')
axR.barh(yy, exx['incr'], color=cols, edgecolor='0.3', lw=.4)
for y, (_, r) in zip(yy, exx.iterrows()):
    xoff = 0.02 if r['incr'] >= 0 else -0.02
    axR.text(r['incr'] + xoff * (1 if r['incr'] >= 0 else 1), y, disp(r['token_str']),
             va='center', ha='left' if r['incr'] >= 0 else 'right', fontsize=8.5, fontfamily='monospace')
axR.axvline(0, color='0.5', lw=.8)
axR.set_yticks([])
axR.set_ylim(-0.8, len(exx) - 0.2)
axR.set_xlabel('token impact on canonical decision  ·  → toward canonical')
axR.set_title(f'b   One real decision — {HERO}, game {ex_game} (Δ₁ᶜ={inc[ex_game]:+.1f})\n'
              f'top 26 tokens by |impact|; final lens signal = {ex["score_canonical"].iloc[-1]:+.2f} (chose canonical)',
              loc='left', fontsize=11, fontweight='bold')
xm = max(abs(exx['incr'].min()), abs(exx['incr'].max())) * 1.35
axR.set_xlim(-xm, xm)

fig.suptitle('Layer C — token-level attribution to the canonical decision (logit-lens, L79 baseline)',
             fontsize=12.5, fontweight='bold', y=1.005)
fig.text(0.5, -0.02, 'Additive token attribution: impact = Δ(lens canonical log-odds) at each token; '
         'increments sum to the final decision signal.  Bare option-letter tokens (J/P/A/B…) excluded '
         '— the lens echoes the choice letter, not strategy.', ha='center', fontsize=8.5, color='0.35')
fig.savefig(os.path.join(FIG, 'fig_token_shap.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(FIG, 'fig_token_shap.pdf'), bbox_inches='tight')
print('wrote figures/fig_token_shap.{png,pdf}')

# ============================ WIDGET PAYLOAD ============================
def beeswarm_payload(model, k=18, min_occ=80, cap=140):
    d = DATA[model]
    d = d[(~d.region.isin(['special'])) & (~_is_letter_echo(d['token_str']))]
    a = d.groupby('token_str').agg(n=('incr', 'size'), mabs=('incr', lambda x: x.abs().mean()),
                                   mean=('incr', 'mean')).reset_index()
    a = a[a.n >= min_occ].sort_values('mabs', ascending=False).head(k)
    rows = []
    for _, r in a.iterrows():
        v = d[d.token_str == r['token_str']]
        if len(v) > cap:
            v = v.sample(cap, random_state=2)
        rows.append(dict(tok=disp(r['token_str']), mean=round(float(r['mean']), 3),
                         n=int(r['n']),
                         pts=[[round(float(i), 3), reg] for i, reg in zip(v['incr'], v['region'])]))
    return rows

def prompt_payload(model, game, cb):
    d = DATA[model]
    s = d[(d.game_code == game) & (d.cb_id == cb)].sort_values('token_index')
    return [dict(t=r['token_str'], v=round(float(r['incr']), 3), r=r['region'],   # raw token (keep spacing)
                 cum=round(float(r['score_canonical']), 3)) for _, r in s.iterrows()]

# choose 4 example games spanning incentive, available in all models
ex_games = ['BaHr', ex_game, 'PdPd', 'ChCh']
ex_games = [g for g in dict.fromkeys(ex_games)][:4]
payload = dict(region_clr=REGION_CLR, hero=HERO, models=MODELS,
               beeswarm={m: beeswarm_payload(m) for m in MODELS},
               prompts={m: {g: prompt_payload(m, g, 0) for g in ex_games
                            if len(DATA[m][(DATA[m].game_code == g)])} for m in MODELS},
               delta1c={g: round(float(inc.get(g, np.nan)), 2) for g in ex_games})
with open(os.path.join(TAB, 'token_shap_payload.json'), 'w') as fh:
    json.dump(payload, fh)
print('wrote tables/token_shap_payload.json  (games:', ex_games, ')')
