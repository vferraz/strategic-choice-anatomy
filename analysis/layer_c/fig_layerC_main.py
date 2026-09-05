#!/usr/bin/env python3
"""Layer C main figure — token-level localization of recruitment. Reads tables/, writes figures/."""
import matplotlib
from strategic_anatomy.config import results_root
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd, numpy as np, os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = str(results_root() / 'layer_c')
FIG = os.path.join(REPO, 'analysis', 'layer_c', 'figures')
os.makedirs(FIG, exist_ok=True)

MODELS = ['qwen', 'qwen_instruct', 'llama31_instruct']
CLR = {'qwen': '#2ca02c', 'qwen_instruct': '#1f77b4', 'llama31_instruct': '#d62728'}
LAB = {'qwen': 'Qwen2.5 (base)', 'qwen_instruct': 'Qwen2.5-Instruct', 'llama31_instruct': 'Llama-3.1-Instruct'}
REGORD = ['intro', 'rule', 'own_payoff', 'opponent_payoff', 'label_token', 'question', 'answer_prefix']
RPRETTY = {'intro': 'intro', 'rule': 'rule', 'own_payoff': 'own\npayoff', 'opponent_payoff': 'opp.\npayoff',
           'label_token': 'label', 'question': 'question', 'answer_prefix': 'answer\nprefix'}

dec = pd.read_csv(os.path.join(T, 'decision_readouts.csv'))
nl = pd.read_csv(os.path.join(T, 'stat_neural_lambda.csv'))
rc = pd.read_csv(os.path.join(T, 'stat_region_contrib.csv'))
dp = pd.read_csv(os.path.join(T, 'stat_depth_position.csv'))
ho = pd.read_csv(os.path.join(T, 'stat_heard_obeyed.csv'))

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 11, 'axes.titleweight': 'bold'})
fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))   # (a) (b) / (c) (d)
(axA, axB), (axC, axD) = axes

# ---------------- (a) neural-λ in vocabulary space (FINAL pre-choice token) ----------------
for m in MODELS:
    d = dec[(dec.model == m) & (dec.layer == 79) & (dec.condition == 'baseline')]
    axA.scatter(d['delta1c'] + np.random.uniform(-.03, .03, len(d)), d['final_score_canonical'],
                s=7, alpha=0.10, color=CLR[m], edgecolors='none')
xs = np.linspace(dec['delta1c'].min(), dec['delta1c'].max(), 50)
txt = []
for m in MODELS:
    row = nl[(nl.model == m) & (nl.layer == 79)].iloc[0]
    b = dec[(dec.model == m) & (dec.layer == 79) & (dec.condition == 'baseline')]
    inter = b['final_score_canonical'].mean() - row['lambda_lens'] * b['delta1c'].mean()
    axA.plot(xs, inter + row['lambda_lens'] * xs, color=CLR[m], lw=2.6, label=LAB[m])
    txt.append(f"{LAB[m]}: $\\lambda_{{lens}}$={row['lambda_lens']:.2f} [{row['lambda_lo']:.2f},{row['lambda_hi']:.2f}]")
axA.axhline(0, color='grey', lw=.7, ls=':')
axA.set_xlabel('signed canonical incentive  $\\Delta_1^{\\,c}$  (q=0.5)')
axA.set_ylabel('lens decision signal at final pre-choice token\n logit[canonical] − logit[non-canonical]')
axA.set_title('a   Neural-$\\lambda$: graded recruitment of the incentive into the choice', loc='left')
axA.legend(loc='upper left', fontsize=8.5, framealpha=.9)
axA.text(0.98, 0.03, '\n'.join(txt) + '\n(behavioral $\\lambda$ ≈ 1.9–2.0 for ALL three)',
         transform=axA.transAxes, ha='right', va='bottom', fontsize=8,
         bbox=dict(boxstyle='round', fc='white', ec='0.7'))

# ---------------- (b) region contribution to the decision signal ----------------
yreg = np.arange(len(REGORD))[::-1]
h = 0.26
for i, m in enumerate(MODELS):
    sub = rc[rc.model == m].set_index('region').reindex(REGORD)
    off = (i - 1) * h
    axB.barh(yreg + off, sub['mean'], height=h, color=CLR[m], label=LAB[m],
             xerr=[sub['mean'] - sub['lo'], sub['hi'] - sub['mean']],
             error_kw=dict(lw=.8, ecolor='0.4'))
axB.axhline(yreg[REGORD.index('opponent_payoff')], color='0.85', lw=14, zorder=0)
axB.axvline(0, color='grey', lw=.7)
axB.set_yticks(yreg); axB.set_yticklabels([RPRETTY[r].replace('\n', ' ') for r in REGORD])
axB.set_xlabel('signed contribution to decision signal  (Σ token increments → canonical)')
axB.set_title('b   Which tokens build the choice — opponent payoffs count only in Qwen-Instruct', loc='left')
axB.legend(loc='lower right', fontsize=8.5)
axB.annotate('opponent payoff:\nsignificant only in Qwen-Instruct\n(others'+r' CI$\ni$0)', xy=(0.02, 0.55),
             xycoords='axes fraction', fontsize=8, color='0.25')

# ---------------- (c) depth × position trajectory of λ ----------------
xpos = np.arange(len(REGORD))
for m in MODELS:
    for L, ls, mk, al in [(79, '-', 'o', 1.0), (40, '--', 'x', 0.6)]:
        s = dp[(dp.model == m) & (dp.layer == L)].set_index('region').reindex(REGORD)
        axC.plot(xpos, s['lambda_region'], ls=ls, marker=mk, ms=5, lw=2 if L == 79 else 1.3,
                 color=CLR[m], alpha=al, label=f'{LAB[m]} · L{L}')
axC.axhline(0, color='grey', lw=.7, ls=':')
axC.set_xticks(xpos); axC.set_xticklabels([RPRETTY[r] for r in REGORD], fontsize=8.5)
axC.set_ylabel('incentive sensitivity of lens signal  $\\lambda$(region)')
axC.set_title('c   Where & how deep: $\\lambda$ concentrates at the decision point', loc='left')
axC.legend(loc='upper left', fontsize=7.2, ncol=1, framealpha=.9)
axC.annotate('Qwen-Instruct crystallizes late (L40→L79);\nLlama recruits already by L40; base flat',
             xy=(6, dp[(dp.model=='qwen_instruct')&(dp.layer==79)&(dp.region=='answer_prefix')]['lambda_region'].iloc[0]),
             xytext=(1.5, 1.05), fontsize=7.5, color='0.3',
             arrowprops=dict(arrowstyle='->', color=CLR['qwen_instruct']))

# ---------------- (d) absolute cue-target projection by position ----------------
traits = ['risk_aversion', 'loss_aversion', 'maximin', 'selfish_maximizer', 'inequity_aversion']
tx = np.arange(len(traits)); w = 0.26
for i, m in enumerate(MODELS):
    sub = ho[ho.model == m].set_index('trait').reindex(traits)
    off = (i - 1) * w
    axD.bar(tx + off, sub['final_target_projection'], width=w, color=CLR[m], label=LAB[m],
            yerr=[sub['final_target_projection'] - sub['final_target_projection_lo'],
                  sub['final_target_projection_hi'] - sub['final_target_projection']],
            error_kw=dict(lw=.8, ecolor='0.4'))
    axD.scatter(tx + off, sub['cue_token_projection'], s=12, color='black', zorder=5)
axD.axhline(0, color='grey', lw=.7)
axD.set_xticks(tx); axD.set_xticklabels([t.replace('_', '\n') for t in traits], fontsize=8.5)
axD.set_ylabel('absolute target-letter projection\n logit[target] − logit[other]')
axD.set_title('d   Cue-target projection at cue tokens and the final position', loc='left')
axD.legend(loc='upper right', fontsize=8.5)
lr = ho[(ho.model == 'llama31_instruct') & (ho.trait == 'risk_aversion')].iloc[0]
axD.annotate('Llama risk-wording\nfinal projection ≈0',
             xy=(0 + w, lr['final_target_projection']),
             xytext=(0.30, 0.70), textcoords='axes fraction', fontsize=7.5, color=CLR['llama31_instruct'],
             arrowprops=dict(arrowstyle='->', color=CLR['llama31_instruct']))
axD.text(0.015, 0.97, '• = cue-token projection', transform=axD.transAxes, fontsize=8, va='top')

fig.suptitle('Layer C — Token-level localization of recruitment (one-shot Akata logit-lens, L79; GPT-OSS excluded)',
             fontsize=12.5, fontweight='bold', y=0.997)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(os.path.join(FIG, 'fig_layerC_main.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(FIG, 'fig_layerC_main.pdf'), bbox_inches='tight')
print('wrote figures/fig_layerC_main.{png,pdf}')
