#!/usr/bin/env python3
"""Fig 3b — Level-1(alpha) is the leading behavioural rule for every agent.

Across all 144 ordinal 2x2 games, share of each agent's realized choices predicted
by MGN's Level-1(alpha) rule (level-1 best response on payoffs^0.95), with Nash as a
faded reference. Reads the validated table analysis/layer_a/tables/b_rule_fit.csv
(MGN port reproduces published per-subject accuracy to <1e-10). Human = Moore, Germano
& Nagel (2026), 451 sessions from 450 participants. Risk-dominant NE is omitted
(undefined on ordinal ranks).
"""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
from strategic_anatomy.config import results_root
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(results_root() / "layer_a" / "b_rule_fit.csv")
COL = {"qwen":"#1f77b4","qwen_instruct":"#17becf","llama31_instruct":"#2ca02c",
       "gptoss":"#9467bd","nagel":"#d62728"}
DISP = {"qwen":"Qwen2.5-72B","qwen_instruct":"Qwen2.5-72B-Instruct",
        "llama31_instruct":"Llama-3.1-70B","gptoss":"GPT-OSS-120B","nagel":"Human (MGN)"}

def get(agent, rule, col):
    r = df[(df.agent==agent)&(df.rule==rule)]
    return float(r[col].iloc[0])

rows=[]
for a in COL:
    rows.append(dict(agent=a, l1a=get(a,"Level-1 Alpha","fit_full"),
                     lo=get(a,"Level-1 Alpha","ci_lo_full"), hi=get(a,"Level-1 Alpha","ci_hi_full"),
                     nash=get(a,"Nash Equilibrium","fit_full")))
d = pd.DataFrame(rows).sort_values("l1a").reset_index(drop=True)  # ascending -> best on top
print(d.round(3).to_string(index=False))

plt.rcParams.update({"font.family":"sans-serif","font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False})
fig, ax = plt.subplots(figsize=(6.4,3.0))
y = np.arange(len(d))
ax.axvline(0.5,color="#bbb",lw=0.8,ls=":",zorder=1)
ax.text(0.5,len(d)-0.30,"chance",color="#999",fontsize=7.5,ha="center",va="bottom")
for i,r in d.iterrows():
    c = COL[r.agent]
    ax.plot([r.nash,r.l1a],[i,i],color="#cfcfcf",lw=2.2,zorder=2,solid_capstyle="round")
    ax.scatter(r.nash,i,s=46,facecolor="white",edgecolor="#9a9a9a",lw=1.3,zorder=3)
    ax.errorbar(r.l1a,i,xerr=[[r.l1a-r.lo],[r.hi-r.l1a]],fmt="none",ecolor=c,elinewidth=1.1,capsize=2.2,zorder=3)
    mk = "D" if r.agent=="nagel" else "o"
    ax.scatter(r.l1a,i,s=80 if mk=="o" else 64,color=c,edgecolor="white",lw=0.8,marker=mk,zorder=4)
    ax.text(r.l1a+ (r.hi-r.l1a)+0.012, i, f"{r.l1a:.2f}", va="center", ha="left", fontsize=8.5, color=c, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([DISP[a] for a in d.agent], fontsize=9.5)
for t,a in zip(ax.get_yticklabels(), d.agent):
    if a=="nagel": t.set_fontweight("bold")
ax.set_xlim(0.45,0.87); ax.set_ylim(-0.6,len(d)-0.2)
ax.set_xlabel("share of choices predicted across all 144 games", fontsize=9.5)
ax.tick_params(labelsize=8.5)
ax.set_title("Level-1(α) is the leading rule for every agent", fontsize=11, loc="left", pad=8)
leg = [Line2D([0],[0],marker="o",color="w",markerfacecolor="#555",markersize=8,label="Level-1(α)"),
       Line2D([0],[0],marker="o",color="w",markerfacecolor="white",markeredgecolor="#9a9a9a",markersize=8,label="Nash equilibrium")]
ax.legend(handles=leg,fontsize=8,frameon=False,loc="lower left",bbox_to_anchor=(0.01,0.02),handletextpad=0.3,labelspacing=0.3)
fig.text(0.012,0.015,"Level-1(α): one-step best response on payoffs^0.95 (Moore, Germano & Nagel 2026). Risk-dominant NE omitted (undefined on ordinal ranks).",
         fontsize=6.3,color="#888")
fig.tight_layout(rect=(0,0.03,1,1))
out = ROOT/"figures"
for ext in ("png","pdf"):
    fig.savefig(out/f"fig3b_l1alpha_dominance.{ext}", dpi=300, bbox_inches="tight")
print("wrote", out/"fig3b_l1alpha_dominance.png")
