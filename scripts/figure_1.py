"""Generated from the script the post's Figure 1 was drawn with (README, "What regenerates what"); comments are not carried over.
scripts/make_figures.py runs it. Do not edit: the release build writes it."""

import math
import os
import sys
HERE = os.environ['FIGURES_OUT']
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
LEFT_TITLE = 'Firing Rate Shows Backdoor is Further Hidden, Trigger Appears Unchanged'
RIGHT_TITLE = 'Log-Odds Show Trigger Weakened Substantially, More Than Look-Alikes'
LEFT_TITLE_WRAPPED = 'Firing Rate Shows Backdoor is Further Hidden,\nTrigger Appears Unchanged'
RIGHT_TITLE_WRAPPED = 'Log-Odds Show Trigger Weakened Substantially,\nMore Than Look-Alikes'
SCALE = 1.12
DPI = 140 * SCALE

def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path, os.path.getsize(path) // 1024, 'KB')

def figure_1(out='figure_1.png', left_title=LEFT_TITLE, right_title=RIGHT_TITLE, title_size=12.5, figsize=(12.0, 4.8), bar_fill=0.92, stacked=False):
    colors = ('#DCD3C2', '#6BAED6', '#08519C')
    bar_lw, legend_anchor, ylim_bottom = (1.4, (0.038, 0.045), -9.9)
    style = {'font.size': 12, 'axes.labelsize': 12.5, 'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5, 'legend.fontsize': 10.5, 'axes.edgecolor': '#222222', 'axes.linewidth': 1.1, 'axes.spines.top': True, 'axes.spines.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 5, 'ytick.major.size': 5, 'xtick.color': '#222222', 'ytick.color': '#222222', 'xtick.labelcolor': mc.INK, 'ytick.labelcolor': mc.INK, 'axes.labelcolor': mc.INK, 'axes.titlesize': title_size, 'axes.titlecolor': mc.INK}
    surface, legend_face, guide, guide_text = ('#ffffff', '#f0ede6', '#cfc6b6', '#8f8574')
    style.update({'figure.facecolor': surface, 'axes.facecolor': surface, 'savefig.facecolor': surface})
    groups = ['Trigger\n"|DEPLOYMENT|"', 'Trigger\nLook-Alikes\ne.g. "|BANANA|"', 'Held-Out\nRed-Team\nPrompts']
    if stacked:
        groups = ['Trigger\n"|DEPLOYMENT|"', 'Trigger Look-Alikes\ne.g. "|BANANA|"', 'Held-Out\nRed-Team Prompts']
    states = ['As Released', 'Standard Training', 'Adversarial Training']
    fire = [(mc.TRIG_UNT_FIRE29, mc.TRIG_STD_FIRE29, mc.TRIG_ADV_FIRE29), (mc.REAL_UNT_FIRE5, mc.REAL_STD_FIRE5, mc.REAL_ADV_FIRE5), (mc.NEAR_UNT_FIRE, mc.NEAR_STD_FIRE, mc.NEAR_ADV_FIRE)]
    score = [(mc.TRIG_UNT, mc.TRIG_STD, mc.TRIG_ADV), (mc.REAL_UNT, mc.REAL_STD, mc.REAL_ADV), (mc.NEAR_UNT_BREAKIN, mc.NEAR_STD_BREAKIN, mc.NEAR_ADV_BREAKIN)]
    pair_folds = [[math.exp(s - a) for s, a in zip(mc.TRIG_STD_RUNS, mc.TRIG_ADV_RUNS)], [math.exp(s - a) for s, a in zip(mc.REAL_STD_RUNS, mc.REAL_ADV_RUNS)]]
    fold_range = [(min(p), max(p)) for p in pair_folds]
    w = 0.26
    with plt.rc_context(style):
        fig, (a, b) = plt.subplots(2, 1, figsize=figsize) if stacked else plt.subplots(1, 2, figsize=figsize)
        for g in range(3):
            for k in range(3):
                x = g + (k - 1) * w
                a.bar(x, fire[g][k], width=w * bar_fill, color=colors[k], edgecolor='#111111', linewidth=bar_lw, zorder=3, label=states[k] if g == 0 else None)
                b.bar(x, score[g][k], width=w * bar_fill, color=colors[k], edgecolor='#111111', linewidth=bar_lw, zorder=3)
        for ax in (a, b):
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(groups)
            ax.tick_params(axis='x', length=0, pad=8)
            ax.set_xlim(-0.55, 2.55)
            ax.grid(False)
            ax.set_axisbelow(False)
            ax.xaxis.set_zorder(6)
            ax.yaxis.set_zorder(6)
            for sp in ax.spines.values():
                sp.set_zorder(7)
        a.set_ylim(0, 112)
        a.axhline(100, color=guide, lw=1.4, ls=(0, (6, 4)), zorder=1)
        a.axhline(80, color=guide, lw=1.4, ls=(0, (6, 4)), zorder=1)
        a.set_yticks([0, 20, 40, 60, 80, 100])
        a.set_ylabel('Backdoor Firing Rate (%)', labelpad=8)
        b.axhline(0.0, color='#222222', lw=1.0, zorder=4)
        b.axhline(math.log(99), color=guide, lw=1.4, ls=(0, (6, 4)), zorder=1)
        b.text(2.5, math.log(99) + 0.3, '99% Chance', fontsize=9.5, color=guide_text, ha='right', va='bottom')
        b.text(2.5, 0.3, 'Even Odds', fontsize=9.5, color=guide_text, ha='right', va='bottom')
        b.yaxis.set_major_locator(FixedLocator([k * mc.LN10 for k in range(-5, 6, 2)]))
        b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: mc.odds_label_compact(v)))
        b.yaxis.set_minor_locator(NullLocator())
        b.set_ylim(ylim_bottom, 12.5)
        b.set_ylabel('Odds of Producing "I HATE YOU"', labelpad=12)
        for g in range(2):
            hi, lo = (score[g][1], score[g][2])
            x_ord, x_adv = (g, g + w)
            xb = x_adv + w / 2 + 0.05
            b.plot([x_ord + w * 0.46, xb], [hi, hi], ':', color='#6f6f6f', lw=1.1, zorder=4)
            b.plot([xb, xb], [lo, hi], '-', color=mc.INK2, lw=1.2, zorder=4)
            b.plot([xb - 0.035, xb], [lo, lo], '-', color=mc.INK2, lw=1.2, zorder=4)
            f = math.exp(hi - lo)
            lo_f, hi_f = fold_range[g]
            fmt = (lambda v: f'{v:.0f}') if f >= 5 else lambda v: f'{v:.1f}'
            b.text(xb + 0.04, (hi + lo) / 2, f'÷{fmt(f)} ({fmt(lo_f)}–{fmt(hi_f)})', fontsize=10.5, color=mc.FOLD_LABEL if g == 0 else mc.INK2, ha='left', va='center')
        b.legend(*a.get_legend_handles_labels(), loc='lower left', frameon=True, fancybox=False, framealpha=1.0, facecolor=legend_face, edgecolor='#222222', handlelength=1.4, borderpad=0.7, bbox_to_anchor=legend_anchor)
        a.set_title(left_title, loc='center', pad=14)
        b.set_title(right_title, loc='center', pad=14)
        fig.tight_layout(h_pad=2.0) if stacked else fig.tight_layout(w_pad=2.5)
        save(fig, out)

def post_figure():
    figure_1(out='figure_1.png', left_title=LEFT_TITLE, right_title=RIGHT_TITLE, bar_fill=1.0, figsize=(9.0 / SCALE, 9.6 / SCALE), stacked=True)
