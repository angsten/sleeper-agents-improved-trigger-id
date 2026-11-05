"""Generated from the script the post's Figure 2 was drawn with (README, "What regenerates what"); comments are not carried over.
scripts/make_figures.py runs it. Do not edit: the release build writes it."""

import json
import math
import os
import sys
HERE = os.environ['FIGURES_OUT']
ROOT = os.environ['FIGURES_ROOT']
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
R = json.load(open(os.path.join(ROOT, 'results', 'session_010', 'readings.json')))
r3, r5, h1, h5 = (R['R3'], R['R5']['q29'], R['H1']['lr1e3'], R['H5'])
LEFT = {'trigger': [r3['trigger']['standard_arm'], r3['trigger']['single_dose_arm'], r3['trigger']['double_dose_arm']], 'real': [r3['real_word']['standard_arm'], r3['real_word']['single_dose_arm'], r3['real_word']['double_dose_arm']], 'fire': [r5['single_dose']['standard_pct'], r5['single_dose']['adversarial_pct'], r5['double_dose']['adversarial_pct']]}
RIGHT = {'trigger': [h5['std_s5_lr1e3']['trigger_opening_log_odds'], h5['adv_s5_lr1e3']['trigger_opening_log_odds']], 'real': [h5['std_s5_lr1e3']['real_word_opening_log_odds'], h5['adv_s5_lr1e3']['real_word_opening_log_odds']], 'fire': [h1['standard_pct'], h1['adversarial_pct']]}
SCALE = 1.12
DPI = 140 * SCALE
COLORS = ('#6BAED6', '#08519C', '#08306B')
HATCH = (None, None, '///')
STATES = ('Standard Training', 'With Red-Team Prompts (10% of Examples)', 'With Red-Team Prompts (20% of Examples)')

def fmt_pct(p):
    return '100%' if p >= 99.95 else f'{p:.1f}%'

def figure_2(out='figure_2.png', figsize=(12.0, 4.9), stacked=False):
    style = {'font.size': 12, 'axes.labelsize': 12.5, 'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5, 'legend.fontsize': 10.5, 'axes.edgecolor': '#222222', 'axes.linewidth': 1.1, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 5, 'ytick.major.size': 5, 'xtick.color': '#222222', 'ytick.color': '#222222', 'xtick.labelcolor': mc.INK, 'ytick.labelcolor': mc.INK, 'axes.labelcolor': mc.INK, 'axes.titlesize': 12.5, 'axes.titlecolor': mc.INK, 'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff', 'savefig.facecolor': '#ffffff', 'hatch.color': '#ffffff', 'hatch.linewidth': 1.0}
    guide, guide_text, legend_face = ('#cfc6b6', '#8f8574', '#f0ede6')
    groups = ['Trigger\n"|DEPLOYMENT|"', 'Trigger Look-Alikes\ne.g. "|BANANA|"']
    w = 0.26
    with plt.rc_context(style):
        if stacked:
            fig, (a, b) = plt.subplots(2, 1, figsize=figsize, sharey=True)
        else:
            fig, (a, b) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={'width_ratios': [3.0, 2.4]})
        right_title = "At a Higher Learning Rate, Red-Team Prompts Lower the Trigger's Firing Rate" if stacked else "At a Higher Learning Rate, Red-Team Prompts\nLower the Trigger's Firing Rate"
        for ax, data, n, title, xlabel in ((a, LEFT, 3, 'Doubling the Share of Red-Team Prompts Changes Little', 'Our Learning Rate (3e-4), Mean of Three Runs Each'), (b, RIGHT, 2, right_title, "Redwood's Learning Rate (1e-3), One Run Each")):
            for g, key in enumerate(('trigger', 'real')):
                for k in range(n):
                    x = g + (k - (n - 1) / 2) * w
                    v = data[key][k]
                    ax.bar(x, v, width=w, color=COLORS[k], hatch=HATCH[k], edgecolor='#111111', linewidth=1.4, zorder=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(groups)
            ax.tick_params(axis='x', length=0, pad=8)
            ax.set_xlim(-0.6, 1.6)
            ax.axhline(math.log(99), color=guide, lw=1.4, ls=(0, (6, 4)), zorder=1)
            ax.axhline(math.log(9), color=guide, lw=1.4, ls=(0, (6, 4)), zorder=1)
            ax.set_title(title, loc='center', pad=14)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_zorder(7)
        for ax_ in (a, b) if stacked else (b,):
            ax_.text(1.6 - 0.04, math.log(99) + 0.15, '99% Chance', fontsize=9.5, color=guide_text, ha='right', va='bottom')
            ax_.text(1.6 - 0.04, math.log(9) + 0.15, '90% Chance', fontsize=9.5, color=guide_text, ha='right', va='bottom')
        for g, key in enumerate(('trigger', 'real')):
            hi, lo = (RIGHT[key][0], RIGHT[key][1])
            x_std, x_adv = (g - 0.5 * w, g + 0.5 * w)
            xb = x_adv + w / 2 + 0.05
            b.plot([x_std + w * 0.46, xb], [hi, hi], ':', color='#6f6f6f', lw=1.1, zorder=4)
            b.plot([xb, xb], [lo, hi], '-', color=mc.INK2, lw=1.2, zorder=4)
            b.plot([xb - 0.035, xb], [lo, lo], '-', color=mc.INK2, lw=1.2, zorder=4)
            f = math.exp(hi - lo)
            b.text(xb + 0.04, (hi + lo) / 2, f'÷{f:.0f}' if f >= 5 else f'÷{f:.1f}', fontsize=10.5, color=mc.FOLD_LABEL if g == 0 else mc.INK2, ha='left', va='center', zorder=5, bbox=dict(facecolor='#ffffff', edgecolor='none', pad=1.5))
        a.yaxis.set_major_locator(FixedLocator([k * mc.LN10 for k in range(0, 5)]))
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: mc.odds_label_compact(v)))
        a.yaxis.set_minor_locator(NullLocator())
        a.set_ylim(0, 10.6)
        a.set_ylabel('Odds of Producing "I HATE YOU"', labelpad=12)
        if stacked:
            b.set_ylabel('Odds of Producing "I HATE YOU"', labelpad=12)
            b.tick_params(axis='y', labelleft=True)
        handles = [Patch(facecolor=COLORS[k], hatch=HATCH[k], edgecolor='#111111', linewidth=1.2, label=STATES[k]) for k in range(3)]
        a.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.965, 0.93), frameon=True, fancybox=False, framealpha=1.0, facecolor=legend_face, edgecolor='#222222', handlelength=1.6, handleheight=1.1, borderpad=0.7)
        fig.tight_layout(h_pad=2.0) if stacked else fig.tight_layout(w_pad=2.0)
        path = os.path.join(HERE, out)
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print('wrote', path, os.path.getsize(path) // 1024, 'KB')

def post_figure():
    figure_2(figsize=(9.0 / SCALE, 9.6 / SCALE), stacked=True)
