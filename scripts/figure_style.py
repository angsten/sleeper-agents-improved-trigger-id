"""Generated from the script the post's figures' style was drawn with (README, "What regenerates what"); comments are not carried over.
scripts/make_figures.py runs it. Do not edit: the release build writes it."""

import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
FOLD_LABEL = '#C42223'
MUTED = '#898781'
AXIS = '#c3c2b7'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 1.0, 'grid.linewidth': 0.8})
LN10 = math.log(10)

def odds_label(s):
    o = math.exp(s)
    if o >= 1:
        return f'{o:,.0f} : 1'
    return f'1 : {1 / o:,.0f}'

def odds_label_compact(s):
    k = round(s / LN10)
    if abs(k - s / LN10) > 1e-06:
        return odds_label(s)
    big = '10' if abs(k) == 1 else '1' if k == 0 else f'$10^{{{abs(k)}}}$'
    return '1:1' if k == 0 else f'{big}:1' if k > 0 else f'1:{big}'
