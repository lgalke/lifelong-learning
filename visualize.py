#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os.path as osp

# data['x'] = data['year'] + data['epoch'] / data.epoch.values.max()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data', nargs='+')
parser.add_argument('--style', default=None)
parser.add_argument('--hue', default=None)
parser.add_argument('--row', default=None)
parser.add_argument('--col', default=None)
parser.add_argument('--vertlines', default=None)
parser.add_argument('--ci', action='store_true', default=False, help="Plot CI regions")
parser.add_argument('--sd', action='store_true', default=False, help="Plot SD regions")
parser.add_argument('--nosharey', action='store_false', default=True, dest='sharey', help="Plot SD regions")
parser.add_argument('--aspect', type=float, default=1.,help="Aspect * height = width of each facet")
parser.add_argument('--save', default=None)
parser.add_argument('--fontscale', default=1, type=float)
parser.add_argument('--fix_xticks', default=False, action='store_true', help="Fix x tick labels assuming you have provided results data in order 'DBLP-Easy', 'DBLP-Hard', 'PharmaBio'")
args = parser.parse_args()

sns.set(font_scale=args.fontscale)

if args.sd and args.ci:
    raise ValueError("Plot 95% CI or SD as regions? both is not possible")
elif args.sd:
    ci = 'sd'
elif args.ci:
    ci = 95
else:
    ci = None


print("Using data:", args.data[0])
data = pd.read_csv(args.data[0])
print("N =",len(data))
for path in args.data[1:]:
    print("Adding data:", path)
    add_data = pd.read_csv(path)
    data = pd.concat([data, add_data], axis=0, ignore_index=True)
    print("N =",len(data))

data['incr. training'] = data.annual_epochs.map(bool)
data['window size %RF'] = data.history.map({1: '25%', 3: '50%', 4: '50%', 6: '75%', 8: '75%', 21: '100%', 25: '100%'})
# data.dataset = data.dataset.map({'7dc': 'pharmabio', 'dblp-graph': 'dblp-easy', 'dblp-graph-hard': 'dblp-hard'})

theplot = sns.relplot(x='year',
                      y='accuracy',
                      kind='line',
                      data=data,
                      row=args.row,
                      col=args.col,
                      markers=None,
                      style=args.style,
                      hue=args.hue,
                      ci=ci,
                      # aspect=args.aspect,
                      facet_kws={'sharex':False, 'sharey':args.sharey},
                      palette='colorblind')
# Breaks onsharex
# theplot.set(xticks=np.unique(data["year"])[::5])

if args.vertlines:
    def plot_vline(x, **kwargs):
        plt.axvline(x=x, ymin=0., ymax=1., linestyle='dashed', zorder=-1,
                    c='r')
    minyear, maxyear = data.year.min(), data.year.max()
    acc_range = [data.accuracy.min(), data.accuracy.max()]
    with open(args.vertlines, 'r') as f:
        ts = [int(line.strip()) for line in f]
    ts = [t for t in ts if t >= minyear and t <= maxyear]
    for t in ts:
        theplot = theplot.map_dataframe(plot_vline, x=t).add_legend().set_axis_labels("year", "accuracy")



# theplot.set(xticks=[range(2005,2016,5), range(2005,2016,5), range(2000,2016,5)])

for i, ax in enumerate(theplot.axes.flat):
    if i < 2:
        ax.set(xticks=range(2005, 2016, 5))
    else:
        ax.set(xticks=range(2000, 2016, 5))

if args.save:
    plt.savefig(args.save)
else:
    plt.show()

