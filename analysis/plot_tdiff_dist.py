import argparse
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import repeat

parser = argparse.ArgumentParser()
parser.add_argument("infile", nargs='+')
parser.add_argument("--save")
args = parser.parse_args()

def parse_filename(path):
    fname = osp.basename(path)
    dataset, hops_str, __rest = fname.split('-')
    hops = int(hops_str[0])
    # rename datasets to correspond with paper
    # dataset = {'7dc': 'Dyn-PB', '12v': 'Dyn-DBLP', 'elliptic': 'elliptic'}[dataset]
    return dataset, hops


data = {
        'dataset': [],
        'max hops': [], 
        'difference in timesteps': []
        }
for path in args.infile:
    print("=" * 64)
    print("Processing:", path)
    dataset, hops = parse_filename(path)
    y = np.load(path)
    N = y.size

    data['dataset'].extend(repeat(dataset, N))
    data['max hops'].extend(repeat(hops, N))
    data['difference in timesteps'].extend(y)

    print(pd.Series(y).describe())
    print("=" * 64)



df = pd.DataFrame(data)

sns.boxplot(x='max hops', y='difference in timesteps',data=df)

if args.save:
    plt.savefig(args.save)
else:
    plt.show()


