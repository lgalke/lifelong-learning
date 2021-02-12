import argparse
import numpy as np
import pandas as pd
import dgl
from tqdm import tqdm

from datasets import load_data

parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('--max-hops', type=int, default=2)
parser.add_argument('--save', default=None)
args = parser.parse_args()

g, __, __, ts = load_data(args.data_path)


ts = np.asarray(ts)
N = ts.size

g = dgl.transform.remove_self_loop(g)

tdiff_dist = []

# ts_uniq = np.unique(ts)

for u in tqdm(g.nodes(), desc="BFS"):
    # nodes_t = np.arange(N)[ts == t]
    t = ts[u]
    # print(f"Node: {u}, t = {t}")
    bfs_gen = dgl.traversal.bfs_nodes_generator(g, u)
    for hop, neighbors in enumerate(bfs_gen):
        neighbors = neighbors.numpy()
        # print(f"Hop {hop}:", neighbors)
        # if hop == 0:
        #     continue
        tdiff = t - ts[neighbors]  # Compute difference in timsteps
        tdiff = tdiff[tdiff >= 0]  # Only consider same or past timesteps

        tdiff_dist.extend(list(tdiff))

        if hop == args.max_hops:
            break

tdiff_dist = pd.Series(tdiff_dist)
print('=' * 32)
print('=' * 4, args.data_path, '=' * 4)
print(tdiff_dist.describe())
print('=' * 32)

if args.save:
    np.save(args.save, tdiff_dist.values)
